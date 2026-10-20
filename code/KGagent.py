"""
KG Agent — Neo4j + Cypher RAG (Domain-Aware + LLM Summarizer, fixed OpenAI client)

Fixes in this version
----------------------
- Use `load_dotenv()` + `OpenAI` client consistently (no `openai.*` legacy calls)
- Separate `Neo4jClient` (graph) and `llm_client` (LLM)
- Correctly handle new OpenAI SDK response format (no subscripting)

Requirements
-------------
- pip install python-dotenv openai neo4j rapidfuzz fastapi
- Add `.env` file with `OPENAI_API_KEY=sk-xxxx`

Run
---
python KGagent.py "what is ella"
"""

import os
from typing import List, Dict, Any
from dataclasses import dataclass
from neo4j import GraphDatabase
from rapidfuzz import process, fuzz
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv
from openai import OpenAI

# -------------------- Load env --------------------
load_dotenv()

# Neo4j config
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "morningafter_001")

# LLM client
llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_SEED_ENTITIES = 3
EXPANSION_DEPTH = 2
MAX_FACTS = 50

REL_TEMPLATES = {
    "HAS_INGREDIENT": "{h} contains {t}",
    "HAS_MECHANISM": "{h} works by {t}",
    "HAS_EFFECTIVENESS": "{h} is effective {t}",
    "HAS_SIDE_EFFECT": "{h} may cause {t}",
    "NOT_RECOMMENDED_FOR": "{h} is not recommended for {t}",
}

# -------------------- Neo4j Client --------------------
@dataclass
class Neo4jClient:
    uri: str
    user: str
    password: str

    def __post_init__(self):
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))

    def close(self):
        self.driver.close()

    def sample_entity_names(self, limit: int = 2000) -> List[str]:
        q = """
        MATCH (e) WHERE e.name IS NOT NULL
        RETURN toLower(e.name) AS name
        LIMIT $limit
        """
        with self.driver.session() as s:
            return [r["name"] for r in s.run(q, limit=limit)]

    def expand_neighborhood(self, name: str, depth: int = 2, max_facts: int = 50) -> List[Dict[str, Any]]:
        q = f"""
        MATCH (seed {{name:$name}})
        CALL (seed) {{
          WITH seed
          MATCH (seed)-[r*1..{depth}]-(n)
          WITH seed, n LIMIT 50
          OPTIONAL MATCH path=(seed)-[re*1..{depth}]-(n)
          UNWIND re AS rel
          WITH DISTINCT startNode(rel) AS h, rel, endNode(rel) AS t
          RETURN toLower(h.name) AS h, type(rel) AS r, toLower(t.name) AS t, properties(t) AS t_props
        }}
        RETURN h, r, t, t_props LIMIT $max_facts
        """
        with self.driver.session() as s:
            rows = s.run(q, name=name.lower(), max_facts=max_facts)
            out = []
            for r in rows:
                out.append({
                    "h": r["h"],
                    "r": r["r"],
                    "t": r["t"],
                    "t_props": r["t_props"],
                })
            return out

    def find_nearest_entity_names(self, query: str, topk: int = 5) -> List[str]:
        names = self.sample_entity_names(limit=5000)
        scored = process.extract(query.lower(), names, scorer=fuzz.WRatio, limit=topk)
        return [n for n, score, _ in scored if score >= 70]

# -------------------- Answer Composer --------------------

def fact_to_text(h: str, r: str, t: str, t_props: Dict[str, Any]) -> str:
    if not h or not t or not r:
        return None
    if r == "HAS_EFFECTIVENESS" and t_props and "timeframe" in t_props:
        return f"{h.title()} is effective if taken {t_props['timeframe']}"
    tmpl = REL_TEMPLATES.get(r)
    if tmpl:
        return tmpl.format(h=h.title(), t=t.title())
    return f"{h.title()} {r.lower()} {t.title()}"


def dedup_facts(facts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out = []
    for f in facts:
        key = (f["h"], f["r"], f["t"])
        if key not in seen:
            seen.add(key)
            out.append(f)
    return out

# -------------------- LLM Helper --------------------

def llm_generate(question: str, facts: List[str]) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        return None

    prompt = "You are a medical assistant. Answer the question using only the provided FACTS.\n\n"
    for f in facts:
        prompt += f"- {f}\n"
    prompt += "\nNow write a concise answer:\n"

    try:
        resp = llm_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a careful assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[LLM error: {e}]"

# -------------------- Agent --------------------
class KGAgent:
    def __init__(self, client: Neo4jClient):
        self.client = client

    def answer(self, question: str) -> str:
        q = question.strip().lower()
        if not q:
            return "Please provide a question."

        candidates = [q] + [t.strip(",.?! ") for t in q.split()]
        seed_names: List[str] = []
        all_names = self.client.sample_entity_names(limit=5000)
        for cand in candidates:
            best = process.extractOne(cand, all_names, scorer=fuzz.WRatio)
            if best and best[1] >= 80:
                seed_names.append(best[0])
        seed_names = list(dict.fromkeys(seed_names))

        if not seed_names:
            return "I couldn’t find any entities in the graph for that question."

        all_facts: List[Dict[str, Any]] = []
        for name in seed_names[:MAX_SEED_ENTITIES]:
            facts = self.client.expand_neighborhood(name, depth=EXPANSION_DEPTH, max_facts=MAX_FACTS)
            all_facts.extend(facts)
        all_facts = dedup_facts(all_facts)

        if not all_facts:
            return "I couldn’t find supporting facts in the graph."

        fact_texts = []
        for f in all_facts:
            text = fact_to_text(f.get("h"), f.get("r"), f.get("t"), f.get("t_props"))
            if text:
                fact_texts.append(text)

        llm_ans = llm_generate(question, fact_texts)
        if llm_ans:
            return llm_ans

        return "Here are the facts I found:\n" + "\n".join(["- " + f for f in fact_texts])

# -------------------- CLI & API --------------------
client = Neo4jClient(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
agent = KGAgent(client)

class AskBody(BaseModel):
    question: str

app = FastAPI()

@app.post("/ask")
def ask(body: AskBody):
    return {"answer": agent.answer(body.question)}

if __name__ == "__main__":
    import sys
    q = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Question: ")
    print("\n" + agent.answer(q) + "\n")
    client.close()