import json
import os
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from neo4j import GraphDatabase
import re

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------
# Knowledge base for intent scoring
# -----------------------
knowledge_base = {
    "educate": [
        "How does ibuprofen reduce inflammation?",
        "What’s the difference between amoxicillin and ampicillin?",
        "Why is metformin the first-line treatment for type 2 diabetes?",
        "What does ‘extended-release’ mean in medications?"
    ],
    "warn": [
        "What are the side effects of long-term steroid use?",
        "Can I drink alcohol while on metronidazole?",
        "Why is isotretinoin not safe in pregnancy?",
        "What happens if you take too much paracetamol?"
    ],
    "suggest": [
        "What over-the-counter options are there for mild seasonal allergies?",
        "If I get muscle pain from statins, what are my alternatives?",
        "What are non-drug ways to lower blood pressure?",
        "Could a topical cream work instead of oral antibiotics for acne?"
    ]
}

model = SentenceTransformer('all-MiniLM-L6-v2')
embedded_kb = {cat: model.encode(questions) for cat, questions in knowledge_base.items()}

def score_query(query):
    q_vec = model.encode([query])
    scores = {}
    for cat, vectors in embedded_kb.items():
        sims = cosine_similarity(q_vec, vectors)[0]
        scores[cat] = {
            "max_score": float(np.max(sims)),
            "closest_example": knowledge_base[cat][int(np.argmax(sims))]
        }
    return scores


# RAG Agent with URLs

class RAGAgent:
    def __init__(self, kb_file, urls=None, top_k=3, threshold=0.3):
        """
        kb_file: path to KB document
        urls: list of (text, url) tuples
        """
        with open(kb_file, "r", encoding="utf-8") as f:
            self.kb_entries = [line.strip() for line in f if line.strip()]
        
        # Append URLs if provided
        self.url_entries = []
        if urls:
            for text, url in urls:
                self.url_entries.append(f"{text} (Source: {url})")
        self.kb_entries += self.url_entries
        
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedded_kb = self.model.encode(self.kb_entries)
        self.top_k = top_k
        self.threshold = threshold

    def handle(self, question):
        q_vec = self.model.encode([question])
        sims = cosine_similarity(q_vec, self.embedded_kb)[0]
        top_indices = [i for i in sims.argsort()[::-1] if sims[i] >= self.threshold][:self.top_k]
        if not top_indices:
            return "RAGAgent: No relevant entries found."
        results = [self.kb_entries[i] + f" (score: {sims[i]:.2f})" for i in top_indices]
        return "\n".join(results)


# Rule-based agent

class RuleBasedAgent:
    def __init__(self):
        self.rules = {
            "effectiveness": "Most effective within 24 hours, can be taken up to 72 hours after unprotected sex.",
            "contraindications": "Avoid if allergic to levonorgestrel or ulipristal acetate.",
            "side_effects": "May cause nausea, fatigue, headache, or changes in menstrual cycle."
        }

    def handle(self, question):
        q_lower = question.lower()
        for key, rule in self.rules.items():
            if key in q_lower or any(word in q_lower for word in rule.lower().split()[:3]):
                return f"RuleBasedAgent ({key}): {rule}"
        return "RuleBasedAgent: No matching rule found."

# -----------------------
# KG Agent
# -----------------------
schema_info = """
Labels: Condition, Effectiveness, MEchanism, Medication, SideEffect, ActiveIngredient
Properties on Medication: name, timeframe
Relationships: CONTAINS, EFFECTIVE, EFFECTIVENESS_UNCHANGED_IN, HAS_SIDE_EFFECT, INEFFECTIVE_IF, USES, LESS_EFFECTIVE_IN
"""
import re

class KGAgent:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.cache = {}
        # Define possible relationships to try
        self.relationships = ["CONTAINS", "USES"]

    def generate_query(self, question):
        prompt = f"""
You are a Neo4j Cypher query generator.
Database schema info:
{schema_info}

Convert the following question into a Cypher query using only the labels, properties, and relationships above.
IMPORTANT: ONLY return the query. Do NOT include any explanation, natural language text, or Markdown.
Question: \"\"\"{question}\"\"\"
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a Neo4j query generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()

    def execute_query(self, cypher_query):
        with self.driver.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]

    def handle(self, question):
        if question in self.cache:
            return self.cache[question]

        cypher = self.generate_query(question)

        # Strip code blocks
        cypher = cypher.replace("```cypher", "").replace("```", "").strip()

        # Extract first valid Cypher statement
        match = re.search(r"(MATCH|CREATE|CALL|MERGE|UNWIND).*", cypher, flags=re.IGNORECASE | re.DOTALL)
        if match:
            cypher = match.group(0).strip()
        else:
            self.cache[question] = "KGAgent: No valid Cypher query generated."
            return self.cache[question]

        # LOG for debugging
        print("KGAgent executing Cypher:\n", cypher)

        results = []
        # Try multiple relationship types if applicable
        for rel in self.relationships:
            query_to_execute = re.sub(r":\w+", f":{rel}", cypher)
            try:
                data = self.execute_query(query_to_execute)
                if data:
                    results.extend(data)
            except Exception as e:
                print(f"KGAgent warning executing {rel}: {e}")

        if not results:
            results = "KGAgent: Query executed successfully but returned no results."

        self.cache[question] = results
        return results



# -----------------------
# Dynamic Agent Manager (LLM selects agents based on question + KB scores)
# -----------------------
class DynamicAgentManager:
    def __init__(self, agents, model_name="gpt-4o-mini"):
        self.agents = agents
        self.model_name = model_name

    def handle_query(self, question):
        # Step 1: Compute KB similarity scores
        similarity_scores = score_query(question)
        scores_str = "\n".join(
            f"- {cat}: max similarity {info['max_score']:.3f}, closest example: \"{info['closest_example']}\""
            for cat, info in similarity_scores.items()
        )

        # Step 2: Ask LLM which agents to call
        prompt = f"""
You are an AI planner that dynamically decides which agents to call for a medical question.
The available agents are: {list(self.agents.keys())}.
The user's question is:
\"\"\"{question}\"\"\"

Here are the intent similarity scores with example KB entries:
{scores_str}

Based on the question and these scores, select the most relevant agents to call.
Return a JSON array of agent names.
Example: ["educate", "warn"]
"""
        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful agent selection assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=100
        )

        try:
            selected_agents = json.loads(response.choices[0].message.content)
            if not isinstance(selected_agents, list):
                selected_agents = list(self.agents.keys())
        except Exception:
            selected_agents = list(self.agents.keys())

        # Step 3: Call selected agents
        results = []
        for agent_name in selected_agents:
            agent = self.agents.get(agent_name)
            if agent:
                results.append(f"{agent_name.upper()}:\n{agent.handle(question)}")

        return "\n---\n".join(results)

# -----------------------
# DEMO
# -----------------------
if __name__ == "__main__":
    urls = [
        ("Morning-after pill guide", "https://www.drugs.com/mtm/morning-after.html"),
        ("Emergency contraception info", "https://www.drugs.com/condition/postcoital-contraception.html")
    ]
    agents = {
        "educate": RAGAgent("KBdrug.txt", urls=urls),
        "warn": RuleBasedAgent(),
        "suggest": KGAgent("bolt://localhost:7687", "neo4j", "morningafter_001")
    }

    manager = DynamicAgentManager(agents)

    test_queries = [
        "Can I take the morning-after pill if I missed a dose of another medication?",
        "What is the morning after pill made of?",
        "What are the side effects of the morning after pill?"
    ]

    for q in test_queries:
        print(f"\nQuery: {q}\n")
        print(manager.handle_query(q))
        print("="*50)
