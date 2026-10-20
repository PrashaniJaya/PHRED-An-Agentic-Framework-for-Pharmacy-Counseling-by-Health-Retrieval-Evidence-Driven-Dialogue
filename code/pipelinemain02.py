from collections import deque
from sentence_transformers import SentenceTransformer, util
from RAGagentmultianswer import JSONRAGChroma
from KGfacts import KGAgent, Neo4jClient
from decomposeragent import DecomposerAgent
from openai import OpenAI
import os
from collections import deque
from riskagent import RiskAssessmentAgent
#from followup import SemanticStateTracker
from followup import FollowUpAgent
from followup import HTA_RULES
import uuid
from datetime import datetime
import json
import logging
logging.getLogger("chromadb").setLevel(logging.ERROR)
from followup import FollowUpAgent
from followup import SemanticStateTracker
# ------------------------------
# Goal classification examples
# ------------------------------
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

# ------------------------------
# Embedding model for semantic classification
# ------------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")
goal_embeddings = {
    goal: model.encode(examples, convert_to_tensor=True)
    for goal, examples in knowledge_base.items()
}

def classify_goal(question: str) -> str:
    q_emb = model.encode(question, convert_to_tensor=True)
    best_goal, best_score = None, -1
    for goal, emb_list in goal_embeddings.items():
        sim = util.cos_sim(q_emb, emb_list).max().item()
        if sim > best_score:
            best_goal, best_score = goal, sim
    return best_goal if best_goal else "educate"

# ------------------------------
# Rule-based synthesis
# ------------------------------
def synthesize_response(goal, rag_text, kg_text):
    #print("Rag answer", rag_text, "kganswer", kg_text)
    if not rag_text and not kg_text:
        return "Sorry, I don’t have enough information to answer that."

    if goal == "educate":
        return f"Explanation:\n{rag_text and kg_text}"

    elif goal == "warn":
        return f"Important safety information:\n{kg_text and rag_text}"

    elif goal == "suggest":
        parts = []
        if rag_text:
            parts.append(f"Based on medical dialogues: {rag_text}")
        if kg_text:
            parts.append(f"From medical facts: {kg_text}")
        return "\n\n".join(parts)

    return rag_text or kg_text

# ------------------------------
# LLM-polished synthesis
# ------------------------------
def llm_polish(goal, user_input, rag_text, kg_text, llm_model="gpt-4o-mini"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context = f"Retrieved answers:\n{rag_text}\n\n{kg_text}"

    prompt = f"""
You are a concise and factual medical assistant.

The user asked: "{user_input}"
Their intent (goal) is: {goal}.

You have multiple retrieved answers from different sources below.

You have multiple retrieved answers from different sources below.

Your task:
- Combine these answers into one coherent, **concise**, and **evidence-based** explanation.
- **Mimic the tone, phrasing, and sentence structure** of the retrieved answers.
- Use **the same medical terminology and formal tone** seen in those answers.
- Avoid long empathetic introductions or numbered lists.
- Merge overlapping information naturally and summarize key points into 3–5 sentences.
- ONLY use the retrieved information below; do not invent or add knowledge beyond it.
- If uncertain, mention that explicitly.

Retrieved context:
{context}
"""


    resp = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip() # type: ignore

#def llm_polish(goal, user_input, rag_text, kg_text, llm_model="gpt-4o-mini"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context = f"Retrieved answers:\n{rag_text}\n\n{kg_text}"
    prompt = f"""
You are a helpful medical assistant.

The user asked: "{user_input}"
Their intent (goal) is: {goal}.

Please COMBINE and SUMMARIZE them into a SINGLE clear, concise, and user-friendly explanation.

Rules:
- Do NOT repeat the same points.
- Merge overlapping information naturally.
- Keep it medically accurate and easy to read but use any meidcal terms mentioned to address the answer.
- If uncertain, mention that explicitly.
- ONLY use context answers and not your training based output
- ONLY use the retrieved information below to answer. Do not add knowledge beyond the sources of information in context.
{context}
"""

    resp = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip() # type: ignore

#def llm_polish(goal, user_input, rag_text, kg_text, llm_model="gpt-4o-mini"):
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    context = f"Retrieved answers:\n{rag_text}\n\n{kg_text}"
    prompt = f"""
You are a helpful medical assistant writing a single, medically detailed final answer.

User question:
"{user_input}"
Intent (goal): {goal}

You have multiple retrieved answers from different sources below.
Combine them into one coherent, evidence-based explanation.

Guidelines:
- Use the medical details present in the sources (drug names, hormone names, time windows, dosages, test names, numbers, etc.).
- Preserve quantitative and pharmacological information; do not summarize it away.
- Merge overlapping ideas naturally and order them logically (cause → effect → recommendation).
- If any uncertainty exists, mention it explicitly and suggest a clinical next step.
- Maintain a calm, professional tone like a real doctor explaining to a patient.
- Be concise but *not* at the cost of losing medical specificity.
- Use only the retrieved information below; do not add new facts beyond the sources.

Retrieved context:
{context}
"""

    resp = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=350,
    )
    return resp.choices[0].message.content.strip() # type: ignore

from collections import deque
from datetime import datetime
import json

from datetime import datetime
import json
from collections import deque

def chat_loop(log_file="chat_log_pipelinemain02.json"):
    print("💊 Medical QA Chat (type 'exit' to quit)\n")
    history = deque(maxlen=10)

    # --- Init agents ---
    urls = [
    ("Morning-after pill guide", "https://www.drugs.com/mtm/morning-after.html"),
    ("Emergency contraception info", "https://www.drugs.com/condition/postcoital-contraception.html"),
    ]

    rag = JSONRAGChroma(
        kb_file="meddialog_test.json",  # your full MedDialog or subset JSON
        answer_fields=["answer_icliniq", "answer_chatdoctor"],
        mode="concat",
        urls=urls,
        use_llm=False  # or True if you want synthesized answers
    )
    #rag = JSONRAGChroma(kb_file="meddialog.json", answer_fields=["answer_chatgpt", "answer_icliniq", "answer_chatdoctor"], mode="concat", use_llm=True)
    client = Neo4jClient("bolt://localhost:7687", "neo4j", "morningafter_001")
    kg = KGAgent(client)
    decomposer = DecomposerAgent(use_llm=True)
    risk_agent = RiskAssessmentAgent()
    state_tracker = SemanticStateTracker()
    followup_agent = FollowUpAgent(HTA_RULES, state_tracker)

    log_data = {
        "session_start": datetime.now().isoformat(),
        "turns": [],
        "followups_suggested": [],
        "user_followup_decision": None,
    }

    print("Type your medical question below.\n")

    # --- Main chat loop ---
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("\n🩺 Checking if we’ve covered everything important...")
            break

        # --- Step 1: Decompose if complex ---
        sub_questions = decomposer.decompose(user_input)
        sub_answers = []

        # --- Step 2: Answer each sub-question ---
        for sub_q in sub_questions:
            goal = classify_goal(sub_q)

            rag_resp = rag.answer(sub_q)
            kg_resp = kg.answer(sub_q)
            sub_answer = synthesize_response(goal, rag_resp, kg_resp)
            final_answer = risk_agent.manage_response(sub_q, sub_answer, medicine_class=None)
            sub_answers.append(f"A ({goal}): {final_answer}")

        # --- Step 1: Decompose if complex ---
        if len(user_input.split(".")) > 2:
            sub_questions = decomposer.decompose(user_input)
            full_context_query = " ".join(sub_questions)
        else:
            full_context_query = user_input

        # --- Step 2: Unified retrieval ---
        goal = classify_goal(full_context_query)
        rag_resp = rag.answer(full_context_query)
        rag_resp, rag_docs = rag.answer(full_context_query)
        kg_resp = kg.answer(full_context_query)

        # --- Step 3: Unified synthesis ---
        #sub_answers = synthesize_response(goal, rag_resp, kg_resp)
        #final_answer = risk_agent.manage_response(full_context_query, sub_answers, medicine_class=None)

        # --- Step 4: Polish ---
        #polished = llm_polish(
        #    goal="mixed",
        #    user_input=user_input,
        #    rag_text=final_answer,
        #    kg_text="",
        #)

        # --- Step 3: Merge & polish ---
        def strip_qa_labels(text: str) -> str:
            return text.replace("Q:", "").replace("A (educate):", "").replace("A (warn):", "").replace("A (suggest):", "").strip()

        cleaned = [strip_qa_labels(a) for a in sub_answers]
        merged = "\n".join(cleaned)

        polished = llm_polish(
            goal="mixed",
            user_input=user_input,
            rag_text=merged,
            kg_text=""
        )

        final_output = risk_agent.manage_response(user_input, polished, medicine_class=None)
        print(f"\n💬 Combined final answer:\n{final_output}\n")

        # --- Step 4: Log this turn ---
        log_data["turns"].append({
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "goal": goal,
            "retrieved_docs": rag_docs,  # <-- structured list from RAG
            "rag_text": rag_resp,        # <-- the combined raw text
            "final_output": polished,
        })

        history.append((user_input, final_output))

    # === After user types "exit" ===
    print("\n🧩 Analysing conversation for uncovered HTA topics...\n")

    # Combine the conversation into a plain text transcript
    chat_history_text = "\n".join([f"You: {q}\nBot: {a}" for q, a in history])

    # --- Step 5: Use LLM to find uncovered topics per goal ---
    for goal in ["educate", "warn", "suggest"]:
        followups = followup_agent.propose_followups(goal, chat_history_text)
        if followups:
            print(f"\n📘 Suggested follow-ups for {goal}:")
            for f in followups:
                print(f"  - {f}")
            log_data["followups_suggested"].append({goal: followups})

    # --- Step 6: Ask user if they want follow-ups ---
    if log_data["followups_suggested"]:
        reply = input("\nWould you like to go through these follow-ups before ending? (yes/no): ").strip().lower()
        if reply in ["yes", "y", "sure", "ok"]:
            log_data["user_followup_decision"] = "accepted"
            for group in log_data["followups_suggested"]:
                for goal, prompts in group.items():
                    for p in prompts:
                        print(f"\n💬 {p}")
                        follow_q = f"{p} (related to {goal})"
                        rag_resp = rag.answer(follow_q)
                        kg_resp = kg.answer(follow_q)
                        sub_ans = synthesize_response(goal, rag_resp, kg_resp)
                        print(f"🧠 Answer: {sub_ans}\n")

                        log_data["turns"].append({
                            "timestamp": datetime.now().isoformat(),
                            "followup_goal": goal,
                            "followup_prompt": p,
                            "followup_answer": sub_ans
                        })
        else:
            log_data["user_followup_decision"] = "declined"
            print("\n👍 No problem. Ending session.\n")
    else:
        print("\n✅ Looks like we’ve covered all important HTA topics. Great job!\n")
        log_data["user_followup_decision"] = "all_covered"

    # --- Step 7: Save log ---
    log_data["session_end"] = datetime.now().isoformat()
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"📝 Conversation saved to {log_file}")

    client.close()


# ------------------------------
# # Run
# # ------------------------------
if __name__ == "__main__":
    chat_loop()



        