from collections import deque
from sentence_transformers import SentenceTransformer, util
from RAGagentmultianswer import JSONRAGChroma
from KGfacts import KGAgent, Neo4jClient
from decomposeragent import DecomposerAgent
from openai import OpenAI
import os
from collections import deque
from riskagent import RiskAssessmentAgent
from followup import SemanticStateTracker
from followup import FollowUpAgent
from followup import HTA_RULES
import json
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

    context = f"RAG answer:\n{rag_text}\n\nKG answer:\n{kg_text}"
    prompt = f"""
You are a helpful medical assistant. ONLY use the retrieved information below to answer. Do not add knowledge beyond the sources of information.
The user asked: "{user_input}"
Their intent (goal) is: {goal}.

Here are two sources of information:
{context}

Please write a clear, concise, and user-friendly answer.
- Blend both sources if relevant.
- Keep it medically accurate but easy to read.
- If uncertain, say so.
- Always end with: "This information is for educational purposes only and not a substitute for professional medical advice."
"""

    resp = client.chat.completions.create(
        model=llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip() # type: ignore


# --- Helper: classify follow-up reply intent ---
def parse_followup_reply(reply: str):
    """Classify reply as 'affirmative', 'new_question', or 'decline'."""
    text = reply.lower().strip()

    affirmations = ["yes", "yeah", "yep", "sure", "ok", "okay", "please", "go ahead"]
    negatives = ["no", "nah", "nope", "skip", "not now", "later"]

    if any(a in text for a in affirmations):
        # If it's long and has more than 3 words, treat as new question with context
        if len(text.split()) > 3:
            return "new_question"
        return "affirmative"

    if any(n in text for n in negatives):
        return "decline"

    # default: probably a new question or unclear statement
    return "new_question"


# --- Main chat loop ---
def chat_loop(log_file="chat_log.json"):
    print("Medical QA Chat (type 'exit' to quit)\n")
    history = deque(maxlen=10)

    # --- Init agents ---
    rag = JSONRAGChroma(kb_file="meddialog.json", use_llm=True)
    client = Neo4jClient("bolt://localhost:7687", "neo4j", "morningafter_001")
    kg = KGAgent(client)
    decomposer = DecomposerAgent(use_llm=False)
    risk_agent = RiskAssessmentAgent()
    state = SemanticStateTracker()
    followup_agent = FollowUpAgent(HTA_RULES, state)

    log_data = []

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            print(" Goodbye!")
            break

        # --- Step 1: Decompose ---
        sub_questions = decomposer.decompose(user_input)

        sub_answers = []
        detected_risks = []
        followups_triggered = []

        # --- Step 2: Process each sub-question ---
        for sub_q in sub_questions:
            goal = classify_goal(sub_q)
            state.initialize_goal(goal, HTA_RULES)

            rag_resp = rag.answer(sub_q)
            kg_resp = kg.answer(sub_q)
            sub_answer = synthesize_response(goal, rag_resp, kg_resp)

            # --- Step 3: Risk Assessment ---
            final_answer = risk_agent.manage_response(sub_q, sub_answer, medicine_class=None)

            #print(f"\nBot ({goal}):\n{final_answer}\n")

            sub_answers.append(f"Q: {sub_q}\nA ({goal}): {final_answer}")
            detected_risks.extend(risk_agent.assess_risk(sub_q + " " + final_answer))

            # --- Step 4: Update semantic follow-up state ---
            full_text = sub_q + " " + final_answer
            followup_agent.update_state(goal, full_text)

            # --- Step 5: Generate follow-up prompts ---
            prompts = followup_agent.propose_followups(goal)
            if prompts:
                for p in prompts:
                    print(f"{p}")
                    reply = input("You: ")
                    if reply.strip().lower() == "exit":
                        print(" Goodbye!")
                        client.close()
                        exit(0) 
                    reply_type = parse_followup_reply(reply)

                    if reply_type == "affirmative":
                        sub_followup = f"{p} about {sub_q}"
                        rag2 = rag.answer(sub_followup)
                        kg2 = kg.answer(sub_followup)
                        sub_ans = synthesize_response(goal, rag2, kg2)
                        #print(f"Bot: {sub_ans}\n")

                        # mark topic covered
                        for topic in state.topics[goal]:
                            if topic in p.lower():
                                state.mark_topic_covered(goal, topic)

                        followups_triggered.append(p)

                    elif reply_type == "new_question":
                        print("Bot: I see you added more context — let me answer that specifically.\n")
                        # Treat reply as a new question
                        rag2 = rag.answer(reply)
                        kg2 = kg.answer(reply)
                        sub_ans = synthesize_response(goal, rag2, kg2)
                        #print(f"Bot: {sub_ans}\n")

                    else:  # decline
                        print("Okay, skipping that for now.\n")

        # --- Step 6: Combine all sub-answers ---
        if sub_answers:
            combined_output = "\n\n".join(sub_answers)
            print(f"\n Summary:\n{combined_output}\n")
        else:
            print("\n No answers were generated for this query.\n")

        # --- Step 7: Log conversation turn ---
        log_data.append({
            "user_input": user_input,
            "sub_questions": sub_questions,
            "final_answer": combined_output,
            "risks_detected": detected_risks,
            "followups_triggered": followups_triggered
        })           
        history.append((user_input, combined_output))

    # --- Save logs on exit ---
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)
    print(f"Conversation saved to {log_file}")

    client.close()

# ------------------------------
# # Run
# # ------------------------------
if __name__ == "__main__":
    chat_loop()
