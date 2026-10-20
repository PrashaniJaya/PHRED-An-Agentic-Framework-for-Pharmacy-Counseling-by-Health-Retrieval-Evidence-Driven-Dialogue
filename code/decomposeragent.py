# decomposer_agent.py
import re
from openai import OpenAI
import os

class DecomposerAgent:
    def __init__(self, use_llm=True, llm_model="gpt-4o-mini"):
        self.use_llm = use_llm
        self.llm_model = llm_model
        if use_llm:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def rule_based(self, question: str) -> list[str]:
        parts = re.split(r"\band\b|\bor\b|\?", question, flags=re.IGNORECASE)
        parts = [p.strip() for p in parts if p.strip()]
        return [p if p.endswith("?") else p + "?" for p in parts]

    def semantic_llm(self, question: str) -> list[str]:
        prompt = f"""
        Break down the following medical question into the smallest 
        possible sub-questions. Each sub-question must be independently 
        answerable. Return them as a numbered list.

        Question: "{question}"
        """
        resp = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=200,
        )
        output = resp.choices[0].message.content.strip() # type: ignore
        sub_questions = [line.split(".", 1)[-1].strip() for line in output.split("\n") if line.strip()]
        return sub_questions

    def decompose(self, question: str, num_entities: int = 1) -> list[str]:
        # Rule-based for obvious multi-question
        if question.count("?") > 1 or re.search(r"\b(and|or)\b", question, flags=re.IGNORECASE):
            return self.rule_based(question)

        # Semantic fallback
        if self.use_llm and len(question.split()) > 15 and num_entities > 1:
            return self.semantic_llm(question)

        # Default: no split
        return [question]
