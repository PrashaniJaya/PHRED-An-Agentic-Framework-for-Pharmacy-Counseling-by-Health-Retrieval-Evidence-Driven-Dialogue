# -*- coding: utf-8 -*-
"""
Evaluate your RAG system on MedDialog-style data with RAGAS (LangChain version).

Inputs:
  1) meddialog_test.json — list[dict] with fields:
       - input: str (the question)
       - answer_icliniq: str (gold answer to use)
  2) model_log.json — list[dict] with fields:
       - question: str (should match `input`)
       - model_answer: str (your system answer)
       - retrieved_context: list[dict] with "text" (top-k = 8)

Outputs:
  - ragas_results.csv  (per-sample metrics)
  - ragas_results.json (aggregate + per-sample)
"""

import os
import json
import argparse
import pandas as pd
from typing import List, Dict, Any, Optional
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_correctness,
)

from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY, etc. from .env

# --- LangChain LLM + Embeddings (no LiteLLM) ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# For robust alignment if there are tiny text differences
from rapidfuzz import process, fuzz


# ----------------------------
# Loading helpers
# ----------------------------
def load_json_any(path: str):
    """
    Supports:
      - JSON array file
      - JSONL file (one JSON object per line)
    """
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out
    else:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert isinstance(data, list), "Expected a JSON array of rows"
            return data


# ----------------------------
# Normalization + joining
# ----------------------------
def norm_q(s: str) -> str:
    """Normalize question for stable joining."""
    return " ".join(s.strip().split()).lower()


def build_index_by_question(items: List[Dict[str, Any]], key: str):
    """
    Build a dict: normalized_question -> item
    If duplicates occur, keep the first (warn).
    """
    idx = {}
    for it in items:
        q = it.get(key, "")
        nq = norm_q(q)
        if not nq:
            continue
        if nq in idx:
            # You could log/print if needed
            pass
        idx[nq] = it
    return idx


def fuzzy_lookup(nq: str, candidates: List[str], threshold: int = 95) -> Optional[str]:
    """
    Fuzzy fallback: find closest candidate key if >= threshold similarity.
    """
    if not candidates:
        return None
    match, score, _ = process.extractOne(nq, candidates, scorer=fuzz.token_set_ratio)
    return match if score >= threshold else None


# ----------------------------
# Mapping to RAGAS schema
# ----------------------------
def to_contexts(retrieved_context_field) -> List[str]:
    """
    Convert your retrieved_context field into list[str] of text chunks.
    """
    ctx = []
    for d in (retrieved_context_field or []):
        if isinstance(d, dict):
            t = d.get("text", "")
            if t:
                ctx.append(t)
        elif isinstance(d, str):
            ctx.append(d)
    return ctx


def build_ragas_dataframe(
    gold_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
    k: Optional[int] = None,  # you can cut contexts to top-k if desired
) -> pd.DataFrame:
    """
    gold_rows: from meddialog_test.json (keys: input, answer_icliniq)
    pred_rows: from model_log.json (keys: question, model_answer, retrieved_context)
    """
    gold_idx = build_index_by_question(gold_rows, key="input")
    pred_idx = build_index_by_question(pred_rows, key="question")

    # For fuzzy fallback:
    pred_keys = list(pred_idx.keys())

    ragas_rows = []
    missing = 0
    fuzzy_used = 0

    for g in gold_rows:
        q = g.get("input", "")
        nq = norm_q(q)
        pred = pred_idx.get(nq)

        if pred is None:
            # Try fuzzy match (helps if whitespace/punct differs)
            candidate = fuzzy_lookup(nq, pred_keys, threshold=95)
            if candidate is not None:
                pred = pred_idx[candidate]
                fuzzy_used += 1

        if pred is None:
            missing += 1
            continue

        contexts = to_contexts(pred.get("retrieved_context", []))
        if k is not None:
            contexts = contexts[:k]

        ragas_rows.append({
            "question": q,
            "answer": pred.get("model_answer", ""),
            "contexts": contexts,
            "ground_truth": g.get("answer_icliniq", ""),  # use icliniq as the gold
        })

    if missing:
        print(f"[WARN] {missing} gold questions had no matching prediction.")
    if fuzzy_used:
        print(f"[INFO] Used fuzzy join for {fuzzy_used} question(s).")

    # Minimal sanity
    for i, r in enumerate(ragas_rows[:3]):
        assert isinstance(r["question"], str) and r["question"], f"Row {i} missing question"
        assert isinstance(r["answer"], str), f"Row {i} answer must be str"
        assert isinstance(r["contexts"], list), f"Row {i} contexts must be list[str]"

    return pd.DataFrame(ragas_rows)


# ----------------------------
# Judge makers (LangChain)
# ----------------------------
def make_judges(
    judge_model: Optional[str],
    embed_model: Optional[str],
):
    """
    Construct LangChain judge LLM + embeddings.
    Defaults chosen to work with OpenAI if OPENAI_API_KEY is set.
    """
    # Fail early if API key missing (avoids confusing 401s later)
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY")):
        raise RuntimeError(
            "Missing API key. Set OPENAI_API_KEY (or AZURE_OPENAI_API_KEY) "
            "in your environment / .env before running."
        )

    llm = ChatOpenAI(
        model=judge_model or "gpt-4o-mini",
        temperature=0,
        #timeout=1000, #forexceptionerror
    )

    emb = OpenAIEmbeddings(
        model=embed_model or "text-embedding-3-small"
    )
    return llm, emb


# ----------------------------
# Run RAGAS
# ----------------------------
def run_ragas(
    df: pd.DataFrame,
    out_prefix: str = "ragas_results",
    judge_model: Optional[str] = None,
    embed_model: Optional[str] = None,
):
    """
    df columns: question, answer, contexts, ground_truth
    """
    judge_llm, judge_emb = make_judges(judge_model, embed_model)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,  # we have ground truth
    ]

    ds = Dataset.from_pandas(df)
    result = evaluate(
        ds,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_emb,
    )

    # ---- Robust aggregate extraction across RAGAS versions ----
    agg = None
    try:
        if isinstance(result, dict):  # some older versions
            agg = result.get("aggregated_metrics") or result.get("aggregated_scores")
            per = result["results"].to_pandas()
        else:
            # newer versions: Mapping-like object
            try:
                agg = result["aggregated_metrics"]
            except Exception:
                try:
                    agg = result["aggregated_scores"]
                except Exception:
                    agg = None
            # per-sample
            try:
                per = result["results"].to_pandas()
            except Exception:
                # fallback: many versions expose to_pandas() on the result itself
                per = result.to_pandas()
    except Exception:
        # final fallback
        per = result.to_pandas()
        agg = None

    # If aggregates weren’t provided, compute simple means from per-sample metrics
    if agg is None:
        metric_cols = [
            c for c in per.columns
            if c in {"faithfulness","answer_relevancy","context_precision","context_recall","answer_correctness"}
        ]
        # If names differ, include any float columns
        if not metric_cols:
            metric_cols = [c for c in per.columns if pd.api.types.is_numeric_dtype(per[c])]
        agg = {m: float(per[m].mean()) for m in metric_cols if m in per.columns}

    print("\n=== Aggregate metrics ===")
    for k, v in agg.items():
        try:
            print(f"{k}: {float(v):.3f}")
        except Exception:
            print(f"{k}: {v}")

    # Save outputs
    per.to_csv(f"{out_prefix}.csv", index=False)
    with open(f"{out_prefix}.json", "w", encoding="utf-8") as f:
        json.dump({
            "aggregated_metrics": agg,
            "per_sample": per.to_dict(orient="records"),
        }, f, ensure_ascii=False, indent=2)

    print(f"\nSaved: {out_prefix}.csv and {out_prefix}.json")
    return agg, per


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate your RAG system on MedDialog-style data using RAGAS."
    )
    ap.add_argument(
        "--gold",
        default="meddialog_test.json",
        help="Path to MedDialog test file (default: meddialog_test.json)",
    )
    ap.add_argument(
        "--preds",
        default="qa_log_20251118_031101.json",
        help="Path to model output log file (default: model_log.json)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=4,
        help="Use only top-k retrieved contexts per query (default: 4)",
    )
    ap.add_argument(
        "--out",
        default="ragas_results",
        help="Output prefix for CSV/JSON results (default: ragas_results)",
    )
    ap.add_argument(
        "--judge_model",
        default="gpt-4o",
        help="OpenAI model name for evaluation (default: gpt-4o-mini)",
    )
    ap.add_argument(
        "--embed_model",
        default="text-embedding-3-small",
        help="OpenAI embedding model for evaluation (default: text-embedding-3-small)",
    )
    args = ap.parse_args()

    # --- Load the input data ---
    gold_rows = load_json_any(args.gold)
    pred_rows = load_json_any(args.preds)

    df = build_ragas_dataframe(gold_rows, pred_rows, k=args.topk)
    if df.empty:
        raise RuntimeError(
            "No aligned rows found. Check that `input` in gold matches `question` in preds."
        )

    # --- Run evaluation ---
    run_ragas(
        df,
        out_prefix=args.out,
        judge_model=args.judge_model,
        embed_model=args.embed_model,
    )


if __name__ == "__main__":
    main()