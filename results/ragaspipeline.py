# -*- coding: utf-8 -*-
"""
RAGAS (LangChain) evaluation for MedDialog-style gold + session/turn prediction logs.

Gold file (unchanged):
  meddialog_test.json — list[dict] with:
    - input: str (question)
    - answer_icliniq: str (gold)

Predictions file (this schema):
{
  "session_start": "...",
  "turns": [
    {
      "timestamp": "...",
      "user_input": "<question>",
      "sub_questions": [...],          # optional
      "answers": [...],                # optional, not used by default
      "final_output": "<model_answer>",# <-- default model answer
      "retrieved_docs": [              # contexts
        {"text": "<chunk text>", "source": "...", "similarity": 0.185},
        ...
      ]
    },
    ...
  ]
}

Outputs:
  - <out_prefix>.csv  (per-sample RAGAS metrics)
  - <out_prefix>.json (aggregates + per-sample)

Robustness:
  - first-match duplicate handling
  - fuzzy join (token_set + partial, default threshold 85)
  - context key fallbacks
  - provides ground_truth + ground_truths (list[str]) for RAGAS versions
  - OpenAI/Azure OpenAI with longer timeouts + retries
  - batched evaluate() with backoff + resume
"""

import os
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import pandas as pd
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
load_dotenv()

# LangChain LLM + Embeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
try:
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    HAVE_AZURE = True
except Exception:
    HAVE_AZURE = False

# Fuzzy join helpers
from rapidfuzz import process, fuzz


# ----------------------------
# Loading helpers
# ----------------------------
def load_json_any(path: str):
    """
    Supports:
      - JSON array file
      - JSON object file
      - JSONL file (one JSON object per line)
    Returns Python object as loaded (list or dict).
    """
    if path.endswith(".jsonl"):
        out = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    out.append(json.loads(line))
        return out
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ----------------------------
# Normalization + joining
# ----------------------------
def norm_q(s: str) -> str:
    return " ".join((s or "").strip().split()).lower()


def build_index_by_question(items: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    idx = {}
    for it in items:
        q = it.get(key, "")
        nq = norm_q(q)
        if not nq:
            continue
        if nq not in idx:
            idx[nq] = it  # keep FIRST
    return idx


def fuzzy_lookup(nq: str, candidates: List[str], threshold: int = 85) -> Optional[Tuple[str, int]]:
    if not candidates:
        return None
    m1 = process.extractOne(nq, candidates, scorer=fuzz.token_set_ratio)
    m2 = process.extractOne(nq, candidates, scorer=fuzz.partial_ratio)
    best = None
    if m1:
        best = m1
    if m2 and (best is None or m2[1] > best[1]):
        best = m2
    if best and best[1] >= threshold:
        return (best[0], int(best[1]))
    return None


# ----------------------------
# Parse your session/turn predictions into flat rows
# ----------------------------
def extract_pred_rows_from_sessions(
    raw_pred,
    answer_from: str = "final_output",   # "final_output" | "answers_concat" | "first_answer"
) -> List[Dict[str, Any]]:
    """
    Returns list of rows with:
      - question
      - model_answer
      - retrieved_context   (list[dict or str])
    Accepts:
      - A single session dict
      - A list of session dicts
    """
    sessions = raw_pred if isinstance(raw_pred, list) else [raw_pred]
    rows = []
    for sess in sessions:
        turns = (sess or {}).get("turns", [])
        for t in turns:
            q = t.get("user_input", "")
            # choose model answer field
            if answer_from == "final_output":
                a = t.get("final_output", "") or ""
            elif answer_from == "answers_concat":
                ans_list = t.get("answers", [])
                if isinstance(ans_list, list):
                    a = "\n\n".join([str(x) for x in ans_list if x])
                else:
                    a = str(ans_list or "")
            elif answer_from == "first_answer":
                ans_list = t.get("answers", [])
                a = str(ans_list[0]) if isinstance(ans_list, list) and ans_list else ""
            else:
                a = t.get("final_output", "") or ""

            ctx = t.get("retrieved_docs", []) or []
            rows.append({
                "question": q,
                "model_answer": a,
                "retrieved_context": ctx,
            })
    return rows


# ----------------------------
# Mapping to RAGAS schema
# ----------------------------
def to_contexts(retrieved_context_field) -> List[str]:
    """
    Accepts dicts with 'text' | 'content' | 'page_content' | 'metadata.text' or raw strings.
    """
    ctx = []
    for d in (retrieved_context_field or []):
        if isinstance(d, dict):
            meta = d.get("metadata") or {}
            t = d.get("text") or d.get("content") or d.get("page_content") or meta.get("text")
            if t:
                ctx.append(str(t))
        elif isinstance(d, str):
            ctx.append(d)
    return ctx


def build_ragas_dataframe(
    gold_rows: List[Dict[str, Any]],
    pred_rows: List[Dict[str, Any]],
    k: Optional[int] = None,      # None keep as-is; 0 means all; >0 means top-k
    min_fuzzy: int = 85,
    consume_preds: bool = True,
) -> pd.DataFrame:
    gold_idx = build_index_by_question(gold_rows, key="input")
    pred_idx = build_index_by_question(pred_rows, key="question")

    pred_keys = list(pred_idx.keys())
    used_pred_keys = set()

    ragas_rows = []
    missing = 0
    fuzzy_used = 0

    for g in gold_rows:
        q = g.get("input", "")
        if not q:
            continue
        nq = norm_q(q)
        pred = pred_idx.get(nq)
        matched_key = nq
        used_fuzzy = False

        if pred is None:
            found = fuzzy_lookup(nq, pred_keys, threshold=min_fuzzy)
            if found is not None:
                matched_key, score = found
                pred = pred_idx.get(matched_key)
                used_fuzzy = True

        if pred is None:
            missing += 1
            continue

        if consume_preds and matched_key in used_pred_keys:
            remaining = [k for k in pred_keys if k not in used_pred_keys]
            found2 = fuzzy_lookup(nq, remaining, threshold=min_fuzzy)
            if found2 is not None:
                matched_key, score = found2
                pred = pred_idx.get(matched_key)
                used_fuzzy = True

        if pred is None:
            missing += 1
            continue

        if consume_preds:
            used_pred_keys.add(matched_key)
        if used_fuzzy:
            fuzzy_used += 1

        contexts = to_contexts(pred.get("retrieved_context", []))
        if k is not None and k > 0:
            contexts = contexts[:k]

        ragas_rows.append({
            "question": q,
            "answer": pred.get("model_answer", "") or "",
            "contexts": contexts,
            "ground_truth": g.get("answer_icliniq", "") or "",
        })

    total = len(gold_rows) or 1
    print(f"[COVERAGE] matched: {len(ragas_rows)}/{total} = {len(ragas_rows)/total:.1%}")
    if missing:
        print(f"[WARN] {missing} gold questions had no matching prediction.")
    if fuzzy_used:
        print(f"[INFO] Used fuzzy join for {fuzzy_used} question(s).")

    df = pd.DataFrame(ragas_rows)
    if not df.empty:
        df["ground_truth"] = df["ground_truth"].astype(str)
        df["ground_truths"] = df["ground_truth"].apply(lambda x: [x])
    return df


# ----------------------------
# Judge makers (OpenAI + optional Azure)
# ----------------------------
def make_judges(judge_model: Optional[str], embed_model: Optional[str]):
    # Prefer Azure if fully configured
    azure_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_ep = os.getenv("AZURE_OPENAI_ENDPOINT")
    chat_deploy = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    emb_deploy = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
    api_ver = os.getenv("AZURE_OPENAI_API_VERSION", "2024-06-01")

    if azure_key and azure_ep and chat_deploy and emb_deploy and HAVE_AZURE:
        llm = AzureChatOpenAI(
            azure_endpoint=azure_ep,
            api_key=azure_key,
            api_version=api_ver,
            deployment_name=chat_deploy,
            temperature=0,
            max_retries=8,
            timeout=180,
            max_tokens=2048,
        )
        emb = AzureOpenAIEmbeddings(
            azure_endpoint=azure_ep,
            api_key=azure_key,
            api_version=api_ver,
            deployment=emb_deploy,
            max_retries=8,
            timeout=180,
        )
        return llm, emb

    # Regular OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("Missing OPENAI_API_KEY (or Azure envs).")

    llm = ChatOpenAI(
        model=judge_model or "gpt-4o-mini",
        temperature=0,
        max_tokens=2048,   # avoid LLMDidNotFinishException
        timeout=180,
        max_retries=8,
    )
    emb = OpenAIEmbeddings(
        model=embed_model or "text-embedding-3-small",
        max_retries=8,
        timeout=180,
    )
    return llm, emb


# ----------------------------
# Batch-safe evaluate with resume
# ----------------------------
def evaluate_in_batches(
    df: pd.DataFrame,
    metrics,
    judge_llm,
    judge_emb,
    out_prefix: str,
    batch_size: int = 50,
    sleep_between: float = 1.0,
    resume: bool = True,
):
    out_csv = Path(f"{out_prefix}.csv")
    out_json = Path(f"{out_prefix}.json")

    done_df = pd.DataFrame()
    start_idx = 0
    if resume and out_csv.exists():
        try:
            done_df = pd.read_csv(out_csv)
            start_idx = len(done_df)
            print(f"[RESUME] Found {out_csv}, resuming after {start_idx} rows.")
        except Exception:
            pass

    per_rows = []
    if start_idx > 0 and len(done_df) > 0:
        per_rows.append(done_df)

    total = len(df)
    while start_idx < total:
        end_idx = min(start_idx + batch_size, total)
        print(f"[BATCH] {start_idx}–{end_idx-1} / {total-1}")

        ds = Dataset.from_pandas(df.iloc[start_idx:end_idx][
            ["question","answer","contexts","ground_truth","ground_truths"]
        ])

        attempts, last_err = 0, None
        while attempts < 5:
            try:
                result = evaluate(
                    ds,
                    metrics=metrics,
                    llm=judge_llm,
                    embeddings=judge_emb,
                )
                try:
                    per = result["results"].to_pandas()
                except Exception:
                    per = result.to_pandas()
                per_rows.append(per)
                cur = pd.concat(per_rows, ignore_index=True)
                cur.to_csv(out_csv, index=False)
                time.sleep(sleep_between)
                break
            except Exception as e:
                attempts += 1
                last_err = e
                backoff = min(30, 2 ** attempts) + (attempts * 0.5)
                print(f"[RETRY {attempts}/5] Batch {start_idx}-{end_idx-1} failed: {e}. Sleeping {backoff:.1f}s.")
                time.sleep(backoff)
        else:
            raise RuntimeError(f"Batch {start_idx}-{end_idx-1} failed repeatedly") from last_err

        start_idx = end_idx

    per_df = pd.concat(per_rows, ignore_index=True)

    metric_cols = [
        c for c in per_df.columns
        if c in {"faithfulness","answer_relevancy","context_precision","context_recall","answer_correctness"}
    ]
    if not metric_cols:
        metric_cols = [c for c in per_df.columns if pd.api.types.is_numeric_dtype(per_df[c])]
    agg = {m: float(per_df[m].mean()) for m in metric_cols if m in per_df.columns}

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"aggregated_metrics": agg,
                   "per_sample": per_df.to_dict(orient="records")},
                  f, ensure_ascii=False, indent=2)

    print("\n=== Aggregate metrics ===")
    for k, v in agg.items():
        print(f"{k}: {v:.3f}" if isinstance(v, (int,float)) else f"{k}: {v}")
    print(f"\nSaved: {out_csv} and {out_json}")

    return agg, per_df


# ----------------------------
# Run RAGAS (calls batched evaluator)
# ----------------------------
def run_ragas(
    df: pd.DataFrame,
    out_prefix: str = "ragas_results",
    judge_model: Optional[str] = None,
    embed_model: Optional[str] = None,
    batch_size: int = 50,
):
    if df.empty:
        raise RuntimeError("Empty dataframe—nothing to evaluate.")

    judge_llm, judge_emb = make_judges(judge_model, embed_model)

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        answer_correctness,
    ]

    return evaluate_in_batches(
        df, metrics, judge_llm, judge_emb,
        out_prefix=out_prefix,
        batch_size=batch_size,
        sleep_between=1.0,
        resume=True,
    )


# ----------------------------
# CLI
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Evaluate your RAG system (session/turn JSON) against MedDialog gold using RAGAS."
    )
    ap.add_argument("--gold", default="meddialog_test.json",
                    help="Path to MedDialog test file (default: meddialog_test.json)")
    ap.add_argument("--preds", default="chat_log_llm_followup.json",
                    help="Path to predictions in session/turn schema")
    ap.add_argument("--answer_from", default="final_output",
                    choices=["final_output", "answers_concat", "first_answer"],
                    help="Which field to use for model answer (default: final_output)")
    ap.add_argument("--topk", type=int, default=4,
                    help="Use only top-k retrieved contexts per query; 0 = all (default: 4)")
    ap.add_argument("--out", default="ragas_resultsfollowup",
                    help="Output prefix for CSV/JSON results (default: ragas_results)")
    ap.add_argument("--judge_model", default="gpt-4o-mini",
                    help="OpenAI model name for evaluation (default: gpt-4o-mini)")
    ap.add_argument("--embed_model", default="text-embedding-3-small",
                    help="OpenAI embedding model (default: text-embedding-3-small)")
    ap.add_argument("--min_fuzzy", type=int, default=85,
                    help="Minimum fuzzy match score for joining questions (default: 85)")
    ap.add_argument("--no_consume_preds", action="store_true",
                    help="If set, allow the same prediction row to match multiple gold rows.")
    ap.add_argument("--batch_size", type=int, default=50,
                    help="Evaluate in batches (default: 50)")
    args = ap.parse_args()

    # --- Load data ---
    gold_raw = load_json_any(args.gold)
    if not isinstance(gold_raw, list):
        raise RuntimeError("Gold file must be a JSON array of rows.")
    pred_raw = load_json_any(args.preds)

    # Flatten your session/turns into rows
    pred_rows = extract_pred_rows_from_sessions(pred_raw, answer_from=args.answer_from)

    topk = None if args.topk == 0 else args.topk

    df = build_ragas_dataframe(
        gold_rows=gold_raw,
        pred_rows=pred_rows,
        k=topk,
        min_fuzzy=args.min_fuzzy,
        consume_preds=not args.no_consume_preds,
    )
    if df.empty:
        raise RuntimeError("No aligned rows found. Check that gold 'input' matches session 'user_input'.")

    # --- Run evaluation ---
    run_ragas(
        df,
        out_prefix=args.out,
        judge_model=args.judge_model,
        embed_model=args.embed_model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
