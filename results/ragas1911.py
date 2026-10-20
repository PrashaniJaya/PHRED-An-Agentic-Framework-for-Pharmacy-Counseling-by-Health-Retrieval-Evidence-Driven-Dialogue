# -*- coding: utf-8 -*-
"""
RAGAS (LangChain) evaluation for MedDialog-style data.

Inputs:
  1) meddialog_test.json — list[dict] with fields:
       - input: str (the question)
       - answer_icliniq: str (gold answer to use)
  2) model_log.json — list[dict] with fields:
       - question: str (should match `input`)
       - model_answer: str (your system answer)
       - retrieved_context: list[dict|str] with "text" (top-k = 8)
         (also accepted: content, page_content, metadata.text)

Outputs:
  - <out_prefix>.csv  (per-sample metrics)
  - <out_prefix>.json (aggregate + per-sample)

Hardening:
  - Keeps FIRST duplicate question
  - Robust fuzzy join (token_set + partial; default threshold 85)
  - Context key fallbacks
  - Provides ground_truth and ground_truths
  - OpenAI clients with longer timeouts + retries
  - Batched evaluate() with retries, backoff, resume from CSV
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
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings  # optional
    HAVE_AZURE = True
except Exception:
    HAVE_AZURE = False

# Fuzzy join helpers
from rapidfuzz import process, fuzz


# ----------------------------
# Loading helpers
# ----------------------------
def load_json_any(path: str) -> List[Dict[str, Any]]:
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
    return " ".join((s or "").strip().split()).lower()


def build_index_by_question(items: List[Dict[str, Any]], key: str) -> Dict[str, Dict[str, Any]]:
    """
    Build a dict: normalized_question -> item
    Keep the FIRST occurrence (do not silently overwrite).
    """
    idx = {}
    for it in items:
        q = it.get(key, "")
        nq = norm_q(q)
        if not nq:
            continue
        if nq not in idx:
            idx[nq] = it
        # else: could log a warning about duplicates
    return idx


def fuzzy_lookup(
    nq: str,
    candidates: List[str],
    threshold: int = 85
) -> Optional[Tuple[str, int]]:
    """
    Fuzzy fallback: choose best of token_set_ratio and partial_ratio.
    Returns (match_key, score) if >= threshold, else None.
    """
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
# Mapping to RAGAS schema
# ----------------------------
def to_contexts(retrieved_context_field) -> List[str]:
    """
    Convert your retrieved_context field into list[str] of text chunks.
    Accepts:
      - dicts with 'text' OR 'content' OR 'page_content' OR 'metadata'.'text'
      - raw strings
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
    k: Optional[int] = None,          # top-k cut; None means keep as-is; 0 means keep all
    min_fuzzy: int = 85,
    consume_preds: bool = True,       # avoid mapping the same pred to many golds
) -> pd.DataFrame:
    """
    gold_rows: from meddialog_test.json (keys: input, answer_icliniq)
    pred_rows: from model_log.json (keys: question, model_answer, retrieved_context)
    """
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
            # Try fuzzy match (helps if whitespace/punct differs)
            found = fuzzy_lookup(nq, pred_keys, threshold=min_fuzzy)
            if found is not None:
                matched_key, score = found
                pred = pred_idx.get(matched_key)
                used_fuzzy = True

        if pred is None:
            missing += 1
            continue

        # Optionally avoid reusing the same pred row for multiple gold rows
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
            "ground_truth": g.get("answer_icliniq", "") or "",  # gold
        })

    total = len(gold_rows) or 1
    print(f"[COVERAGE] matched: {len(ragas_rows)}/{total} = {len(ragas_rows)/total:.1%}")
    if missing:
        print(f"[WARN] {missing} gold questions had no matching prediction.")
    if fuzzy_used:
        print(f"[INFO] Used fuzzy join for {fuzzy_used} question(s).")

    # Minimal sanity
    for i, r in enumerate(ragas_rows[:3]):
        assert isinstance(r["question"], str) and r["question"], f"Row {i} missing question"
        assert isinstance(r["answer"], str), f"Row {i} answer must be str"
        assert isinstance(r["contexts"], list), f"Row {i} contexts must be list[str]"

    df = pd.DataFrame(ragas_rows)

    # RAGAS cross-version compatibility: both single and list ground truths
    if not df.empty:
        df["ground_truth"] = df["ground_truth"].astype(str)
        df["ground_truths"] = df["ground_truth"].apply(lambda x: [x])

    return df


# ----------------------------
# Judge makers (OpenAI + optional Azure)
# ----------------------------
def make_judges(
    judge_model: Optional[str],
    embed_model: Optional[str],
):
    """
    Construct LangChain judge LLM + embeddings.
    OpenAI by default. Azure OpenAI if env vars present:
      AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT,
      AZURE_OPENAI_CHAT_DEPLOYMENT (for chat),
      AZURE_OPENAI_EMBED_DEPLOYMENT (for embeddings),
      AZURE_OPENAI_API_VERSION (optional)
    """
    # Prefer Azure if fully configured, else fall back to OpenAI
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
            timeout=120,
            max_tokens=2048,
        )
        emb = AzureOpenAIEmbeddings(
            azure_endpoint=azure_ep,
            api_key=azure_key,
            api_version=api_ver,
            deployment=emb_deploy,
            max_retries=8,
            timeout=120,
        )
        return llm, emb

    # Regular OpenAI
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "Missing API key. Set OPENAI_API_KEY (or Azure envs) in your environment/.env before running."
        )

    # Longer timeout + more retries; cap output length for faster eval
    llm = ChatOpenAI(
        model=judge_model or "gpt-4o-mini",
        temperature=0,
        max_tokens=2048,   # short judge outputs
        timeout=120,      # seconds (LangChain param)
        max_retries=8,    # exponential backoff built-in
    )
    emb = OpenAIEmbeddings(
        model=embed_model or "text-embedding-3-small",
        max_retries=8,
        timeout=120,
    )
    return llm, emb


# ----------------------------
# Batch-safe RAGAS evaluate
# ----------------------------
def evaluate_in_batches(
    df: pd.DataFrame,
    metrics,
    judge_llm,
    judge_emb,
    out_prefix: str,
    batch_size: int = 50,
    sleep_between: float = 1.0,   # seconds between batches
    resume: bool = True,
):
    """
    Runs RAGAS in batches. Writes intermediate CSV/JSON so you can resume.
    Returns (aggregates, per_df)
    """
    out_csv = Path(f"{out_prefix}.csv")
    out_json = Path(f"{out_prefix}.json")

    # Resume if a previous per-sample CSV exists
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

        # Retry the whole batch a few times if 520/timeout happens
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
                # Write incremental CSV
                cur = pd.concat(per_rows, ignore_index=True)
                cur.to_csv(out_csv, index=False)
                time.sleep(sleep_between)
                break
            except Exception as e:
                attempts += 1
                last_err = e
                # Small exponential backoff with jitter
                backoff = min(30, 2 ** attempts) + (attempts * 0.5)
                print(f"[RETRY {attempts}/5] Batch {start_idx}-{end_idx-1} failed: {e}. Sleeping {backoff:.1f}s.")
                time.sleep(backoff)
        else:
            # All attempts failed
            raise RuntimeError(f"Batch {start_idx}-{end_idx-1} failed repeatedly") from last_err

        start_idx = end_idx

    # Combine all per-sample
    per_df = pd.concat(per_rows, ignore_index=True)

    # Aggregates from per_df (works across RAGAS versions)
    metric_cols = [
        c for c in per_df.columns
        if c in {"faithfulness","answer_relevancy","context_precision","context_recall","answer_correctness"}
    ]
    if not metric_cols:
        metric_cols = [c for c in per_df.columns if pd.api.types.is_numeric_dtype(per_df[c])]
    agg = {m: float(per_df[m].mean()) for m in metric_cols if m in per_df.columns}

    # Save final JSON too
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
        answer_correctness,  # needs ground truth(s)
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
        description="Evaluate your RAG system on MedDialog-style data using RAGAS (LangChain)."
    )
    ap.add_argument(
        "--gold",
        default="meddialog_test.json",
        help="Path to MedDialog test file (default: meddialog_test.json)",
    )
    ap.add_argument(
        "--preds",
        default= "qa_log_20251118_031101.json", #"qa_log_20251118_015223.json",
        help="Path to model output log file (default: model_log.json)",
    )
    ap.add_argument(
        "--topk",
        type=int,
        default=8,
        help="Use only top-k retrieved contexts per query; 0 = all (default: 4)",
    )
    ap.add_argument(
        "--out",
        default="ragas_results07",
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
    ap.add_argument(
        "--min_fuzzy",
        type=int,
        default=85,
        help="Minimum fuzzy match score (0-100) for joining questions (default: 85)",
    )
    ap.add_argument(
        "--no_consume_preds",
        action="store_true",
        help="If set, allow the same prediction row to match multiple gold rows.",
    )
    ap.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Evaluate in batches to survive transient errors (default: 50)",
    )
    args = ap.parse_args()

    # --- Load the input data ---
    gold_rows = load_json_any(args.gold)
    pred_rows = load_json_any(args.preds)

    topk = None if args.topk == 0 else args.topk

    df = build_ragas_dataframe(
        gold_rows,
        pred_rows,
        k=topk,
        min_fuzzy=args.min_fuzzy,
        consume_preds=not args.no_consume_preds,
    )
    if df.empty:
        raise RuntimeError(
            "No aligned rows found. Check that `input` in gold matches `question` in preds "
            "(consider lowering --min_fuzzy or disabling consume)."
        )

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
