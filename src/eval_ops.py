"""Operational numbers: speed, tokens, and size.

Part 1 (free, no LLM): search latency, with and without the reranker.
Part 2 (uses Groq tokens): end-to-end latency and tokens per question.

Run:
    python src/eval_ops.py --no-llm          (Part 1 only, zero tokens)
    python src/eval_ops.py --n 10            (Part 1 + 10 real answers)
"""
import argparse
import json
import os
import time
from pathlib import Path

import httpx
import numpy as np

import answer
from search import HybridSearcher

ROOT = Path(__file__).resolve().parent.parent
PAUSE = float(os.environ.get("ANSWER_PAUSE", "15"))
out = {}


def pct(xs, p):
    return round(float(np.percentile(xs, p)), 2)


def size_mb(path):
    p = Path(path)
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()

    tests = json.loads((ROOT / "data/test_set.json").read_text(encoding="utf-8"))
    questions = [t["question"] for t in tests]
    s = HybridSearcher(rerank=True)
    reranker = s.reranker

    # ---- Part 1: search speed ----
    for q in questions[:2]:                       # warm-up, not counted
        s.search(q, k=5)
    times = {}
    for label, rr in (("without reranker", None), ("with reranker", reranker)):
        s.reranker = rr
        ts = []
        for q in questions*3:
            t0 = time.perf_counter()
            s.search(q, k=5)
            ts.append(time.perf_counter() - t0)
        times[label] = ts
    s.reranker = reranker
    print(f"\n== 1. SEARCH SPEED ({len(questions)*3} questions, seconds)")
    for label, ts in times.items():
        print(f"  {label:<18} p50 {pct(ts, 50)}   p95 {pct(ts, 95)}   max {pct(ts, 100)}")
        out[f"search {label}"] = {"p50": pct(ts, 50), "p95": pct(ts, 95)}

    # ---- size ----
    print("\n== SIZE")
    for name, path in (("source PDFs", ROOT / "data/pdfs"), ("chunks file", ROOT / "output/chunks.jsonl"),
                       ("vector index", ROOT / "output/qdrant_db")):
        print(f"  {name:<14} {size_mb(path):.1f} MB")
        out[f"size {name}"] = round(size_mb(path), 1)
    print(f"  chunks: {len(s.chunks)}")

    if a.no_llm:
        return save()

    # ---- Part 2: end-to-end speed and tokens ----
    usage = []
    real_post = httpx.post

    def spy(*args, **kwargs):                     # records token usage of every API call
        r = real_post(*args, **kwargs)
        if r.status_code == 200:
            usage.append(r.json().get("usage", {}))
        return r

    answer.httpx.post = spy
    print(f"\n== 2. END-TO-END ({a.n} questions, real answers)")
    totals, search_t, llm_t = [], [], []
    for q in questions[:a.n]:
        t0 = time.perf_counter()
        chunks = s.search(q, k=5)
        t1 = time.perf_counter()
        answer.ask_llm(q, chunks, time.strftime("%Y-%m-%d"))
        t2 = time.perf_counter()
        search_t.append(t1 - t0)
        llm_t.append(t2 - t1)
        totals.append(t2 - t0)
        time.sleep(PAUSE)
    answer.httpx.post = real_post

    print(f"  total per question   p50 {pct(totals, 50)}   p95 {pct(totals, 95)} seconds")
    print(f"  of which search      p50 {pct(search_t, 50)}   answer model p50 {pct(llm_t, 50)}")
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    avg = {k: round(sum(u.get(k, 0) for u in usage) / max(len(usage), 1)) for k in keys}
    print(f"  tokens per question  input {avg['prompt_tokens']}  output {avg['completion_tokens']}  total {avg['total_tokens']}")
    out["end_to_end"] = {"p50": pct(totals, 50), "p95": pct(totals, 95), "tokens": avg, "n": a.n}
    save()


def save():
    (ROOT / "output/eval_ops.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nsaved output/eval_ops.json")


if __name__ == "__main__":
    main()