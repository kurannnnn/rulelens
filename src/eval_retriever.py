"""Layer 1: retriever alone. Exact metrics from gold chunk IDs (no LLM).

Run from the project root:
    python src/eval_retriever.py --state full
    python src/eval_retriever.py --state base
    python src/eval_retriever.py --state full --all     (status filter OFF)
"""
import argparse
import json
from pathlib import Path

from search import HybridSearcher
from status import load_registry

ROOT = Path(__file__).resolve().parent.parent
JULY10 = "MF-CIR-2026-07-10"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--state", default="full", choices=["full", "base"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--all", action="store_true")   # turn status filter OFF
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    a = ap.parse_args()
    w = {"hybrid": (0.5, 0.5), "dense": (1.0, 0.0), "bm25": (0.0, 1.0)}[a.mode]

    tests = [t for t in json.loads((ROOT / "data/test_set.json").read_text(encoding="utf-8"))
             if t["state"] in (a.state, "any") and not t.get("registry_only")]

    searcher = HybridSearcher(rerank=not a.no_rerank)
    # base = Jul 10 circular NOT uploaded yet, full = uploaded
    all_docs = set(load_registry(str(ROOT / "data/document_registry.json")))
    searcher.loaded = all_docs if a.state == "full" else all_docs - {JULY10}

    recalls, precisions, rrs, hits, leaks, misses = [], [], [], 0, 0, []
    for t in tests:
        got = searcher.search(t["question"], k=a.k, as_of=t.get("as_of"),
                              w_dense=w[0], w_keyword=w[1],
                              include_superseded=a.all)
        ids = [c["chunk_id"] for c in got]
        gold = set(t.get("gold_chunk_ids", []))
        bad = set(t.get("forbidden_docs", []))

        if any(c["doc_id"] in bad or c["status"] == "superseded" for c in got):
            leaks += 1

        if gold:                          # skip "not in documents" questions
            found = gold & set(ids)
            recalls.append(len(found) / len(gold))
            precisions.append(len(found) / a.k)
            hits += bool(found)
            first = next((r for r, i in enumerate(ids, 1) if i in gold), None)
            rrs.append(1 / first if first else 0)
            if not found:
                misses.append(t["id"])

    n, g = len(tests), len(recalls)
    print(f"questions={n} (with gold={g}) k={a.k} state={a.state} mode={a.mode} filter_off={a.all}")
    print(f"recall@{a.k}:        {sum(recalls)/g:.2f}")
    print(f"precision@{a.k}:     {sum(precisions)/g:.2f}")
    print(f"hit rate@{a.k}:      {hits}/{g}")
    print(f"MRR:               {sum(rrs)/g:.2f}")
    print(f"superseded leakage: {leaks}/{n}")
    print("missed questions:  ", misses)


if __name__ == "__main__":
    main()