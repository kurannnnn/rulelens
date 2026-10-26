"""Layer 3: retriever + generator together (the real RAG).

Run (start small):
    python src/eval_rag.py --state full --limit 3
    python src/eval_rag.py --state full            (filter ON)
    python src/eval_rag.py --state full --all      (filter OFF = plain RAG baseline)
Add --no-ctx to skip the slow context-relevance metric.
"""
import argparse
import json
import os
import re
import time
from datetime import date
from pathlib import Path

from deepeval.metrics import (AnswerRelevancyMetric, ContextualRelevancyMetric,
                              FaithfulnessMetric, GEval)
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from answer import ask_llm, check_citations
from judge import GroqJudge
from search import HybridSearcher
from status import load_registry

ROOT = Path(__file__).resolve().parent.parent
JULY10 = "MF-CIR-2026-07-10"
PAUSE = float(os.environ.get("ANSWER_PAUSE", "15"))   # seconds between answer calls (token limit)


def score(metric, case):
    if metric is None:                 # --no-judge: skip LLM-judged metrics
        return None
    try:
        metric.measure(case)
        return metric.score
    except Exception as e:
        print("   metric error:", str(e)[:100])
        return None


def avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="full", choices=["full", "base"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--all", action="store_true")      # status filter OFF
    ap.add_argument("--no-ctx", action="store_true")
    ap.add_argument("--no-judge", action="store_true")   # exact metrics only, no judge calls
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    tests = [t for t in json.loads((ROOT / "data/test_set.json").read_text(encoding="utf-8"))
             if t["state"] in (a.state, "any") and not t.get("registry_only")]
    answerable = [t for t in tests if t["gold_chunk_ids"]]
    refuse = [t for t in tests if t["type"] == "not_in_docs"]
    if a.limit:
        answerable, refuse = answerable[:a.limit], refuse[:a.limit]

    searcher = HybridSearcher(rerank=not a.no_rerank)
    all_docs = set(load_registry(str(ROOT / "data/document_registry.json")))
    searcher.loaded = all_docs if a.state == "full" else all_docs - {JULY10}

    if a.no_judge:
        faith = relev = ctxrel = correct = None
    else:
        judge = GroqJudge()
        faith = FaithfulnessMetric(model=judge, async_mode=False)
        relev = AnswerRelevancyMetric(model=judge, async_mode=False)
        ctxrel = None if a.no_ctx else ContextualRelevancyMetric(model=judge, async_mode=False)
        correct = GEval(
            name="Correctness",
            criteria="Does the actual output state the same facts as the expected output? "
                     "Ignore wording and citation markers like [1]. Penalise wrong or missing facts.",
            evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
            model=judge, async_mode=False)

    rows, refusals = [], []
    for t in answerable:
        today = t.get("as_of") or date.today().isoformat()
        got = searcher.search(t["question"], k=a.k, as_of=today,
                              include_superseded=a.all)
        out = ask_llm(t["question"], got, today)
        time.sleep(PAUSE)
        ans = out.get("answer", "")

        cited = {int(n) for n in re.findall(r"\[(\d+)\]", ans)}
        cited_chunks = [got[n - 1] for n in cited if 1 <= n <= len(got)]
        bad = set(t.get("forbidden_docs", []))
        outdated = any(c["doc_id"] in bad or c["status"] == "superseded"
                       for c in cited_chunks)
        hit = bool(set(t["gold_chunk_ids"]) & {c["chunk_id"] for c in got})

        case = LLMTestCase(input=t["question"], actual_output=ans,
                           expected_output=t["gold_answer"],
                           retrieval_context=[c["text"] for c in got])
        row = {"id": t["id"], "answer": ans, "gold_hit": hit, "outdated": outdated,
               "citation_ok": not check_citations(out, got),
               "faithfulness": score(faith, case),
               "answer_relevance": score(relev, case),
               "context_relevance": score(ctxrel, case) if ctxrel else None,
               "correctness": score(correct, case)}
        rows.append(row)
        print(t["id"], {k: (round(v, 2) if isinstance(v, float) else v)
                        for k, v in row.items() if k != "answer"})

    for t in refuse:                       # real retrieval, expect "Not found"
        today = date.today().isoformat()
        got = searcher.search(t["question"], k=a.k, as_of=today, include_superseded=a.all)
        out = ask_llm(t["question"], got, today)
        time.sleep(PAUSE)
        refused = (not out.get("enough_evidence")) or "not found" in out.get("answer", "").lower()
        refusals.append(refused)
        print(t["id"], "refused:", refused)

    tag = f"{a.state}_k{a.k}" + ("_nofilter" if a.all else "") + ("_nojudge" if a.no_judge else "")
    (ROOT / f"output/eval_rag_{tag}.json").write_text(
        json.dumps({"rows": rows, "refusals": refusals}, indent=2), encoding="utf-8")
    n = len(rows)
    print(f"\nRAG ({tag}), {n} questions")
    print("gold hit:          ", sum(r["gold_hit"] for r in rows), "/", n)
    print("outdated answers:  ", sum(r["outdated"] for r in rows), "/", n)
    print("faithfulness:      ", avg([r["faithfulness"] for r in rows]))
    print("answer relevance:  ", avg([r["answer_relevance"] for r in rows]))
    print("context relevance: ", avg([r["context_relevance"] for r in rows]))
    print("correctness:       ", avg([r["correctness"] for r in rows]))
    print("valid citations:   ", sum(r["citation_ok"] for r in rows), "/", n)
    print("refusal rate:      ", sum(refusals), "/", len(refusals))


if __name__ == "__main__":
    main()