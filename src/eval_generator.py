"""Layer 2: generator alone. We hand it the CORRECT chunks, so any mistake
is the generator's fault, not the search's.

Setup (PowerShell):
    $env:LLM_API_KEY = "..."            $env:LLM_MODEL = "openai/gpt-oss-120b"
    $env:JUDGE_MODEL = "qwen/qwen3.8-27b"
    $env:DEEPEVAL_TELEMETRY_OPT_OUT = "YES"
Run (try --limit 3 first):
    python src/eval_generator.py --state full --limit 3
    python src/eval_generator.py --state full
"""
import argparse
import json
import random
from datetime import date
from pathlib import Path

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from answer import ask_llm, check_citations
from judge import GroqJudge

ROOT = Path(__file__).resolve().parent.parent


def score(metric, case):
    # Run one metric. If the judge fails, return None so the run continues.
    try:
        metric.measure(case)
        return metric.score
    except Exception as e:
        print("   metric error:", str(e)[:120])
        return None


def avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", default="full", choices=["full", "base"])
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    chunks = {}
    for line in open(ROOT / "output/chunks.jsonl", encoding="utf-8"):
        c = json.loads(line)
        c["status"] = "current"            # gold chunks are treated as current
        chunks[c["chunk_id"]] = c
    tests = [t for t in json.loads((ROOT / "data/test_set.json").read_text(encoding="utf-8"))
             if t["state"] in (a.state, "any") and not t.get("registry_only")]

    judge = GroqJudge()
    faith = FaithfulnessMetric(model=judge, async_mode=False)
    relev = AnswerRelevancyMetric(model=judge, async_mode=False)
    correct = GEval(
        name="Correctness",
        criteria="Does the actual output state the same facts as the expected output? "
                 "Ignore wording and citation markers like [1]. Penalise wrong or missing facts.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=judge, async_mode=False)

    rows, refusals = [], []
    answerable = [t for t in tests if t["gold_chunk_ids"]]
    refuse_tests = [t for t in tests if t["type"] == "not_in_docs"]
    if a.limit:
        answerable, refuse_tests = answerable[:a.limit], refuse_tests[:a.limit]

    # --- Part 1: answer with the gold chunks ---
    for t in answerable:
        src = [chunks[i] for i in t["gold_chunk_ids"]]
        out = ask_llm(t["question"], src, t.get("as_of") or date.today().isoformat())
        ans = out.get("answer", "")
        case = LLMTestCase(input=t["question"], actual_output=ans,
                           expected_output=t["gold_answer"],
                           retrieval_context=[c["text"] for c in src])
        row = {"id": t["id"], "answer": ans,
               "citation_ok": not check_citations(out, src),
               "faithfulness": score(faith, case),
               "answer_relevance": score(relev, case),
               "correctness": score(correct, case)}
        rows.append(row)
        print(t["id"], {k: v for k, v in row.items() if k not in ("answer",)})

    # --- Part 2: refusal test. Give IRRELEVANT chunks, expect "Not found" ---
    rnd = random.Random(0)
    pool = list(chunks.values())
    for t in refuse_tests:
        src = rnd.sample(pool, 3)
        out = ask_llm(t["question"], src, date.today().isoformat())
        refused = (not out.get("enough_evidence")) or "not found" in out.get("answer", "").lower()
        refusals.append(refused)
        print(t["id"], "refused:", refused)

    (ROOT / f"output/eval_generator_{a.state}.json").write_text(
        json.dumps({"rows": rows, "refusals": refusals}, indent=2), encoding="utf-8")
    print(f"\nGENERATOR ALONE ({a.state}), {len(rows)} questions")
    print("faithfulness:     ", avg([r["faithfulness"] for r in rows]))
    print("answer relevance: ", avg([r["answer_relevance"] for r in rows]))
    print("correctness:      ", avg([r["correctness"] for r in rows]))
    print("valid citations:  ", sum(r["citation_ok"] for r in rows), "/", len(rows))
    print("refusal rate:     ", sum(refusals), "/", len(refusals))


if __name__ == "__main__":
    main()