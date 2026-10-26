"""Judge calibration: you label answers, then we compare with the LLM judge.

Step 1 (about 20 minutes):   python src/label_answers.py
   You see question, gold answer, and the system's answer. Type:
      1 = correct   2 = partly correct   0 = wrong   s = skip   q = quit and save
   Progress is saved after every answer, so you can stop and continue later.
Step 2:                      python src/label_answers.py --report
   Compares your labels with the judge's correctness scores (kappa, agreement).
"""
import argparse
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "output/human_labels.json"


def load_items():
    tests = {t["id"]: t for t in json.loads((ROOT / "data/test_set.json").read_text(encoding="utf-8"))}
    items = []
    for state in ("full", "base"):
        f = ROOT / f"output/eval_generator_{state}.json"
        if not f.exists():
            continue
        for r in json.loads(f.read_text(encoding="utf-8"))["rows"]:
            if r.get("correctness") is None:
                continue                      # no judge score, nothing to compare
            t = tests[r["id"]]
            items.append({"key": f"{state}:{r['id']}", "question": t["question"],
                          "gold": t["gold_answer"], "answer": r["answer"],
                          "judge": r["correctness"]})
    cf = ROOT / "output/controls.json"
    if cf.exists():                           # deliberately damaged answers (see --make-controls)
        items += json.loads(cf.read_text(encoding="utf-8"))
    random.Random(0).shuffle(items)           # fixed shuffle, so labelling stays blind
    return items


def label():
    items = load_items()
    labels = json.loads(LABELS.read_text()) if LABELS.exists() else {}
    todo = [i for i in items if i["key"] not in labels]
    print(f"{len(labels)} labelled, {len(todo)} left. Judge scores are hidden on purpose.\n")
    for n, it in enumerate(todo, 1):
        print("=" * 70)
        print(f"[{len(labels) + 1}/{len(items)}]  QUESTION: {it['question']}")
        print(f"GOLD ANSWER:   {it['gold']}")
        print(f"SYSTEM ANSWER: {it['answer']}")
        while True:
            c = input("\nDoes the system answer state the same facts as the gold?  "
                      "1=correct 2=partly 0=wrong s=skip q=quit > ").strip().lower()
            if c in ("1", "2", "0", "s", "q"):
                break
        if c == "q":
            break
        if c == "s":
            continue
        labels[it["key"]] = int(c)
        LABELS.write_text(json.dumps(labels, indent=2))
    print(f"\nSaved {len(labels)} labels to {LABELS}")


def corrupt(text, other):
    """Return a damaged copy of an answer, or None if no damage rule applies."""
    rules = [lambda t: t.replace("20", "50", 1) if "20" in t and "2026" not in t[:t.find("20") + 4] else None,
             lambda t: t.replace("80%", "60%", 1) if "80%" in t else None,
             lambda t: t.replace("2026", "2025", 1) if "2026" in t else None,
             lambda t: t.replace("AMC", "Trustee", 1) if "AMC" in t else None,
             lambda t: other,                                   # answer to a different question
             lambda t: t.split(". ")[0] + "." if ". " in t else None]   # only the first sentence
    return rules


def make_controls(n=12):
    """Damage some real answers on purpose, then let the judge score them."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from judge import GroqJudge

    items = [i for i in load_items() if not i["key"].startswith("ctrl:")]
    judge = GroqJudge()
    metric = GEval(
        name="Correctness",
        criteria="Does the actual output state the same facts as the expected output? "
                 "Ignore wording and citation markers like [1]. Penalise wrong or missing facts.",
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
        model=judge, async_mode=False)
    out, used = [], set()
    for k, it in enumerate(items):
        rules = corrupt(it["answer"], items[(k + 7) % len(items)]["answer"])
        for r_i, rule in enumerate(rules):
            if r_i in used and len(used) < len(rules):
                continue
            new = rule(it["answer"])
            if new and new != it["answer"]:
                used.add(r_i)
                case = LLMTestCase(input=it["question"], actual_output=new, expected_output=it["gold"])
                metric.measure(case)
                out.append({"key": f"ctrl:{len(out)}", "question": it["question"], "gold": it["gold"],
                            "answer": new, "judge": metric.score})
                break
        if len(out) >= n:
            break
        if len(used) == len(rules):
            used = set()
    (ROOT / "output/controls.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Made {len(out)} damaged answers. Now run:  python src/label_answers.py")


RUBRIC_V2 = ("Look at the QUESTION. Check whether the actual output answers it correctly, using the "
             "expected output as the reference. Ignore details in the expected output that the question "
             "did not ask for. Ignore wording and citation markers like [1]. Penalise any wrong number, "
             "date, name or entity, and any fact the question asks for that is missing or wrong.")


def rescore():
    """Score every labelled answer again with the improved rubric (judge only, no new answers)."""
    from deepeval.metrics import GEval
    from deepeval.test_case import LLMTestCase, LLMTestCaseParams
    from judge import GroqJudge

    judge = GroqJudge()
    metric = GEval(name="CorrectnessV2", criteria=RUBRIC_V2,
                   evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                                      LLMTestCaseParams.EXPECTED_OUTPUT],
                   model=judge, async_mode=False)
    scores = {}
    for it in load_items():
        case = LLMTestCase(input=it["question"], actual_output=it["answer"], expected_output=it["gold"])
        metric.measure(case)
        scores[it["key"]] = metric.score
        print(f"  {it['key']:<12} v1 {it['judge']:.2f}  v2 {metric.score:.2f}")
    (ROOT / "output/judge_v2.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print("Saved output/judge_v2.json. Now run:  python src/label_answers.py --report --v2")


def kappa(a, b, k=3, weighted=False):
    m = np.zeros((k, k))
    for x, y in zip(a, b):
        m[x, y] += 1
    n = m.sum()
    po_w = np.fromfunction(lambda i, j: 1 - (abs(i - j) / (k - 1) if weighted else (i != j)), (k, k))
    row, col = m.sum(1) / n, m.sum(0) / n
    po = (po_w * m).sum() / n
    pe = (po_w * np.outer(row, col)).sum()
    return (po - pe) / (1 - pe) if pe < 1 else float("nan")


def ranks(x):
    order = np.argsort(x)
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in set(x):                          # average the ranks of ties
        idx = [i for i, y in enumerate(x) if y == v]
        r[idx] = np.mean(r[idx])
    return r


def report(v2=False):
    items = {i["key"]: i for i in load_items()}
    if v2:
        for k, v in json.loads((ROOT / "output/judge_v2.json").read_text()).items():
            if k in items:
                items[k]["judge"] = v
        print("JUDGE RUBRIC: v2 (ignores facts the question did not ask for)")
    else:
        print("JUDGE RUBRIC: v1 (original)")
    labels = json.loads(LABELS.read_text())
    keys = [k for k in labels if k in items]
    cat = {1: 2, 2: 1, 0: 0}                                    # human: correct=2, partly=1, wrong=0
    human = [cat[labels[k]] for k in keys]
    judge_score = [items[k]["judge"] for k in keys]
    judge = [2 if s >= 0.85 else (0 if s <= 0.35 else 1) for s in judge_score]
    n = len(keys)
    agree = sum(h == j for h, j in zip(human, judge)) / n
    rh, rj = ranks(human), ranks(judge_score)
    spear = float(np.corrcoef(rh, rj)[0, 1])
    print(f"Labelled answers: {n}")
    print(f"Exact agreement:          {agree:.0%}  ({sum(h == j for h, j in zip(human, judge))}/{n})")
    print(f"Cohen's kappa:            {kappa(human, judge):.2f}")
    print(f"Weighted kappa (linear):  {kappa(human, judge, weighted=True):.2f}")
    print(f"Spearman correlation:     {spear:.2f}")
    names = ["wrong", "partly", "correct"]
    print("\nRows = you, columns = judge")
    print("            " + "  ".join(f"{x:>7}" for x in names))
    for i in range(3):
        print(f"{names[i]:>10}  " + "  ".join(f"{sum(1 for h, j in zip(human, judge) if h == i and j == jj):>7}" for jj in range(3)))
    print("\nBiggest disagreements (you vs judge score):")
    for k in sorted(keys, key=lambda k: -abs(cat[labels[k]] / 2 - items[k]["judge"]))[:5]:
        print(f"  {k:<10} you={names[cat[labels[k]]]:<8} judge={items[k]['judge']:.2f}")
    print("\nKappa guide: below 0.4 weak, 0.4-0.6 moderate, 0.6-0.8 good, above 0.8 very good.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--make-controls", action="store_true")
    ap.add_argument("--rescore", action="store_true")
    ap.add_argument("--v2", action="store_true")
    a = ap.parse_args()
    if a.make_controls:
        make_controls()
    elif a.rescore:
        rescore()
    elif a.report:
        report(a.v2)
    else:
        label()
 