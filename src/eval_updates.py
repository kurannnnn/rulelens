"""Update tests: does the system change its answers correctly when documents
are added? Search only. No LLM, no judge, zero Groq tokens.

Run:  python src/eval_updates.py
"""
import argparse
import json
import random
from datetime import date
from pathlib import Path

from search import HybridSearcher
from status import doc_status, load_registry

ROOT = Path(__file__).resolve().parent.parent
J, A, M26, M23 = "MF-CIR-2026-07-10", "MF-CIR-2026-03-25", "MF-MC-2026", "MF-MC-2023"
TODAY = date.today().isoformat()
results = []


def check(name, ok, detail=""):
    results.append({"test": name, "pass": bool(ok), "detail": detail})
    print(("PASS  " if ok else "FAIL  ") + name + (f"   [{detail}]" if detail else ""))


def has_doc(chunks, doc):
    return any(c["doc_id"] == doc for c in chunks)


def has_clause(chunks, doc, clause):
    return any(c["doc_id"] == doc and clause in c["clauses"] for c in chunks)


def run(s, loaded, q, as_of=None, k=5):
    s.loaded = set(loaded)             # pretend only these documents are uploaded
    return s.search(q, k=k, as_of=as_of or TODAY)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-rerank", action="store_true")
    a = ap.parse_args()

    reg = load_registry(str(ROOT / "data/document_registry.json"))
    ALL = set(reg)
    s = HybridSearcher(rerank=not a.no_rerank)

    # ---- Test 1: forward update (upload the Jul 10 circular) -------------
    print("\n== 1. FORWARD UPDATE: upload the Jul 10 2026 circular")
    qs = [("What are the rules for intraday borrowing by mutual funds?", lambda r: has_clause(r, M26, "5.9.1")),
          ("What conditions apply to intraday borrowing?", lambda r: has_clause(r, M26, "5.9.1")),
          ("From what date do the intraday borrowing guidelines apply?",
           lambda r: has_clause(r, M26, "5.9.1") or has_doc(r, A))]
    for i, (q, before_expect) in enumerate(qs, 1):
        before = run(s, ALL - {J}, q)
        after = run(s, ALL, q)
        check(f"Q{i} before upload: old rule found, new circular absent",
              before_expect(before) and not has_doc(before, J))
        check(f"Q{i} after upload: new circular found, old 5.9.1 and addendum gone",
              has_doc(after, J) and not has_clause(after, M26, "5.9.1") and not has_doc(after, A))
    q = qs[0][0]
    check("As of 2026-08-01 (before Sep 1): old rule still applies",
          has_clause(run(s, ALL, q, "2026-08-01"), M26, "5.9.1")
          and not has_doc(run(s, ALL, q, "2026-08-01"), J))
    check("As of today: new rule applies", has_doc(run(s, ALL, q, TODAY), J))

    # ---- Test 2: backfill (old master circular uploaded late) --------------
    print("\n== 2. BACKFILL: upload the 2023 master circular after the 2026 one")
    qb = "What minimum percentage of total assets must a Large Cap Fund invest in large cap stocks?"
    for label, loaded in (("without 2023", ALL - {M23}), ("with 2023 uploaded late", ALL)):
        r = run(s, loaded, qb)
        check(f"Today, {label}: 2023 circular is never used for a current-rule question",
              not has_doc(r, M23))
    st_today = doc_status(reg, ALL, TODAY)
    st_2023 = doc_status(reg, ALL, "2023-12-01")
    check("2023 master circular marked superseded today", st_today[M23] == "superseded")
    check("2026 master circular stays current", st_today[M26] == "current")
    check("As of 2023-12-01 the 2023 circular is current", st_2023[M23] == "current")
    check("As of 2023-12-01, search can use the 2023 circular",
          has_doc(run(s, ALL, qb, "2023-12-01"), M23))

    # ---- Test 3: load order must not matter -----------------------------------
    print("\n== 3. ORDER: same documents, different upload order")
    orders = {"registry order": list(reg), "reversed": list(reg)[::-1]}
    rnd = random.Random(0)
    shuffled = list(reg)
    rnd.shuffle(shuffled)
    orders["shuffled"] = shuffled
    finals, answers = {}, {}
    for name, order in orders.items():
        loaded, ok = set(), True
        for d in order:                       # upload one by one, check every step
            loaded.add(d)
            st = doc_status(reg, loaded, TODAY)
            for x in loaded:
                newer = reg[x].get("superseded_by")
                if newer in loaded and reg[newer]["effective_date"] <= TODAY and st[x] != "superseded":
                    ok = False
        check(f"{name}: no replaced document is ever marked current", ok)
        finals[name] = doc_status(reg, loaded, TODAY)
        answers[name] = [c["chunk_id"] for c in run(s, loaded, q)]
    check("Final statuses identical for all orders", len({json.dumps(v, sort_keys=True) for v in finals.values()}) == 1)
    check("Search results identical for all orders", len({tuple(v) for v in answers.values()}) == 1)

    passed = sum(r["pass"] for r in results)
    print(f"\nUPDATE TESTS: {passed}/{len(results)} passed")
    (ROOT / "output/eval_updates.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()