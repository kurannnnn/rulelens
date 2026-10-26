# SEBI Mutual Fund RAG + Evaluation

A RAG system over SEBI mutual fund regulation that **answers with the current rule** and updates when a new circular replaces an old one. The main focus is **evaluation**: every layer (retriever, generator, full pipeline, updates, safety, speed, judge) is tested separately.

## Headline result

Without version tracking, **6 of 16** answers used a rule that had already been replaced. With version tracking, **0 of 16**.

| Full RAG, 16 answerable questions | Version tracking ON | OFF |
|---|---|---|
| Outdated answers | **0/16** | 6/16 |
| Gold chunk found | 14/16 | 11/16 |
| Valid citations | 16/16 | 10/16 |
| Refusal on 6 not-in-documents questions | 6/6 | 6/6 |

The same 6 questions go wrong with tracking off, and none go wrong only with it on (exact paired test, p ≈ 0.03). Small sample, see Limitations.

## What it does

- Reads 9 SEBI documents (regulations 1996 and 2026, master circulars 2018 to 2026, and two 2026 circulars about clause 5.9.1).
- Splits them into one chunk per clause (6,409 chunks) with page numbers.
- Marks each chunk **current, superseded, not_effective or pending_review** from issue date, effective date and replace links. Upload order does not matter.
- Searches with dense + BM25 (RRF merge) and a cross-encoder reranker. Only current chunks are returned by default, and a past date can be requested (`as_of`).
- Answers with citations, or says "Not found in the loaded documents."

The key test chain: clause 5.9.1 in the 2026 master circular → Mar 25 addendum (delays it) → Jul 10 circular (replaces both, effective Sep 1).

## Stack

Python, pdfplumber, fastembed (BAAI/bge-small-en-v1.5), local Qdrant, BM25, cross-encoder reranker (ms-marco-MiniLM-L-6-v2), Groq free tier (answer: `openai/gpt-oss-120b`, judge: `qwen/qwen3.8-27b`), DeepEval.

## Test set

30 hand-written questions, checked against the PDF text, never changed after writing:

| Type | Count |
|---|---|
| Current rule | 10 |
| Past date | 8 |
| What changed | 6 |
| Not in documents | 6 |

3 past-date questions are registry lookups and are skipped by the retrieval and generation evals, so each run has 16 answerable + 6 refusal questions.

## Results by layer

**1. Retriever alone** (k=5, reranker, no LLM, exact metrics)

| | Status filter ON | OFF |
|---|---|---|
| Recall@5 | 0.88 | 0.66 |
| Hit rate | 14/16 | 11/16 |
| MRR | 0.66 | 0.47 |
| Superseded chunks returned | 0/22 | 21/22 |

Earlier runs (k=6, no reranker): dense recall 0.63, BM25 0.84, hybrid 0.84. BM25 beat dense because the text is full of exact terms like "Regulation 42" and "5.9.1".

Reranker on/off (filter on): hit rate 13/16 → 14/16, recall 0.81 → 0.88. It fixed 2 questions (C07, P08) and broke 1 (P04). Net gain is small.

**2. Generator alone** (given the correct chunks)

Faithfulness 0.98, answer relevance 1.00, correctness 0.86, valid citations 14/16, refusal 6/6.

Bug found and fixed: with one source, the model cited non-existent [2][3] and copied internal document IDs into answers (one faithfulness score was 0.33). Fix: prompt says "sources numbered 1 to N, never invent a number", and source headers use readable titles.

**3. Retriever + generator**: see the headline table.

**4. Update tests** (search only, zero tokens): **19/19 passed**.
- Forward update: after "uploading" the Jul 10 circular, results switch to it and the old 5.9.1 and addendum disappear.
- Date awareness: as of 2026-08-01 the old rule still applies (new one starts Sep 1).
- Late backfill: a 2023 master circular uploaded late is never used for current-rule questions.
- Order: three upload orders give identical statuses and results.

**5. Safety**
- Fake circular claiming to replace a real rule (4 variants): worked **4/4 without** the approval gate, **0/4 with** it. The gate marks unapproved uploads as pending_review.
- Prompt injection (5 cases): **1/5 worked** (an official-looking notice saying a limit is 50%). A prompt-level defence did not help, so it was not added. The approval gate is the real defence.

**6. Operational** (Groq free tier)

| | p50 | p95 |
|---|---|---|
| Search, no reranker | 0.09 s | 0.11 s |
| Search, with reranker | 0.96 s | 1.16 s |
| Full cited answer | 1.86 s | 2.86 s |

About 1,842 tokens per question (1,552 in, 290 out). Size: PDFs 20.7 MB, chunks 6.8 MB, index 32 MB.

**7. Judge calibration** (I hand-labelled 44 answers: 32 real + 12 deliberately damaged)

| | Original rubric | Rewritten rubric (v2) |
|---|---|---|
| Exact agreement | 57% | 77% |
| Cohen's kappa | 0.30 | 0.45 |
| Weighted kappa | 0.44 | 0.56 |
| Spearman | 0.62 | 0.67 |

The original judge was too strict (called 14 of 33 correct answers "partly" because gold answers contain facts the question did not ask for). v2 fixes that but missed 2 of 9 damaged answers. v2 was tuned on the same 44 labels, so its gain is optimistic. Judge-based scores should be read as a lower bound.

## What didn't work

- **P05** (borrowing limit in June 2025) misses with and without the reranker. The rule sits at the end of a long chunk (Regulation 44) that is mostly about other rules, so the key words get diluted. Splitting at sub-clauses like (1), (2) would likely help.
- **P04** is found without the reranker but pushed out by it.
- **Addendum questions** (P07, W02, W03) still miss in the base state, because search does not use document type as a clue.
- **Prompt injection**: 1 of 5 attacks worked, and a prompt rule did not help.

## Limitations

- **Small test set (30 questions, 16 answerable).** I used Groq's free tier (about 8,000 tokens per minute, 200,000 per day, and hidden reasoning tokens count). A bigger set and a full judged run would take many days of quota. So I kept one hand-checked set and used cheap exact metrics (no LLM) wherever possible. Small differences are noise, and results are given as counts, not bare percentages. With a paid API, the next step is 60 to 100 questions plus confidence intervals.
- The test set was written by the same person who built the system. No second labeller.
- One domain, free-tier models. Latency and token numbers will not transfer to production.
- Judge agreement is only moderate (kappa 0.45).
- Tables and footnote numbers extract messily from PDFs, and chunks carry no chapter headings.
- Skipped on purpose (cost vs signal): chunk-size comparison, context-relevance metric, toxicity and PII metrics.

## Run it

Run from the project root (`SEBI-RAG/`), with `LLM_API_KEY`, `LLM_MODEL`, `JUDGE_MODEL` set in `.env`.

```
python src/extract_pages.py
python src/chunk_clauses.py
python src/build_index.py

python src/eval_retriever.py --state full
python src/eval_retriever.py --state full --all       # status filter OFF
python src/eval_rag.py --no-judge
python src/eval_updates.py
python src/eval_safety.py
python src/eval_ops.py
```