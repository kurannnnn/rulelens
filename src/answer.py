"""Step 5: answer a question using only the retrieved, current chunks.

Works with any OpenAI-compatible API (Groq, Google Gemini, OpenRouter...).
Setup for Groq (free, PowerShell, once per terminal):
    $env:LLM_API_KEY = "your-key-here"
    $env:LLM_MODEL = "model-name-from-groq-console"
Optional (default is Groq):
    $env:LLM_BASE_URL = "https://api.groq.com/openai/v1"

Run:
    python src/answer.py "What are the rules for intraday borrowing by mutual funds?"
    python src/answer.py "same question" --asof 2026-08-01
"""
import argparse
import json
import os
import re
from datetime import date

import time

import httpx

import env_loader  # noqa: F401  (loads .env)

from search import HybridSearcher

BASE_URL = os.environ.get("LLM_BASE_URL", "https://api.groq.com/openai/v1")
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")  # check Groq's list
TOP_K = 5

SYSTEM_PROMPT = """You answer questions about SEBI mutual fund regulation.

Rules:
1. Use ONLY the numbered sources given. Do not use outside knowledge.
2. Every sentence of your answer must end with a source number in square brackets,
   like [1] or [2][3]. This applies even to one-line answers. Only use source
   numbers that exist in the list. Never invent a number.
3. If the sources do not contain enough to answer, set "enough_evidence" to false
   and write "Not found in the loaded documents." as the answer.
4. When a rule has an effective date, say it. Today's date is {today}.
5. Never give investment advice. Only explain what the documents say.

Reply with JSON only, no other text:
{{"answer": "...", "enough_evidence": true}}"""


def build_sources(chunks):
    # Each source gets a number plus a header the model can quote.
    blocks = []
    for i, c in enumerate(chunks, start=1):
        title = c["context"].split(" | ")[0]      # readable name, not the internal ID
        head = (f"[{i}] {title} | issued {c['issue_date']} | "
                f"effective {c['effective_date']} | "
                f"clauses {', '.join(c['clauses'][:4])} | "
                f"pages {c['page_start']}-{c['page_end']}")
        blocks.append(head + "\n" + c["text"])
    return "\n\n".join(blocks)


def ask_llm(question, chunks, today):
    body = {
        "model": MODEL,
        "temperature": 0,
        "max_tokens": 1500,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT.format(today=today)},
            {"role": "user",
             "content": f"SOURCES (numbered 1 to {len(chunks)}):\n{build_sources(chunks)}\n\nQUESTION: {question}"},
        ],
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    for attempt in range(8):                 # free tiers rate-limit: wait, retry
        r = httpx.post(f"{BASE_URL}/chat/completions", json=body,
                       headers=headers, timeout=60)
        if r.status_code != 429:
            break
        low = r.text.lower()
        if "per day" in low or "tpd" in low:   # daily cap: waiting a minute won't help
            break
        time.sleep(10 * (attempt + 1))
    if r.status_code >= 400:
        print("API ERROR", r.status_code, r.text[:600])   # shows which limit and when it resets
    r.raise_for_status()
    raw = r.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(raw[raw.index("{"): raw.rindex("}") + 1])
    except ValueError:
        return {"answer": raw, "enough_evidence": False, "parse_error": True}


def check_citations(result, chunks):
    # Returns a list of problems. Empty list = citations look fine.
    problems = []
    if not result.get("enough_evidence"):
        return problems
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", result["answer"])}
    if not cited:
        problems.append("answer has no citations")
    for n in cited:
        if n < 1 or n > len(chunks):
            problems.append(f"citation [{n}] does not exist")
        elif chunks[n - 1]["status"] != "current":
            problems.append(f"citation [{n}] is not a current rule")
    return problems


def answer_question(searcher, question, as_of=None):
    today = as_of or date.today().isoformat()
    chunks = searcher.search(question, k=TOP_K, as_of=today)

    result = ask_llm(question, chunks, today)
    problems = check_citations(result, chunks)
    if problems:                      # one retry, then give up honestly
        result = ask_llm(question, chunks, today)
        problems = check_citations(result, chunks)

    return {"question": question, "as_of": today, "result": result,
            "citation_problems": problems, "sources": chunks}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="+")
    ap.add_argument("--asof", help="date, e.g. 2026-08-01 (default: today)")
    a = ap.parse_args()

    searcher = HybridSearcher(rerank=True)
    out = answer_question(searcher, " ".join(a.question), a.asof)

    print("\nANSWER:", out["result"]["answer"])
    if out["citation_problems"]:
        print("\nCITATION PROBLEMS:", out["citation_problems"])
    print("\nSOURCES:")
    for i, c in enumerate(out["sources"], start=1):
        print(f"[{i}] {c['doc_id']} | clauses {c['clauses'][:3]} | "
              f"pages {c['page_start']}-{c['page_end']} | {c['status']}")