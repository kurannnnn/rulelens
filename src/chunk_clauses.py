"""Step 2: split pages into chunks, one chunk per clause (5.9.1, 1.5.2 ...).

Run from the project folder:
    python src/chunk_clauses.py
"""
import json
import re
from collections import defaultdict
from pathlib import Path

PAGES_FILE = Path("output/pages.jsonl")
REGISTRY_FILE = Path("data/document_registry.json")
OUT_FILE = Path("output/chunks.jsonl")

MAX_CHARS = 1800   # split chunks longer than this
MIN_CHARS = 200    # glue chunks shorter than this onto the next one

# A clause starts a line like "5.9.1. text", "8.4.4.1 text" or "42. text".
# Multi-level numbers may omit the final dot (older circulars do this).
CLAUSE_RE = re.compile(r"^(?:(\d{1,2}(?:\.\d{1,3}){1,4})\.?|(\d{1,2})\.)\s+\S")


def load_pages():
    # Group page records by document, in page order.
    docs = defaultdict(list)
    with open(PAGES_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            docs[r["doc_id"]].append(r)
    return docs


def split_into_clauses(pages):
    # Walk through every line. Start a new clause when a line begins with
    # a clause number. Remember which page each clause started on.
    clauses = []
    current = {"clause": "start", "ids": ["start"], "lines": [], "pages": [],
               "page_start": 1, "page_end": 1}
    for p in pages:
        for line in p["text"].split("\n"):
            m = CLAUSE_RE.match(line.strip())
            if m:
                num = m.group(1) or m.group(2)
                clauses.append(current)
                current = {"clause": num, "ids": [num], "lines": [], "pages": [],
                           "page_start": p["page"], "page_end": p["page"]}
            current["lines"].append(line)
            current["pages"].append(p["page"])
            current["page_end"] = p["page"]
    clauses.append(current)
    return [c for c in clauses if "".join(c["lines"]).strip()]


def merge_short(clauses):
    # Tiny pieces (usually just a heading) are glued onto the next clause.
    merged, carry = [], None
    for c in clauses:
        if carry:
            c["lines"] = carry["lines"] + c["lines"]
            c["pages"] = carry["pages"] + c["pages"]
            c["page_start"] = carry["page_start"]
            c["clause"] = carry["clause"]
            c["ids"] = carry["ids"] + c["ids"]   # keep every clause number inside
            carry = None
        if len("\n".join(c["lines"])) < MIN_CHARS:
            carry = c
        else:
            merged.append(c)
    if carry:
        merged.append(carry)
    return merged


def split_long(clause):
    # Cut a long clause into pieces under MAX_CHARS, at line breaks.
    pieces, buf, buf_pages = [], [], []
    for line, page in zip(clause["lines"], clause["pages"]):
        if sum(len(x) + 1 for x in buf) + len(line) > MAX_CHARS and buf:
            pieces.append((buf, buf_pages))
            buf, buf_pages = [], []
        buf.append(line)
        buf_pages.append(page)
    if buf:
        pieces.append((buf, buf_pages))
    return pieces


def build_context(doc_title, clause, headings):
    # Example: "Master Circular ... | 5. Investment ... > 5.9. Borrowings ..."
    parts = clause.split(".")
    trail = []
    for i in range(1, len(parts)):
        key = ".".join(parts[:i])
        if "." in key and key in headings and key != clause:
            trail.append(f"{key}. {headings[key]}")
    return doc_title + (" | " + " > ".join(trail) if trail else "")


def main():
    registry = {d["doc_id"]: d
                for d in json.load(open(REGISTRY_FILE, encoding="utf-8"))["documents"]}
    docs = load_pages()
    total = 0
    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for doc_id, pages in docs.items():
            info = registry[doc_id]
            raw = split_into_clauses(pages)

            # Remember the first line of every clause as its heading.
            headings = {}
            for c in raw:
                first = c["lines"][0].strip()
                headings.setdefault(c["clause"], re.sub(r"^[\d.]+\s*", "", first)[:70])

            chunks = merge_short(raw)
            n = 0
            for c in chunks:
                context = build_context(info["title"], c["clause"], headings)
                for part_no, (piece, piece_pages) in enumerate(split_long(c), start=1):
                    n += 1
                    rec = {
                        "chunk_id": f"{doc_id}#{n:05d}",
                        "doc_id": doc_id,
                        "doc_type": info["type"],
                        "issue_date": info["issue_date"],
                        "effective_date": info["effective_date"],
                        "clause": c["clause"],
                        "clauses": c["ids"],
                        "part": part_no,
                        "page_start": min(piece_pages),
                        "page_end": max(piece_pages),
                        "context": context,
                        "text": "\n".join(piece).strip(),
                    }
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            total += n
            print(f"{doc_id:<20} chunks={n}")
    print(f"\nDone. {total} chunks saved to {OUT_FILE}")


if __name__ == "__main__":
    main()