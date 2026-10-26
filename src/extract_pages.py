"""Step 1: read every PDF page by page and save the text with registry info.

Run from the project folder:
    pip install pdfplumber
    python src/extract_pages.py
"""
import json
from pathlib import Path

import pdfplumber

PDF_DIR = Path("data/pdfs")
REGISTRY_FILE = Path("data/document_registry.json")
OUT_FILE = Path("output/pages.jsonl")


def load_registry():
    # Read the registry and index it by file name for quick lookup.
    with open(REGISTRY_FILE, encoding="utf-8") as f:
        docs = json.load(f)["documents"]
    return {d["file"]: d for d in docs}


def extract_pdf(pdf_path, doc):
    # Return one record per page: text plus info from the registry.
    records = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_no, page in enumerate(pdf.pages, start=1):
            text = (page.extract_text() or "").strip()
            records.append({
                "doc_id": doc["doc_id"],
                "file": doc["file"],
                "doc_type": doc["type"],
                "issue_date": doc["issue_date"],
                "effective_date": doc["effective_date"],
                "page": page_no,
                "text": text,
            })
    return records


def main():
    registry = load_registry()
    OUT_FILE.parent.mkdir(exist_ok=True)

    total_pages = 0
    with open(OUT_FILE, "w", encoding="utf-8") as out:
        for file_name, doc in registry.items():
            pdf_path = PDF_DIR / file_name
            if not pdf_path.exists():
                print(f"MISSING: {file_name}")
                continue

            records = extract_pdf(pdf_path, doc)
            empty = sum(1 for r in records if len(r["text"]) < 20)
            for r in records:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")

            total_pages += len(records)
            print(f"{doc['doc_id']:<20} pages={len(records):<4} "
                  f"empty_pages={empty}")

    print(f"\nDone. {total_pages} pages saved to {OUT_FILE}")


if __name__ == "__main__":
    main()