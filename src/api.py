"""RuleLens API.
Run from project root:  python -m uvicorn src.api:app --reload
Docs page:              http://127.0.0.1:8000/docs
"""
import json
import re
import sys
import time
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

# Lets files in src/ import each other (search.py imports status, etc.)
sys.path.insert(0, str(Path(__file__).resolve().parent))

from search import HybridSearcher
from answer import answer_question

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
REGISTRY = DATA / "document_registry.json"
LOADED = DATA / "loaded_docs.json"

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the searcher once (slow), reuse for every request
    state["searcher"] = HybridSearcher(rerank=True)
    yield
    state.clear()


app = FastAPI(title="RuleLens", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str
    as_of: Optional[str] = None  # "YYYY-MM-DD"


def parse_date(value: Optional[str], name: str) -> Optional[str]:
    """Return the date string if valid, else raise a clear 422 error."""
    if not value:
        return None
    try:
        date.fromisoformat(value)
    except ValueError:
        raise HTTPException(422, f"{name} must look like YYYY-MM-DD")
    return value


@app.get("/health")
def health():
    return {"ok": True, "docs_loaded": len(state["searcher"].loaded)}


@app.post("/ask")
def ask(req: AskRequest):
    if not req.question.strip():
        raise HTTPException(422, "question is empty")
    as_of = parse_date(req.as_of, "as_of")
    start = time.time()
    result = answer_question(state["searcher"], req.question, as_of=as_of)
    result["seconds"] = round(time.time() - start, 2)
    return result


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    doc_type: str = Form("circular"),
    issue_date: Optional[str] = Form(None),
    effective_date: Optional[str] = Form(None),
):
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "only PDF files are accepted")
    issue_date = parse_date(issue_date, "issue_date")
    effective_date = parse_date(effective_date, "effective_date")

    # Safe doc id from the filename (no paths, no odd characters)
    doc_id = re.sub(r"[^A-Za-z0-9_-]", "_", Path(file.filename).stem)[:80]
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    loaded = json.loads(LOADED.read_text(encoding="utf-8"))
    if doc_id in registry or doc_id in loaded["loaded"]:
        raise HTTPException(409, f"{doc_id} already exists")

    # 1. Save the PDF
    (DATA / "pdfs" / f"{doc_id}.pdf").write_bytes(await file.read())

    # 2. Registry entry, NOT approved -> status.py calls it pending_review
    registry[doc_id] = {
        "type": doc_type,
        "issue_date": issue_date,
        "effective_date": effective_date,
        "superseded_by": None,
        "clause_links": [],
        "approved": False,
    }
    REGISTRY.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    # 3. Add to the loaded list
    loaded["loaded"].append(doc_id)
    LOADED.write_text(json.dumps(loaded, indent=2), encoding="utf-8")

    return {
        "doc_id": doc_id,
        "status": "pending_review",
        "note": "Not searchable until approved and the index is rebuilt.",
    }