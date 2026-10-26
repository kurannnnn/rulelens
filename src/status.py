"""Step 4: decide which documents and clauses are current.

Status is computed from dates and links in the registry, and only counts
documents that are loaded. Nothing here depends on upload order.
"""
import json


def load_registry(path="data/document_registry.json"):
    docs = json.load(open(path, encoding="utf-8"))["documents"]
    return {d["doc_id"]: d for d in docs}


def load_loaded(path="data/loaded_docs.json"):
    return set(json.load(open(path, encoding="utf-8"))["loaded"])


def doc_status(registry, loaded, as_of):
    """Return {doc_id: 'current' | 'superseded' | 'not_effective'}.
    as_of is an ISO date string like '2026-10-16' (these compare correctly)."""
    status = {}
    for doc_id in loaded:
        d = registry[doc_id]
        if not d.get("approved", True):           # uploaded but not yet approved by a human
            status[doc_id] = "pending_review"
            continue
        if d["effective_date"] > as_of:
            status[doc_id] = "not_effective"      # rule has not started yet
            continue
        newer = d.get("superseded_by")
        # Replaced only if the newer document is loaded, APPROVED, and already in force.
        if (newer and newer in loaded and registry[newer].get("approved", True)
                and registry[newer]["effective_date"] <= as_of):
            status[doc_id] = "superseded"
        else:
            status[doc_id] = "current"
    return status


def clause_overrides(registry, loaded, as_of):
    """List of (target_doc, target_clause, replaced_by) for single clauses
    replaced by a newer loaded document that is already in force."""
    out = []
    for doc_id in loaded:
        d = registry[doc_id]
        if d["effective_date"] > as_of or not d.get("approved", True):
            continue                              # unapproved uploads cannot replace rules
        for link in d.get("clause_links", []):
            if link["relation"] == "supersedes" and link["target_doc"] in loaded:
                out.append((link["target_doc"], link["target_clause"], doc_id))
    return out


def chunk_status(chunk, doc_st, overrides):
    """Return 'current', 'superseded', 'not_effective' or 'not_loaded'."""
    st = doc_st.get(chunk["doc_id"])
    if st is None:
        return "not_loaded"
    if st != "current":
        return st
    for target_doc, clause, _by in overrides:
        if chunk["doc_id"] != target_doc:
            continue
        # clause 5.9.1 also covers 5.9.1.1 but not 5.9.10
        if any(c == clause or c.startswith(clause + ".") for c in chunk["clauses"]):
            return "superseded"
    return "current"