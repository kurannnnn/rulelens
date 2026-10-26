"""Step 4: hybrid search that only returns rules that are in force.

Try it:
    python src/search.py "what are the rules for intraday borrowing"
    python src/search.py "expense ratio rules" --asof 2023-06-01
    python src/search.py "intraday borrowing" --all      (include old rules)
"""
import argparse
import json
import re
from datetime import date
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny
from rank_bm25 import BM25Okapi

from status import (chunk_status, clause_overrides, doc_status,
                    load_loaded, load_registry)

CHUNKS_FILE = Path("output/chunks.jsonl")
DB_PATH = "output/qdrant_db"
COLLECTION = "sebi_mf"
MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"   # small free cross-encoder
RERANK_CANDIDATES = 20
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class HybridSearcher:
    def __init__(self, rerank=False):
        self.chunks = [json.loads(l) for l in open(CHUNKS_FILE, encoding="utf-8")]
        self.registry = load_registry()
        self.loaded = load_loaded()          # documents "uploaded" so far
        self.model = TextEmbedding(MODEL_NAME)
        self.reranker = TextCrossEncoder(model_name=RERANK_MODEL) if rerank else None
        self.client = QdrantClient(path=DB_PATH)
        self.bm25 = BM25Okapi([tokenize(c["context"] + " " + c["text"])
                               for c in self.chunks])

    def allowed(self, as_of, include_superseded):
        # Which chunk positions may we search? Computed fresh on every call,
        # so a new upload or a different date changes the answer at once.
        doc_st = doc_status(self.registry, self.loaded, as_of)
        overrides = clause_overrides(self.registry, self.loaded, as_of)
        ok = {}
        for i, c in enumerate(self.chunks):
            st = chunk_status(c, doc_st, overrides)
            if st == "current" or (include_superseded and st == "superseded"):
                ok[i] = st
        return ok

    def dense(self, query, ok, k=20):
        vec = list(self.model.embed([QUERY_PREFIX + query]))[0].tolist()
        docs = list({self.chunks[i]["doc_id"] for i in ok})
        flt = Filter(must=[FieldCondition(key="doc_id", match=MatchAny(any=docs))])
        # Ask for extra hits, then drop clauses that were replaced.
        hits = self.client.query_points(COLLECTION, query=vec, limit=k * 4,
                                        query_filter=flt).points
        return [h.id for h in hits if h.id in ok][:k]

    def keyword(self, query, ok, k=20):
        scores = self.bm25.get_scores(tokenize(query))
        masked = np.full(len(scores), -1.0)
        idx = list(ok)
        masked[idx] = scores[idx]
        return [int(i) for i in np.argsort(-masked)[:k] if int(i) in ok]

    def search(self, query, k=10, as_of=None, include_superseded=False,
               w_dense=0.5, w_keyword=0.5, rrf_k=60):
        as_of = as_of or date.today().isoformat()
        ok = self.allowed(as_of, include_superseded)
        scores = {}
        for weight, ranking in ((w_dense, self.dense(query, ok)),
                                (w_keyword, self.keyword(query, ok))):
            for rank, idx in enumerate(ranking, start=1):
                scores[idx] = scores.get(idx, 0) + weight / (rrf_k + rank)
        n = RERANK_CANDIDATES if self.reranker else k
        top = sorted(scores, key=scores.get, reverse=True)[:n]
        if self.reranker and top:          # re-score each chunk against the question
            docs = [self.chunks[i]["context"] + "\n" + self.chunks[i]["text"] for i in top]
            rs = list(self.reranker.rerank(query, docs))
            top = [i for _, i in sorted(zip(rs, top), reverse=True)][:k]
        return [{**self.chunks[i], "status": ok[i]} for i in top]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+")
    ap.add_argument("--asof", help="date, e.g. 2023-06-01 (default: today)")
    ap.add_argument("--all", action="store_true", help="include old rules")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rerank", action="store_true")
    a = ap.parse_args()

    searcher = HybridSearcher(rerank=a.rerank)
    for c in searcher.search(" ".join(a.query), k=a.k, as_of=a.asof,
                             include_superseded=a.all):
        print(f"{c['doc_id']} [{c['status']}] | clauses {c['clauses'][:3]} | "
              f"pages {c['page_start']}-{c['page_end']}")
        print("   ", c["text"][:150].replace("\n", " "), "\n")