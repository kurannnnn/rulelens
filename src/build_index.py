"""Step 3a: turn every chunk into a vector and store it in Qdrant.

Run from the project folder:
    pip install qdrant-client fastembed rank-bm25
    python src/build_index.py
"""
import json
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import numpy as np
from fastembed import TextEmbedding

CHUNKS_FILE = Path("output/chunks.jsonl")
DB_PATH = "output/qdrant_db"          # Qdrant saves here, no Docker needed
COLLECTION = "sebi_mf"
MODEL_NAME = "BAAI/bge-small-en-v1.5"  # small and free, 384 numbers per chunk


def main():
    chunks = [json.loads(line) for line in open(CHUNKS_FILE, encoding="utf-8")]
    print(f"{len(chunks)} chunks loaded")

    # We embed the context line plus the text, so the vector knows the source.
    texts = [c["context"] + "\n" + c["text"] for c in chunks]

    model = TextEmbedding(MODEL_NAME)
    print("Embedding... this takes a few minutes")
    vectors = np.array(list(model.embed(texts, batch_size=32)))

    client = QdrantClient(path=DB_PATH)
    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vectors.shape[1],
                                    distance=Distance.COSINE),
    )

    # Upload in batches. Each point = vector + the full chunk as payload.
    batch = 256
    for start in range(0, len(chunks), batch):
        points = [
            PointStruct(id=i, vector=vectors[i].tolist(), payload=chunks[i])
            for i in range(start, min(start + batch, len(chunks)))
        ]
        client.upsert(collection_name=COLLECTION, points=points)

    print(f"Done. {len(chunks)} vectors stored in {DB_PATH}")


if __name__ == "__main__":
    main()