"""Step 3 of RAG: find the chunks closest in meaning to a question.

Embed the question, then compare it to every stored vector with cosine
similarity (1.0 = same direction/meaning, 0 = unrelated). Return the top k.

Try it:  uv run python -m rag.retrieval "how cold do refrigerated trucks stay?"
"""

import sys

import numpy as np

from rag.embedding import VectorStore, build_vector_store, embed_query


def cosine_similarity(query: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    return (vectors @ query) / (np.linalg.norm(vectors, axis=1) * np.linalg.norm(query))


def search(store: VectorStore, query: str, k: int = 3) -> list[tuple[float, str]]:
    scores = cosine_similarity(embed_query(query), store.vectors)
    best = np.argsort(scores)[::-1][:k]
    return [(float(scores[i]), store.chunks[i]) for i in best]


if __name__ == "__main__":
    question = sys.argv[1] if len(sys.argv) > 1 else "Can you carry gas bottles?"
    for score, chunk in search(build_vector_store(), question):
        print(f"--- score {score:.3f} ---\n{chunk}\n")
