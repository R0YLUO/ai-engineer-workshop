"""Step 2 of RAG: turn each chunk into a vector and store it.

An embedding model maps text to a list of numbers (a vector) so that texts with
similar MEANING end up close together - even when they share no words.
"How cold do your trucks stay?" lands near "maintain a range of 2°C to 8°C".

We use a small open-source model (BAAI/bge-small-en-v1.5, 384 dimensions) that
runs locally via `fastembed` - no API key needed. The first run downloads it.

Our "vector database" is just a Python list of chunks plus a numpy matrix of
their vectors, held in memory.

Try it:  uv run python -m rag.embedding
"""

import numpy as np
from fastembed import TextEmbedding

from rag.chunking import DOCUMENT_PATH, chunk_document

_model = None


def get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding("BAAI/bge-small-en-v1.5")
    return _model


def embed_documents(texts: list[str]) -> np.ndarray:
    return np.array(list(get_model().passage_embed(texts)))


def embed_query(text: str) -> np.ndarray:
    # Queries are embedded slightly differently from documents for this model.
    return np.array(list(get_model().query_embed(text)))[0]


class VectorStore:
    def __init__(self):
        self.chunks: list[str] = []
        self.vectors = np.empty((0, 384))

    def add(self, chunks: list[str]) -> None:
        self.chunks.extend(chunks)
        self.vectors = np.vstack([self.vectors, embed_documents(chunks)])


def build_vector_store() -> VectorStore:
    """Load the service guide, chunk it, embed it, store it."""
    store = VectorStore()
    store.add(chunk_document(DOCUMENT_PATH.read_text()))
    return store


if __name__ == "__main__":
    store = build_vector_store()
    print(f"Stored {len(store.chunks)} chunks. Vector matrix shape: {store.vectors.shape}")
    print(f"First chunk's vector (first 8 of 384 numbers): {store.vectors[0][:8].round(3)}")
