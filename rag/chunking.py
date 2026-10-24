"""Step 1 of RAG: split a long document into small, self-contained chunks.

Why chunk? We want to put only the RELEVANT parts of a document in front of the
model. Small chunks = precise retrieval. But chunks that are too small lose
their meaning ("It costs $30" - what costs $30?).

Strategy used here (simple and works well for structured docs):
  1. Split the document on its "## " section headings.
  2. If a section is still too long, split it on paragraph boundaries.
  3. Prefix every chunk with its section heading so it keeps its context.

Try it:  uv run python -m rag.chunking
"""

from pathlib import Path

DOCUMENT_PATH = Path(__file__).parent.parent / "data" / "documents" / "service_guide.md"


def chunk_document(text: str, max_chars: int = 800) -> list[str]:
    chunks = []
    for section in text.split("\n## ")[1:]:  # [0] is the document title
        heading, _, body = section.partition("\n")
        paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]

        current = ""
        for paragraph in paragraphs:
            if current and len(current) + len(paragraph) > max_chars:
                chunks.append(f"## {heading}\n{current.strip()}")
                current = ""
            current += paragraph + "\n\n"
        if current:
            chunks.append(f"## {heading}\n{current.strip()}")
    return chunks


if __name__ == "__main__":
    chunks = chunk_document(DOCUMENT_PATH.read_text())
    for i, chunk in enumerate(chunks):
        print(f"--- chunk {i} ({len(chunk)} chars) ---\n{chunk}\n")
    print(f"Total chunks: {len(chunks)}")
