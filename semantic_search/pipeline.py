from __future__ import annotations

from pathlib import Path

from .ingestion import discover_documents, extract_document, pages_to_chunks
from .vector_store import VectorStore


def ingest_path(path: str | Path, model_key: str = "minilm") -> dict:
    """Extract, chunk, embed, and index every PDF/TXT file found at path."""
    input_path = Path(path).expanduser().resolve()
    files = discover_documents(input_path)
    if not files:
        raise ValueError(f"No PDF or TXT documents found at: {input_path}")

    store = VectorStore()
    indexed_chunks = 0
    indexed_files = 0
    skipped_files: list[str] = []

    for file_path in files:
        try:
            pages = extract_document(file_path)
            chunks = pages_to_chunks(pages)
            if not chunks:
                skipped_files.append(f"{file_path.name} (no extractable text)")
                continue
            indexed_chunks += store.index_chunks(chunks, model_key)
            indexed_files += 1
            print(f"Indexed {file_path.name}: {len(chunks)} chunks")
        except Exception as exc:
            skipped_files.append(f"{file_path.name} ({type(exc).__name__}: {exc})")

    return {
        "files_found": len(files),
        "files_indexed": indexed_files,
        "chunks_indexed": indexed_chunks,
        "skipped": skipped_files,
        "model": model_key,
    }
