from __future__ import annotations

import hashlib
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF

from .config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENTS_DIR

SUPPORTED_SUFFIXES = {".pdf", ".txt"}


@dataclass
class ExtractedPage:
    text: str
    source: str
    file_type: str
    page: int


@dataclass
class TextChunk:
    chunk_id: str
    text: str
    metadata: dict


def clean_text(text: str) -> str:
    """Normalize extracted text while preserving paragraph boundaries."""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\t\f\v]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(path: Path) -> list[ExtractedPage]:
    pages: list[ExtractedPage] = []
    with fitz.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(
                    ExtractedPage(
                        text=text,
                        source=path.name,
                        file_type="pdf",
                        page=page_number,
                    )
                )
    return pages


def extract_txt(path: Path) -> list[ExtractedPage]:
    # utf-8-sig handles regular UTF-8 as well as UTF-8 files with a BOM.
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raw = path.read_text(encoding="utf-8", errors="replace")
    text = clean_text(raw)
    if not text:
        return []
    return [ExtractedPage(text=text, source=path.name, file_type="txt", page=1)]


def extract_document(path: Path) -> list[ExtractedPage]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf(path)
    if suffix == ".txt":
        return extract_txt(path)
    raise ValueError(f"Unsupported file type: {path.suffix}. Use PDF or TXT.")


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split one long block on whitespace, using a character window as a fallback."""
    words = text.split()
    chunks: list[str] = []
    current_words: list[str] = []
    current_length = 0

    for word in words:
        extra = len(word) + (1 if current_words else 0)
        if current_words and current_length + extra > chunk_size:
            chunks.append(" ".join(current_words))
            # Retain a suffix near the requested overlap size.
            retained: list[str] = []
            retained_length = 0
            for previous_word in reversed(current_words):
                addition = len(previous_word) + (1 if retained else 0)
                if retained and retained_length + addition > overlap:
                    break
                retained.append(previous_word)
                retained_length += addition
            current_words = list(reversed(retained))
            current_length = len(" ".join(current_words))

        current_words.append(word)
        current_length = len(" ".join(current_words))

    if current_words:
        chunks.append(" ".join(current_words))
    return chunks


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Create overlapping chunks, preferring paragraph boundaries."""
    if chunk_size < 100:
        raise ValueError("chunk_size must be at least 100 characters")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_text(paragraph, chunk_size, overlap))
            continue

        candidate = f"{current}\n\n{paragraph}" if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            # Carry a small text suffix from the previous chunk.
            suffix = current[-overlap:].strip() if overlap and current else ""
            current = f"{suffix}\n\n{paragraph}".strip() if suffix else paragraph
            if len(current) > chunk_size:
                chunks.extend(_split_long_text(current, chunk_size, overlap))
                current = ""

    if current:
        chunks.append(current)

    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _stable_chunk_id(source: str, page: int, chunk_index: int, text: str) -> str:
    payload = f"{source}|{page}|{chunk_index}|{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pages_to_chunks(pages: Iterable[ExtractedPage]) -> list[TextChunk]:
    output: list[TextChunk] = []
    for page in pages:
        page_chunks = chunk_text(page.text)
        for chunk_index, text in enumerate(page_chunks):
            output.append(
                TextChunk(
                    chunk_id=_stable_chunk_id(page.source, page.page, chunk_index, text),
                    text=text,
                    metadata={
                        "source": page.source,
                        "file_type": page.file_type,
                        "page": int(page.page),
                        "chunk_index": int(chunk_index),
                    },
                )
            )
    return output


def discover_documents(path: Path) -> list[Path]:
    if path.is_file():
        candidates = [path]
    elif path.is_dir():
        candidates = sorted(p for p in path.rglob("*") if p.is_file())
    else:
        raise FileNotFoundError(f"Input path does not exist: {path}")
    return [p for p in candidates if p.suffix.lower() in SUPPORTED_SUFFIXES]


def copy_into_library(paths: Iterable[str | Path], destination: Path = DOCUMENTS_DIR) -> list[Path]:
    """Copy uploaded files into the project document library, avoiding name collisions."""
    destination.mkdir(parents=True, exist_ok=True)
    copied: list[Path] = []
    for raw_path in paths:
        source_path = Path(raw_path)
        if not source_path.is_file() or source_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue
        target = destination / source_path.name
        if source_path.resolve() != target.resolve():
            if target.exists() and source_path.read_bytes() != target.read_bytes():
                stem, suffix = target.stem, target.suffix
                digest = hashlib.sha1(source_path.read_bytes()).hexdigest()[:8]
                target = destination / f"{stem}_{digest}{suffix}"
            shutil.copy2(source_path, target)
        copied.append(target)
    return copied
