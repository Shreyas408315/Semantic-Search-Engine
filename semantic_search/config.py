from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
CHROMA_DIR = DATA_DIR / "chroma"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# Each model has an independent ChromaDB collection. Do not share a collection
# between models because their embedding spaces are not interchangeable.
MODELS = {
    "minilm": {
        "label": "MiniLM (all-MiniLM-L6-v2)",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "collection": "semantic_search_minilm",
        "prefix_query": "",
        "prefix_document": "",
    },
    "e5": {
        "label": "E5 (intfloat/e5-small-v2)",
        "model_id": "intfloat/e5-small-v2",
        "collection": "semantic_search_e5",
        "prefix_query": "query: ",
        "prefix_document": "passage: ",
    },
}

DEFAULT_MODEL = os.getenv("SEMANTIC_SEARCH_MODEL", "minilm").lower()
if DEFAULT_MODEL not in MODELS:
    DEFAULT_MODEL = "minilm"

CHUNK_SIZE = int(os.getenv("SEMANTIC_SEARCH_CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("SEMANTIC_SEARCH_CHUNK_OVERLAP", "150"))
EMBEDDING_BATCH_SIZE = int(os.getenv("SEMANTIC_SEARCH_BATCH_SIZE", "32"))
DEFAULT_TOP_K = 5
MAX_TOP_K = 20

for directory in (DATA_DIR, DOCUMENTS_DIR, CHROMA_DIR, EVALUATION_DIR):
    directory.mkdir(parents=True, exist_ok=True)
