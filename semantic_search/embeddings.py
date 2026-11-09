from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_BATCH_SIZE, MODELS


@lru_cache(maxsize=2)
def load_model(model_key: str) -> SentenceTransformer:
    if model_key not in MODELS:
        raise ValueError(f"Unknown model '{model_key}'. Choose one of: {', '.join(MODELS)}")
    model_id = MODELS[model_key]["model_id"]
    print(f"Loading embedding model: {model_id}")
    return SentenceTransformer(model_id)


def embed_documents(texts: list[str], model_key: str) -> np.ndarray:
    model = load_model(model_key)
    prefix = MODELS[model_key]["prefix_document"]
    prepared = [prefix + text for text in texts]
    vectors = model.encode(
        prepared,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=len(prepared) > EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(query: str, model_key: str) -> np.ndarray:
    model = load_model(model_key)
    prepared = MODELS[model_key]["prefix_query"] + query.strip()
    vector = model.encode(
        prepared,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(vector, dtype=np.float32)
