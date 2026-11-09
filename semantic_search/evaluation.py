from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import EVALUATION_DIR, MODELS
from .vector_store import VectorStore


def load_queries(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or (EVALUATION_DIR / "queries.json")
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Evaluation JSON must contain a list of query objects.")
    for index, item in enumerate(data):
        if not item.get("query") or not item.get("relevant_sources"):
            raise ValueError(f"Evaluation item {index} needs 'query' and 'relevant_sources'.")
    return data


def evaluate_model(model_key: str, queries: list[dict[str, Any]], k: int = 5) -> dict[str, float]:
    store = VectorStore()
    precisions: list[float] = []
    recalls: list[float] = []
    reciprocal_ranks: list[float] = []
    evaluated = 0

    for item in queries:
        relevant = {str(source) for source in item["relevant_sources"]}
        results = store.search(item["query"], model_key=model_key, top_k=k)
        ranked_sources = [str(result["metadata"].get("source", "")) for result in results]
        hits = [source in relevant for source in ranked_sources]
        relevant_retrieved = len({source for source in ranked_sources if source in relevant})

        precision_denominator = min(k, len(ranked_sources))
        precisions.append(sum(hits) / precision_denominator if precision_denominator else 0.0)
        recalls.append(relevant_retrieved / len(relevant))
        first_relevant_rank = next((i + 1 for i, hit in enumerate(hits) if hit), None)
        reciprocal_ranks.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)
        evaluated += 1

    if evaluated == 0:
        raise ValueError("No evaluation queries were found.")
    return {
        f"precision@{k}": sum(precisions) / evaluated,
        f"recall@{k}": sum(recalls) / evaluated,
        "mrr": sum(reciprocal_ranks) / evaluated,
        "queries_evaluated": float(evaluated),
    }


def compare_models(k: int = 5, query_path: Path | None = None) -> pd.DataFrame:
    queries = load_queries(query_path)
    rows = []
    for model_key, model_config in MODELS.items():
        metrics = evaluate_model(model_key, queries, k=k)
        rows.append({"model": model_config["label"], **metrics})
    return pd.DataFrame(rows)
