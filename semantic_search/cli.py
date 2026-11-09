from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import DOCUMENTS_DIR, MODELS
from .evaluation import compare_models
from .pipeline import ingest_path
from .vector_store import VectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local semantic search engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Index PDF/TXT files")
    ingest_parser.add_argument("--path", default=str(DOCUMENTS_DIR), help="File or directory to ingest")
    ingest_parser.add_argument("--model", choices=MODELS, default="minilm")

    search_parser = subparsers.add_parser("search", help="Search indexed documents")
    search_parser.add_argument("query", help="Natural-language search query")
    search_parser.add_argument("--model", choices=MODELS, default="minilm")
    search_parser.add_argument("--top-k", type=int, default=5)
    search_parser.add_argument("--file-type", choices=["pdf", "txt"])
    search_parser.add_argument("--source", help="Exact source filename filter")
    search_parser.add_argument("--page-min", type=int)
    search_parser.add_argument("--page-max", type=int)

    evaluate_parser = subparsers.add_parser("evaluate", help="Compare MiniLM and E5")
    evaluate_parser.add_argument("--k", type=int, default=5)
    evaluate_parser.add_argument("--queries", type=Path, default=None)

    subparsers.add_parser("stats", help="Show indexed chunk counts")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "ingest":
        summary = ingest_path(args.path, args.model)
        print(json.dumps(summary, indent=2))
    elif args.command == "search":
        results = VectorStore().search(
            args.query,
            model_key=args.model,
            top_k=args.top_k,
            file_type=args.file_type,
            source=args.source,
            page_min=args.page_min,
            page_max=args.page_max,
        )
        if not results:
            print("No results. Check that documents are indexed and filters match.")
        for rank, result in enumerate(results, start=1):
            metadata = result["metadata"]
            print(f"\n#{rank} | similarity={result['similarity']:.4f} | {metadata.get('source')} | page {metadata.get('page')}")
            print(result["text"][:1200])
    elif args.command == "evaluate":
        print(compare_models(k=args.k, query_path=args.queries).to_string(index=False))
    elif args.command == "stats":
        print(json.dumps(VectorStore().collection_stats(), indent=2))


if __name__ == "__main__":
    main()
