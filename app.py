from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

from semantic_search.config import DEFAULT_MODEL, DOCUMENTS_DIR, MODELS
from semantic_search.ingestion import copy_into_library
from semantic_search.pipeline import ingest_path
from semantic_search.vector_store import VectorStore

store = VectorStore()
MODEL_CHOICES = [(config["label"], key) for key, config in MODELS.items()]


def _render_results(results: list[dict]) -> str:
    if not results:
        return "### No matches found\n\nTry a different query, remove filters, or index some documents first."

    sections = [f"### Found {len(results)} relevant chunk(s)"]
    for rank, result in enumerate(results, start=1):
        meta = result["metadata"]
        source = html.escape(str(meta.get("source", "Unknown")))
        file_type = html.escape(str(meta.get("file_type", ""))).upper()
        page = html.escape(str(meta.get("page", "?")))
        score = result["similarity"]
        text = html.escape(result["text"])
        sections.append(
            f"<div class='result'>"
            f"<div class='result-head'><b>#{rank} · {source}</b>"
            f"<span>{file_type} · Page {page} · Similarity {score:.3f}</span></div>"
            f"<p>{text}</p></div>"
        )
    return "\n".join(sections)


def search_documents(query, model_key, top_k, file_type, source, page_min, page_max):
    if not query or not query.strip():
        return "Enter a search query."
    try:
        results = store.search(
            query=query,
            model_key=model_key,
            top_k=int(top_k),
            file_type=None if file_type == "All" else file_type,
            source=source.strip() or None,
            page_min=int(page_min) if page_min not in (None, "") else None,
            page_max=int(page_max) if page_max not in (None, "") else None,
        )
        return _render_results(results)
    except Exception as exc:
        return f"Search error: {html.escape(str(exc))}"


def ingest_uploads(uploaded_files, model_key):
    if not uploaded_files:
        return "Choose at least one PDF or TXT file."
    try:
        copied = copy_into_library(uploaded_files, DOCUMENTS_DIR)
        if not copied:
            return "No supported PDF/TXT files were selected."
        summary = ingest_path(DOCUMENTS_DIR, model_key)
        skipped = "\n".join(f"- {item}" for item in summary["skipped"]) or "None"
        return (
            f"### Ingestion complete\n"
            f"- Files found: {summary['files_found']}\n"
            f"- Files indexed in this run: {summary['files_indexed']}\n"
            f"- Chunks processed: {summary['chunks_indexed']}\n"
            f"- Model: {MODELS[model_key]['label']}\n"
            f"- Skipped: {skipped}\n\n"
            "The database is saved locally in `data/chroma`."
        )
    except Exception as exc:
        return f"Ingestion error: {html.escape(str(exc))}"


def refresh_sources(model_key):
    return gr.Dropdown(choices=["", *store.list_sources(model_key)], value="")


def show_stats():
    stats = store.collection_stats()
    return "\n".join(f"- **{MODELS[key]['label']}**: {count} chunks" for key, count in stats.items())


with gr.Blocks(title="Semantic Search Engine", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        "# Semantic Search Engine\n"
        "Search your local PDF and TXT documents using Sentence Transformer embeddings. "
        "No LLM or API key is required."
    )

    with gr.Tab("Search"):
        query = gr.Textbox(label="Search query", placeholder="e.g. How do solar panels convert sunlight into electricity?", lines=2)
        with gr.Row():
            model = gr.Dropdown(choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="Embedding model")
            top_k = gr.Slider(1, 20, value=5, step=1, label="Top-K results")
        with gr.Accordion("Metadata filters", open=False):
            with gr.Row():
                file_type = gr.Dropdown(["All", "pdf", "txt"], value="All", label="File type")
                source = gr.Dropdown(choices=[""], value="", allow_custom_value=True, label="Exact source filename")
                refresh_button = gr.Button("Refresh sources", size="sm")
            with gr.Row():
                page_min = gr.Number(label="Minimum page (optional)", precision=0, value=None)
                page_max = gr.Number(label="Maximum page (optional)", precision=0, value=None)
        search_button = gr.Button("Search documents", variant="primary")
        results = gr.HTML(label="Search results")
        search_button.click(search_documents, [query, model, top_k, file_type, source, page_min, page_max], results)
        query.submit(search_documents, [query, model, top_k, file_type, source, page_min, page_max], results)
        refresh_button.click(refresh_sources, [model], source)
        model.change(refresh_sources, [model], source)

    with gr.Tab("Add documents"):
        gr.Markdown(
            "Upload PDF or TXT files. The app copies them into `data/documents`, "
            "extracts text, splits it into chunks and indexes them with the selected model. "
            "For scanned/image-only PDFs, OCR is not included; use a text-based PDF."
        )
        uploads = gr.File(file_count="multiple", file_types=[".pdf", ".txt"], type="filepath", label="Choose PDF/TXT files")
        ingest_model = gr.Dropdown(choices=MODEL_CHOICES, value=DEFAULT_MODEL, label="Index with model")
        ingest_button = gr.Button("Ingest and index", variant="primary")
        ingest_output = gr.Markdown()
        ingest_button.click(ingest_uploads, [uploads, ingest_model], ingest_output)

    with gr.Tab("Database status"):
        stats_button = gr.Button("Refresh database status")
        stats_output = gr.Markdown(value=show_stats())
        stats_button.click(show_stats, outputs=stats_output)

    gr.Markdown(
        "---\n"
        "**Local-first:** uploaded files and the ChromaDB index remain on this computer. "
        "The first use of each embedding model downloads its model files from Hugging Face."
    )


if __name__ == "__main__":
    demo.launch()
