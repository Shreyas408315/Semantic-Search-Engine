# Semantic Search Engine (No LLM Required)

A local semantic-search capstone built with Python, Sentence Transformers, ChromaDB, PyMuPDF, and Gradio. It ingests PDF/TXT files, extracts and chunks text, creates embeddings, stores vectors and metadata, searches by semantic similarity, supports metadata filters, compares MiniLM with E5, and evaluates retrieval quality.

## What is included

- **PDF and TXT ingestion** using PyMuPDF and UTF-8 text reading.
- **Text cleaning and overlapping chunking** with source, file type, page, and chunk-index metadata.
- **MiniLM:** `sentence-transformers/all-MiniLM-L6-v2`.
- **E5 comparison:** `intfloat/e5-small-v2`, using `query: ` and `passage: ` prefixes.
- **ChromaDB persistence:** separate cosine-distance collections for each embedding model under `data/chroma/`.
- **Search filters:** file type, exact source filename, and page range.
- **Evaluation:** Precision@K, Recall@K, and Mean Reciprocal Rank (MRR), using editable source-level relevance labels.
- **Gradio interface:** upload documents, index them, search, and inspect database counts.
- Three small sample TXT documents and a starter evaluation set.

There is no LLM, answer generation, API key, or cloud vector database. The first time you use a model, Sentence Transformers downloads its model files from Hugging Face; subsequent runs use the local model cache.

## 1. Requirements

- Python **3.10, 3.11, or 3.12** recommended.
- Internet access for the initial Python package and embedding-model downloads.
- About 2–4 GB of free disk space is a reasonable starting allowance for Python dependencies, model cache, and project data; actual usage varies by platform and installed PyTorch build.
- CPU is sufficient for a small document set. A compatible NVIDIA GPU can speed up embedding for larger collections, but is optional.

Scanned/image-only PDFs are not OCR-processed in this project. Use text-based PDFs, or OCR them separately before ingestion.

## 2. Installation

Open a terminal in the extracted project folder.

### Windows (PowerShell)

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, you can use Command Prompt instead:

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If the `python3.11` command is not available, use another installed Python version in the recommended 3.10–3.12 range.

## 3. Run the application

With the virtual environment activated:

```bash
python app.py
```

Open the local URL printed in the terminal (usually `http://127.0.0.1:7860`).

### First run walkthrough

1. Open the **Add documents** tab.
2. Upload PDF/TXT files, or use the three sample TXT documents already in `data/documents/`.
3. Select **MiniLM** and click **Ingest and index**.
4. Open the **Search** tab and try a natural-language question, such as `How do solar panels generate electricity?`.
5. Adjust Top-K and expand **Metadata filters** to filter by type, filename, or page range.
6. To compare E5, return to **Add documents**, select E5 and ingest the same document library. This builds the E5 collection separately.
7. Switch the Search model to E5 and rerun the same queries.
8. Use the **Database status** tab to see chunk counts for each model.

The app copies uploaded files into `data/documents/`. The persistent ChromaDB database is stored in `data/chroma/`. Both are local project data.

## 4. Index documents from the command line

Index the project's document folder with MiniLM:

```bash
python -m semantic_search.cli ingest --path data/documents --model minilm
```

Index the same folder with E5:

```bash
python -m semantic_search.cli ingest --path data/documents --model e5
```

You can also ingest a single file or a different folder:

```bash
python -m semantic_search.cli ingest --path "C:/Users/you/Documents/handbook.pdf" --model minilm
```

The CLI supports PDF/TXT files. Re-indexing is safe: chunk IDs are deterministic and ChromaDB upserts existing IDs rather than creating duplicate records for unchanged chunks.

## 5. Search from the command line

```bash
python -m semantic_search.cli search "How do solar panels convert sunlight into electricity?" --model minilm --top-k 5
```

Metadata-filtered examples:

```bash
python -m semantic_search.cli search "phishing and stolen passwords" --model e5 --file-type txt
python -m semantic_search.cli search "solar energy storage" --model minilm --source renewable_energy.txt
python -m semantic_search.cli search "machine learning evaluation" --model minilm --page-min 1 --page-max 3
```

The similarity displayed by the app is `1 - cosine_distance` from ChromaDB. It is a retrieval similarity score, not a calibrated probability that a result is correct.

## 6. Evaluate MiniLM vs E5

First index the same document set with **both** models. Then run:

```bash
python -m semantic_search.cli evaluate --k 5
```

The evaluation set is `evaluation/queries.json`. Edit it to add your own queries and relevant source filenames:

```json
[
  {
    "query": "What is the purpose of a firewall?",
    "relevant_sources": ["cybersecurity_basics.txt"]
  }
]
```

Metrics:

- **Precision@K:** fraction of the top K retrieved chunks whose source is labeled relevant.
- **Recall@K:** fraction of labeled relevant source documents retrieved at least once in the top K chunks.
- **MRR:** reciprocal rank of the first result whose source is labeled relevant, averaged over queries.

The included evaluation set is a demonstration, not a statistically robust benchmark. For a meaningful comparison, create a larger set of realistic queries and manually label all relevant documents. Use the exact same corpus, chunking settings, K, and relevance judgments for both models.

## 7. Project structure

```text
semantic_search_engine/
├── app.py                     # Gradio interface
├── run_app.py                 # Convenience launcher
├── requirements.txt
├── README.md
├── semantic_search/
│   ├── config.py               # Models, paths, chunking settings
│   ├── ingestion.py            # PDF/TXT extraction and chunking
│   ├── embeddings.py           # MiniLM/E5 embedding generation
│   ├── vector_store.py         # ChromaDB indexing, filtering, search
│   ├── pipeline.py             # End-to-end document ingestion
│   ├── evaluation.py           # P@K, R@K, MRR comparison
│   └── cli.py                  # Command-line operations
├── data/
│   ├── documents/              # Place or upload source files here
│   └── chroma/                 # Persistent vector database (created on run)
└── evaluation/
    └── queries.json            # Editable evaluation queries and labels
```

## 8. Configuration

Default settings are in `semantic_search/config.py`:

- `CHUNK_SIZE = 1000` characters.
- `CHUNK_OVERLAP = 150` characters.
- `EMBEDDING_BATCH_SIZE = 32`.
- Top-K can be set from 1 to 20.

You can override the chunk and batch settings with environment variables before indexing:

```bash
# macOS / Linux
export SEMANTIC_SEARCH_CHUNK_SIZE=1200
export SEMANTIC_SEARCH_CHUNK_OVERLAP=180
export SEMANTIC_SEARCH_BATCH_SIZE=16
```

```powershell
# Windows PowerShell
$env:SEMANTIC_SEARCH_CHUNK_SIZE="1200"
$env:SEMANTIC_SEARCH_CHUNK_OVERLAP="180"
$env:SEMANTIC_SEARCH_BATCH_SIZE="16"
```

Changing chunk settings affects future ingestion. For a fair model comparison, use the same settings for both models and re-index both collections after changing them.

## 9. Troubleshooting

**The first model load takes time:** expected. The model is downloaded once and cached locally.

**`No results` appears:** confirm that the same model used for indexing is selected for searching, that the collection has chunks (Database status tab), and that metadata filters are not too restrictive.

**A scanned PDF returns no text:** OCR is not included. Convert it to a searchable PDF or extract its text with an OCR tool first.

**Out of memory:** reduce `SEMANTIC_SEARCH_BATCH_SIZE`, try CPU mode, or ingest fewer documents at a time.

**Port 7860 is busy:** Gradio will usually select another available local port; use the URL printed in the terminal.

**Want to reset the index?** Stop the app, close the Python process, and remove `data/chroma/`. This deletes both model collections; it does not delete source files in `data/documents/`.

## 10. Scope and limitations

This is a semantic retrieval engine. It returns the most relevant document chunks and their metadata; it does not synthesize answers, summarize documents, or cite a generated answer. Retrieval quality depends on the corpus, chunking strategy, embedding model, and relevance labels. The local app is intended for development and personal use, not as a hardened multi-user production service.
