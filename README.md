# DocuQuery — Hybrid RAG Pipeline with Reranking

**DocuQuery** is a full-stack document question-answering application being built to demonstrate a practical Retrieval-Augmented Generation (RAG) pipeline. The planned pipeline combines keyword and semantic retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, and evidence-grounded LLM answers with source citations.

> **Foundation and PDF ingestion.** The current implementation supports PDF upload, validation, page-aware text extraction, overlapping chunk creation, and a React interface for inspecting the result. Retrieval, reranking, LLM generation, persistence, and deployment are planned milestones—not completed features.

> **Dense and keyword retrieval implementation in progress.**
> The project includes PDF ingestion and page-aware chunking, plus independently
> testable Sentence Transformers/FAISS dense retrieval and BM25 keyword retrieval.
> Day 2 is complete only after local dependency installation, tests, and evaluation
> have been verified. RRF, reranking, LLM generation, persistence, and deployment
> remain future milestones.

## Current features

- **FastAPI backend** with a health endpoint and a PDF ingestion endpoint.
- **PDF validation** for filename/extension, content type, size limit, PDF signature, encrypted files, malformed files, and PDFs with no extractable text.
- **Page-aware extraction** using PyMuPDF, preserving 1-based page numbers.
- **Overlapping text chunks** with configurable chunk size and overlap.
- **Source metadata** on every chunk: document ID, chunk ID, page number, and character offsets.
- **React + Vite frontend** for uploading a PDF, displaying loading/errors, and inspecting extracted chunks.
- **Automated backend tests** for health checks, ingestion, invalid files, and chunking behavior.

**Retrieval pipeline**

Dense retrieval: Sentence Transformers generates semantic embeddings for document chunks, and FAISS retrieves semantically similar passages.

Keyword retrieval: BM25 identifies passages based on lexical relevance.

Reciprocal Rank Fusion (RRF): Combines the dense and BM25 rankings into a unified candidate ranking.

Cross-encoder reranking: Scores query-candidate pairs to refine the ranking of retrieved passages.

The pipeline preserves document IDs, chunk IDs, page numbers, and source metadata throughout retrieval.

**Retrieval evaluation**

The retrieval pipeline was evaluated using 18 labelled queries and the following metrics:

Recall@5: Whether relevant passages appear in the top five results.

MRR: How highly the first relevant passage is ranked.

nDCG@5: Ranking quality, accounting for the positions of relevant passages.

Average latency: Mean retrieval time per query.

### Evaluation Results

| Retrieval Configuration | Recall@5 | MRR | nDCG@5 | Avg. Latency |
|-------------------------|:--------:|:---:|:------:|-------------:|
| Dense Only | 1.0000 | 1.0000 | 0.9955 | 41.44 ms |
| BM25 Only | 1.0000 | 0.9722 | 0.9795 | 0.28 ms |
| Hybrid RRF | 1.0000 | 1.0000 | 1.0000 | 26.38 ms |
| Hybrid RRF + Reranking | 1.0000 | 1.0000 | 1.0000 | 549.56 ms |

Observations:

Hybrid RRF achieved perfect scores across all three retrieval quality metrics on the evaluation dataset.

Dense retrieval achieved strong semantic retrieval performance.

BM25 provided a lightweight and fast keyword-based retrieval baseline.

Cross-encoder reranking maintained the hybrid pipeline's perfect scores but did not improve the measured metrics on this dataset.

Reranking introduces additional computational latency, which should be considered when optimizing the pipeline.



## Architecture today

```text
React + Vite (JavaScript)
        |
        | HTTP / multipart PDF upload
        v
FastAPI REST API
        |
        v
PDF validation (size, type, signature, readability)
        |
        v
PyMuPDF text extraction (page metadata preserved)
        |
        v
Character-based chunking with overlap
        |
        v
Structured JSON response: document + chunks + source metadata
```

## Planned RAG pipeline

The project roadmap will extend ingestion with the following independently testable stages:

Next milestone: Integrate grounded LLM answer generation using the retrieved and reranked document passages, with source citations and insufficient-evidence handling.

## Technology stack

| Area | Technology |
|---|---|
| Backend/API | Python, FastAPI, Pydantic, Uvicorn |
| PDF ingestion | PyMuPDF |
| Frontend | React, Vite, JavaScript, CSS |
| Testing | pytest, HTTPX |
| Planned retrieval | Sentence Transformers, FAISS, `rank-bm25` |
| Planned reranking | Sentence Transformers cross-encoder |
| Planned generation | Hosted LLM API behind a small provider interface |

## Prerequisites

- Python 3.11 or 3.12 recommended
- Node.js 22.12 or newer
- npm
- Git

The development setup was exercised by the project author with Python 3.12, Node.js 24, and npm 11. Versions can differ, but use a supported Python environment if a future ML dependency requires it.

## Run locally (Windows PowerShell)

### 1. Clone the repository

```powershell
git clone <your-github-repository-url>
cd docuquery-day1
```

If you already have the project folder, start from that directory instead.

### 2. Start the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
python -m pytest -q
python -m uvicorn app.main:app --reload
```

Backend URLs:

- Health check: <http://localhost:8000/health>
- Interactive API docs: <http://localhost:8000/docs>

Run tests from the `backend` directory with `python -m pytest -q`. Using `python -m pytest` explicitly runs pytest under the active interpreter and avoids some Windows import-path issues that can occur with the standalone `pytest` command.

### 3. Start the frontend

Open a second PowerShell terminal at the repository root:

```powershell
cd frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open the local address printed by Vite, usually <http://localhost:5173>.

### 4. Try the current workflow

1. Keep both backend and frontend running.
2. Open the frontend in your browser.
3. Upload a text-based PDF.
4. Check the displayed filename, page/chunk information, and extracted passages.
5. Try a non-PDF file or a PDF containing no extractable text to inspect the error handling.

No LLM API key is required for Day 1.

## Configuration

Backend configuration is read from `backend/.env` using Pydantic Settings. See `backend/.env.example` for the available values.

| Setting | Purpose | Default |
|---|---|---:|
| `MAX_UPLOAD_MB` | Maximum upload size in megabytes | `15` |
| `CHUNK_SIZE` | Maximum chunk length in characters | `900` |
| `CHUNK_OVERLAP` | Shared characters between adjacent chunks | `150` |
| `CORS_ORIGINS` | Comma-separated frontend origins allowed by the API | `http://localhost:5173` |

Chunk size and overlap are measured in **characters**, not tokens.

## API endpoints implemented today

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Confirms that the API process is responding. |
| `POST` | `/api/documents` | Validates a PDF, extracts text, creates chunks, and returns structured metadata. |

Additional endpoints for listing, querying, and deleting documents are planned for later milestones, alongside storage and retrieval.

## Testing

From `backend/` with the virtual environment activated:

```powershell
python -m pytest -q
```

## retrieval and evaluation

Install the updated backend dependencies from `backend/`:

```powershell
python -m pip install -r requirements.txt
python -m pytest -q


## Current limitations

- Uploaded documents and chunks are returned in the response and are **not persisted** between requests.
- The current implementation does not yet answer questions or run retrieval, reranking, or LLM generation.
- Scanned/image-only PDFs need OCR, which is not included.
- Chunking is character-based and page-local; it does not cross page boundaries.
- Authentication, multi-user authorization, and production document isolation are not implemented.
- Production deployment and persistent index/storage behavior are not yet configured.





## License

No license has been added yet. Add a license file before inviting reuse or contributions.
