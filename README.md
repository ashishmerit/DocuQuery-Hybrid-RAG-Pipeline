# DocuQuery — Hybrid RAG Pipeline with Reranking

**DocuQuery** is a full-stack document question-answering application being built to demonstrate a practical Retrieval-Augmented Generation (RAG) pipeline. The planned pipeline combines keyword and semantic retrieval, Reciprocal Rank Fusion (RRF), cross-encoder reranking, and evidence-grounded LLM answers with source citations.

> **Project status: Day 1 — Foundation and PDF ingestion.** The current implementation supports PDF upload, validation, page-aware text extraction, overlapping chunk creation, and a React interface for inspecting the result. Retrieval, reranking, LLM generation, persistence, and deployment are planned milestones—not completed features.

## Current features

- **FastAPI backend** with a health endpoint and a PDF ingestion endpoint.
- **PDF validation** for filename/extension, content type, size limit, PDF signature, encrypted files, malformed files, and PDFs with no extractable text.
- **Page-aware extraction** using PyMuPDF, preserving 1-based page numbers.
- **Overlapping text chunks** with configurable chunk size and overlap.
- **Source metadata** on every chunk: document ID, chunk ID, page number, and character offsets.
- **React + Vite frontend** for uploading a PDF, displaying loading/errors, and inspecting extracted chunks.
- **Automated backend tests** for health checks, ingestion, invalid files, and chunking behavior.

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

1. Sentence Transformers embeddings and FAISS dense retrieval.
2. BM25 lexical retrieval.
3. Reciprocal Rank Fusion to combine retrieval rankings.
4. Cross-encoder reranking of the candidate passages.
5. Grounded answer generation through a replaceable hosted-LLM client.
6. Source citations, insufficient-evidence handling, retrieval evaluation, and deployment.

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

### 4. Try the Day 1 workflow

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

The author’s Day 1 run reported **8 passing tests**. A deprecation warning from the earlier PDF import was addressed by using the current `pymupdf` module name; rerun the tests in your environment to verify the current checkout.

## Current limitations

- Uploaded documents and chunks are returned in the response and are **not persisted** between requests.
- The current implementation does not yet answer questions or run retrieval, reranking, or LLM generation.
- Scanned/image-only PDFs need OCR, which is not included.
- Chunking is character-based and page-local; it does not cross page boundaries.
- Authentication, multi-user authorization, and production document isolation are not implemented.
- Production deployment and persistent index/storage behavior are not yet configured.

## Development roadmap

- [x] Day 1 — Foundation, PDF ingestion, page metadata, chunking, initial frontend, tests
- [ ] Day 2 — Dense retrieval, BM25, and evaluation query set
- [ ] Day 3 — Hybrid retrieval with RRF and cross-encoder reranking
- [ ] Day 4 — Grounded LLM generation, citations, and insufficient-evidence behavior
- [ ] Day 5 — Complete frontend workflow and document management
- [ ] Day 6 — Integration tests, retrieval evaluation, latency/cost measurements
- [ ] Day 7 — Deployment, documentation, demo, and interview preparation

Milestones are marked complete only after their acceptance checks have been run; roadmap checkboxes reflect the author’s latest recorded progress.

## Learning notes

The internal `day1_concepts_guide.md` is a study guide for the concepts implemented in this milestone. `developer_notes.md` and `project_progress.md` are development notes and may be kept out of the public repository if preferred.

## License

No license has been added yet. Add a license file before inviting reuse or contributions.
