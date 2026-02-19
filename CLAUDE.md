# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LitReview is an automated literature review generation tool. Users upload a CSV of research papers (`id`, `title`, `abstract` columns), enter a research idea, and the system generates a "Related Work" section with inline citations.

The workflow is a deliberate 3-step pipeline: **Upload & Index → Retrieve & Rank → Generate**. This split lets users inspect and adjust paper selection before committing to generation.

## Development Setup

**Prerequisites:** Node.js 20+, Python 3.12+, `uv` package manager, OpenAI API key.

```bash
# Install dependencies
npm install
uv sync

# Required environment variables (copy to .env)
OPENAI_API_KEY=...
VECTOR_STORAGE_MODE=in_memory   # or "persistent"
VECTOR_PERSIST_DIRECTORY=./corpus-data/chroma_db
```

**Run in development (two terminals):**

```bash
# Terminal 1 — Frontend (port 3000)
npm run dev

# Terminal 2 — Backend (port 8000)
cd api && uvicorn index:app --reload --port 8000
```

Next.js proxies `/api/*` to FastAPI at port 8000 in dev (configured in `next.config.ts`).

**Other commands:**

```bash
npm run build    # Export static frontend to ./out/
npm run lint     # ESLint

# Docker (production — single container on port 8000)
docker build -t lit-review .
docker run -p 8000:8000 -e OPENAI_API_KEY=... lit-review
```

There is no test suite currently.

## Architecture

### Stack
- **Frontend:** Next.js 15 (Pages Router, static export), React 19, Tailwind CSS 4, TypeScript
- **Backend:** FastAPI + Uvicorn, Python 3.12
- **AI:** OpenAI `gpt-4o-mini` (generation + relevance scoring via `openai-agents` SDK), `text-embedding-3-small` (embeddings)
- **Vector Store:** ChromaDB via LangChain, with hybrid search (semantic ChromaDB + keyword BM25 via `EnsembleRetriever`)

### Request Flow

```
pages/index.tsx
  POST /api/upload-and-index   → pipeline.py: load CSV → chunk → embed → index into ChromaDB
  POST /api/retrieve-and-rank  → pipeline.py: hybrid_search → score_papers_relevance (parallel AI agent) → top-k
  POST /api/generate           → pipeline.py: stream SSE from OpenAI chat completion
```

### Key Files

| File | Role |
|------|------|
| `api/index.py` | FastAPI routes + in-memory session store (`SESSIONS` dict keyed by 8-char UUID cookie) |
| `api/pipeline.py` | All ML logic: `LiteratureReviewPipeline` class, retrieval, AI-based relevance scoring, streaming generation |
| `api/vectorstore.py` | `VectorStoreAbstract` wrapping LangChain + ChromaDB; handles chunking (size=150, overlap=20), embedding, and hybrid search |
| `api/config.py` | Loads env vars: `OPENAI_API_KEY`, `VECTOR_STORAGE_MODE`, `VECTOR_PERSIST_DIRECTORY` |
| `pages/index.tsx` | Entire frontend UI — drag-and-drop CSV upload, paper selection, SSE streaming display |

### Session Model

Each browser session gets an 8-char UUID cookie. ChromaDB collections are namespaced as `{session_id}_{sanitized_filename}`. In `in_memory` mode, the `VectorStoreAbstract` object is stored directly in the `SESSIONS` dict (does not survive restarts). Ranked papers from step 2 are also stored in `SESSIONS` and read by step 3.

### Relevance Scoring Pattern

`score_papers_relevance()` in `pipeline.py` runs scoring in parallel using the `openai-agents` SDK. Each paper goes through a "debate" pattern: the agent generates arguments for/against inclusion, then outputs a structured `AbstractRelevance` Pydantic model with a score 1–100.

### SSE Streaming Protocol

`/api/generate` streams three event types parsed by the frontend:
- `data: <text chunk>` — incremental generated text
- `data: [METADATA]{...}` — citation metadata (JSON)
- `data: [DONE]` — end of stream

### Production Build

The Dockerfile is a two-stage build: Stage 1 (Node 22 Alpine) exports static Next.js files to `./out/`; Stage 2 (Python 3.12 slim) runs FastAPI and serves both the API and the static files from `./static/`. Targeted for AWS App Runner (health check at `GET /health`).
