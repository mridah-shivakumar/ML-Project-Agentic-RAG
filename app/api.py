"""
app/api.py
──────────
FastAPI production API — wraps the LangGraph agent.

Endpoints
─────────
  POST /ingest        upload a PDF and index it
  POST /query         ask a question, get an answer + sources
  GET  /health        liveness probe (required for cloud deploy)
  GET  /metrics       RAGAs evaluation scores (if run)

Run locally:
    uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from pydantic import BaseModel

from app.agent import ask
from ingestion.ingest import load_pdfs
from ingestion.tabular_store import tabular_store
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from vectorstore.store import build_index


def parse_history(history: Optional[List[dict]]) -> List[BaseMessage]:
    """Convert API history payload into LangChain BaseMessage objects."""
    if not history:
        return []
    messages: List[BaseMessage] = []
    for item in history:
        role = item.get("role", "").lower()
        content = item.get("content", "")
        if role in ("user", "human"):
            messages.append(HumanMessage(content=content))
        elif role in ("assistant", "ai"):
            messages.append(AIMessage(content=content))
    return messages


# ── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Agentic RAG Document Assistant",
    description=(
        "Production RAG API with hybrid retrieval, "
        "cross-encoder reranking, LangGraph agentic loop, "
        "and Corrective RAG (query rewriting + web fallback)."
    ),
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR     = Path("data")
METRICS_FILE = Path("evaluation/latest_scores.json")
RESULTS_FILE = Path("evaluation/results.json")
DATA_DIR.mkdir(exist_ok=True)


# ── Pydantic schemas ─────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question:   str
    history:    Optional[List[dict]] = None   # [{"role": "user"|"assistant", "content": "..."}]


class SourceItem(BaseModel):
    text:   str
    source: str
    page:   int


class QueryResponse(BaseModel):
    answer:        str
    sources:       List[SourceItem]
    rewrite_count: int
    used_web:      bool
    latency_ms:    int
    trace:         Optional[Dict[str, Any]] = None


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Liveness probe — required for Docker/cloud deploy."""
    return {"status": "ok", "version": app.version}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """
    Upload a document (PDF or CSV).
    - PDFs are chunked and added to the hybrid FAISS+BM25 vector index.
    - CSVs are loaded into the structured tabular store for safe pandas analysis.
    """
    fname = file.filename.lower()
    if fname.endswith(".csv"):
        return await ingest_csv(file)

    if not fname.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF and CSV files are supported.")

    dest = DATA_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)
    logger.info(f"Saved uploaded file: {dest}")

    try:
        chunks, metas = load_pdfs(DATA_DIR)
        build_index(chunks, metas)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    return {
        "status":    "indexed",
        "type":      "pdf",
        "filename":  file.filename,
        "chunks":    len(chunks),
    }


@app.post("/ingest/csv")
async def ingest_csv(file: UploadFile = File(...)):
    """
    Upload and load a CSV dataset for structured tabular analysis.
    Preserves data types and enables safe pandas querying without arbitrary code execution.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV (.csv) files are supported for tabular ingestion.")

    content = await file.read()
    try:
        summary = tabular_store.load_csv(content, file.filename)
    except Exception as e:
        logger.error(f"Failed to ingest CSV: {e}")
        raise HTTPException(status_code=400, detail=f"CSV ingestion failed: {e}")

    return {
        "status":    "indexed",
        "type":      "csv",
        "filename":  summary["filename"],
        "rows":      summary["row_count"],
        "columns":   summary["columns"],
    }


@app.get("/csv/schema")
def get_csv_schema():
    """Return active tabular dataset schema and metadata."""
    if not tabular_store.has_data():
        return {"loaded": False, "detail": "No CSV dataset is currently loaded."}
    return {
        "loaded": True,
        "filename": tabular_store.active_filename,
        "rows": len(tabular_store.active_df),
        "columns": list(tabular_store.active_df.columns),
        "dtypes": {col: str(dtype) for col, dtype in tabular_store.active_df.dtypes.items()},
    }


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """
    Ask a question. The LangGraph agent:
    1. Routes (documents vs LLM-only)
    2. Retrieves with hybrid FAISS+BM25
    3. Reranks with cross-encoder
    4. Grades relevance — rewrites query if needed (Corrective RAG)
    5. Falls back to web search if docs fail after max rewrites
    6. Generates a cited answer
    """
    t0 = time.time()
    try:
        history_messages = parse_history(req.history)
        result = ask(req.question, history=history_messages)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="No index found. Upload a PDF via POST /ingest first."
        )
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    latency = int((time.time() - t0) * 1000)
    logger.info(f"Query answered in {latency}ms | rewrites={result['rewrite_count']}")

    return QueryResponse(
        answer        = result["answer"],
        sources       = [SourceItem(**s) for s in result["sources"]],
        rewrite_count = result["rewrite_count"],
        used_web      = result["used_web"],
        latency_ms    = latency,
        trace         = result.get("trace"),
    )


@app.get("/metrics")
def metrics():
    """Return latest evaluation scores if available."""
    if METRICS_FILE.exists():
        return json.loads(METRICS_FILE.read_text())
    if RESULTS_FILE.exists():
        data = json.loads(RESULTS_FILE.read_text())
        return data.get("aggregate", data)
    return {"detail": "No evaluation scores yet. Run: python -m evaluation.evaluate"}


@app.get("/evaluation/results")
def evaluation_results():
    """Return complete evaluation report and per-question telemetry if available."""
    if RESULTS_FILE.exists():
        return json.loads(RESULTS_FILE.read_text())
    if METRICS_FILE.exists():
        return json.loads(METRICS_FILE.read_text())
    return {"detail": "No evaluation scores yet. Run: python -m evaluation.evaluate"}
