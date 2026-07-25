"""api/main.py — Week 3 reference with /metrics middleware.

W3 lab built /ask, /ask_batched, /health. The W3 activity adds:
- A module-level request_counts dict (path → count)
- A FastAPI middleware that increments the counter on every request
- A /metrics endpoint that returns the dict + total

Run with:
    uvicorn api.main:app --reload --port 8000
"""
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()


# ─── W3 activity addition — counter + middleware ────────────────────────
# Module-level; lives in memory; resets on process restart.

request_counts: dict[str, int] = {}


@app.middleware("http")
async def count_requests(request, call_next):
    """Increment the per-path counter on every request.

    Limitations:
      - In-memory, lost on restart
      - Not atomic under heavy concurrency (Python GIL makes this
        unlikely in single-process uvicorn but doesn't guarantee it)
    """
    path = request.url.path
    request_counts[path] = request_counts.get(path, 0) + 1
    response = await call_next(request)
    return response


# ─── Endpoints (W3 lab) ─────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask")
async def ask(req: AskRequest):
    # Stub — real implementation calls the LLM
    return {"answer": f"You asked: {req.question}"}


@app.post("/ask_batched")
async def ask_batched(req: AskRequest):
    # Stub — real implementation routes via the async batch queue
    return {"answer": f"(batched) You asked: {req.question}"}


# ─── W3 activity addition — /metrics endpoint ───────────────────────────

@app.get("/metrics")
async def metrics():
    """Self-observability endpoint.

    Returns:
      - endpoints: per-path request counts since process start
      - total: sum of all counts
    """
    return {
        "endpoints": request_counts,
        "total": sum(request_counts.values()),
    }
