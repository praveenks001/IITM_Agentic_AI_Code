# Week 3 — Hands-on Activity

**Title:** Add a `/metrics` endpoint — your first piece of self-observability.
**Time:** ~15–20 minutes
**Builds on:** the FastAPI service you built during Day 1 of the W3 lab (the one with `/ask`, `/ask_batched`, and `/health`).
**When to do this:** as a 20-minute warm-up at the start of Day 2 — or as a take-home between sessions.

> The point of this activity isn't to add a new feature. It's to feel what happens when you give your own service the ability to report on itself — and to notice the trade-offs (state in memory, no atomicity, no persistence) that real observability solves in different ways. The pattern recurs in W4 (Prometheus-style counters) and W22 (full instrumentation).

---

## Goal

Add a `/metrics` endpoint that returns a JSON object with one counter per path: how many requests has each endpoint served since the process started? Hit your service a few times, then `curl /metrics` and see your activity reflected back. Write two sentences in your run notes on what you observed.

## Materials

- `api/main.py` from class (the one with `/ask`, `/ask_batched`, `/health`).
- Your `uvicorn api.main:app --port 8000` running in a second terminal.
- A `docs/runs/week3-activity.md` file you'll create as you go.

---

## Steps

### 1. Add a counter and a middleware

Open `api/main.py`. Near the top of the file (after the imports, before the endpoint definitions), add:

- A module-level counter: `request_counts: dict[str, int] = {}`. This is a plain dict — `path → count`. It lives in memory, resets on restart. (That's the *first* thing you'll notice as a limitation.)
- An async middleware function `count_requests(request, call_next)` decorated with `@app.middleware("http")`. The body:
  1. Reads `path = request.url.path`.
  2. Increments `request_counts[path]` (use `.get(path, 0) + 1` to handle the first hit).
  3. Awaits `response = await call_next(request)` to let the actual endpoint run.
  4. Returns `response`.

The middleware sits between the HTTP layer and your endpoint handlers — every request flows through it. FastAPI runs all registered middlewares in order, so even `/health` and `/docs` get counted.

### 2. Add the `/metrics` endpoint

Below your existing endpoints, add:

- `@app.get("/metrics")` decorating a small async function `metrics()`.
- The function returns a dict with two keys: `"endpoints"` (the `request_counts` dict itself) and `"total"` (sum of the values).

That's it. No Pydantic model needed — plain dict-as-JSON is fine for an internal observability endpoint. (You'll harden this in W22 when metrics become a *contract* of their own.)

### 3. Restart uvicorn and exercise the service

In the terminal running uvicorn:

```bash
# Ctrl-C to stop, then:
uvicorn api.main:app --reload --port 8000
```

In a second terminal:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health
curl http://localhost:8000/health

curl -X POST http://localhost:8000/ask_batched \
     -H "Content-Type: application/json" \
     -d '{"question": "What is RAG?"}'

curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"question": "Explain async in two sentences."}'

curl http://localhost:8000/metrics
```

### 4. Read the output

Your `/metrics` response should look something like:

```
{
  "endpoints": {
    "/health":       3,
    "/ask_batched":  1,
    "/ask":          1,
    "/metrics":      1
  },
  "total": 6
}
```

Notice `/metrics` counted itself — the middleware fires for every path, including this one. (Some libraries exclude their own observability endpoints; that's a deliberate design choice, not an obvious right answer.)

### 5. Reflect — two sentences in `docs/runs/week3-activity.md`

Open the file and answer:

- In one sentence: what's lost if the process restarts? (i.e., what is the `request_counts` dict missing that a real metrics system would provide?)
- In one sentence: under heavy concurrency, would two simultaneous requests to `/health` always result in the counter going from N to N+2? Why or why not?

---

## What you should notice

The middleware pattern made every endpoint observable *without* touching any endpoint code. That's the engineering payoff: instrumentation as a separate concern. You added one function and got per-path counts on every existing handler.

Two limitations you'll feel immediately:

1. **The counter is in-memory.** Kill the process, the counts disappear. A real metrics system (Prometheus, StatsD, OTLP) persists outside the process — usually by scraping or pushing the values somewhere durable. This is the W22 lesson preview.

2. **The counter isn't atomic.** Two concurrent requests hitting `/health` at the same moment could both read the same `count`, both add 1, and both write the same `count + 1` — losing one increment. Python's GIL makes this rare in single-process uvicorn, but it's not *guaranteed* safe; under heavy load or with multiple workers, you'd see lost counts. Real systems use atomic counters (`threading.Lock`, or a Redis `INCR`).

These limitations are not flaws *of this activity* — they're the reasons real observability tooling exists. You can see now why the answer isn't "just count things in a dict".

---

## Stretch (if you have ~5 extra minutes)

- **Track status codes too.** Change the counter to a nested dict: `{path: {status: count}}`. After `call_next`, you have `response.status_code` available. Now your /metrics shows you not just *how many* requests per path, but *how many failed*.
- **Add latency.** Wrap the `await call_next(request)` in a `time.time()` measurement. Track total time per path and compute mean per request. (Mean is the wrong stat — see Quiz 2 Q6 — but you can build to histograms in W22.)
- **Exclude `/metrics` from the counter.** One line: `if path == "/metrics": return await call_next(request)` before incrementing. Now you're not counting your own observability. Argue with yourself for thirty seconds about whether that's right.

---

## Submit (optional but recommended)

Commit your changes:

```bash
git add api/main.py docs/runs/week3-activity.md
git commit -m "feat(w3): add /metrics endpoint with request counter middleware"
```

The `/metrics` endpoint and its middleware can stay in your `api/main.py` permanently — they're a real piece of functionality, not a throwaway. In W22, you'll swap the in-memory dict for a proper Prometheus client; the *shape* (one endpoint, one middleware) won't change.

**Don't add this to the W3 ADR.** It's an internal observability endpoint — see the ADR's "Don't bump for" list. If someone later proposes making `/metrics` part of the public contract (e.g., for an external monitoring tool to consume), *then* it gets its own ADR.
