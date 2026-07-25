# W3 Activity Solution — `/metrics` endpoint

> *Instructor reference for the W3 self-observability activity. The
> point of this exercise is to make the instrumentation-as-separate-
> concern pattern concrete, and to surface its limitations.*

**Activity time:** 15-20 min
**Stretch time:** +5 min
**Files involved:** `api/main.py` (existing) + `docs/runs/week3-activity.md` (new)

---

## What this activity is testing

Three outcomes:

1. **Middleware pattern.** Learners add one middleware function and
   get per-path counts on every existing endpoint — without touching
   any endpoint code. This is the engineering payoff: instrumentation
   as a separate concern.

2. **Self-observability awareness.** The `/metrics` endpoint counts
   itself by default. Whether that's right is a design question, not
   an obvious answer.

3. **Limitations of naive in-memory counters.** Two issues should be
   surfaced: (a) state lost on restart, (b) not atomic under
   concurrency.

This is the foundation for W4 (Prometheus-style counters), W22
(full instrumentation), and any FastAPI service the cohort builds
later.

---

## Reference solution walkthrough

### The two additions

Near the top of `api/main.py` (after imports, before endpoints):

```python
# Module-level counter — path → count
request_counts: dict[str, int] = {}


@app.middleware("http")
async def count_requests(request, call_next):
    path = request.url.path
    request_counts[path] = request_counts.get(path, 0) + 1
    response = await call_next(request)
    return response
```

Below the existing endpoints:

```python
@app.get("/metrics")
async def metrics():
    return {
        "endpoints": request_counts,
        "total": sum(request_counts.values()),
    }
```

That's the whole solution. ~10 lines of code.

### Expected `/metrics` output

After running these requests in order:
```bash
curl http://localhost:8000/health
curl http://localhost:8000/health
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask_batched -d '{"question":"x"}' -H "Content-Type: application/json"
curl -X POST http://localhost:8000/ask -d '{"question":"y"}' -H "Content-Type: application/json"
curl http://localhost:8000/metrics
```

The final `/metrics` response should look like:

```json
{
  "endpoints": {
    "/health": 3,
    "/ask_batched": 1,
    "/ask": 1,
    "/metrics": 1
  },
  "total": 6
}
```

The `/metrics` entry shows 1 because the middleware counted the
request that fetched `/metrics` itself. This is the
*observe-the-observer* property.

### The reflection — what learners should write

**Q1:** *"What's lost if the process restarts?"*

> The entire `request_counts` dict lives in memory. Kill the process,
> the counts go to zero. A real metrics system (Prometheus, StatsD)
> persists outside the process — usually by scraping or pushing
> values to a durable store.

**Q2:** *"Under heavy concurrency, would two simultaneous requests to
`/health` always result in the counter going from N to N+2?"*

> Not guaranteed. Two coroutines could both read the same value of
> `count`, both compute `count + 1`, both write that — losing one
> increment. Python's GIL makes this unlikely in single-process
> uvicorn but doesn't *guarantee* it; with multiple workers or
> heavy CPU concurrency, you'd see lost counts. Real systems use
> atomic counters (`threading.Lock`, `Redis INCR`, etc.).

---

## What to look for in submissions

**Strong signals:**
- Middleware is registered with `@app.middleware("http")`
- Counter dict initialised correctly (with `.get(path, 0) + 1`)
- `/metrics` returns the dict + a `total` (showing they sum the
  values, not just dump the dict)
- Reflection explicitly names both limitations (persistence +
  atomicity)

**Weak signals:**
- Decorator on every endpoint instead of middleware (defeats the
  separate-concern point)
- Counter on each endpoint manually (same issue)
- Reflection that only notes "it works" without identifying the
  trade-offs

**Common mistakes:**
- Forgets `await call_next(request)` — every endpoint returns None
  because the middleware doesn't pass through
- Uses a list instead of dict — can't query per-path counts
- `/metrics` excludes itself from the counter without justification
  (this is *defensible* but should be a deliberate choice, not an
  accident)

---

## Stretch outcomes

For learners who attempted the stretch:

- **Status codes too.** Nested dict `{path: {status: count}}` lets
  them see how many of each path failed. Good preview for W22.
- **Latency tracking.** They'll measure `time.time()` around
  `call_next` and store totals. Right shape; wrong stat (mean is
  misleading for latency — histograms are right, but only at W22).
- **Exclude `/metrics` from itself.** One-line guard. They should
  argue for/against in writing — either side is fine if defended.

---

## Office hours hot questions

- *"Should I put this in an ADR?"* — No. The W3 lab's ADR explicitly
  lists "internal observability" in the "don't bump for" section.
  This is internal tooling, not a public contract.
- *"Why doesn't the middleware see the body of POST requests?"* —
  It does, but reading the body in middleware consumes the stream,
  preventing the actual endpoint from reading it. Don't read the
  body in observability middleware.
- *"Can I use Prometheus instead?"* — Yes, in W22. The shape (one
  middleware, one endpoint) stays the same; only the internals
  change (in-memory dict → prometheus_client Counter).

---

## Files in this solution package

- `main_with_metrics.py` — complete `api/main.py` with `/ask`,
  `/ask_batched`, `/health`, plus the new middleware + `/metrics`
- `sample_curl_session.sh` — the exact curl sequence to demo the
  `/metrics` output
- `sample_week3-activity.md` — what a strong learner submission looks like

---

*End of W3 solution.*
