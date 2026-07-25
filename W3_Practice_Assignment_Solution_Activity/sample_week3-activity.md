# Week 3 Activity — `/metrics` endpoint

## Changes to `api/main.py`

Added a module-level counter dict, an HTTP middleware that increments
it, and a `/metrics` endpoint that returns the dict + total.

## Sample `/metrics` response

After 3× `/health`, 1× `/ask_batched`, 1× `/ask`, and the `/metrics`
call itself:

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

The middleware counted `/metrics` calling itself — the
observe-the-observer effect.

## What I observed

**Q1 — What's lost if the process restarts?**

> Everything. The `request_counts` dict lives in memory; killing the
> process resets it to `{}`. A real metrics system (Prometheus,
> StatsD, OTLP exporter) keeps data outside the process — either
> scraped by an external collector or pushed to a durable store.

**Q2 — Under heavy concurrency, would two simultaneous requests to
`/health` always result in the counter going from N to N+2?**

> Not guaranteed. Two coroutines could both read `count` at the same
> moment, both compute `count + 1`, both write back — losing one
> increment. Python's GIL makes this unlikely in single-process
> uvicorn, but not impossible. Under heavy load or with multiple
> workers (e.g. `--workers 4`), I'd see lost counts. Production
> systems use atomic counters: `threading.Lock`, `Redis INCR`, or
> a proper Prometheus client which handles this internally.
