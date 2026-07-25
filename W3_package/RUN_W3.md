# W3 — How to run and verify on Vocareum

This package is a fully laid-out W3 capstone repo at the end of Lab Step 4
(ADR done; tests passing; service runnable). All files are byte-identical to
the published references — no modifications.

## What's inside

```
W3_package/
├── api/
│   ├── __init__.py
│   └── main.py                 ← W3 reference (FastAPI app)
├── ui/
│   ├── __init__.py
│   └── app_streamlit.py        ← W3 reference (NOT auto-tested; UI verified manually)
├── src/                         ← W2 carry-over
│   ├── __init__.py
│   └── pipeline/
│       ├── __init__.py
│       ├── settings.py
│       ├── logging_config.py
│       ├── fake_llm.py
│       ├── pipeline.py
│       ├── store.py
│       └── query_results.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py
│   ├── test_pipeline.py
│   └── test_api.py
├── scripts/
│   └── stress_test.py
├── data/
│   └── questions.csv
├── reference/                   ← cohort-repo/week3/ extras
├── requirements.txt             ← W3 additions (fastapi, uvicorn, pytest, …)
├── requirements_w2.txt          ← W2 deps (installed first by the script)
├── pytest.ini
├── logs/                        ← the script fills this in
├── RUN_W3.sh                    ← run this
└── RUN_W3.md                    ← this file
```

Every `.py` file is md5-identical to `/mnt/user-data/outputs/AI-RAG_W{2,3}_*.py`.

## How to run on Vocareum

```bash
unzip AI-RAG_W3_Package.zip
cd W3_package
bash RUN_W3.sh
```

Wall-clock: ~75 seconds (most of it is pip install + stress test wall time).

**Important — port 8000.** The script spins up uvicorn on port 8000 because
that's what `scripts/stress_test.py` hardcodes. If you already have something
running on 8000 (e.g. a previous uvicorn from manual lab work), stop it before
running this script. The trap handler cleans up the script's own uvicorn at
exit, but it won't kill processes it didn't start.

## What the script verifies (24 steps)

| # | Group | Steps | What's checked |
|---|---|---|---|
| **0** | Environment | 5 | Python, key, W2 + W3 pip installs, all packages importable |
| **1** | Layout | 1 | Every canonical file at the expected path |
| **2** | FastAPI (TestClient) | 7 | `/health`, `/ask_batched` happy path + 3 validation cases, `/ask` streaming |
| **3** | Pytest | 1 | `pytest tests/ -q` — 4 tests across `test_pipeline.py` + `test_api.py` |
| **4** | **Live uvicorn + Lab §3** | 10 | **One uvicorn on port 8000, every lab Step 3 procedure runs against it** |

### What Step 4 (the headline) actually does

One uvicorn process is started in the background. While it runs, the script
executes the **exact lab guide §3 procedures**:

1. `step4_uvicorn_start` — spawn `uvicorn api.main:app --port 8000`, wait up to 10s for `/health` to respond
2. `step4_curl_health` — `curl /health` returns 200 + `{"status":"ok"}`
3. `step4_curl_ask_batched_happy` — `curl POST /ask_batched` returns 200 + W3 contract (`content/cost_usd/retries`)
4. `step4_case1_empty_body` — **Lab §3a Case 1**: `{}` → 422
5. `step4_case2_wrong_field` — **Lab §3a Case 2**: `{"q":"…"}` → 422
6. `step4_case3_wrong_type` — **Lab §3a Case 3**: `{"question": 42}` → 422
7. `step4_case4_malformed_json` — **Lab §3a Case 4**: broken JSON (missing `}`) → 422/400
8. `step4_long_5000_chars` — **Lab §3b**: 5200-char question → 200
9. `step4_stress_test_50` — **Lab §3d**: `scripts/stress_test.py --requests 50 --concurrent 10` against the live server
10. `step4_uvicorn_stop` — clean kill

A successful run reports the **stress-test latency stats** (p50, p95, max) for
50 real concurrent requests — exactly what the lab guide §3e asks the learner
to document.

## What's deliberately skipped

- **Streamlit UI**: UI surface, instructor verifies in browser (lab §1g).
- **Lab §3c disconnect mid-stream**: a runtime behaviour test (Ctrl+C
  an active stream); not a code-correctness check, and brittle to automate.

## What to send back

```bash
cat logs/_SUMMARY.log
```

Paste that. If any step shows FAIL, also include the relevant individual log:

```bash
cat logs/<failed_step_id>.log
```

Or `tar czf w3_logs.tgz logs/` for everything in one go.

## What this script does NOT do

- Modify any code anywhere.
- Auto-fix failures — every step's output is captured raw.
- Use port 8000 if it's already occupied (uvicorn will fail to bind and the
  script will report that clearly; clean up and retry).
