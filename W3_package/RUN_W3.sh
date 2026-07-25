#!/usr/bin/env bash
# ============================================================================
# RUN_W3.sh — Verifies W3 reference code on Vocareum (v2: no Streamlit).
#
# What this script does:
#   1. Confirms Python + OPENAI_API_KEY.
#   2. Installs W2 + W3 requirements.
#   3. Runs in-process TestClient smokes against every FastAPI endpoint.
#   4. Runs the full pytest suite.
#   5. Starts a real uvicorn server (port 8000, as the lab guide does) and
#      executes every Lab Guide Step 3 procedure against it:
#        - §1e/§3a Cases 1-4 (malformed JSON via curl — all expect 422)
#        - §3b 5000-character question (expects 200)
#        - §3d 50 parallel requests via scripts/stress_test.py
#      Then stops uvicorn cleanly.
#   6. Captures each step's output to logs/<step_id>.log.
#   7. Writes logs/_SUMMARY.log with a one-line PASS/FAIL per step.
#
# Skipped (deliberately):
#   - Streamlit UI checks: it's a UI surface, instructor verifies in browser.
#   - Lab Guide §3c (disconnect mid-stream): a runtime behaviour test
#     (Ctrl+C an active stream); not a code-correctness check.
# ============================================================================

set -u
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT" || exit 2

LOG_DIR="$ROOT/logs"
SUMMARY="$LOG_DIR/_SUMMARY.log"
UVICORN_PORT=8000   # matches the hardcoded URL in scripts/stress_test.py
UVICORN_LOG="$LOG_DIR/_uvicorn_real.log"
UVICORN_PID_FILE="$LOG_DIR/_uvicorn.pid"

mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log "$UVICORN_PID_FILE"
rm -f "$ROOT/results.json" "$ROOT/results.db"
find "$ROOT" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null
find "$ROOT" -name '.pytest_cache' -type d -exec rm -rf {} + 2>/dev/null

# Ensure background uvicorn dies if the script is interrupted
cleanup() {
    if [ -f "$UVICORN_PID_FILE" ]; then
        local pid
        pid=$(cat "$UVICORN_PID_FILE")
        kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
        rm -f "$UVICORN_PID_FILE"
    fi
}
trap cleanup EXIT INT TERM

{
    echo "W3 Verification Run"
    echo "==================="
    echo "Timestamp:    $(date -Iseconds)"
    echo "PWD:          $ROOT"
    echo "Python:       $(python --version 2>&1)"
    echo "Has API key:  $(python -c "import os; print(bool(os.getenv('OPENAI_API_KEY')))" 2>&1)"
    echo "Uvicorn port: $UVICORN_PORT"
    echo
    printf "%-44s  %-30s  %s\n" "STEP" "VERDICT" "DESCRIPTION"
    printf "%-44s  %-30s  %s\n" "----" "-------" "-----------"
} > "$SUMMARY"

# ─── helper ────────────────────────────────────────────────────────────────
run_step () {
    local step_id="$1"
    local description="$2"
    local expected_rc="$3"
    local cmd="$4"
    local log_file="$LOG_DIR/${step_id}.log"

    {
        echo "STEP:        $step_id"
        echo "DESCRIPTION: $description"
        echo "EXPECTED RC: $expected_rc"
        echo "TIMESTAMP:   $(date -Iseconds)"
        echo "COMMAND:"
        echo "$cmd" | sed 's/^/    /'
        echo "--- OUTPUT ---"
    } > "$log_file"

    bash -c "$cmd" >> "$log_file" 2>&1
    local rc=$?

    local verdict
    if [ "$expected_rc" = "any" ]; then
        verdict="DONE  (rc=$rc)"
    elif [ "$rc" = "$expected_rc" ]; then
        verdict="PASS"
    else
        verdict="FAIL  (rc=$rc, expected $expected_rc)"
    fi

    echo "  [$verdict]  $step_id  —  $description"
    printf "%-44s  %-30s  %s\n" "$step_id" "$verdict" "$description" >> "$SUMMARY"

    {
        echo "--- END ---"
        echo "EXIT_CODE: $rc"
        echo "VERDICT:   $verdict"
    } >> "$log_file"
}

# ===========================================================================
# Step 0 — Environment
# ===========================================================================
echo ""
echo "=========================================="
echo " Step 0 — Environment"
echo "=========================================="

run_step "step0_python_version"     "Python version"                       0 \
    "python --version"

run_step "step0_openai_key"         "OPENAI_API_KEY present"               0 \
    "python -c \"import os; ok=bool(os.getenv('OPENAI_API_KEY')); print('OPENAI_API_KEY present:', ok); raise SystemExit(0 if ok else 1)\""

run_step "step0_pip_install_w2"     "pip install -r requirements_w2.txt"   0 \
    "pip install --break-system-packages -q -r requirements_w2.txt"

run_step "step0_pip_install_w3"     "pip install -r requirements.txt"      0 \
    "pip install --break-system-packages -q -r requirements.txt"

run_step "step0_imports_ok"         "All required packages importable"     0 \
    "python -c \"import openai, pydantic, dotenv, httpx, fastapi, uvicorn, pytest, requests; print('OK · fastapi', fastapi.__version__, '· pytest', pytest.__version__, '· httpx', httpx.__version__)\""

# ===========================================================================
# Step 1 — File layout
# ===========================================================================
echo ""
echo "=========================================="
echo " Step 1 — File layout"
echo "=========================================="

run_step "step1_layout_complete"    "All canonical files exist at expected paths"  0 \
    "for f in api/main.py src/pipeline/pipeline.py src/pipeline/settings.py tests/conftest.py tests/test_pipeline.py tests/test_api.py scripts/stress_test.py pytest.ini data/questions.csv; do
        if [ -f \"\$f\" ]; then echo \"  ✓  \$f\"; else echo \"  ✗  MISSING: \$f\" && exit 1; fi
    done"

# ===========================================================================
# Step 2 — FastAPI via TestClient (in-process, fast)
# ===========================================================================
echo ""
echo "=========================================="
echo " Step 2 — FastAPI endpoints (TestClient)"
echo "=========================================="

run_step "step2_app_imports"        "from api.main import app — imports cleanly"  0 \
    "python -c \"
import sys; sys.path.insert(0, '.')
from api.main import app
print('app type:', type(app).__name__)
print('routes:', sorted([r.path for r in app.routes if hasattr(r, 'path')]))
\""

run_step "step2_health"             "GET /health returns 200"              0 \
    "python -c \"
import sys; sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from api.main import app
client = TestClient(app)
r = client.get('/health')
print('status:', r.status_code)
print('body:', r.json())
assert r.status_code == 200
\""

run_step "step2_ask_batched_valid"  "POST /ask_batched with valid question" 0 \
    "python -c \"
import sys; sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from api.main import app
client = TestClient(app)
r = client.post('/ask_batched', json={'question': 'What is RAG in one sentence?'})
print('status:', r.status_code)
body = r.json()
print('keys:', sorted(body.keys()))
print('content head:', body.get('content', '')[:80])
print('cost_usd:', body.get('cost_usd'))
print('retries:', body.get('retries'))
assert r.status_code == 200
for required in ('content', 'cost_usd', 'retries'):
    assert required in body, f'W3 contract missing: {required}'
\""

run_step "step2_ask_batched_empty"  "POST /ask_batched with empty body → 422"  0 \
    "python -c \"
import sys; sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from api.main import app
client = TestClient(app)
r = client.post('/ask_batched', json={})
print('status:', r.status_code)
assert r.status_code == 422
print('PASS — 422 returned as expected')
\""

run_step "step2_ask_batched_wrong_field"  "POST /ask_batched with wrong field name → 422"  0 \
    "python -c \"
import sys; sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from api.main import app
client = TestClient(app)
r = client.post('/ask_batched', json={'q': 'wrong field'})
print('status:', r.status_code)
assert r.status_code == 422
print('PASS — 422 returned as expected')
\""

run_step "step2_ask_batched_wrong_type"   "POST /ask_batched with wrong type → 422"  0 \
    "python -c \"
import sys; sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from api.main import app
client = TestClient(app)
r = client.post('/ask_batched', json={'question': 42})
print('status:', r.status_code)
assert r.status_code == 422
print('PASS — 422 returned as expected')
\""

run_step "step2_ask_streaming"      "POST /ask streaming returns chunks"   0 \
    "python -c \"
import sys; sys.path.insert(0, '.')
from fastapi.testclient import TestClient
from api.main import app
client = TestClient(app)
with client.stream('POST', '/ask', json={'question': 'What is RAG?'}) as r:
    print('status:', r.status_code)
    assert r.status_code == 200
    chunks = []
    for chunk in r.iter_text():
        chunks.append(chunk)
    full = ''.join(chunks)
    print('chunks received:', len(chunks))
    print('content head:', full[:80])
    assert len(full) > 0, 'streaming response was empty'
\""

# ===========================================================================
# Step 3 — Pytest formal tests
# ===========================================================================
echo ""
echo "=========================================="
echo " Step 3 — Pytest tests/"
echo "=========================================="

run_step "step3_pytest"             "pytest tests/ -q"                     0 \
    "python -m pytest tests/ -q --tb=short"

# ===========================================================================
# Step 4 — Live uvicorn server + Lab Guide §3 procedures
# ===========================================================================
echo ""
echo "=========================================="
echo " Step 4 — Live uvicorn (port $UVICORN_PORT)"
echo "=========================================="

run_step "step4_uvicorn_start"      "Start uvicorn api.main:app + wait for ready"  0 \
    "uvicorn api.main:app --port $UVICORN_PORT --log-level warning > '$UVICORN_LOG' 2>&1 &
    UV_PID=\$!
    echo \$UV_PID > '$UVICORN_PID_FILE'
    echo \"uvicorn started, pid=\$UV_PID\"
    for i in 1 2 3 4 5 6 7 8 9 10; do
        sleep 1
        if curl -s -f -o /dev/null http://localhost:$UVICORN_PORT/health 2>/dev/null; then
            echo \"server ready after \${i}s\"
            exit 0
        fi
    done
    echo 'SERVER NEVER BECAME READY'
    echo '--- uvicorn log ---'
    cat '$UVICORN_LOG'
    exit 1"

run_step "step4_curl_health"        "curl GET /health → 200 / {status:ok}"  0 \
    "set -o pipefail
    out=\$(curl -s -i http://localhost:$UVICORN_PORT/health)
    echo \"\$out\"
    echo \"\$out\" | head -1 | grep -q '200' || { echo 'EXPECTED 200'; exit 1; }
    echo \"\$out\" | tail -1 | grep -q '\"status\":\"ok\"' || { echo 'EXPECTED status:ok in body'; exit 1; }"

run_step "step4_curl_ask_batched_happy"  "curl POST /ask_batched (happy path) → 200 + W3 contract"  0 \
    "out=\$(curl -s -i -X POST http://localhost:$UVICORN_PORT/ask_batched \\
                   -H 'Content-Type: application/json' \\
                   -d '{\"question\": \"What is RAG?\"}')
    echo \"\$out\"
    echo \"\$out\" | head -1 | grep -q '200' || { echo 'EXPECTED 200'; exit 1; }
    body=\$(echo \"\$out\" | tail -1)
    for field in content cost_usd retries; do
        echo \"\$body\" | grep -q \"\\\"\$field\\\"\" || { echo \"MISSING FIELD: \$field\"; exit 1; }
    done"

# Lab Guide §3a — 4 malformed JSON cases, all expecting 422

run_step "step4_case1_empty_body"   "Lab §3a Case 1: empty body → 422"     0 \
    "out=\$(curl -s -i -X POST http://localhost:$UVICORN_PORT/ask \\
                   -H 'Content-Type: application/json' \\
                   -d '{}')
    echo \"\$out\"
    echo \"\$out\" | head -1 | grep -q '422' || { echo 'EXPECTED 422'; exit 1; }"

run_step "step4_case2_wrong_field"  "Lab §3a Case 2: wrong field name → 422"  0 \
    "out=\$(curl -s -i -X POST http://localhost:$UVICORN_PORT/ask \\
                   -H 'Content-Type: application/json' \\
                   -d '{\"q\": \"What is the leave policy?\"}')
    echo \"\$out\"
    echo \"\$out\" | head -1 | grep -q '422' || { echo 'EXPECTED 422'; exit 1; }"

run_step "step4_case3_wrong_type"   "Lab §3a Case 3: wrong type (int) → 422"  0 \
    "out=\$(curl -s -i -X POST http://localhost:$UVICORN_PORT/ask \\
                   -H 'Content-Type: application/json' \\
                   -d '{\"question\": 42}')
    echo \"\$out\"
    echo \"\$out\" | head -1 | grep -q '422' || { echo 'EXPECTED 422'; exit 1; }"

run_step "step4_case4_malformed_json"  "Lab §3a Case 4: malformed JSON → 422 or 400"  0 \
    "out=\$(curl -s -i -X POST http://localhost:$UVICORN_PORT/ask \\
                   -H 'Content-Type: application/json' \\
                   -d '{\"question\": \"What is the leave policy?\"')
    echo \"\$out\"
    if echo \"\$out\" | head -1 | grep -qE '4(00|22)'; then
        echo 'PASS — got 4xx as expected'
    else
        echo 'EXPECTED 422 or 400'
        exit 1
    fi"

# Lab Guide §3b — 5000-character question

run_step "step4_long_5000_chars"    "Lab §3b: 5000-char question → 200"    0 \
    "python -c \"
import httpx, sys
q = 'Please summarize the company policy on remote work. ' * 100
print(f'question length: {len(q)} chars')
r = httpx.post('http://localhost:$UVICORN_PORT/ask_batched', json={'question': q}, timeout=60.0)
print('status:', r.status_code)
if r.status_code == 200:
    print('content head:', r.json().get('content', '')[:120])
else:
    print('body:', r.text[:300])
sys.exit(0 if r.status_code == 200 else 1)
\""

# Lab Guide §3d — 50 parallel requests via scripts/stress_test.py
# Run with the lab guide's default args; fake LLM means this finishes in seconds.

run_step "step4_stress_test_50"     "Lab §3d: scripts/stress_test.py --requests 50 --concurrent 10"  0 \
    "python scripts/stress_test.py --requests 50 --concurrent 10"

run_step "step4_uvicorn_stop"       "Stop uvicorn cleanly"                 0 \
    "if [ -f '$UVICORN_PID_FILE' ]; then
        pid=\$(cat '$UVICORN_PID_FILE')
        echo \"stopping uvicorn, pid=\$pid\"
        kill \$pid 2>/dev/null
        sleep 1
        if kill -0 \$pid 2>/dev/null; then
            echo 'force kill'
            kill -9 \$pid 2>/dev/null
        fi
        rm -f '$UVICORN_PID_FILE'
        echo 'uvicorn stopped'
    else
        echo 'no pid file — uvicorn was not running'
        exit 1
    fi"

# ===========================================================================
# Done — final summary
# ===========================================================================
echo ""
echo "=========================================="
echo " Summary"
echo "=========================================="
cat "$SUMMARY"
echo ""
echo "All step logs are in: $LOG_DIR/"
echo "Top-line summary is in: $SUMMARY"
echo ""
echo "To share, either zip the logs/ directory or just paste the contents of:"
echo "  $SUMMARY"
echo "plus any individual log file where you see FAIL."
