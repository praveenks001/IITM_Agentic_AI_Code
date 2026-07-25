#!/usr/bin/env bash
# W3 activity — demo curl sequence to show /metrics counting requests.
# Run this against `uvicorn api.main:app --reload --port 8000`.

set -e

BASE="http://localhost:8000"

echo "=== Hitting /health three times ==="
curl -s "${BASE}/health"
echo
curl -s "${BASE}/health"
echo
curl -s "${BASE}/health"
echo

echo
echo "=== POST /ask_batched ==="
curl -s -X POST "${BASE}/ask_batched" \
    -H "Content-Type: application/json" \
    -d '{"question": "What is RAG?"}'
echo

echo
echo "=== POST /ask ==="
curl -s -X POST "${BASE}/ask" \
    -H "Content-Type: application/json" \
    -d '{"question": "Explain async in two sentences."}'
echo

echo
echo "=== Fetch /metrics ==="
curl -s "${BASE}/metrics" | python -m json.tool
