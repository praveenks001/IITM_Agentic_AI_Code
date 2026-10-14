"""MP2 · Mini-RAG — Starter Template
====================================

You'll build a complete RAG pipeline over the Sherlock Holmes corpus in this
file. Fill in every TODO. The reference solution is ~250 lines, but yours can
be shorter or longer — what matters is that it works end-to-end.

Pipeline you're building:
    corpus/*.txt  →  chunks  →  embeddings  →  Qdrant
                                                  ↓
                              question  →  retrieve  →  answer + citations

Run sequence (once you've filled in the TODOs):
    pip install -r requirements.txt
    source .env                 # exports your OpenAI + Qdrant credentials
    python mp2_rag.py ingest    # builds the collection (run once)
    python mp2_rag.py ask       # interactive Q&A loop
    python mp2_rag.py validate  # runs against data/predefined_questions.jsonl

Tip: get the CORE pipeline working FIRST (Steps 1-7 below), THEN come back to
polish and add your 3 questions. Don't try to perfect each step before moving
on — you'll learn more from a rough end-to-end loop than a polished half.
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from dotenv import load_dotenv


# ─── Configuration ──────────────────────────────────────────────────────

CORPUS_DIR        = Path(__file__).parent / "corpus"
DATA_DIR          = Path(__file__).parent / "data"
COLLECTION_NAME   = "mp2_sherlock"
EMBEDDING_MODEL   = "text-embedding-3-small"
EMBEDDING_DIM     = 1536
CHAT_MODEL        = "gpt-4o-mini"
TARGET_CHUNK_SIZE = 500   # characters
CHUNK_OVERLAP     = 80    # characters
PRICE_INPUT_PER_1M  = {"gpt-4o-mini": 0.15}
PRICE_OUTPUT_PER_1M = {"gpt-4o-mini": 0.60}


load_dotenv()
assert os.environ.get("QDRANT_URL"),     "Set QDRANT_URL — get free-tier at cloud.qdrant.io"
assert os.environ.get("QDRANT_API_KEY"), "Set QDRANT_API_KEY — from your Qdrant Cloud cluster"


openai = OpenAI()
qdrant = QdrantClient(
    url=os.environ["QDRANT_URL"],
    api_key=os.environ.get("QDRANT_API_KEY"),
)

print(f"Qdrant Keys loaded")
print("Qdrant is:::",qdrant)


# ─── Step 1: Load the corpus ────────────────────────────────────────────

def load_corpus(corpus_dir: Path) -> list[dict[str, Any]]:
    """Read every .txt file in the corpus directory.

    Returns a list of dicts, each with: source (filename), title (first line),
    and text (full content).

    TODO:
      - Iterate over every *.txt file in corpus_dir (use Path.glob)
      - For each file, read its text and extract the first non-empty line as title
      - Return the list of doc dicts
    """
    documents = []

    for file_path in sorted(corpus_dir.glob("*.txt")):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        documents.append({
            "source": file_path.name,
            "text": text
        })

    return documents


#─── Step 2: Chunk each document ────────────────────────────────────────
def chunk_text(text: str, size: int = 500, overlap: int = 40) -> list[str]:
    """Sliding window over characters. Simplest possible chunker."""
    if len(text) <= size:
        return [text]
    chunks, i = [], 0
    while i < len(text):
        end = min(i + size, len(text))
        chunks.append(text[i:end])
        if end == len(text):
            break
        i = end - overlap
    return chunks


def chunk_document(doc: dict[str, Any]) -> list[dict[str, Any]]:
    """Split a document into smaller chunks.

    Each chunk should be a dict with: source, title, section, text.

    Approach (your choice):
      - Simple: fixed-size windows (split text into N-character chunks with overlap)
      - Smarter: split on paragraph boundaries (\\n\\n), then pack paragraphs
        into chunks up to TARGET_CHUNK_SIZE characters

    The reference solution uses the smarter approach, plus heuristic
    section-header detection (short lines without terminal punctuation).
    Either approach is acceptable.

    TODO:
      - Pick an approach
      - Implement it
      - Return list of chunk dicts
    """
    all_chunks = []
    for chunk_idx, chunk in enumerate(chunk_text(doc["text"], TARGET_CHUNK_SIZE ,CHUNK_OVERLAP)):
        all_chunks.append({
            "chunk_id":  f"{doc['source']}#{chunk_idx}",
            "source_id": doc["source"],
            "text":      chunk,
        })
    return all_chunks


# ─── Step 3: Embed text ─────────────────────────────────────────────────

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch-embed a list of texts using OpenAI's embedding model.

    Returns a list of 1536-dim float vectors (same order as inputs).

    TODO:
      - Call openai.embeddings.create with EMBEDDING_MODEL and the texts
      - Extract the embedding vectors from the response
    """

    resp = openai.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in resp.data]


# ─── Step 4: Set up the Qdrant collection ───────────────────────────────

def setup_collection() -> None:
    """Create (or recreate) the Qdrant collection.

    TODO:
      - Use qdrant.recreate_collection
      - VectorParams with EMBEDDING_DIM and Distance.COSINE
    """

    existing = qdrant.get_collections()
    existing_names = [c.name for c in existing.collections]
    print("Existing collectiond names are:::",existing_names)

    if COLLECTION_NAME in existing_names:

        print(f"Collection {COLLECTION_NAME!r} already exists, so no need to create new collection")

        info = qdrant.get_collection(COLLECTION_NAME)

        print(f"Dimensions: {info.config.params.vectors.size}")
        print(f"Distance: {info.config.params.vectors.distance}")
        print(f"Points: {info.points_count}")

        return #return COLLECTION_NAME


    print(f"New Collection is going to create.")
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),         #For Large embedding model
    )

    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"Created collection {COLLECTION_NAME!r}")
    print(f"  points:   {info.points_count}")
    #return COLLECTION_NAME


# ─── Step 4a: Check Qdrant exists for created collection ───────────────────────────────

def is_qdrant_collection_populated(qdrant, collection_name):

    info = qdrant.get_collection(collection_name)
    print(f"Qdrant collection points: {info.points_count}")

    return info.points_count > 0




# ─── Step 4b: Upsert the embedded chunks into qdrant collection ───────────────────────────────

def upsert_embedded_chunks_into_qdrant(all_chunks, vectors,qdrantCollection,qdrant):
    points = [
        PointStruct(
            id=idx,
            vector=vectors[idx],
            payload={
                "chunk_id": chunk["chunk_id"],
                "source": chunk.get("source_id"),
                "text": chunk["text"],
            },
        )
        for idx, chunk in enumerate(all_chunks)
    ]

    print(f"Total corpus chunks : {len(all_chunks)}")
    print(f"Total points created: {len(points)}")
    
        
    # ------------------------------------------------------------
    # Upload to Qdrant in batches since it cannot upload all chunks in one time
    # ------------------------------------------------------------

    BATCH_SIZE = 100

    print(f"\nUploading to Qdrant in batches of {BATCH_SIZE}...\n")

    for start in range(0, len(points), BATCH_SIZE):

        end = min(
            start + BATCH_SIZE,
            len(points)
        )

        batch = points[start:end]

        qdrant.upsert(
            collection_name=qdrantCollection,
            points=batch,
            wait=True
        )

        print(
            f"Uploaded {start + 1:4d} - {end:4d} "
            f"({end}/{len(points)})"
        )



    # Verify exact number of points
    info = qdrant.get_collection(qdrantCollection)
    print(f"Upserted {len(points)} points.")
    print(f"Collection now has {info.points_count} points.")




# ─── Step 5: Ingest chunks into Qdrant ──────────────────────────────────

def ingest_chunks(chunks: list[dict[str, Any]]) -> None:
    """Embed every chunk and upsert into Qdrant.

    TODO:
      - Call embed_texts on the chunk texts
      - Build PointStruct objects (id=uuid, vector, payload=chunk dict)
      - qdrant.upsert
    """

    if is_qdrant_collection_populated(qdrant, COLLECTION_NAME):
        print("Qdrant collection already contains embedded chunks.so we can skip")

    else:

        chunk_texts = [c["text"] for c in chunks]
        print(len(chunk_texts))

        vectors = embed_texts(chunk_texts)

        for chunk, vec in zip(chunks, vectors):
            chunk["vector"] = vec

        print(f"Embedded {len(vectors)} chunks.")
        print(f"Each embedding shape: {len(vectors[0])} dimensions")
        print(f"\nFirst 8 dims of chunk 0 ({chunks[0]['chunk_id']}):")
        print(f"  {vectors[0][:8]}")
        print(f"\nFirst 8 dims of chunk 1 ({chunks[1]['chunk_id']}):")
        print(f"  {vectors[1][:8]}")


    if is_qdrant_collection_populated(qdrant, COLLECTION_NAME):
        print("Qdrant collection already contains embedded chunks. Skipping upsert.")

    else:
        upsert_embedded_chunks_into_qdrant(chunks, vectors, COLLECTION_NAME, qdrant)




# ─── Step 6: Retrieve ───────────────────────────────────────────────────

def retrieve(query: str, k: int = 3) -> list[dict[str, Any]]:
    """Retrieve top-k chunks for a query.

    TODO:
      - Embed the query
      - qdrant.search with the query vector, limit=k
      - Return list of chunk dicts (include score for citations)
    """

    # Embed the question
    q_vec = embed_texts([query])[0]

    if is_qdrant_collection_populated(qdrant, COLLECTION_NAME):
        print("Qdrant collection already contains embedded chunks. so pass the embeeded query to retrieve the cosine similarity chunks")

        # Qdrant performs cosine similarity search
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=q_vec,
            limit=k,
            with_payload=True
        ).points

        # Convert Qdrant results into same structure used by existing RAG
        retrieved = []

        for hit in results:

            payload = hit.payload

            retrieved.append({
                "chunk_id": payload.get("chunk_id", ""),
                "source_id": payload.get("source", ""),
                "text": payload.get("text", ""),
                "score": hit.score,
            })

        return retrieved


    else:
        q_vec = embed_texts([query])[0]
        scored = [(cosine(q_vec, c["vector"]), c) for c in index]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [{**c, "score": s} for s, c in scored[:k]]


    


# ─── Step 7: Generate the answer ────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful assistant answering questions about a small
collection of Sherlock Holmes stories. You will be given the user's question and
several relevant excerpts. Use ONLY the provided excerpts to answer. If the
excerpts don't contain the answer, say so plainly. Cite the source (story title
+ section) in your answer."""

def build_prompt(question: str, retrieved: list[dict],
                 system: str = SYSTEM_PROMPT) -> tuple[str, str]:
    """Return (system_message, user_message) so we can inspect them."""
    context = "\n\n".join(
        f"[{hit['chunk_id']}]\n{hit['text']}"
        for hit in retrieved
    )
    user_msg = f"Context:\n{context}\n\n---\n\nQuestion: {question}"
    return system, user_msg




def answer(question: str, k: int = 3) -> dict[str, Any]:
    """End-to-end: retrieve, format context, call LLM, return result.

    TODO:
      - Call retrieve(question, k=k)
      - Format the retrieved chunks into a context string
        (include "[Source: <title> — <section>]" before each)
      - Call openai.chat.completions.create with SYSTEM_PROMPT and the user message
      - Return dict with: question, answer, citations, latency_ms
    """
    start_time = time.perf_counter()

    # call retrieve
    retrieved = retrieve(question,k=k)

    total_cost = 0.0
    total_latency = 0.0
    total_tokens_in = 0
    total_tokens_out = 0

    system_msg, user_msg = build_prompt(question, retrieved, system=SYSTEM_PROMPT)

    resp = openai.chat.completions.create(
        model=CHAT_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": user_msg},
        ],
    )
    
    # Token counts
    tokens_in = resp.usage.prompt_tokens
    tokens_out = resp.usage.completion_tokens

    # Calculate chat cost
    cost = (
        tokens_in * PRICE_INPUT_PER_1M[CHAT_MODEL] / 1_000_000 +
        tokens_out * PRICE_OUTPUT_PER_1M[CHAT_MODEL] / 1_000_000
    )

    # Calculate total RAG latency in seconds
    latency = time.perf_counter() - start_time
    
    print("The answer for below question is::",resp.choices[0].message.content)

    return {
        "question":   question,
        "answer":     resp.choices[0].message.content,
        "sources":    [hit["chunk_id"] for hit in retrieved],
        "tokens_in":  resp.usage.prompt_tokens,
        "tokens_out": resp.usage.completion_tokens,
        "retrieved":  retrieved,  # full retrieved chunks for inspection,
        "cost_usd": cost,
        "latency_s":  latency,
        "latency_ms":  latency * 1000,
        "citations": [
            {
                "source": hit["source_id"],
                "chunk_id": hit["chunk_id"],
                "title": hit["source_id"],
                "section": hit["chunk_id"]
            }
            for hit in retrieved]
    }



# ─── Validation harness (provided — do not modify) ──────────────────────

def validate_against(jsonl_path: Path) -> None:
    questions = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()]
    print(f"\n  Validating {len(questions)} questions from {jsonl_path.name}…\n")

    hits = 0
    for q in questions:
        result = answer(q["question"], k=3)
        cited_sources = {cit["source"] for cit in result["citations"]}
        source_hit = q["expected_source"] in cited_sources

        ans_lower = result["answer"].lower()
        facts_hit = sum(1 for fact in q.get("expected_facts", []) if fact.lower() in ans_lower)
        facts_total = len(q.get("expected_facts", []))

        verdict = "✓" if source_hit else "✗"
        print(f"  {verdict} {q['id']}")
        print(f"      Q: {q['question']}")
        print(f"      Cited: {', '.join(cited_sources)}")
        print(f"      Expected: {q['expected_source']}")
        print(f"      Facts matched: {facts_hit}/{facts_total}")
        print(f"      Latency: {result.get('latency_ms', '?')}ms")
        print()
        if source_hit:
            hits += 1

    print(f"  Source-match: {hits}/{len(questions)}")


# ─── CLI (provided — do not modify) ─────────────────────────────────────

def cmd_ingest() -> None:
    print("→ Loading corpus…")
    docs = load_corpus(CORPUS_DIR)
    print(f"  {len(docs)} documents loaded")

    print("→ Chunking…")
    all_chunks: list[dict[str, Any]] = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"→ Total chunks: {len(all_chunks)}")
    print("→ Setting up Qdrant collection…")
    setup_collection()

    print("→ Ingesting…")
    ingest_chunks(all_chunks)
    print("\n✓ Done. Try: python mp2_rag.py ask")


def cmd_ask() -> None:
    print("Mini-RAG over the Sherlock Holmes corpus.")
    print("Type your question. Empty line or Ctrl-C to exit.\n")
    while True:
        try:
            q = input("? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not q:
            return
        result = answer(q, k=3)
        print(f"\n{result['answer']}\n")
        print("  Sources:")
        for c in result["citations"]:
            print(f"    - {c['title']} — {c['section']}")
        print(f"  Latency: {result.get('latency_ms', '?')}ms\n")


def cmd_validate() -> None:
    validate_against(DATA_DIR / "predefined_questions.jsonl")
    learner_path = DATA_DIR / "learner_questions.jsonl"
    if learner_path.exists():
        first = json.loads(learner_path.read_text().splitlines()[0])
        if not first["question"].startswith("Replace this"):
            validate_against(learner_path)


def main() -> None:
    print("Executing main:::")
    print("sys.argv is:::",sys.argv)

    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    print("calling main cmd", cmd)

    if cmd == "ingest":   cmd_ingest()
    elif cmd == "ask":    cmd_ask()
    elif cmd == "validate": cmd_validate()
    else:
        print(f"Unknown command: {cmd}\n")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    print("calling main")
    main()
