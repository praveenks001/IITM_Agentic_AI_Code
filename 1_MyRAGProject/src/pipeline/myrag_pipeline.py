import os
import numpy as np
import re
import json
import time
from openai import OpenAI
from pathlib import Path
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before running this notebook"

_client = OpenAI()

EMBED_MODEL = "text-embedding-3-large"  #"text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to get Qdrant key & URL
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def get_qdrant_client():
    """Build a QdrantClient using QDRANT_URL + QDRANT_API_KEY env vars.

    Resolution order (same as capstone src/rag/qdrant_store.py):
      1. QDRANT_URL + QDRANT_API_KEY → Qdrant Cloud
      2. QDRANT_URL only → local (no auth)
      3. Default → http://localhost:6333 (local Docker fallback)
    """

    load_dotenv()

    assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before running this notebook"
    assert os.environ.get("QDRANT_URL"),     "Set QDRANT_URL — get free-tier at cloud.qdrant.io"
    assert os.environ.get("QDRANT_API_KEY"), "Set QDRANT_API_KEY — from your Qdrant Cloud cluster"
    print(f"Qdrant Keys loaded")

    url = os.environ.get("QDRANT_URL", "http://localhost:6333")
    api_key = os.environ.get("QDRANT_API_KEY") or None
    if api_key:
        return QdrantClient(url=url, api_key=api_key)
    return QdrantClient(url=url)




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Qdrant - Method to check the exsiting collection and create a new collection
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def create_qdrant_collection(qdrant):

    COLLECTION_NAME = "myrag_collection"

    existing = qdrant.get_collections()
    existing_names = [c.name for c in existing.collections]

    if COLLECTION_NAME in existing_names:

        print(f"Collection {COLLECTION_NAME!r} already exists.")

        info = qdrant.get_collection(COLLECTION_NAME)

        print(f"  dim:      {info.config.params.vectors.size}")
        print(f"  metric:   {info.config.params.vectors.distance}")
        print(f"  points:   {info.points_count}")

        return COLLECTION_NAME
    
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,

        #vectors_config=VectorParams(size=1536, distance=Distance.COSINE),         #For small embedding model
         vectors_config=VectorParams(size=3072, distance=Distance.COSINE),         #For Large embedding model
    )

    info = qdrant.get_collection(COLLECTION_NAME)
    print(f"Created collection {COLLECTION_NAME!r}")
    print(f"  dim:      {info.config.params.vectors.size}")
    print(f"  metric:   {info.config.params.vectors.distance}")
    print(f"  points:   {info.points_count}")
    return COLLECTION_NAME




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to check if qdrant contains any points (meaning embedded chunks are upserted already)
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def is_qdrant_collection_populated(qdrant, collection_name):

    info = qdrant.get_collection(collection_name)
    print(f"Qdrant collection points: {info.points_count}")

    return info.points_count > 0




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to Load the Reference documents in ascending order
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_documents():
    folder_path = Path("./data/corpus")

    documents = []

    for file_path in sorted(folder_path.glob("*.txt")):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        documents.append({
            "id": file_path.name,
            "text": text
        })

    return documents



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to Load the golden set questions
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_golden_set():
    """Load the golden set questions from JSON file."""

    golden_path = Path("./data/goldenSet.json")

    with open(golden_path, "r", encoding="utf-8") as f:
        golden_set = json.load(f)

    return golden_set




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to chunk the document
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────

# 1. Sliding Window - using overlap method

# a) Method to chunk the single document
def chunk_text(text: str, size: int = 200, overlap: int = 40) -> list[str]:
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


# b)Method to chunk the multiple documents 
def chunk_documents(documents: list[dict], size: int = 200,
                    overlap: int = 40) -> list[dict]:
    """Chunk every document. Returns flat list with source pointers."""
    all_chunks = []
    for doc in documents:
        for chunk_idx, chunk in enumerate(chunk_text(doc["text"], size, overlap)):
            all_chunks.append({
                "chunk_id":  f"{doc['id']}#{chunk_idx}",
                "source_id": doc["id"],
                "text":      chunk,
            })
    return all_chunks


# 2. Chunk By Sentence

# a) Method to chunk the single document
def chunk_by_sentence(text: str) -> list[str]:
    """Split on sentence boundaries. Simpler than a proper NLP splitter."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s for s in sentences if s]



# b)Method to chunk the multiple documents
def chunk_by_sentence_documents(documents: list[dict]) -> list[dict]:
    """Chunk every document. Returns flat list with source pointers."""
    all_chunks = []
    for doc in documents:
        for chunk_idx, chunk in enumerate(chunk_by_sentence(doc["text"])):
            all_chunks.append({
                "chunk_id":  f"{doc['id']}#{chunk_idx}",
                "source_id": doc["id"],
                "text": chunk_by_sentence(doc["text"]),
            })
    return all_chunks


# 3. Chunk By Paragraph 

# a) Method to chunk the single document
def chunk_by_paragraph(text: str) -> list[str]:
    """One chunk per paragraph."""
    return [p.strip() for p in text.split("\n\n") if p.strip()]


# b)Method to chunk the multiple documents
def chunk_by_paragraph_documents(documents: list[dict]) -> list[dict]:
    """Chunk every document. Returns flat list with source pointers."""
    all_chunks = []
    for doc in documents:
        for chunk_idx, chunk in enumerate(chunk_by_paragraph(doc["text"])):
            all_chunks.append({
                "chunk_id":  f"{doc['id']}#{chunk_idx}",
                "source_id": doc["id"],
                "text": chunk_by_sentence(doc["text"]),
            })
    return all_chunks



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to Embed the chunks / texts 
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """One API call, list of vectors back."""
    resp = _client.embeddings.create(model=model, input=texts)
    return [item.embedding for item in resp.data]


def build_index(chunks: list[dict], model: str = EMBED_MODEL) -> list[dict]:
    """Attach 'vector' field to each chunk. Returns the same list mutated."""
    texts = [c["text"] for c in chunks]
    vectors = embed_batch(texts, model=model)
    for chunk, vec in zip(chunks, vectors):
        chunk["vector"] = vec
    return chunks





#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to upsert the embbeded chunks into Qdrant
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def upsert_embedded_chunks_into_qdrant(all_chunks, vectors,qdrantCollection):
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

    # qdrant.upsert(collection_name=qdrantCollection, points=points)
    
        
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




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to determine the Cosine similarity 
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to determine the top k results
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def show_top_k(queries, all_chunks, k=5):
    """Print top-K chunks for multiple queries."""

    query_vectors = embed_batch(queries)

    for query, q_vec in zip(queries, query_vectors):
        scored = [(cosine(q_vec, chunk["vector"]), chunk) for chunk in all_chunks]
        scored.sort(key=lambda pair: pair[0], reverse=True)



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to retrieve the top k results
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def retrieve(query: str, index: list[dict], k: int = 3,
             embed_model: str = EMBED_MODEL) -> list[dict]:
    """Embed the query, rank chunks by cosine, return top-K with scores."""
    q_vec = embed_batch([query], model=embed_model)[0]
    scored = [(cosine(q_vec, c["vector"]), c) for c in index]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**c, "score": s} for s, c in scored[:k]]



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to Prompt + generate
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
SYSTEM = (
    "You are a helpful assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain the answer, say so "
    "plainly. Cite the source id in square brackets after any fact you use."
)


def build_prompt(question: str, retrieved: list[dict],
                 system: str = SYSTEM) -> tuple[str, str]:
    """Return (system_message, user_message) so we can inspect them."""
    context = "\n\n".join(
        f"[{hit['chunk_id']}]\n{hit['text']}"
        for hit in retrieved
    )
    user_msg = f"Context:\n{context}\n\n---\n\nQuestion: {question}"
    return system, user_msg


def ask_rag(question: str, index: list[dict], k: int = 3,
            system: str = SYSTEM,
            embed_model: str = EMBED_MODEL,
            chat_model: str = CHAT_MODEL) -> dict:
    """Full pipeline: retrieve → prompt → generate. Returns dict with
    answer, sources, cost, latency-relevant token counts."""

    # Start timer
    start_time = time.perf_counter()

    retrieved = retrieve(question, index, k=k, embed_model=embed_model)
    system_msg, user_msg = build_prompt(question, retrieved, system=system)
    resp = _client.chat.completions.create(
        model=chat_model,
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
        tokens_in * PRICE_INPUT_PER_1M[chat_model] / 1_000_000 +
        tokens_out * PRICE_OUTPUT_PER_1M[chat_model] / 1_000_000
    )

    # Calculate total RAG latency in seconds
    latency = time.perf_counter() - start_time

    return {
        "question":   question,
        "answer":     resp.choices[0].message.content,
        "sources":    [hit["chunk_id"] for hit in retrieved],
        "tokens_in":  resp.usage.prompt_tokens,
        "tokens_out": resp.usage.completion_tokens,
        "retrieved":  retrieved,  # full retrieved chunks for inspection,
        "cost_usd": cost,
        "latency_s":  latency,
    }




# Pricing constants (from W4 multi-model week)
PRICE_INPUT_PER_1M  = {"gpt-4o-mini": 0.15}
PRICE_OUTPUT_PER_1M = {"gpt-4o-mini": 0.60}
PRICE_EMBED_PER_1M  = {"text-embedding-3-small": 0.02,
                       "text-embedding-3-large": 0.13}


def cost_usd(result: dict, chat_model: str = CHAT_MODEL) -> float:
    """Compute cost of a single ask_rag call from token counts."""
    return (
        result["tokens_in"]  * PRICE_INPUT_PER_1M[chat_model]  / 1_000_000 +
        result["tokens_out"] * PRICE_OUTPUT_PER_1M[chat_model] / 1_000_000
    )
