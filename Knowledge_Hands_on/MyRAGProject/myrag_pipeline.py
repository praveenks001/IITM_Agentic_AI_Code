import os
import numpy as np
import re
from openai import OpenAI
from pathlib import Path

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before running this notebook"

_client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to Load the Reference documents
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_documents():
    folder_path = Path("./reference_docs")

    documents = []

    for file_path in sorted(folder_path.glob("*.txt")):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        documents.append({
            "id": file_path.name,
            "text": text
        })

    print(f"Loaded {len(documents)} documents")

    return documents



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
def chunk_text_documents(documents: list[dict], size: int = 200,
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
# Method to determine the Cosine similarity 
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def cosine(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb)))



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to determine the top k results
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def show_top_k(query, k=3):
    """Given a query, print the top-K chunks with their cosine scores."""
    q_vec = embed_batch([query])[0]
    scored = [(cosine(q_vec, c["vector"]), c) for c in all_chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    print(f"\nQ: {query!r}")
    for i, (score, chunk) in enumerate(scored[:k], 1):
        print(f"  [{i}] {score:.3f}  {chunk['chunk_id']:<25s} {chunk['text'][:60]}...")




def retrieve(query: str, index: list[dict], k: int = 3,
             embed_model: str = EMBED_MODEL) -> list[dict]:
    """Embed the query, rank chunks by cosine, return top-K with scores."""
    q_vec = embed_batch([query], model=embed_model)[0]
    scored = [(cosine(q_vec, c["vector"]), c) for c in index]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return [{**c, "score": s} for s, c in scored[:k]]