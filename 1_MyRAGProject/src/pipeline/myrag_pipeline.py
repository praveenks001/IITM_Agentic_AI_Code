import os
import numpy as np
import re
import json
import time
import pymupdf  # aka fitz
import pdfplumber
from openai import OpenAI
from pathlib import Path 
from qdrant_client import QdrantClient
from dotenv import load_dotenv
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import PointStruct
from bs4 import BeautifulSoup
from docx import Document
from rank_bm25 import BM25Okapi


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
    
    if EMBED_MODEL == "text-embedding-3-large":
        size = 3072
    else:
        size = 1536

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
         vectors_config=VectorParams(size=size, distance=Distance.COSINE),         #For Large embedding model
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
# Qdrant - Method to delete and create a collection using HNSW
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def create_qdrant_collection_with_HNSW(qdrant, distance=Distance.COSINE, hnsw=None):
    """Delete-then-create so cells are re-runnable."""

    COLLECTION_NAME = "myrag_hnsw_collection"

    if EMBED_MODEL == "text-embedding-3-large":
        size = 3072
    else:
        size = 1536

    try:
        qdrant.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=size, distance=distance),
        hnsw_config=hnsw,
    )

    return COLLECTION_NAME



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
# Method to Load the PDF documents in ascending order - if the data contains is only text
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_documents_PDF():
    folder_path = Path("./data/corpus_pdf")

    pdf_files = sorted(folder_path.glob("*.pdf"))
    assert pdf_files, (f"No PDF files found in: {folder_path}")

    documents = []

    for file_path in pdf_files:

        # Open PDF
        pdf = pymupdf.open(file_path)

        # Extract text from all pages
        text = ""

        for page_num, page in enumerate(pdf, 1):
            text += page.get_text() + "\n"

        documents.append({
            "id": file_path.name,
            "text": text
        })

        pdf.close()

    #SAMPLE_HTML = SAMPLE_DIR / "product_page.html"
    #SAMPLE_DOCX = SAMPLE_DIR / "onboarding.docx"

    # Hard-fail if any are missing — the notebook depends on them
    # for path in [SAMPLE_PDF]:    # SAMPLE_HTML, SAMPLE_DOCX
    #     assert path.exists(), (
    #         f"Missing {path.name}. Run: python demos/generate_sample_docs.py"
    #     )

    return documents





#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to Load the PDF documents in ascending order - if the data contains tabular data + text
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_documents_PDF_for_tabular_and_text():
    folder_path = Path("./data/corpus_pdf")

    pdf_files = sorted(folder_path.glob("*.pdf"))
    assert pdf_files, (f"No PDF files found in: {folder_path}")

    documents = []

    for file_path in pdf_files:

        with pdfplumber.open(file_path) as pdf:
            
            # Text extraction — similar to PyMuPDF for prose
            text = ""

            for page_num, page in enumerate(pdf.pages, 1):

                page_text = page.extract_text() or ""
                text += f"\n--- Page {page_num} ---\n"
                text += page_text + "\n"

                # Extract tables from current page
                tables = page.extract_tables()
                print(f"Tables detected on page {page_num}: {len(tables)}")

                for i, table in enumerate(tables, 1):

                    print(f"\n══ Table {i} ══")

                    text += f"\nTable {i} (Page {page_num}):\n"

                    for row in table:
                        row_text = " | ".join(
                            str(c) if c is not None else ""
                            for c in row
                        )

                        print(row_text)

                        # Add table content for RAG
                        text += row_text + "\n"

            documents.append({
                "id": file_path.name,
                "text": text
            })

            print(f"\nLoaded: {file_path.name}")
            print(text[:300], "...\n")

    return documents



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to Load the PDF documents in ascending order - if the data contains image (scanned from paper documents)
#─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_documents_PDF_scanned_images():
    folder_path = Path("./data/corpus_pdf")

    pdf_files = sorted(folder_path.glob("*.pdf"))
    assert pdf_files, (f"No PDF files found in: {folder_path}")

    documents = []

    for file_path in pdf_files:

        scanned_path = SAMPLE_DIR / "scanned_fake.pdf"
        scan_doc = pymupdf.open()
        page = scan_doc.new_page()
        # Draw a filled rectangle to simulate a scanned image with no text layer
        page.draw_rect(pymupdf.Rect(100, 100, 500, 700), fill=(0.9, 0.9, 0.9))
        scan_doc.save(str(scanned_path))
        scan_doc.close()

        # Now try to extract text
        doc = pymupdf.open(scanned_path)
        extracted = doc[0].get_text()
        doc.close()

        print(f"Fake 'scanned' PDF: {scanned_path.name}")
        print(f"Extracted text length: {len(extracted)} chars")
        print(f"Extracted content: {extracted!r}")
        print()
        print("That's the trap. The parser succeeded — no crash. But zero content.")
        print("Your ingestion silently produced 0 chunks from this document.")

        # Cleanup — don't clutter sample_docs/
        scanned_path.unlink()



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to Load the HTML documents in ascending order
#─────────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_documents_HTML():

    folder_path = Path("./data/corpus_html")
    html_files = sorted(folder_path.glob("*.html"))
    assert html_files, (f"No HTML files found in: {folder_path}")

    documents = []

    for file_path in html_files:
        html = file_path.read_text()  # html = file_path.read_text(encoding="utf-8", errors="ignore")

        # Naive: get all text, no stripping
        # soup_naive = BeautifulSoup(html, "html.parser")
        # text = soup_naive.get_text(separator="\n", strip=True)

        soup_clean = BeautifulSoup(html, "html.parser")
        for tag in soup_clean(["nav", "footer", "script", "style"]):
            tag.decompose()

        # Get <main> if present, else body
        main = soup_clean.find("main") or soup_clean.find("body") or soup_clean
        text = main.get_text(separator="\n", strip=True)

        documents.append({
            "id": file_path.name,
            "text": text
        })

        print("══ CLEAN extraction (nav + footer + scripts included) ══")
        print(text[:400])
        print(f"\n... {len(text)} chars total\n")
    
    return documents




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to Load the docx documents in ascending order
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def load_documents_DOCX():

    folder_path = Path("./data/corpus_docx")
    docx_files = sorted(folder_path.glob("*.docx"))
    assert docx_files, (f"No docx files found in: {folder_path}")
   
    documents = []

    for file_path in docx_files:
        
        doc = Document(str(file_path))

        print(f"DOCX: {file_path.name}")
        print(f"Paragraphs: {len(doc.paragraphs)}\n")

        text = ""

        for i, para in enumerate(doc.paragraphs):
            if para.text.strip():
                style = para.style.name
                preview = para.text[:70]
                print(f"  [{i:2d}] {style:15s}  {preview}...")

                # Preserve heading information for RAG
                if style.startswith("Heading") or style == "Title":
                    text += f"\n{style}: {para.text.strip()}\n"
                else:
                    text += para.text.strip() + "\n"

        documents.append({
            "id": file_path.name,
            "text": text
        })

        print(f"\nLoaded: {file_path.name}")
        print(f"Extracted {len(text)} characters\n")

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



# 4. Recursive chunking - First chunk by paragraph and if a paragraph is too long then fallback to sentence splits

# a) Method to chunk a single document
def chunk_recursive(text: str, max_size: int = 400) -> list[str]:
    """Split by paragraphs. If a paragraph exceeds max_size, split by sentences."""

    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    for para in paragraphs:
        if len(para) <= max_size:
            chunks.append(para)

        else:
            # Sentence-level fallback
            sentences = re.split(r'(?<=[.!?])\s+', para)

            current = ""

            for sent in sentences:

                # Handle sentences longer than max_size
                if len(sent) > max_size:

                    if current:
                        chunks.append(current)
                        current = ""

                    for start in range(0, len(sent), max_size):
                        chunks.append(sent[start:start + max_size])

                    continue

                if len(current) + len(sent) + (1 if current else 0) <= max_size:
                    current = (current + " " + sent).strip()

                else:
                    if current:
                        chunks.append(current)

                    current = sent

            if current:
                chunks.append(current)

    return chunks


# b) Method to chunk multiple documents
def chunk_recursive_documents(documents: list[dict], max_size: int = 400) -> list[dict]:
    """Recursively chunk every document. Returns flat list with source pointers."""

    all_chunks = []

    for doc in documents:

        for chunk_idx, chunk in enumerate(chunk_recursive(doc["text"], max_size)):

            all_chunks.append({
                "chunk_id": f"{doc['id']}#{chunk_idx}",
                "source_id": doc["id"],
                "text": chunk
            })

    return all_chunks




# 5. Structure aware chunking - Best for docx
def chunk_docx_structure_aware(path):
    """Split DOCX using Heading 1 / Heading 2 and preserve section metadata."""

    doc = Document(str(path))

    chunks = []
    current_h1 = None
    current_h2 = None
    current_text = []

    def flush():
        if current_text:
            section_path = " > ".join(
                p for p in [current_h1, current_h2] if p
            )

            chunks.append({
                "section_path": section_path,
                "text": "\n".join(current_text)
            })

    for para in doc.paragraphs:

        if not para.text.strip():
            continue

        style = para.style.name

        if style == "Title" or style == "Heading 1":

            flush()

            current_text = []
            current_h1 = para.text.strip()
            current_h2 = None

        elif style == "Heading 2":

            flush()

            current_text = []
            current_h2 = para.text.strip()

        else:
            current_text.append(para.text.strip())

    flush()

    return chunks


# b) Method to chunk multiple DOCX documents
def chunk_docx_structure_aware_documents(folder_path="./data/corpus_docx"):
    """Chunk all DOCX files and return chunks with source pointers."""

    folder_path = Path(folder_path)

    docx_files = sorted(folder_path.glob("*.docx"))
    assert docx_files, f"No DOCX files found in: {folder_path}"

    all_chunks = []

    for file_path in docx_files:

        chunks = chunk_docx_structure_aware(file_path)

        for chunk_idx, chunk in enumerate(chunks):

            all_chunks.append({
                "chunk_id": f"{file_path.name}#{chunk_idx}",
                "source_id": file_path.name,
                "section_path": chunk["section_path"],
                "text": chunk["text"]
            })

        print(f"Created {len(chunks)} structure-aware chunks from {file_path.name}")

    print(f"Total DOCX structure-aware chunks: {len(all_chunks)}")

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
# Method to retrieve Top K results from Qdrant
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def retrieve_from_qdrant(query: str, qdrant, collection_name: str, k: int = 3,
                         embed_model: str = EMBED_MODEL) -> list[dict]:

    # Embed the question
    q_vec = embed_batch([query], model=embed_model)[0]

    # Qdrant performs cosine similarity search
    results = qdrant.query_points(
        collection_name=collection_name,
        query=q_vec,
        limit=k,
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
            chat_model: str = CHAT_MODEL,
            qdrant=None,
            collection_name=None) -> dict:
    """Full pipeline: retrieve → prompt → generate. Returns dict with
    answer, sources, cost, latency-relevant token counts."""

    # Start timer
    start_time = time.perf_counter()

    #retrieved = retrieve(question, index, k=k, embed_model=embed_model)

        # If Qdrant contains embedded chunks, retrieve from Qdrant
    if (qdrant is not None and collection_name is not None and is_qdrant_collection_populated(qdrant, collection_name)):
        print(f"Retrieving Top {k} chunks from Qdrant...")

        retrieved = retrieve_from_qdrant(
            question,
            qdrant,
            collection_name,
            k=k,
            embed_model=embed_model
        )

    else:

        # Existing non-Qdrant retrieval
        print(f"Retrieving Top {k} chunks using local cosine similarity...")

        retrieved = retrieve(
            question,
            index,
            k=k,
            embed_model=embed_model
        )


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




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to do Dense search
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def dense_search(query: str, k: int = 3) -> list[dict]:
    """Query Qdrant for top-K by cosine similarity."""
    q_vec = openai_client.embeddings.create(
        model="text-embedding-3-small", input=[query]
    ).data[0].embedding
    hits = qdrant.query_points(collection_name=COLLECTION, query=q_vec, limit=k).points
    return [
        {"id": h.payload["id"], "title": h.payload["title"], "score": h.score, "doc": h.payload}
        for h in hits
    ]

print(f"══ Dense retrieval on all 6 queries (top-1) ══\n")
print(f"  {'Expected':10s}  {'Query':<50s}  {'Retrieved':<20s}  Verdict")
print(f"  {'--------':10s}  {'-----':<50s}  {'---------':<20s}  -------")




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to do BM25
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def simple_tokenize(text: str) -> list[str]:
    """Lowercase + word-and-alphanumeric split, KEEPING hyphens/slashes inside tokens.
    
    Critical: this pattern preserves 'ac-1042' and 'v2/dashboards' as single
    tokens rather than splitting them. That's what makes BM25 catch exact IDs.
    """
    return re.findall(r'[a-z0-9][a-z0-9\-/_]*', text.lower())




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Method to do BM25 search
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
def bm25_search(query: str, k: int = 3) -> list[dict]:
    """Query BM25 index; return top-K."""
    tokens = simple_tokenize(query)
    scores = bm25.get_scores(tokens)
    ranked = sorted(zip(scores, corpus), key=lambda pair: pair[0], reverse=True)
    return [
        {"id": doc["id"], "title": doc["title"], "score": float(score), "doc": doc}
        for score, doc in ranked[:k]
    ]