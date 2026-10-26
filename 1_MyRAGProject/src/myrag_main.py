import os
import numpy as np
import json
import time

from openai import OpenAI
from pathlib import Path
from pipeline import myrag_pipeline as pipeline
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, HnswConfigDiff, Filter, FieldCondition, MatchValue,IsEmptyCondition, PayloadField
from rank_bm25 import BM25Okapi
from pipeline.document_refresh import sync_documents
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)


from config import (
    EMBED_MODEL,
    CHAT_MODEL,
    TOP_K,
    SEARCH_METHOD,
    QDRANT_TYPE,
    DOCUMENT_TYPE,
    CHUNKING_STRATEGY,
    COMPARISON_CHART_REQD,
    USE_RERANKER,
    CACHE_DB,
    ENABLE_CACHE_DB,
    REFRESH_DOCUMENT,
    AUTO_SYNC_DOCUMENTS,
    ENABLE_RAGAS_EVALUATION,
    RAGAS_OUTPUT_DIR,
)

        

embed_model = EMBED_MODEL
top_k = TOP_K
search_method = SEARCH_METHOD
qdrant_type = QDRANT_TYPE
document_type = DOCUMENT_TYPE
chunking_strategy = CHUNKING_STRATEGY
comparison_chart_reqd = COMPARISON_CHART_REQD
use_reranker = USE_RERANKER
bm25 = None
isCacheDBEnabled = ENABLE_CACHE_DB

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before running this notebook"

client = OpenAI()


#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Load Qdrant 
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
qdrant = pipeline.get_qdrant_client()
print("Qdrant isss:::", qdrant)



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Qdrant - Create collection to store the embedded chunks
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
if qdrant_type == "NORMAL":
    if search_method == "HYBRID_RRF": 
        qdrantCollection_dense = pipeline.create_qdrant_collection(qdrant,"DENSE_SEARCH")  #Normal
        qdrantCollection_bm25 = pipeline.create_qdrant_collection(qdrant,"BM25")  #Normal
    else:
        qdrantCollection = pipeline.create_qdrant_collection(qdrant,search_method)  #Normal

else:
    hnsw_config = HnswConfigDiff(
        m=16, #32 16 is enough
        ef_construct=100  #200
    )
    qdrantCollection = pipeline.create_qdrant_collection_with_HNSW(qdrant, Distance.COSINE, hnsw_config)  #create a qdrant collection using HNSW




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The corpus = Load the Reference documents
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
if document_type == "txt":

    print("Going to load txt documents")
    documents = pipeline.load_documents();
    print(f"Loaded {len(documents)} documents")

elif document_type == "PDF":

    print("Going to load PDF documents")
    documents = pipeline.load_documents_PDF();
    print(f"Loaded {len(documents)} documents")
    print("Sample Non txt documents ready:")

elif document_type == "PDF_WITH_TABLE":

    print("Going to load txt documents")
    documents = pipeline.load_documents_PDF_for_tabular_and_text();
    print(f"Loaded {len(documents)} HTML documents")
    print("Sample PDF text + tabular documents Loaded:")


elif document_type == "HTML":

    print("Going to load HTML documents")
    documents = pipeline.load_documents_HTML();
    print(f"Loaded {len(documents)} HTML documents")
    print("Sample HTML documents Loaded:")


elif document_type == "DOCX":

    print("Going to load DOCX documents")
    documents = pipeline.load_documents_DOCX();
    print(f"Loaded {len(documents)} DOCX documents")
    print("Sample DOCX documents Loaded:")

else:

    documents = pipeline.load_documents_PDF_scanned_images();
    print(f"Loaded {len(documents)} DOCX documents")
    print("Sample DOCX documents Loaded:")




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The goldenSet Questions = Load it
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
golden_set = pipeline.load_golden_set()
print(f"Loaded {len(golden_set)} golden-set questions")
print("Golden Set fields:", golden_set[0].keys())



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Using chunk_text method - Chunk every document;
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The default chunking size was 200 but then the number of chunks were increased to 5250 for the reference documents
# and it supposed to be 2048 or lesser, so increase the chunking size to 500

#1.Sliding window Chunking
if chunking_strategy == "Sliding_window_Chunking":

    all_chunks = pipeline.chunk_documents(documents, 500); #chunked_documents = pipeline.chunk_text_documents(documents, 200);
    print(f"Total chunks from {len(documents)} documents: {len(all_chunks)}\n")


# Write the result chunk into an output file
DATA_DIR = Path('./docs/runs')
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATA_DIR / "01_chunks_using_sliding_window.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for result in all_chunks:
        f.write(str(result))
        f.write("\n")

for c in all_chunks[:8]:  # show first 8 to avoid wall of text
    marker = "…" if len(c["text"]) == 500 else " "
    print(f"  {c['chunk_id']:30s} [{len(c['text']):3d} chars] {c['text'][:60]}{marker}")
print(f"  ... and {len(all_chunks) - 8} more chunks")



# #──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# # Using chunk_text method - Chunk every document;
# #──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# #2.Chunk By Sentence Chunking
# all_chunks = pipeline.chunk_by_sentence_documents(documents);
# print(f"Total chunks from {len(documents)} documents: {len(all_chunks)}\n")


# # Write the result chunk into an output file
# OUTPUT_FILE = DATA_DIR / "02_chunks_by_sentence.txt"

# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for result in all_chunks:
#         f.write(str(result))
#         f.write("\n")

# for c in all_chunks[:8]:  # show first 8 to avoid wall of text
#     marker = "…" if len(c["text"]) == 200 else " "
#     print(f"  {c['chunk_id']:30s} [{len(c['text']):3d} chars] {c['text'][:60]}{marker}")
# print(f"  ... and {len(all_chunks) - 8} more chunks")



# #──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# # Using chunk_text method - Chunk every document;
# #──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# #3.Chunk By Paragraph Chunking
# all_chunks = pipeline.chunk_by_paragraph_documents(documents);
# print(f"Total chunks from {len(documents)} documents: {len(all_chunks)}\n")


# # Write the result chunk into an output file
# OUTPUT_FILE = DATA_DIR / "03_chunks_by_paragraph.txt"

# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for result in all_chunks:
#         f.write(str(result))
#         f.write("\n")

# for c in all_chunks[:8]:  # show first 8 to avoid wall of text
#     marker = "…" if len(c["text"]) == 200 else " "
#     print(f"  {c['chunk_id']:30s} [{len(c['text']):3d} chars] {c['text'][:60]}{marker}")
# print(f"  ... and {len(all_chunks) - 8} more chunks")



# #──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# # Using chunk_text method - Chunk every document;
# #──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# #4.Chunk By Recursive Chunking

# all_chunks = pipeline.chunk_recursive(page1_text, max_size=500)
# print(f"Recursive chunking (max_size=500) → {len(all_chunks)} chunks\n")




# #──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# # Using chunk_text method - Chunk every document;
# #──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# #5.Chunk By Strucutre-aware
# all_chunks = pipeline.chunk_docx_structure_aware_documents("./data/corpus_docx")
# print(f"Total chunks created: {len(all_chunks)}")





#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Embedding the chunks
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# check if qdrant collection has already embedding else embed the chunks
if search_method in ("DENSE_SEARCH", "HYBRID_RRF"):
    if search_method == "HYBRID_RRF":
        qdrantCollection = qdrantCollection_dense

    if pipeline.is_qdrant_collection_populated(qdrant, qdrantCollection):
        print("Qdrant collection already contains embedded chunks.so we can skip")

    else:
        chunk_texts = [c["text"] for c in all_chunks]
        print(len(chunk_texts))

        vectors = pipeline.embed_batch(chunk_texts)

        for chunk, vec in zip(all_chunks, vectors):
            chunk["vector"] = vec

        print(f"Embedded {len(vectors)} chunks.")
        print(f"Each embedding shape: {len(vectors[0])} dimensions")
        print(f"\nFirst 8 dims of chunk 0 ({all_chunks[0]['chunk_id']}):")
        print(f"  {vectors[0][:8]}")
        print(f"\nFirst 8 dims of chunk 1 ({all_chunks[1]['chunk_id']}):")
        print(f"  {vectors[1][:8]}")



if search_method in ("BM25", "HYBRID_RRF"):
    if search_method == "HYBRID_RRF":
        qdrantCollection = qdrantCollection_bm25

    if pipeline.is_qdrant_collection_populated(qdrant, qdrantCollection):
        print("BM25 Qdrant collection already contains embedded chunks.so we can skip")

    else:
        print("══ BM25 Tokenization checks - Tokenize the chunks ══")

        for chunk in all_chunks[:top_k]:
            text = chunk["text"]
            tokens = pipeline.simple_tokenize(text)

            print(f"Chunk ID: {chunk['chunk_id']}")
            print(f"Tokens: {tokens}\n")



        print("\n══ Building BM25 index ══")

        tokenized_corpus = [pipeline.simple_tokenize(d["text"])for d in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        print(f"Indexed {len(all_chunks)} chunks. Ready to query.")




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Qdrant - Upsert the embedded chunks into Qdrant
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
if search_method != "HYBRID_RRF":
    if pipeline.is_qdrant_collection_populated(qdrant, qdrantCollection):
        print("Qdrant collection already contains embedded chunks. Skipping upsert.")

    else:
        if search_method == "DENSE_SEARCH":
            pipeline.upsert_embedded_chunks_into_qdrant(all_chunks, vectors, qdrantCollection, qdrant)
        else:
            pipeline.upsert_BM25_chunks_into_qdrant(all_chunks, qdrantCollection, qdrant)

else:
    if pipeline.is_qdrant_collection_populated(qdrant, qdrantCollection_dense):
        print("Qdrant collection already contains embedded chunks. Skipping upsert.")

    else:
        pipeline.upsert_embedded_chunks_into_qdrant(all_chunks, vectors, qdrantCollection_dense, qdrant)



    if pipeline.is_qdrant_collection_populated(qdrant, qdrantCollection_bm25):
        print("Qdrant collection already contains embedded chunks. Skipping upsert.")

    else:
        pipeline.upsert_BM25_chunks_into_qdrant(all_chunks, qdrantCollection_bm25, qdrant)




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Tombstone - Refresh Updated Document
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
if REFRESH_DOCUMENT and AUTO_SYNC_DOCUMENTS:

    changed_documents = sync_documents(
        qdrant=qdrant,
        embed_model=embed_model
    )

    if changed_documents:
        cache_conn = pipeline.cache_init(CACHE_DB)

        try:
            deleted = pipeline.cache_clear(cache_conn)
        finally:
            cache_conn.close()

        print(f"✓ Synchronized {len(changed_documents)} documents")
        print(f"✓ Invalidated {deleted} cached answers")
    else:
        print("✓ No changes detected. Cache retained.")




#────────────────────────────────────────────────────────────────────────────────────────
# Dense vs BM25 Top-1 comparison (after Qdrant upsert)
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────

if COMPARISON_CHART_REQD and search_method == "HYBRID_RRF":

    DENSE_COLLECTION = qdrantCollection_dense
    BM25_COLLECTION = qdrantCollection_bm25

    # --------------------------------------------------------
    # 1. Configure output files
    # --------------------------------------------------------

    DATA_DIR = Path("./docs/runs")
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    comparison_txt_file = (
        DATA_DIR / "08_dense_vs_bm25_comparison.txt"
    )

    comparison_json_file = (
        DATA_DIR / "08_dense_vs_bm25_comparison.json"
    )

    comparison_output_lines = []

    def log_comparison(message=""):
        message = str(message)
        print(message)
        comparison_output_lines.append(message)

    # --------------------------------------------------------
    # 2. Verify both Qdrant collections
    # --------------------------------------------------------

    for collection in (DENSE_COLLECTION, BM25_COLLECTION):

        if not qdrant.collection_exists(collection):
            raise RuntimeError(
                f"Collection {collection!r} is missing. "
                "Create and index both collections first."
            )

        if not pipeline.is_qdrant_collection_populated(
            qdrant, collection
        ):
            raise RuntimeError(
                f"Collection {collection!r} is empty."
            )

    # --------------------------------------------------------
    # 3. Initialize comparison counters
    # --------------------------------------------------------

    retrieval_comparison_results = []

    counts = {
        "Both Correct": 0,
        "Dense Only": 0,
        "BM25 Only": 0,
        "Both Wrong": 0,
        "Same Source": 0,
        "Different Source": 0,
        "Missing Result": 0
    }

    total_common_chunks = 0
    total_overlap_percentage = 0.0

    # --------------------------------------------------------
    # 4. Print comparison table header
    # --------------------------------------------------------

    log_comparison(
        f"\n══ Dense vs BM25 side-by-side (Top-{top_k}) ══\n"
    )

    log_comparison(
        f"{'ID':<6} "
        f"{'Question':<45} "
        f"{'Dense Hits':<12} "
        f"{'BM25 Hits':<12} "
        f"{'Common':<10} "
        f"{'Overlap':<10} "
        f"{'Verdict':<20}"
    )

    log_comparison("-" * 125)

    # --------------------------------------------------------
    # 5. Compare all Golden Set questions
    # --------------------------------------------------------

    for item in golden_set:

        question_id = item["id"]
        question = item["question"]

        # Dense Top-K retrieval
        dense_hits = pipeline.retrieve_from_qdrant(
            question,
            qdrant,
            DENSE_COLLECTION,
            k=top_k
        )

        # BM25 Top-K retrieval
        bm25_hits = pipeline.retrieve_bm25_from_qdrant(
            question,
            qdrant,
            BM25_COLLECTION,
            k=top_k
        )

        # ----------------------------------------------------
        # 6. Extract Top-K chunk IDs and sources
        # ----------------------------------------------------

        dense_chunk_ids = [
            str(hit["chunk_id"])
            for hit in dense_hits
        ]

        bm25_chunk_ids = [
            str(hit["chunk_id"])
            for hit in bm25_hits
        ]

        dense_sources = [
            str(hit.get("source_id") or "")
            for hit in dense_hits
        ]

        bm25_sources = [
            str(hit.get("source_id") or "")
            for hit in bm25_hits
        ]

        # Find common chunks
        common_chunks = (
            set(dense_chunk_ids) &
            set(bm25_chunk_ids)
        )

        # Find common sources
        common_sources = (
            set(dense_sources) &
            set(bm25_sources)
        )

        common_sources.discard("")

        # Calculate Overlap@K
        overlap_percentage = (
            len(common_chunks) / top_k * 100
        )

        total_common_chunks += len(common_chunks)
        total_overlap_percentage += overlap_percentage

        # ----------------------------------------------------
        # 7. Compare retrieved sources against Golden Set
        # ----------------------------------------------------

        expected_id = item.get("expected_source_id")

        if expected_id is None:

            dense_ok = None
            bm25_ok = None

            if not dense_hits or not bm25_hits:
                verdict = "Missing Result"

            elif common_sources:
                verdict = "Same Source"

            else:
                verdict = "Different Source"

        else:

            expected_source = str(expected_id)

            dense_ok = expected_source in dense_sources
            bm25_ok = expected_source in bm25_sources

            if dense_ok and bm25_ok:
                verdict = "Both Correct"

            elif dense_ok:
                verdict = "Dense Only"

            elif bm25_ok:
                verdict = "BM25 Only"

            else:
                verdict = "Both Wrong"

        counts[verdict] += 1

        # ----------------------------------------------------
        # 8. Store detailed Top-K results
        # ----------------------------------------------------

        retrieval_comparison_results.append({

            "question_id": question_id,
            "question": question,
            "expected_source_id": expected_id,

            "dense_top_k": [
                {
                    "rank": rank,
                    "chunk_id": hit.get("chunk_id"),
                    "source_id": hit.get("source_id"),
                    "score": hit.get("score"),
                    "text": hit.get("text", "")
                }
                for rank, hit in enumerate(dense_hits, 1)
            ],

            "bm25_top_k": [
                {
                    "rank": rank,
                    "chunk_id": hit.get("chunk_id"),
                    "source_id": hit.get("source_id"),
                    "score": hit.get("score"),
                    "text": hit.get("text", "")
                }
                for rank, hit in enumerate(bm25_hits, 1)
            ],

            "dense_chunk_ids": dense_chunk_ids,
            "bm25_chunk_ids": bm25_chunk_ids,

            "dense_sources": dense_sources,
            "bm25_sources": bm25_sources,

            "common_chunks": sorted(common_chunks),
            "common_sources": sorted(common_sources),

            "common_chunk_count": len(common_chunks),
            "overlap_percentage": overlap_percentage,

            "dense_correct": dense_ok,
            "bm25_correct": bm25_ok,

            "verdict": verdict
        })

        # ----------------------------------------------------
        # 9. Print and collect comparison row
        # ----------------------------------------------------

        log_comparison(
            f"{str(question_id):<6} "
            f"{question[:45]:<45} "
            f"{len(dense_hits):<12} "
            f"{len(bm25_hits):<12} "
            f"{len(common_chunks):<10} "
            f"{overlap_percentage:<9.2f}% "
            f"{verdict:<20}"
        )

        # ----------------------------------------------------
        # 10. Print detailed Top-K results
        # ----------------------------------------------------

        log_comparison(
            f"\nQuestion {question_id}: {question}"
        )

        log_comparison(f"\nDense Top-{top_k}:")

        for rank, hit in enumerate(dense_hits, 1):

            log_comparison(
                f"  {rank}. "
                f"Chunk ID: {hit['chunk_id']} | "
                f"Source: {hit.get('source_id', '')} | "
                f"Score: {hit['score']:.4f}"
            )

        log_comparison(f"\nBM25 Top-{top_k}:")

        for rank, hit in enumerate(bm25_hits, 1):

            log_comparison(
                f"  {rank}. "
                f"Chunk ID: {hit['chunk_id']} | "
                f"Source: {hit.get('source_id', '')} | "
                f"Score: {hit['score']:.4f}"
            )

        log_comparison(
            f"\nCommon Chunk IDs: {sorted(common_chunks)}"
        )

        log_comparison(
            f"Common Sources: {sorted(common_sources)}"
        )

        log_comparison(
            f"Overlap@{top_k}: {overlap_percentage:.2f}%"
        )

        log_comparison("-" * 125)

    # --------------------------------------------------------
    # 11. Calculate summary
    # --------------------------------------------------------

    total = len(golden_set)

    evaluated = (
        counts["Both Correct"]
        + counts["Dense Only"]
        + counts["BM25 Only"]
        + counts["Both Wrong"]
    )

    log_comparison("\n" + "=" * 65)
    log_comparison(
        f"DENSE vs BM25 TOP-{top_k} COMPARISON SUMMARY"
    )
    log_comparison("=" * 65)

    log_comparison(f"Total Questions     : {total}")
    log_comparison(f"Evaluated Questions : {evaluated}")

    for verdict, count in counts.items():
        log_comparison(f"{verdict:<20}: {count}")

    # --------------------------------------------------------
    # 12. Calculate average overlap
    # --------------------------------------------------------

    if total > 0:

        average_common_chunks = (
            total_common_chunks / total
        )

        average_overlap = (
            total_overlap_percentage / total
        )

        log_comparison(
            f"\nAverage Common Chunks : "
            f"{average_common_chunks:.2f}"
        )

        log_comparison(
            f"Average Overlap@{top_k}   : "
            f"{average_overlap:.2f}%"
        )

    # --------------------------------------------------------
    # 13. Calculate source-level Hit@K
    # --------------------------------------------------------

    if evaluated > 0:

        dense_correct = (
            counts["Both Correct"]
            + counts["Dense Only"]
        )

        bm25_correct = (
            counts["Both Correct"]
            + counts["BM25 Only"]
        )

        dense_accuracy = (
            dense_correct / evaluated * 100
        )

        bm25_accuracy = (
            bm25_correct / evaluated * 100
        )

        log_comparison(
            f"\nDense Source Hit@{top_k}: "
            f"{dense_accuracy:.2f}%"
        )

        log_comparison(
            f"BM25 Source Hit@{top_k} : "
            f"{bm25_accuracy:.2f}%"
        )

    else:

        log_comparison(
            "Source Hit@K unavailable: "
            "Golden Set lacks expected_source_id."
        )

    # --------------------------------------------------------
    # 14. Save comparison to TXT
    # --------------------------------------------------------

    with open(
        comparison_txt_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(comparison_output_lines) + "\n"
        )

    # --------------------------------------------------------
    # 15. Save detailed comparison to JSON
    # --------------------------------------------------------

    with open(
        comparison_json_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            retrieval_comparison_results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\nComparison completed successfully.")

    print(
        f"TXT output saved to: {comparison_txt_file}"
    )

    print(
        f"JSON output saved to: {comparison_json_file}"
    )



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Ask RAG now
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Initialize cache once
cache = pipeline.cache_init(CACHE_DB) if isCacheDBEnabled else None
pipeline.cache = cache
print(f"✓ Cache ready: {CACHE_DB}")


comparison_results = []
ragas_rows = []

for k in [top_k]:  

    if search_method == "HYBRID_RRF":
        OUTPUT_FILE = DATA_DIR / f"02_rag_answers_top{k}chunks_with_{SEARCH_METHOD}_results_qdrant_{embed_model}.txt"
    else:
        if pipeline.is_qdrant_collection_populated(qdrant, qdrantCollection):
            OUTPUT_FILE = DATA_DIR / f"02_rag_answers_top{k}chunks_with_{SEARCH_METHOD}_results_qdrant_{embed_model}.txt"
        else:
            OUTPUT_FILE = DATA_DIR / f"02_rag_answers_top{k}chunks_{SEARCH_METHOD}_results_{embed_model}.txt"
        


    total_cost = 0.0
    total_latency = 0.0
    total_tokens_in = 0
    total_tokens_out = 0
    total_cache_hits = 0
    total_cache_misses = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

        total_questions = len(golden_set)

        for index, item in enumerate(golden_set, start=1):

            question_id = item["id"]
            question = item["question"]

            print(
                f"Processing RAG question "
                f"{index}/{total_questions} with K={k}"
            )

            start_time = time.perf_counter()

            if isCacheDBEnabled:
                query_vector = pipeline.embed_batch([question], model=embed_model)[0]
                answer, meta = pipeline.cached_rag_answer(question,query_vector)
            else:
                answer, meta = None, {"cache_hit": False, "matched": None}


            if meta["cache_hit"] and meta["matched"] == question:
                total_cache_hits += 1
                print(f"✓ CACHE HIT: {question}")

                result = {
                    "question": question,
                    "answer": answer,
                    "sources": [],
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "retrieved": [],
                    "cost_usd": 0.0,
                    "latency_s": time.perf_counter() - start_time
                }
            
            else:
                total_cache_misses += 1
                print(f"✗ CACHE MISS: {question}")


                result = pipeline.ask_rag(
                    question,
                    all_chunks,
                    k=k,
                    qdrant= qdrant,
                    collection_name= qdrantCollection,
                    search_method=search_method,
                    bm25=bm25 if search_method == "BM25" else None,
                    use_reranker=use_reranker
                )

                # ------------------------------------------------
                # 4. Save generated answer to SQLite
                # ------------------------------------------------
                if isCacheDBEnabled:
                    pipeline.cache_put(
                        cache,
                        question,
                        query_vector,
                        result["answer"]
                    )
                    print(f"✓ Answer stored in cache: {question}")


                # ----------------------------------------------------
                # 5. Calculate totals
                # ----------------------------------------------------
                result["latency_s"] = time.perf_counter() - start_time
                total_latency += result["latency_s"]

                total_cost += result["cost_usd"]
                total_tokens_in += result["tokens_in"]
                total_tokens_out += result["tokens_out"]

                # ----------------------------------------------------
                # 6. Console output
                # ----------------------------------------------------
                print(f"\nK: {k}")
                print(f"\nQ: {result['question']}")
                print(f"A: {result['answer']}")
                print(f"Cache Hit: {meta['cache_hit']}")

                print(
                    f"Sources retrieved: "
                    f"{result['sources']}"
                )

                print(
                    f"Latency: "
                    f"{result['latency_s']:.3f} seconds"
                )

                print(
                    f"Cost USD: "
                    f"${result['cost_usd']:.8f}"
                )

                # ----------------------------------------------------
                # 7. File output
                # ----------------------------------------------------
                f.write("=" * 120 + "\n")

                f.write(f"K: {k}\n")

                f.write(f"Golden Set ID: {question_id}\n")

                f.write(f"question: {result['question']}\n")

                f.write(f"answer: {result['answer']}\n")

                f.write(f"cache_hit: {meta['cache_hit']}\n")

                f.write(f"sources: {result['sources']}\n")

                f.write(f"tokens_in: {result['tokens_in']}\n")

                f.write(f"tokens_out: {result['tokens_out']}\n")

                f.write(f"retrieved: {result['retrieved']}\n")

                f.write(f"cost_usd: {result['cost_usd']:.8f}\n")

                f.write(f"latency_s: {result['latency_s']:.3f}\n")

                f.write("=" * 120 + "\n\n")

            # ------------------------------------------------------------
            # Collect actual RAGAS evaluation data
            # ------------------------------------------------------------
            if ENABLE_RAGAS_EVALUATION:

                ground_truth = item.get("ideal_answer")

                if not ground_truth:
                    raise ValueError(f"Missing ideal_answer for question {question_id}")

                contexts = [
                    hit["text"]
                    for hit in result.get("retrieved", [])
                    if isinstance(hit, dict)
                    and isinstance(hit.get("text"), str)
                    and hit["text"].strip()
                ]

                if not contexts:
                    raise RuntimeError(f"No retrieved contexts for question {question_id}")

                ragas_rows.append({
                    "id": str(question_id),
                    "question": question,
                    "answer": str(result["answer"]),
                    "contexts": contexts,
                    "ground_truth": str(ground_truth),
                })      



    # ------------------------------------------------------------
    # 8. Calculate average latency for this K
    # ------------------------------------------------------------
    average_latency = (
        total_latency / total_questions
        if total_questions > 0 else 0.0
    )

    # ------------------------------------------------------------
    # 9. Store comparison result
    # ------------------------------------------------------------
    comparison_results.append({
        "k": k,
        "total_latency": total_latency,
        "average_latency": average_latency,
        "tokens_in": total_tokens_in,
        "tokens_out": total_tokens_out,
        "cost_usd": total_cost,
        "cache_hits": total_cache_hits,
        "cache_misses": total_cache_misses
    })

    # ------------------------------------------------------------
    # 10. Final summary
    # ------------------------------------------------------------
    print(f"\nRAG processing completed for K={k}.")
    print(f"Total Questions: {total_questions}")
    print(f"Cache Hits: {total_cache_hits}")
    print(f"Cache Misses: {total_cache_misses}")
    print(f"Total Latency for K={k}: {total_latency:.3f} seconds")
    print(f"Average Latency for K={k}: {average_latency:.3f} seconds")
    print(f"Total Tokens In for K={k}: {total_tokens_in}")
    print(f"Total Tokens Out for K={k}: {total_tokens_out}")
    print(f"Total Cost USD for K={k}: ${total_cost:.8f}")
    print(f"Results saved to: {OUTPUT_FILE}")


# ============================================================
# RAGAS Evaluation - Four Metrics
# ============================================================

if ENABLE_RAGAS_EVALUATION:

    if not ragas_rows:
        raise RuntimeError("No RAGAS evaluation rows collected.")

    print("\n========== RAGAS EVALUATION STARTED ==========")
    print(f"Total Questions: {len(ragas_rows)}")

    dataset = Dataset.from_list([
        {
            "question": row["question"],
            "answer": row["answer"],
            "contexts": row["contexts"],
            "ground_truth": row["ground_truth"],
        }
        for row in ragas_rows
    ])

    ragas_result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        raise_exceptions=True,
    )

    score_df = ragas_result.to_pandas()

    score_df.insert(
        0,
        "id",
        [row["id"] for row in ragas_rows]
    )

    metric_names = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]

    output_dir = Path(RAGAS_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "11_ragas_live.csv"
    json_path = output_dir / "11_ragas_live.json"

    score_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    summary = {
        metric: float(score_df[metric].mean())
        for metric in metric_names
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": "live",
                "total_questions": len(ragas_rows),
                "summary": summary,
                "questions": json.loads(
                    score_df.to_json(orient="records")
                ),
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n========== RAGAS RESULTS ==========")

    for metric, score in summary.items():
        print(f"{metric:<20}: {score:.4f}")

    print(f"\nCSV saved: {csv_path}")
    print(f"JSON saved: {json_path}")
    print("========== RAGAS EVALUATION COMPLETED ==========")



if cache is not None:
    cache.close()
