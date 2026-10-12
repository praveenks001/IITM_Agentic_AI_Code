import os
import numpy as np
import json
from openai import OpenAI
from pathlib import Path
from pipeline import myrag_pipeline as pipeline
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.models import HnswConfigDiff
from rank_bm25 import BM25Okapi

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before running this notebook"

client = OpenAI()

EMBED_MODEL = "text-embedding-3-large"  #"text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"
top_k=5
search_method = "HYBRID_RRF"   #  DENSE_SEARCH # BM25 # HYBRID_RRF
qdrant_type = "NORMAL"  #HNSW
document_type = "txt" #PDF,DOCX,HTML,PDF_WITH_TABLE
chunking_strategy = "Sliding_window_Chunking" #
compare_search_methods = True  # Requires both Qdrant collections to be indexed
bm25 = None
use_reranker = True

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
if search_method in ("DENSE_SEARCH", "HYBRID_RRF"):

    if search_method == "HYBRID_RRF":
        qdrantCollection = qdrantCollection_dense
    
    # check if qdrant collection has already embedding else embed the chunks
    if pipeline.is_qdrant_collection_populated(qdrant, qdrantCollection):
        print("Qdrant collection already contains embedded chunks.so we can skip")

    else:
        # Dense search
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

    # check if qdrant collection has already embedding else embed the chunks
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
# Dense vs BM25 Top-1 comparison (after Qdrant upsert)
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────

if compare_search_methods and search_method == "HYBRID_RRF":

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
# Embed each questions and find cosine similarity with the embedded document chunks
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
if search_method in ("DENSE_SEARCH", "HYBRID_RRF"):

    for item in golden_set:

        QUERY = item["question"]

        # 1. Embed this question
        query_vector = pipeline.embed_batch([QUERY])[0]

        if search_method == "HYBRID_RRF":
            qdrantCollection = qdrantCollection_dense

        if pipeline.is_qdrant_collection_populated(qdrant, qdrantCollection):
            print("Qdrant collection already contains embedded chunks.so retrive from qdrant to give cosine similarity chunks")
            
            # --------------------------------------------------------
            # Search Qdrant
            # --------------------------------------------------------

            results = qdrant.query_points(
                collection_name=qdrantCollection,
                query=query_vector,
                limit=top_k,
            ).points


            # --------------------------------------------------------
            # Golden Set information
            # --------------------------------------------------------

            question_id = item["id"]


            # --------------------------------------------------------
            # Print Top 5 retrieved chunks
            # --------------------------------------------------------

            for rank, hit in enumerate(results, 1):
                payload = hit.payload

                result_text = (
                    f"\n[{rank}] "
                    f"score={hit.score:.4f}\n"
                    f"Chunk ID: {payload.get('chunk_id', '')}\n"
                    f"Source: {payload.get('source', '')}\n"
                    f"Text:\n{payload.get('text', '')}\n"
                )

                print(result_text)

        else:
                
            # 2. Compare this question against ALL chunks
            scored = [(pipeline.cosine(query_vector, c["vector"]), c) for c in all_chunks]

            # 3. Sort highest cosine similarity first
            scored.sort(key=lambda pair: pair[0], reverse=True)

            # 4. Print results
            print(f"\nQuestion: {QUERY}")

            # Top 3 results
            # for i, (score, chunk) in enumerate(scored[:3], 1):
            #     print(f"  [{i}] {score:.3f}  {chunk['chunk_id']:<25s} {chunk['text'][:60]}...")

            # # Top 5 results
            for i, (score, chunk) in enumerate(scored[:top_k], 1):
                print(f"  [{i}] {score:.3f}  {chunk['chunk_id']:<25s} {chunk['text'][:60]}...")

            # # Top 7 results
            # for i, (score, chunk) in enumerate(scored[:7], 1):
            #     print(f"  [{i}] {score:.3f}  {chunk['chunk_id']:<25s} {chunk['text'][:60]}...")

    print(f"Embedded each questions and found cosine similarity with the embedded document chunks.")





#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Retrieve top K results
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
if search_method == "DENSE_SEARCH":
        
    if pipeline.is_qdrant_collection_populated(qdrant, qdrantCollection):
            print("Qdrant collection already contains embedded chunks.so retrive from qdrant to give cosine similarity chunks")
    else:
        # for item in golden_set:
        #     QUERY = item["question"]
        #     top3 = pipeline.retrieve(QUERY, all_chunks, k=3)

        for item in golden_set:
            QUERY = item["question"]
            top3 = pipeline.retrieve(QUERY, all_chunks, k=top_k)

        # for item in golden_set:
        #     QUERY = item["question"]
        #     top3 = pipeline.retrieve(QUERY, all_chunks, k=7)


elif search_method == "BM25":
    if pipeline.is_qdrant_collection_populated(qdrant, qdrantCollection):
        for item in golden_set:
            QUERY = item["question"]

            hits = pipeline.retrieve_bm25_from_qdrant(QUERY,qdrant,qdrantCollection,top_k)

    else  :
                
        for item in golden_set:
            QUERY = item["question"]
            hits = pipeline.bm25_search(QUERY,bm25,all_chunks,top_k)
            print(f"\nQuestion: {QUERY}")
        
            for rank, hit in enumerate(hits, 1):
                print(f"\nRank: {rank}")
                print(f"BM25 Score: {hit['score']:.4f}")
                print(f"Chunk ID: {hit['chunk_id']}")
                print(f"Text: {hit['text'][:200]}")





#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Ask RAG now
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Store the final comparison values for each K
comparison_results = []

for k in [top_k]:  #for k in [3, 5, 7]:
    if search_method == "DENSE_SEARCH":
        if pipeline.is_qdrant_collection_populated(qdrant, qdrantCollection):
            print("Qdrant is already created so output file name with qdrant creating")
            OUTPUT_FILE = DATA_DIR / f"07_rag_answers_goldenset_top{k}chunks_with_dense_qdrant_results.txt"
        else:
            OUTPUT_FILE = DATA_DIR / f"07_rag_answers_goldenset_top{k}chunks_densetxt"
    elif search_method == "BM25":
        if pipeline.is_qdrant_collection_populated(qdrant, qdrantCollection):
            print("Qdrant is already created so output file name with qdrant creating")
            OUTPUT_FILE = DATA_DIR / f"07_rag_answers_goldenset_top{k}chunks_with_BM25_qdrant_results.txt"
        else:
            OUTPUT_FILE = DATA_DIR / f"07_rag_answers_goldenset_top{k}chunks_with_BM25_results.txt"
    elif search_method == "HYBRID_RRF":
        OUTPUT_FILE = DATA_DIR / f"07_rag_answers_goldenset_top{k}chunks_with_HYBRID_RRF_results.txt"


    total_cost = 0.0
    total_latency = 0.0
    total_tokens_in = 0
    total_tokens_out = 0

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:


        total_questions = len(golden_set)

        for index, item in enumerate(golden_set, start=1):
            question_id = item["id"]
            question = item["question"]

            print(
                f"Processing RAG question "
                f"{index}/{total_questions} with K={k}"
            )

            # Run RAG for current question
            result = pipeline.ask_rag(
                question,
                all_chunks,
                k=k,
                qdrant=qdrant,
                collection_name=qdrantCollection,
                search_method=search_method,
                bm25=bm25 if search_method == "BM25" else None,
                use_reranker= use_reranker)

            total_cost += result["cost_usd"]
            total_latency += result["latency_s"]
            total_tokens_in += result["tokens_in"]
            total_tokens_out += result["tokens_out"]

            # ----------------------------------------------
            # Console output
            # ----------------------------------------------

            print(f"\nK: {k}")
            print(f"\nQ: {result['question']}")
            print(f"A: {result['answer']}")
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

            # ----------------------------------------------
            # File output
            # ----------------------------------------------

            f.write("=" * 120 + "\n")

            f.write(
                f"K: {k}\n"
            )

            f.write(
                f"Golden Set ID: {question_id}\n"
            )

            f.write(
                f"question: {result['question']}\n"
            )

            f.write(
                f"answer: {result['answer']}\n"
            )

            f.write(
                f"sources: {result['sources']}\n"
            )

            f.write(
                f"tokens_in: {result['tokens_in']}\n"
            )

            f.write(
                f"tokens_out: {result['tokens_out']}\n"
            )

            f.write(
                f"retrieved: {result['retrieved']}\n"
            )

            f.write(
                f"cost_usd: {result['cost_usd']:.8f}\n"
            )

            f.write(
                f"latency_s: {result['latency_s']:.3f}\n"
            )

            f.write("=" * 120 + "\n\n")

    # Calculate average latency for this K
    average_latency = total_latency / total_questions

    # Store comparison result
    comparison_results.append({
        "k": k,
        "total_latency": total_latency,
        "average_latency": average_latency,
        "tokens_in": total_tokens_in,
        "tokens_out": total_tokens_out,
        "cost_usd": total_cost
    })

    print(f"\nRAG processing completed for K={k}.")
    print(f"Total Latency for K={k}: "f"{total_latency:.3f} seconds")
    print(f"Average Latency for K={k}: "f"{average_latency:.3f} seconds")
    print(f"Total Tokens In for K={k}: "f"{total_tokens_in}")
    print(f"Total Tokens Out for K={k}: "f"{total_tokens_out}")
    print(f"Total Cost USD for K={k}: "f"${total_cost:.8f}")
    print(f"Results saved to: {OUTPUT_FILE}")



# #──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# # Final comparison of K = 3, 5, 7
# #──────────────────────────────────────────────────────────────────────────────────────────────────────────────

# print("\n")
# print("=" * 90)
# print("K VALUE COMPARISON")
# print("=" * 90)

# print(
#     f"{'K':>3s} "
#     f"{'total latency':>15s} "
#     f"{'avg latency':>15s} "
#     f"{'tokens_in':>12s} "
#     f"{'tokens_out':>12s} "
#     f"{'cost_usd':>14s}"
# )

# print("-" * 90)

# for r in comparison_results:

#     print(
#         f"{r['k']:>3d} "
#         f"{r['total_latency']:>15.3f} "
#         f"{r['average_latency']:>15.3f} "
#         f"{r['tokens_in']:>12d} "
#         f"{r['tokens_out']:>12d} "
#         f"{r['cost_usd']:>14.8f}"
#     )

# print("=" * 90)