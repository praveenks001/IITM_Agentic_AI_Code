import os
import numpy as np
import json
from openai import OpenAI
from pathlib import Path
from pipeline import myrag_pipeline as pipeline
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before running this notebook"

client = OpenAI()

EMBED_MODEL = "text-embedding-3-large"  #"text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Load Qdrant 
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
qdrant = pipeline.get_qdrant_client()
print("Qdrant isss:::", qdrant)




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Qdrant - Create collection to store the embedded chunks
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
qdrantCollection = pipeline.create_qdrant_collection(qdrant)




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The corpus = Load the Reference documents
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
documents = pipeline.load_documents();
print(f"Loaded {len(documents)} documents")



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
all_chunks = pipeline.chunk_documents(documents, 500); #chunked_documents = pipeline.chunk_text_documents(documents, 200);
print(f"Total chunks from {len(documents)} documents: {len(all_chunks)}\n")


# Write the result chunk into an output file
DATA_DIR = Path('./docs/runs')
DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = DATA_DIR / "01_chunks_sliding_window.txt"

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



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Embedding the chunks
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────

# check if qdrant collection has already embedding else embed the chunks
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




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Qdrant - Upsert the embedded chunks into Qdrant
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
if pipeline.is_qdrant_collection_populated(qdrant, qdrantCollection):
    print("Qdrant collection already contains embedded chunks.so we can skip")

else:
    pipeline.upsert_embedded_chunks_into_qdrant(all_chunks, vectors,qdrantCollection)




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Embed each questions and find cosine similarity with the embedded document chunks
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
for item in golden_set:

    QUERY = item["question"]

    # 1. Embed this question
    query_vector = pipeline.embed_batch([QUERY])[0]

    if pipeline.is_qdrant_collection_populated(qdrant, qdrantCollection):
        print("Qdrant collection already contains embedded chunks.so retrive from qdrant to give cosine similarity chunks")


    else:
            
        # 2. Compare this question against ALL chunks
        scored = [(pipeline.cosine(query_vector, c["vector"]), c) for c in all_chunks]

        # 3. Sort highest cosine similarity first
        scored.sort(key=lambda pair: pair[0], reverse=True)

        # 4. Print results
        print(f"\nQuestion: {QUERY}")

        # Top 3 results
        for i, (score, chunk) in enumerate(scored[:3], 1):
            print(f"  [{i}] {score:.3f}  {chunk['chunk_id']:<25s} {chunk['text'][:60]}...")

        # # Top 5 results
        # for i, (score, chunk) in enumerate(scored[:5], 1):
        #     print(f"  [{i}] {score:.3f}  {chunk['chunk_id']:<25s} {chunk['text'][:60]}...")

        # # Top 7 results
        # for i, (score, chunk) in enumerate(scored[:7], 1):
        #     print(f"  [{i}] {score:.3f}  {chunk['chunk_id']:<25s} {chunk['text'][:60]}...")

print(f"Embedded each questions and found cosine similarity with the embedded document chunks.")





#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Retrieve top K results
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
for item in golden_set:
    QUERY = item["question"]
    top3 = pipeline.retrieve(QUERY, all_chunks, k=3)

# for item in golden_set:
#     QUERY = item["question"]
#     top3 = pipeline.retrieve(QUERY, all_chunks, k=5)

# for item in golden_set:
#     QUERY = item["question"]
#     top3 = pipeline.retrieve(QUERY, all_chunks, k=7)




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Ask RAG now
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Store the final comparison values for each K
comparison_results = []

for k in [3, 5, 7]:
    OUTPUT_FILE = DATA_DIR / f"07_rag_answers_goldenset_top{k}chunks.txt"

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
            result = pipeline.ask_rag(question,all_chunks,k=k)

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



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Final comparison of K = 3, 5, 7
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────

print("\n")
print("=" * 90)
print("K VALUE COMPARISON")
print("=" * 90)

print(
    f"{'K':>3s} "
    f"{'total latency':>15s} "
    f"{'avg latency':>15s} "
    f"{'tokens_in':>12s} "
    f"{'tokens_out':>12s} "
    f"{'cost_usd':>14s}"
)

print("-" * 90)

for r in comparison_results:

    print(
        f"{r['k']:>3d} "
        f"{r['total_latency']:>15.3f} "
        f"{r['average_latency']:>15.3f} "
        f"{r['tokens_in']:>12d} "
        f"{r['tokens_out']:>12d} "
        f"{r['cost_usd']:>14.8f}"
    )

print("=" * 90)