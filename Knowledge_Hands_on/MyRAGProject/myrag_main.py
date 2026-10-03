import os
import numpy as np
import myrag_pipeline as pipeline
import json
from openai import OpenAI
from pathlib import Path

assert os.environ.get("OPENAI_API_KEY"), "Set OPENAI_API_KEY before running this notebook"

client = OpenAI()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL  = "gpt-4o-mini"


#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The corpus = Load the Reference documents
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
documents = pipeline.load_documents();
print(f"Loaded {len(documents)} documents")



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The goldenSet Questions = Load it
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
data_folder = Path("./data")
for file in data_folder.rglob("*"):
    if file.is_file():
        print(file)

golden_path = "./data/goldenSet.json"

with open(golden_path, "r", encoding="utf-8") as f:
    golden_set = json.load(f)

print(f"Loaded {len(golden_set)} golden-set questions")



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Using chunk_text method - Chunk every document;
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# The default chunking size was 200 but then the number of chunks were increased to 5250 for the reference documents
# and it supposed to be 2048 or lesser, so increase the chunking size to 500

#1.Sliding window Chunking
all_chunked_documents = pipeline.chunk_text_documents(documents, 500); #chunked_documents = pipeline.chunk_text_documents(documents, 200);
print(f"Total chunks from {len(documents)} documents: {len(all_chunked_documents)}\n")


# Write the result chunk into an output file
DATA_DIR = Path('./runs_output')   # adjust if your folder layout differs
OUTPUT_FILE = DATA_DIR / "01_chunks_sliding_window.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    for result in all_chunked_documents:
        f.write(str(result))
        f.write("\n")

for c in all_chunked_documents[:8]:  # show first 8 to avoid wall of text
    marker = "…" if len(c["text"]) == 200 else " "
    print(f"  {c['chunk_id']:30s} [{len(c['text']):3d} chars] {c['text'][:60]}{marker}")
print(f"  ... and {len(all_chunked_documents) - 8} more chunks")



# #──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# # Using chunk_text method - Chunk every document;
# #──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# #2.Chunk By Sentence Chunking
# all_chunked_documents = pipeline.chunk_by_sentence_documents(documents);
# print(f"Total chunks from {len(documents)} documents: {len(all_chunked_documents)}\n")


# # Write the result chunk into an output file
# OUTPUT_FILE = DATA_DIR / "02_chunks_by_sentence.txt"

# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for result in all_chunked_documents:
#         f.write(str(result))
#         f.write("\n")

# for c in all_chunked_documents[:8]:  # show first 8 to avoid wall of text
#     marker = "…" if len(c["text"]) == 200 else " "
#     print(f"  {c['chunk_id']:30s} [{len(c['text']):3d} chars] {c['text'][:60]}{marker}")
# print(f"  ... and {len(all_chunked_documents) - 8} more chunks")



# #──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# # Using chunk_text method - Chunk every document;
# #──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# #3.Chunk By Paragraph Chunking
# all_chunked_documents = pipeline.chunk_by_paragraph_documents(documents);
# print(f"Total chunks from {len(documents)} documents: {len(all_chunked_documents)}\n")


# # Write the result chunk into an output file
# OUTPUT_FILE = DATA_DIR / "03_chunks_by_paragraph.txt"

# with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
#     for result in all_chunked_documents:
#         f.write(str(result))
#         f.write("\n")

# for c in all_chunked_documents[:8]:  # show first 8 to avoid wall of text
#     marker = "…" if len(c["text"]) == 200 else " "
#     print(f"  {c['chunk_id']:30s} [{len(c['text']):3d} chars] {c['text'][:60]}{marker}")
# print(f"  ... and {len(all_chunked_documents) - 8} more chunks")



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Embedding the chunks
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
chunk_texts = [c["text"] for c in all_chunked_documents]
print(len(chunk_texts))
vectors = pipeline.embed_batch(chunk_texts)

for chunk, vec in zip(all_chunked_documents, vectors):
    chunk["vector"] = vec

print(f"Embedded {len(vectors)} chunks.")
print(f"Each embedding shape: {len(vectors[0])} dimensions")
print(f"\nFirst 8 dims of chunk 0 ({all_chunked_documents[0]['chunk_id']}):")
print(f"  {vectors[0][:8]}")
print(f"\nFirst 8 dims of chunk 1 ({all_chunked_documents[1]['chunk_id']}):")
print(f"  {vectors[1][:8]}")



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Embed golden set questions and Find the Cosine Similarity
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
QUERY = []
for item in golden_set:
    question_id = item["id"]
    question = item["question"]

    QUERY.append(question)

    print(f"ID: {question_id}")
    print(f"Question: {question}")
    print("-" * 80)

    # Write results to output file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE = DATA_DIR / "04_goldenset_questions_extracted.txt"

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in golden_set:
            f.write(item["question"])
            f.write("\n")

print(f"Loaded {len(QUERY)} questions")


query_vector = pipeline.embed_batch(QUERY)

for item, q_vec in zip(golden_set, query_vector):
    question_id = item["id"]
    question = item["question"]

    scored = [(pipeline.cosine(q_vec, c["vector"]), c) for c in all_chunked_documents]
    scored.sort(key=lambda pair: pair[0], reverse=True)


# SCORE_OUTPUT_FILE = DATA_DIR /"05_chunk_score_for_extracted_goldenset_questions.txt"
# with open(SCORE_OUTPUT_FILE, "w", encoding="utf-8") as f:

#     for item, q_vec in zip(golden_set, query_vector):

#         question_id = item["id"]
#         question = item["question"]

#         # Compare this question against all document chunks
#         scored = [
#             (pipeline.cosine(q_vec, c["vector"]), c)
#             for c in all_chunked_documents
#         ]

#         # Highest cosine similarity first
#         scored.sort(
#             key=lambda pair: pair[0],
#             reverse=True
#         )

#         # --------------------------------------------------
#         # Console
#         # --------------------------------------------------

#         print("\n" + "=" * 100)
#         print(f"ID: {question_id}")
#         print(f"Query: {question}")
#         print("=" * 100)

#         # --------------------------------------------------
#         # File
#         # --------------------------------------------------

#         f.write("=" * 100 + "\n")
#         f.write(f"ID: {question_id}\n")
#         f.write(f"Query: {question}\n")
#         f.write("=" * 100 + "\n")

#         # Top 10 chunks
#         for rank, (score, chunk) in enumerate(
#             scored[:10],
#             start=1
#         ):

#             output_line = (
#                 f"[{rank}] "
#                 f"Cosine: {score:.3f} | "
#                 f"Chunk ID: {chunk['chunk_id']} | "
#                 f"Text: {chunk['text'][:200]}"
#             )

#             print(output_line)

#             f.write(output_line + "\n")

#         f.write("\n")


# print(
#     f"\nChunk scores written to: "
#     f"{SCORE_OUTPUT_FILE}"
# )




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Top K results
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
pipeline.show_top_k(QUERY)


# OUTPUT_FILE = DATA_DIR / "06_top_k_query_results.txt"




#     # Append results to output file
#     with open(OUTPUT_FILE, "a", encoding="utf-8") as f:

#         f.write("=" * 100 + "\n")
#         f.write(f"Q: {query}\n")
#         f.write("=" * 100 + "\n")

#         for i, (score, chunk) in enumerate(scored[:k], 1):

#             output = (
#                 f"[{i}] "
#                 f"{score:.3f}  "
#                 f"{chunk['chunk_id']:<25s} "
#                 f"{chunk['text'][:100]}..."
#             )

#             # Print to console
#             print(output)

#             # Write to file
#             f.write(output + "\n")

#         f.write("\n")
