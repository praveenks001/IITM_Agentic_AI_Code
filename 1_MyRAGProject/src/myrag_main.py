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




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Top K results
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
pipeline.show_top_k(
    QUERY,
    all_chunked_documents,
    k=3
)



#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Retrieve top K results
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
top3 = pipeline.retrieve(QUERY, all_chunked_documents, k=3)




#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# Ask RAG now
#──────────────────────────────────────────────────────────────────────────────────────────────────────────────
# result = ask_rag(QUERY, all_chunks, k=3)

OUTPUT_FILE = DATA_DIR / "07_rag_answers_goldenset.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    total_questions = len(QUERY)

    for index, question in enumerate(QUERY, start=1):

        print(
            f"Processing RAG question "
            f"{index}/{total_questions}"
        )

        # Run RAG for current question
        result = pipeline.ask_rag(question,all_chunked_documents,k=3)

        # ----------------------------------------------
        # Console output
        # ----------------------------------------------

        print(f"\nQ: {result['question']}")
        print(f"A: {result['answer']}")
        print(
            f"Sources retrieved: "
            f"{result['sources']}"
        )

        # ----------------------------------------------
        # File output
        # ----------------------------------------------

        f.write("=" * 120 + "\n")

        f.write(
            f"Question {index}/{total_questions}\n"
        )

        f.write(
            f"Q: {result['question']}\n\n"
        )

        f.write(
            f"A: {result['answer']}\n\n"
        )

        f.write(
            "Sources retrieved:\n"
        )

        for source in result["sources"]:
            f.write(f"  - {source}\n")

        f.write(
            f"\nPrompt tokens: "
            f"{result['tokens_in']}\n"
        )

        f.write(
            f"Completion tokens: "
            f"{result['tokens_out']}\n"
        )

        f.write("=" * 120 + "\n\n")


print("\nRAG processing completed.")
print(f"Results saved to: {OUTPUT_FILE}")