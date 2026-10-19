# ============================================================
# MyRAGProject - Configuration
# ============================================================

# OpenAI Models
EMBED_MODEL = "text-embedding-3-large" #"text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Retrieval Configuration
TOP_K = 5
SEARCH_METHOD = "DENSE_SEARCH" #  DENSE_SEARCH # BM25 # HYBRID_RRF

# Qdrant Configuration
QDRANT_TYPE = "NORMAL" #HNSW

# Document Processing
DOCUMENT_TYPE = "txt" #PDF,DOCX,HTML,PDF_WITH_TABLE
CHUNKING_STRATEGY = "Sliding_window_Chunking"

# Evaluation Configuration
COMPARE_SEARCH_METHODS = False # Requires both Qdrant collections to be indexed

# Cross-Encoder Reranking
USE_RERANKER = False
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CACHE_DB = "myrag_query_cache.db"
CACHE_VERSION = "v1"