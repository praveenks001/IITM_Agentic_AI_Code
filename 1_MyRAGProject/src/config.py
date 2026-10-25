# ============================================================
# MyRAGProject - Configuration
# ============================================================

# OpenAI Models
EMBED_MODEL = "text-embedding-3-large" #"text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"

# Retrieval Configuration
TOP_K = 5
SEARCH_METHOD = "HYBRID_RRF" #  DENSE_SEARCH # BM25 # HYBRID_RRF

# Qdrant Configuration
QDRANT_TYPE = "NORMAL" #HNSW

# Document Processing
DOCUMENT_TYPE = "txt" #PDF,DOCX,HTML,PDF_WITH_TABLE
CHUNKING_STRATEGY = "Sliding_window_Chunking"

# Evaluation Configuration
COMPARISON_CHART_REQD = False # Requires if dense / BM25 comparison is required

# Cross-Encoder Reranking
USE_RERANKER = True
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

ENABLE_CACHE_DB = True
CACHE_DB = "myrag_query_cache.db"
CACHE_VERSION = "v1"


# Tombstone / Document Refresh
REFRESH_DOCUMENT = True
REFRESH_DOCUMENT_NAME = "Client agreement(1).txt"