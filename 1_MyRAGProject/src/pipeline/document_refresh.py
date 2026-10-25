"""Synchronize new and changed TXT documents into existing Dense + BM25 Qdrant collections.

Uses the existing MyRAGProject chunking and dense embedding helpers.
Does not detect source-file removals. Run with a single writer and invalidate
SQLite cache after any successful change, before answering queries.
"""

import hashlib
import time
import uuid
from pathlib import Path

from fastembed import SparseTextEmbedding
from qdrant_client.models import (
    FieldCondition,
    Filter,
    IsEmptyCondition,
    MatchValue,
    PayloadField,
    PayloadSchemaType,
    PointIdsList,
    PointStruct,
    SparseVector,
)

from pipeline import myrag_pipeline as pipeline


CORPUS_DIR = Path("./data/corpus")
DENSE_COLLECTION = "myrag_collection"
BM25_COLLECTION = "myrag_collection_BM25"
COLLECTIONS = (
    (DENSE_COLLECTION, "source"),
    (BM25_COLLECTION, "source_id"),
)


def _ensure_indexes(qdrant, collection, source_field):
    existing = qdrant.get_collection(collection).payload_schema or {}
    for field, schema in (
        (source_field, PayloadSchemaType.KEYWORD),
        ("deleted_at", PayloadSchemaType.FLOAT),
    ):
        if field not in existing:
            qdrant.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=schema,
                wait=True,
            )


def _get_active_points(qdrant, collection, source_field, filename):
    """Return active points belonging to filename, across all scroll pages."""
    source_filter = Filter(
        must=[
            FieldCondition(key=source_field, match=MatchValue(value=filename)),
            IsEmptyCondition(is_empty=PayloadField(key="deleted_at")),
        ]
    )
    points = []
    offset = None
    while True:
        batch, offset = qdrant.scroll(
            collection_name=collection,
            scroll_filter=source_filter,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            return points


def _make_sparse_vectors(texts):
    """Match the existing project's Qdrant/bm25 embedding model."""
    model = SparseTextEmbedding(model_name="Qdrant/bm25")
    return list(model.embed(texts))


def refresh_document(qdrant, filename, embed_model):
    """Insert new or refresh changed TXT document; return True if modified.

    Return False when its active chunks already match the source-file hash.
    A partially failed two-collection update requires inspection/recovery.
    """
    # A filename only, not an arbitrary path outside the corpus directory.
    if Path(filename).name != filename or not filename.endswith(".txt"):
        raise ValueError(f"Expected a TXT filename within the corpus: {filename}")

    file_path = CORPUS_DIR / filename
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    text = file_path.read_text(encoding="utf-8", errors="ignore")
    if not text.strip():
        raise ValueError(f"Document is empty: {filename}")

    current_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    old_points = {}

    for collection, source_field in COLLECTIONS:
        _ensure_indexes(qdrant, collection, source_field)
        old_points[collection] = _get_active_points(
            qdrant, collection, source_field, filename
        )

    dense_count = len(old_points[DENSE_COLLECTION])
    bm25_count = len(old_points[BM25_COLLECTION])

    # A document must either exist in both collections or neither.
    if (dense_count == 0) != (bm25_count == 0):
        raise RuntimeError(
            f"Inconsistent active document state for {filename}: "
            f"Dense={dense_count}, BM25={bm25_count}. "
            "Inspect Qdrant before continuing."
        )

    is_new_document = dense_count == 0

    if not is_new_document:
        unchanged = all(
            point.payload.get("document_hash") == current_hash
            for collection, _ in COLLECTIONS
            for point in old_points[collection]
        )
        if unchanged:
            print(f"✓ Document unchanged: {filename}")
            return False

    print(f"{'New document' if is_new_document else 'Changed document'}: {filename}")

    chunks = pipeline.chunk_documents(
        [{"id": filename, "text": text}], 500, overlap=40
    )
    if not chunks:
        raise RuntimeError(f"No chunks generated for {filename}")

    chunk_texts = [chunk["text"] for chunk in chunks]
    dense_vectors = pipeline.embed_batch(chunk_texts, model=embed_model)
    sparse_vectors = _make_sparse_vectors(chunk_texts)

    if len(dense_vectors) != len(chunks) or len(sparse_vectors) != len(chunks):
        raise RuntimeError(f"Vector count mismatch for {filename}")

    version = uuid.uuid4().hex
    now = time.time()
    dense_points = []
    bm25_points = []

    for chunk, dense_vec, sparse_vec in zip(chunks, dense_vectors, sparse_vectors):
        point_id = str(uuid.uuid4())
        chunk_id = f"{chunk['chunk_id']}@{version}"
        common_payload = {
            "chunk_id": chunk_id,
            "text": chunk["text"],
            "document_hash": current_hash,
            "document_version": version,
            "ingested_at": now,
        }

        dense_points.append(
            PointStruct(
                id=point_id,
                vector=dense_vec,
                payload={**common_payload, "source": filename},
            )
        )
        bm25_points.append(
            PointStruct(
                id=point_id,
                vector={
                    "bm25": SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist(),
                    )
                },
                payload={**common_payload, "source_id": filename},
            )
        )

    new_ids = [point.id for point in dense_points]
    attempted_collections = []
    try:
        for collection, points in (
            (DENSE_COLLECTION, dense_points),
            (BM25_COLLECTION, bm25_points),
        ):
            attempted_collections.append(collection)
            qdrant.upsert(collection_name=collection, points=points, wait=True)
    except Exception:
        # Best-effort rollback of newly inserted points only.
        for collection in attempted_collections:
            try:
                qdrant.delete(
                    collection_name=collection,
                    points_selector=PointIdsList(points=new_ids),
                    wait=True,
                )
            except Exception:
                pass
        raise

    if not is_new_document:
        timestamp = time.time()
        try:
            for collection, _ in COLLECTIONS:
                old_ids = [point.id for point in old_points[collection]]
                qdrant.set_payload(
                    collection_name=collection,
                    payload={"deleted_at": timestamp},
                    points=old_ids,
                    wait=True,
                )
        except Exception as exc:
            raise RuntimeError(
                f"Incomplete tombstone operation for {filename}. "
                "Inspect both Qdrant collections and invalidate the answer "
                "cache before resuming queries."
            ) from exc

    print(f"✓ {'Inserted' if is_new_document else 'Refreshed'}: {filename}")
    print(f"✓ New chunks per collection: {len(chunks)}")
    if not is_new_document:
        print(f"✓ Old dense chunks tombstoned: {dense_count}")
        print(f"✓ Old BM25 chunks tombstoned: {bm25_count}")
    return True


def sync_documents(qdrant, embed_model):
    """Check all TXT files; return names inserted or refreshed successfully."""
    if not CORPUS_DIR.is_dir():
        raise FileNotFoundError(CORPUS_DIR)

    changed_documents = []
    for file_path in sorted(CORPUS_DIR.glob("*.txt")):
        print(f"\nChecking: {file_path.name}")
        if refresh_document(qdrant, file_path.name, embed_model):
            changed_documents.append(file_path.name)
    return changed_documents
