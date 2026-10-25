
import hashlib
import time
import uuid
from pathlib import Path

from fastembed import SparseTextEmbedding
from qdrant_client.models import (
    PointStruct,
    SparseVector,
    Filter,
    FieldCondition,
    MatchValue,
    IsEmptyCondition,
    PayloadField,
    PayloadSchemaType,
    PointIdsList,
)

from pipeline import myrag_pipeline as pipeline


def _ensure_indexes(qdrant, collection, source_field):
    existing = qdrant.get_collection(collection).payload_schema or {}

    for field, schema in [
        (source_field, PayloadSchemaType.KEYWORD),
        ("deleted_at", PayloadSchemaType.FLOAT),
    ]:
        if field not in existing:
            qdrant.create_payload_index(
                collection_name=collection,
                field_name=field,
                field_schema=schema,
                wait=True,
            )


def _get_active_points(qdrant, collection, source_field, filename):
    """Find all active points belonging to one document."""
    points = []
    offset = None

    source_filter = Filter(
        must=[
            FieldCondition(
                key=source_field,
                match=MatchValue(value=filename),
            ),
            IsEmptyCondition(
                is_empty=PayloadField(key="deleted_at")
            ),
        ]
    )

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
            break

    return points


def refresh_document(qdrant, filename, embed_model):
    """
    Refresh one updated TXT document in Dense and BM25 Qdrant.

    Returns True if refreshed, False if unchanged.
    """
    file_path = Path("./data/corpus") / filename

    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    text = file_path.read_text(
        encoding="utf-8", errors="ignore"
    )

    if not text.strip():
        raise ValueError(f"Document is empty: {filename}")

    current_hash = hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()

    collections = [
        ("myrag_collection", "source"),
        ("myrag_collection_BM25", "source_id"),
    ]

    old_points = {}

    # STEP 1: Find current active document chunks
    for collection, source_field in collections:
        _ensure_indexes(qdrant, collection, source_field)

        old_points[collection] = _get_active_points(
            qdrant, collection, source_field, filename
        )

        if not old_points[collection]:
            raise RuntimeError(
                f"No active points for {filename} in {collection}. "
                "Check the existing source metadata."
            )

    # STEP 2: Check whether document already matches Qdrant
    unchanged = all(
        all(
            p.payload.get("document_hash") == current_hash
            for p in old_points[collection]
        )
        for collection, _ in collections
    )

    if unchanged:
        print(f"✓ Document unchanged: {filename}")
        return False

    print(f"Document changed: {filename}")

    # STEP 3: Reuse existing chunking function
    chunks = pipeline.chunk_documents(
        [{"id": filename, "text": text}],
        size=500,
        overlap=40,
    )

    if not chunks:
        raise RuntimeError("No chunks generated")

    # STEP 4: Reuse existing embedding function
    dense_vectors = pipeline.embed_batch(
        [c["text"] for c in chunks],
        model=embed_model,
    )

    # STEP 5: Generate BM25 sparse vectors
    sparse_model = SparseTextEmbedding(
        model_name="Qdrant/bm25"
    )
    sparse_vectors = list(
        sparse_model.embed([c["text"] for c in chunks])
    )

    if len(dense_vectors) != len(chunks) or \
       len(sparse_vectors) != len(chunks):
        raise RuntimeError("Vector count mismatch")

    version = uuid.uuid4().hex
    now = time.time()

    dense_points = []
    bm25_points = []

    for chunk, dense_vec, sparse_vec in zip(
        chunks, dense_vectors, sparse_vectors
    ):
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
                payload={
                    **common_payload,
                    "source": filename,
                },
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
                payload={
                    **common_payload,
                    "source_id": filename,
                },
            )
        )

    # STEP 6: Insert new versions into both collections
    # If insertion fails, attempt to remove newly inserted points.
    new_ids = [p.id for p in dense_points]
    inserted = []

    try:
        for collection, points in [
            ("myrag_collection", dense_points),
            ("myrag_collection_BM25", bm25_points),
        ]:
            # Include collection before attempting the write:
            # partial uploads can occur even if the call fails.
            inserted.append(collection)
            qdrant.upsert(
                collection_name=collection,
                points=points,
                wait=True,
            )

    except Exception:
        for collection in inserted:
            try:
                qdrant.delete(
                    collection_name=collection,
                    points_selector=PointIdsList(
                        points=new_ids
                    ),
                    wait=True,
                )
            except Exception:
                pass
        raise

    # STEP 7: Tombstone the previous versions
    # Do not delete them physically.
    timestamp = time.time()

    try:
        for collection, _ in collections:
            old_ids = [
                p.id for p in old_points[collection]
            ]

            qdrant.set_payload(
                collection_name=collection,
                payload={"deleted_at": timestamp},
                points=old_ids,
                wait=True,
            )
    except Exception as exc:
        raise RuntimeError(
            "Tombstoning did not complete in both collections. "
            "Inspect Qdrant state before retrying or querying."
        ) from exc

    print(f"✓ Refreshed: {filename}")
    print(f"✓ New chunks per collection: {len(chunks)}")
    print(
        "✓ Old dense chunks tombstoned:",
        len(old_points["myrag_collection"]),
    )
    print(
        "✓ Old BM25 chunks tombstoned:",
        len(old_points["myrag_collection_BM25"]),
    )

    return True
