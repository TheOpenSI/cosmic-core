### Core modules ###
import os
from argparse import ArgumentParser
from pathlib import Path
from dotenv import load_dotenv
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.models import (
    PointStruct,
    SparseVector
)


### Type hints ###
from typing import Any
from qdrant_client.models import Record


### Internal modules ###
from ...src.services.vector_database import VectorDatabase
from ...utils.log_tool import set_color


# Root of the CoSMIC repository, where `.env` lives.
REPO_ROOT: Path = Path(__file__).resolve().parent.parent.parent

# Payload key holding the chunk text (langchain-qdrant default).
CONTENT_PAYLOAD_KEY: str = "page_content"


def create_hybrid_collection(
    client: QdrantClient,
    source: str,
    target: str
) -> None:
    """
    Create the target collection with the source's dense vector and a BM25 vector.

    Args:
        client (QdrantClient): Qdrant client.
        source (str): existing dense-only collection.
        target (str): new hybrid collection.

    Raises:
        ValueError: the target collection already exists.
    """
    if client.collection_exists(collection_name=target):
        raise ValueError(f"Collection '{target}' already exists; choose another --target.")

    client.create_collection(
        collection_name=target,
        vectors_config=client.get_collection(collection_name=source).config.params.vectors,
        sparse_vectors_config=VectorDatabase.sparse_vectors_config()
    )

    return None


def to_hybrid_point(
    record: Record,
    sparse_embedding: Any
) -> PointStruct:
    """
    Copy a point, keeping its dense vector and payload and adding a BM25 vector.

    Args:
        record (Record): point read from the source collection, with its vector.
        sparse_embedding (Any): FastEmbed sparse embedding of the point's text.

    Returns:
        point (PointStruct): point for the hybrid collection.
    """
    # The unnamed dense vector is stored under the default name "".
    return PointStruct(
        id=record.id,
        vector={
            "": record.vector,
            VectorDatabase.SPARSE_VECTOR_NAME: SparseVector(
                indices=sparse_embedding.indices.tolist(),
                values=sparse_embedding.values.tolist()
            )
        },
        payload=record.payload
    )


def migrate(
    client: QdrantClient,
    source: str,
    target: str,
    batch_size: int
) -> int:
    """
    Copy every point of the source collection into a new hybrid collection.

    Dense vectors are reused, so the embedding model is not loaded; only the
    BM25 vectors are computed from each point's text.

    Args:
        client (QdrantClient): Qdrant client.
        source (str): existing dense-only collection.
        target (str): new hybrid collection.
        batch_size (int): points read and written per request.

    Returns:
        num_points (int): points copied.
    """
    create_hybrid_collection(
        client=client,
        source=source,
        target=target
    )

    bm25: SparseTextEmbedding = SparseTextEmbedding(model_name=VectorDatabase.BM25_MODEL_NAME)
    num_points: int = 0
    offset: Any = None

    while True:
        (records, offset) = client.scroll(
            collection_name=source,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=True
        )

        if records:
            texts: list[str] = [
                str((record.payload or {}).get(CONTENT_PAYLOAD_KEY, ""))
                for record in records
            ]

            client.upsert(
                collection_name=target,
                points=[
                    to_hybrid_point(record=record, sparse_embedding=sparse_embedding)
                    for (record, sparse_embedding) in zip(records, bm25.embed(texts))
                ]
            )

            num_points += len(records)

            print(
                set_color(
                    status="info",
                    information=f"[migrate_to_hybrid] Copied {num_points} points..."
                )
            )

        if offset is None:
            break

    return num_points


def main() -> None:
    """
    Copy a dense-only collection into a new hybrid collection.

    Run from the folder that contains the repository, so the relative imports
    resolve:
        python -m CoSMIC.scripts.migrations.migrate_to_hybrid_collection --target cosmic_collection_hybrid
    Then set QDRANT_COLLECTION to the target and RAG_HYBRID_ENABLED=true. The
    source collection is left unchanged, so switching back is a config change.
    """
    load_dotenv(dotenv_path=REPO_ROOT / ".env")

    parser: ArgumentParser = ArgumentParser(description="Copy a Qdrant collection into a hybrid (dense + BM25) one.")
    parser.add_argument("--source", default=os.getenv("QDRANT_COLLECTION", "cosmic_collection"), help="existing collection.")
    parser.add_argument("--target", required=True, help="new hybrid collection.")
    parser.add_argument("--batch-size", type=int, default=256, help="points per request. Defaults to 256.")
    args = parser.parse_args()

    client: QdrantClient = QdrantClient(url=os.getenv("QDRANT_URL", "http://qdrant:6333"))

    num_points: int = migrate(
        client=client,
        source=args.source,
        target=args.target,
        batch_size=args.batch_size
    )

    print(
        set_color(
            status="success",
            information=(
                f"[migrate_to_hybrid] Copied {num_points} points from '{args.source}' to '{args.target}'. "
                f"Set QDRANT_COLLECTION={args.target} and RAG_HYBRID_ENABLED=true to use it."
            )
        )
    )

    return None


if __name__ == "__main__":
    main()
