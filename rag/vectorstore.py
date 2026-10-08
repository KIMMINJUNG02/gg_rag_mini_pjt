"""Qdrant collection 연결 및 문서 인덱싱."""

from collections.abc import Sequence
from uuid import NAMESPACE_URL, uuid5

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from common.ai_model import get_embedding_model
from common.qdrant import get_qdrant_client


COLLECTION_NAME = "ai_basic_law_project"


def _point_id(document: Document) -> str:
    """재인덱싱 시 같은 청크를 덮어쓰는 안정적인 UUID를 만든다."""
    source = document.metadata.get("source", "")
    chunk_id = document.metadata.get("chunk_id", "")
    return str(uuid5(NAMESPACE_URL, f"{source}#{chunk_id}"))


def get_vector_store(
    *,
    collection_name: str = COLLECTION_NAME,
    client: QdrantClient | None = None,
    embedding: Embeddings | None = None,
) -> QdrantVectorStore:
    """common 설정의 Qdrant client로 기존 collection에 연결한다."""
    client = client or get_qdrant_client()
    if not client.collection_exists(collection_name):
        raise ValueError(
            f"Qdrant collection이 없습니다: {collection_name}. "
            "index_documents()를 먼저 실행하세요."
        )
    return QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding or get_embedding_model(),
        distance=models.Distance.COSINE,
    )


def index_documents(
    documents: Sequence[Document],
    *,
    collection_name: str = COLLECTION_NAME,
    recreate: bool = False,
    client: QdrantClient | None = None,
    embedding: Embeddings | None = None,
) -> QdrantVectorStore:
    """문서를 cosine-distance Qdrant collection에 upsert한다.

    recreate=True인 경우에만 기존 collection을 삭제하고 다시 만든다.
    기본값에서는 결정적 point ID를 사용하므로 같은 PDF를 재실행해도 중복되지 않는다.
    """
    if not documents:
        raise ValueError("인덱싱할 문서가 없습니다.")

    client = client or get_qdrant_client()
    embedding = embedding or get_embedding_model()
    exists = client.collection_exists(collection_name)

    if recreate and exists:
        client.delete_collection(collection_name)
        exists = False

    if not exists:
        vector_size = len(embedding.embed_query("embedding dimension probe"))
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embedding,
        distance=models.Distance.COSINE,
    )
    vector_store.add_documents(
        documents=list(documents),
        ids=[_point_id(document) for document in documents],
    )
    return vector_store
