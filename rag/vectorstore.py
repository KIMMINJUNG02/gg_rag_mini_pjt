from common.ai_model import get_embedding_model
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.models import Distance
from langchain_chroma import Chroma
from common.config import QDRANT_URL
from pathlib import Path

COLLECTION_NAME = "law_doc_chunks"

# Qdrant Vector Store 저장
def set_vectorstore():
    from rag.chunker import text_split
    embeddings = get_embedding_model()

    chunks = text_split()

    client = QdrantClient(url=QDRANT_URL)

    # Chunking 설정을 수정한 뒤 다시 실습할 때는 True
    # 기존 컬렉션을 유지하려면 False
    RESET_DB = False

    if RESET_DB and client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        print(f"기존 컬렉션 삭제: {COLLECTION_NAME}")

    vector_store = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        distance=Distance.COSINE,
        retrieval_mode=RetrievalMode.DENSE,
    )

    print(
        f"저장된 Chunk 수: "
        f"{client.count(COLLECTION_NAME, exact=True).count}"
    )

    return vector_store

# 기존 Qdrant Vector Store 연결
def load_vectorstore():
    embeddings = get_embedding_model()

    client = QdrantClient(url=QDRANT_URL)

    if not client.collection_exists(COLLECTION_NAME):
        # raise ValueError(
        #     f"'{COLLECTION_NAME}' 컬렉션이 없습니다. "
        #     "set_vectorstore()를 먼저 실행하세요."
        # )
        return set_vectorstore()

    return QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        url=QDRANT_URL,
        retrieval_mode=RetrievalMode.DENSE,
    )