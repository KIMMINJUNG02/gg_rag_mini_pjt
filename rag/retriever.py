"""Qdrant MMR(Maximal Marginal Relevance) 검색기."""

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore

from rag.vectorstore import get_vector_store


DEFAULT_K = 4
DEFAULT_FETCH_K = 12
DEFAULT_LAMBDA_MULT = 0.9


def create_mmr_retriever(
    vector_store: QdrantVectorStore | None = None,
    *,
    k: int = DEFAULT_K,
    fetch_k: int = DEFAULT_FETCH_K,
    lambda_mult: float = DEFAULT_LAMBDA_MULT,
) -> VectorStoreRetriever:
    """관련성과 결과 다양성을 함께 고려하는 MMR retriever를 만든다."""
    if k < 1:
        raise ValueError("k는 1 이상이어야 합니다.")
    if fetch_k < k:
        raise ValueError("fetch_k는 k 이상이어야 합니다.")
    if not 0.0 <= lambda_mult <= 1.0:
        raise ValueError("lambda_mult는 0과 1 사이여야 합니다.")

    vector_store = vector_store or get_vector_store()
    return vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": fetch_k,
            "lambda_mult": lambda_mult,
        },
    )


def retrieve(
    question: str,
    retriever: VectorStoreRetriever | None = None,
) -> list[Document]:
    """질문과 관련된 문서를 MMR 방식으로 검색한다."""
    if not question.strip():
        raise ValueError("질문을 입력하세요.")
    return (retriever or create_mmr_retriever()).invoke(question)
