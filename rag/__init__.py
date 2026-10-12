"""법령 API 데이터 로딩, 청킹, Qdrant 저장 및 MMR 검색 패키지."""


def data_load():
    """법령 API에서 원문을 가져온다."""
    from .loader import data_load as load

    return load()


def text_split():
    """법령 조문을 검색용 청크로 나눈다."""
    from .chunker import text_split as split

    return split()


def set_vectorstore():
    """청크를 Qdrant에 저장한다."""
    from .vectorstore import set_vectorstore as store

    return store()


def load_vectorstore():
    """기존 Qdrant 컬렉션에 연결한다."""
    from .vectorstore import load_vectorstore as load

    return load()


def mmr_search(question, k=4, fetch_k=20, lambda_mult=0.5):
    """MMR 방식으로 관련 청크를 검색한다."""
    from .retriever import mmr_search as search

    return search(question, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult)


__all__ = [
    "data_load",
    "text_split",
    "set_vectorstore",
    "load_vectorstore",
    "mmr_search",
]
