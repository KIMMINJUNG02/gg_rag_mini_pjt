from rag.vectorstore import load_vectorstore

def mmr_search(question, k=4, fetch_k=20, lambda_mult=0.5):
    """MMR을 이용해 유사성과 다양성을 고려한 청크를 검색한다."""
    vector_store = load_vectorstore()
    chunks_k = vector_store.max_marginal_relevance_search(
        query=question,
        k=k,
        fetch_k=fetch_k,
        lambda_mult=lambda_mult
    )

    return chunks_k

