# Qwen 전용 재정렬 모델의 CPU·GPU 실행 장치를 선택한다.
import torch
from sentence_transformers import CrossEncoder
import yaml
from rag.retriever import mmr_search
from langchain_core.documents import Document
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

k = config["mmr"]["k"]
fetch_k = config["mmr"]["k"]
lambda_mult = config["mmr"]["k"]

QWEN_RERANKER_ID = "Qwen/Qwen3-Reranker-0.6B"
rerank_device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Qwen 재정렬 실행 장치:", rerank_device)

qwen_reranker = None

def load_qwen_reranker():
    global qwen_reranker
    if qwen_reranker is None:
        qwen_reranker = CrossEncoder(QWEN_RERANKER_ID)
    return qwen_reranker

def qwen_scores(original_question, candidate_documents):
    """질문·청크 쌍의 Qwen 관련도 점수를 반환한다."""
    model = load_qwen_reranker()
    question_document_pairs = [(original_question, doc.page_content) for doc in candidate_documents]
    scores = qwen_reranker.predict(question_document_pairs, batch_size=1, activation_fn=torch.nn.Sigmoid())
    return [float(score) for score in scores]

def build_reranked_context(original_question, k = k, fetch_k = fetch_k, lambda_mult = lambda_mult):
    """같은 Dense 후보 8개의 순위만 변경하고 최종 4개를 선택한다."""
    candidate_documents = mmr_search(original_question, k=k, fetch_k = fetch_k, lambda_mult = lambda_mult)
    relevance_scores = qwen_scores(original_question, candidate_documents)
    scored_documents = []
    for document, score in zip(candidate_documents, relevance_scores):
        metadata = {**document.metadata, "rerank_score": score}
        scored_documents.append(Document(page_content=document.page_content, metadata=metadata))
    # 안정 정렬: 동점이면 원래 검색 순서를 유지한다.
    # rerank_score가 높은 순으로 정렬
    scored_documents.sort(key=lambda doc: doc.metadata["rerank_score"], reverse=True)
    for document in scored_documents:
        print(document.metadata["chunk_id"], round(document.metadata["rerank_score"], 4))
    return scored_documents