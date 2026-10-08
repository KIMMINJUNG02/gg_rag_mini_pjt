"""Golden Test Set 기반 MMR 검색 및 RAG 답변 평가.

실행 예:
    uv run python eval/evaluate.py --retrieval-only
    uv run python eval/evaluate.py --limit 2
    uv run python eval/evaluate.py --output eval/results.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from common.ai_model import get_llm_model
from rag.pipeline import ANSWER_PROMPT, make_context
from rag.retriever import create_mmr_retriever
from rag.vectorstore import COLLECTION_NAME, get_vector_store


DEFAULT_GOLDEN_PATH = Path(__file__).with_name("golden_set.jsonl")

JUDGE_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 RAG 답변 평가자다. 질문, 기준답변, 검색 근거, 생성답변을 비교해 평가하라.

평가 기준:
- correctness: 기준답변과 비교한 사실적 정답성 (1~5)
- relevance: 질문에 직접적이고 불필요한 내용 없이 답한 정도 (1~5)
- faithfulness: 생성답변의 주장이 검색 근거로 뒷받침되는 정도 (1~5)
- hallucination: 검색 근거로 뒷받침되지 않는 실질적인 주장이 하나라도 있으면 true

기준답변은 정답성 판정에 사용하고, 근거 충실성과 hallucination은 반드시 검색 근거만으로 판정하라.
아래 JSON 객체만 출력하고 Markdown 코드 블록은 사용하지 마라.
{{"correctness": 1, "relevance": 1, "faithfulness": 1, "hallucination": false, "reason": "간단한 판정 이유"}}""",
        ),
        (
            "human",
            """[질문]
{question}

[기준답변]
{reference_answer}

[검색 근거]
{context}

[생성답변]
{answer}""",
        ),
    ]
)


@dataclass(frozen=True)
class GoldenCase:
    id: str
    question: str
    reference_answer: str
    relevant_chunk_ids: list[int]
    relevant_articles: list[str]
    relevant_pages: list[int]
    evidence_phrases: list[str]
    article_evidence: dict[str, list[str]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Golden Set 기반 RAG 평가")
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN_PATH)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--fetch-k", type=int, default=12)
    parser.add_argument("--lambda-mult", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="LLM 답변 생성과 Judge 평가를 생략한다.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="문항별 상세 결과를 저장할 JSONL 경로",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=4,
        help="429 rate limit 발생 시 최대 재시도 횟수",
    )
    parser.add_argument(
        "--retry-wait",
        type=float,
        default=20.0,
        help="429 재시도 전 기본 대기 시간(초)",
    )
    return parser.parse_args()


def load_golden_set(path: Path) -> list[GoldenCase]:
    if not path.is_file():
        raise FileNotFoundError(f"Golden Set을 찾을 수 없습니다: {path}")

    cases: list[GoldenCase] = []
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                relevant_articles = list(data["relevant_articles"])
                evidence_phrases = list(data.get("evidence_phrases", []))
                article_evidence = data.get("article_evidence")
                if article_evidence is None:
                    if not evidence_phrases:
                        article_evidence = {
                            article: []
                            for article in relevant_articles
                        }
                    elif len(relevant_articles) == 1:
                        article_evidence = {
                            relevant_articles[0]: evidence_phrases,
                        }
                    else:
                        raise ValueError(
                            "관련 조문이 여러 개면 article_evidence가 필요합니다."
                        )
                case = GoldenCase(
                    id=data["id"],
                    question=data["question"],
                    reference_answer=data["reference_answer"],
                    relevant_chunk_ids=[
                        int(chunk_id)
                        for chunk_id in data["relevant_chunk_ids"]
                    ],
                    relevant_articles=relevant_articles,
                    relevant_pages=list(data.get("relevant_pages", [])),
                    evidence_phrases=evidence_phrases,
                    article_evidence={
                        article: list(phrases)
                        for article, phrases in article_evidence.items()
                    },
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                raise ValueError(
                    f"{path}의 {line_number}번째 줄 형식이 잘못되었습니다."
                ) from exc
            if not case.relevant_articles:
                raise ValueError(f"{case.id}: relevant_articles가 비어 있습니다.")
            if not case.relevant_chunk_ids:
                raise ValueError(f"{case.id}: relevant_chunk_ids가 비어 있습니다.")
            if len(case.relevant_chunk_ids) != len(set(case.relevant_chunk_ids)):
                raise ValueError(f"{case.id}: relevant_chunk_ids가 중복되었습니다.")
            cases.append(case)
    if not cases:
        raise ValueError("Golden Set이 비어 있습니다.")
    return cases


def normalize_text(text: str) -> str:
    """법령 PDF의 줄바꿈과 공백 차이를 제거해 비교한다."""
    return re.sub(r"\s+", "", text)


def extract_article_headings(document_text: str) -> list[str]:
    """청크 본문에 명시적으로 나타난 '제N조(제목)' 조문 번호를 추출한다."""
    articles = re.findall(r"제\d+조(?:의\d+)?(?=\()", document_text)
    return list(dict.fromkeys(articles))


def matched_articles(
    document_text: str,
    article_evidence: dict[str, list[str]],
) -> set[str]:
    normalized = normalize_text(document_text)
    return {
        article
        for article, phrases in article_evidence.items()
        if (
            any(normalize_text(phrase) in normalized for phrase in phrases)
            if phrases
            else normalize_text(article) in normalized
        )
    }


def retrieval_metrics(
    documents,
    relevant_chunk_ids: list[int],
) -> dict[str, Any]:
    """Golden chunk ID를 기준으로 Hit@K, Recall@K, MRR을 계산한다."""
    retrieved_chunk_ids: list[int] = []
    for document in documents:
        chunk_id = document.metadata.get("chunk_id")
        if chunk_id is None:
            raise ValueError("검색 문서 metadata에 chunk_id가 없습니다.")
        retrieved_chunk_ids.append(int(chunk_id))

    relevant = set(relevant_chunk_ids)
    relevance_by_rank = [
        chunk_id in relevant
        for chunk_id in retrieved_chunk_ids
    ]
    found = relevant.intersection(retrieved_chunk_ids)
    first_relevant_rank = next(
        (
            rank
            for rank, is_relevant in enumerate(relevance_by_rank, start=1)
            if is_relevant
        ),
        None,
    )
    return {
        "hit": int(bool(found)),
        "recall": len(found) / len(relevant),
        "reciprocal_rank": (
            1.0 / first_relevant_rank if first_relevant_rank is not None else 0.0
        ),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "found_chunk_ids": sorted(found),
        "relevance_by_rank": relevance_by_rank,
    }


def parse_judge_output(raw_output: str) -> dict[str, Any]:
    """JSON만 요청했지만 코드 펜스가 섞여도 첫 JSON 객체를 복구한다."""
    start = raw_output.find("{")
    end = raw_output.rfind("}")
    if start == -1 or end < start:
        raise ValueError(f"Judge 출력에서 JSON을 찾을 수 없습니다: {raw_output}")

    result = json.loads(raw_output[start : end + 1])
    for name in ("correctness", "relevance", "faithfulness"):
        score = int(result[name])
        if not 1 <= score <= 5:
            raise ValueError(f"{name} 점수가 1~5 범위가 아닙니다: {score}")
        result[name] = score

    hallucination = result["hallucination"]
    if isinstance(hallucination, str):
        hallucination = hallucination.strip().lower() == "true"
    if not isinstance(hallucination, bool):
        raise ValueError("hallucination은 boolean이어야 합니다.")
    result["hallucination"] = hallucination
    result["reason"] = str(result.get("reason", ""))
    return result


def call_with_rate_limit_retry(
    function,
    *,
    max_retries: int,
    retry_wait: float,
):
    """429 응답일 때만 점증적으로 기다린 뒤 호출을 다시 시도한다."""
    for attempt in range(max_retries + 1):
        try:
            return function()
        except Exception as exc:
            message = str(exc).lower()
            is_rate_limit = "429" in message or "rate limit" in message
            if not is_rate_limit or attempt == max_retries:
                raise
            wait_seconds = min(retry_wait * (attempt + 1), 60.0)
            print(
                f"  Rate limit 감지: {wait_seconds:.0f}초 후 "
                f"재시도 ({attempt + 1}/{max_retries})"
            )
            time.sleep(wait_seconds)


def source_record(
    document,
    rank: int,
    matched_article_names: list[str],
    *,
    is_relevant: bool,
) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": document.metadata.get("chunk_id"),
        "is_relevant": is_relevant,
        "pages": document.metadata.get("pages"),
        "heading": document.metadata.get("heading"),
        "retrieved_articles": extract_article_headings(document.page_content),
        "matched_articles": matched_article_names,
        "content": document.page_content,
    }


def print_case_result(result: dict[str, Any], k: int) -> None:
    retrieval = result["retrieval"]
    print(f"\n[{result['id']}] {result['question']}")
    print(
        f"  Hit@{k}: {retrieval['hit']} | "
        f"Recall@{k}: {retrieval['recall']:.3f} | "
        f"RR: {retrieval['reciprocal_rank']:.3f}"
    )
    retrieved_articles = [
        source["retrieved_articles"]
        or [f"chunk-{source['chunk_id']}(조문 제목 없음)"]
        for source in result["sources"]
    ]
    print(f"  정답 청크 IDs: {result['relevant_chunk_ids']}")
    print(f"  검색된 청크 IDs: {retrieval['retrieved_chunk_ids']}")
    print(f"  정답으로 인정된 청크 IDs: {retrieval['found_chunk_ids']}")
    print(f"  관련 조문: {result['relevant_articles']}")
    print(f"  검색된 조문(Top-{len(result['sources'])}): {retrieved_articles}")
    if "answer_evaluation" in result:
        evaluation = result["answer_evaluation"]
        print(
            "  Answer | "
            f"정답성 {evaluation['correctness']}/5 | "
            f"관련성 {evaluation['relevance']}/5 | "
            f"근거 충실성 {evaluation['faithfulness']}/5 | "
            f"Hallucination {evaluation['hallucination']}"
        )
        print(f"  판정 이유: {evaluation['reason']}")
    if "error" in result:
        print(f"  오류: {result['error']}")
    if "answer_skipped" in result:
        print(f"  Answer 평가 생략: {result['answer_skipped']}")


def aggregate_results(
    results: list[dict[str, Any]],
    *,
    k: int,
    retrieval_only: bool,
) -> dict[str, Any]:
    retrieval_results = [result["retrieval"] for result in results]
    summary: dict[str, Any] = {
        "cases": len(results),
        f"hit@{k}": mean(item["hit"] for item in retrieval_results),
        f"recall@{k}": mean(item["recall"] for item in retrieval_results),
        "mrr": mean(item["reciprocal_rank"] for item in retrieval_results),
    }

    answer_results = [
        result["answer_evaluation"]
        for result in results
        if "answer_evaluation" in result
    ]
    if not retrieval_only and answer_results:
        summary.update(
            {
                "answer_cases": len(answer_results),
                "correctness": mean(item["correctness"] for item in answer_results),
                "relevance": mean(item["relevance"] for item in answer_results),
                "faithfulness": mean(item["faithfulness"] for item in answer_results),
                "hallucination_rate": mean(
                    int(item["hallucination"]) for item in answer_results
                ),
            }
        )
    return summary


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    cases = load_golden_set(args.golden)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit은 1 이상이어야 합니다.")
        cases = cases[: args.limit]

    if args.max_retries < 0:
        raise ValueError("--max-retries는 0 이상이어야 합니다.")
    if args.retry_wait < 0:
        raise ValueError("--retry-wait는 0 이상이어야 합니다.")

    vector_store = call_with_rate_limit_retry(
        lambda: get_vector_store(collection_name=args.collection),
        max_retries=args.max_retries,
        retry_wait=args.retry_wait,
    )
    retriever = create_mmr_retriever(
        vector_store,
        k=args.k,
        fetch_k=args.fetch_k,
        lambda_mult=args.lambda_mult,
    )

    answer_chain = None
    judge_chain = None
    if not args.retrieval_only:
        llm = get_llm_model(temperature=0)
        answer_chain = ANSWER_PROMPT | llm | StrOutputParser()
        judge_chain = JUDGE_PROMPT | llm | StrOutputParser()

    results: list[dict[str, Any]] = []
    for case in cases:
        documents = call_with_rate_limit_retry(
            lambda: retriever.invoke(case.question),
            max_retries=args.max_retries,
            retry_wait=args.retry_wait,
        )
        retrieval = retrieval_metrics(
            documents,
            case.relevant_chunk_ids,
        )
        article_matches_by_rank = [
            sorted(
                matched_articles(
                    document.page_content,
                    case.article_evidence,
                )
            )
            for document in documents
        ]
        result: dict[str, Any] = {
            "id": case.id,
            "question": case.question,
            "reference_answer": case.reference_answer,
            "relevant_chunk_ids": case.relevant_chunk_ids,
            "relevant_articles": case.relevant_articles,
            "relevant_pages": case.relevant_pages,
            "evidence_phrases": case.evidence_phrases,
            "article_evidence": case.article_evidence,
            "retrieval": {
                key: value
                for key, value in retrieval.items()
                if key != "relevance_by_rank"
            },
            "sources": [
                source_record(
                    document,
                    rank,
                    article_matches_by_rank[rank - 1],
                    is_relevant=retrieval["relevance_by_rank"][rank - 1],
                )
                for rank, document in enumerate(documents, start=1)
            ],
        }

        if (
            answer_chain is not None
            and judge_chain is not None
            and case.reference_answer
        ):
            try:
                context = make_context(documents)
                answer = call_with_rate_limit_retry(
                    lambda: answer_chain.invoke(
                        {"question": case.question, "context": context}
                    ),
                    max_retries=args.max_retries,
                    retry_wait=args.retry_wait,
                )
                raw_judgement = call_with_rate_limit_retry(
                    lambda: judge_chain.invoke(
                        {
                            "question": case.question,
                            "reference_answer": case.reference_answer,
                            "context": context,
                            "answer": answer,
                        }
                    ),
                    max_retries=args.max_retries,
                    retry_wait=args.retry_wait,
                )
                result["answer"] = answer
                result["answer_evaluation"] = parse_judge_output(raw_judgement)
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
        elif answer_chain is not None and not case.reference_answer:
            result["answer_skipped"] = "reference_answer가 없는 Retrieval 전용 문항"

        results.append(result)
        print_case_result(result, args.k)

    summary = aggregate_results(
        results,
        k=args.k,
        retrieval_only=args.retrieval_only,
    )
    print("\n" + "=" * 72)
    print("종합 평가")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    if args.output is not None:
        write_results(args.output, results)
        print(f"상세 결과 저장: {args.output.resolve()}")


if __name__ == "__main__":
    main()
