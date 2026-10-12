import json
import csv
from pathlib import Path

from rag.vectorstore import load_vectorstore
from rag.retriever import mmr_search
import yaml
from rag.reranker import build_reranked_context

# ---------------------------------------------------------
# 0. Setting값 불러오기
# ---------------------------------------------------------
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent.parent/ "rag/config.yaml"

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)
fetch_k = config["mmr"]["fetch_k"]
lambda_mult = config["mmr"]["lambda_mult"]
rerank = config["rerank"]

# ---------------------------------------------------------
# 1. Golden Set 불러오기
# ---------------------------------------------------------
def load_golden_set(file_path=Path(__file__).resolve().parent / "golden_set.jsonl"):
    golden_set = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            sample = json.loads(line)

            question = sample["question"]
            relevant_ids = sample["relevant_chunk_ids"]

            if not question or not relevant_ids:
                raise ValueError(
                    f"{line_no}번째 줄에 질문 또는 정답 ID가 없습니다."
                )

            if not isinstance(relevant_ids, list):
                raise TypeError(
                    f"{line_no}번째 줄: relevant_chunk_ids는 리스트여야 합니다."
                )

            golden_set.append(sample)

    if not golden_set:
        raise ValueError("Golden Set이 비어 있습니다.")

    return golden_set


# ---------------------------------------------------------
# 2. 평가 지표 계산
# ---------------------------------------------------------
def calculate_metrics(retrieved_ids, relevant_ids, k):
    retrieved = retrieved_ids[:k]
    relevant = set(relevant_ids)

    # 검색된 정답 ID
    correct_ids = set(retrieved) & relevant

    # Recall@k
    recall = len(correct_ids) / len(relevant)

    # Hit@k
    hit = int(len(correct_ids) > 0)

    # Reciprocal Rank@k
    rr = 0.0

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            rr = 1.0 / rank
            break

    return {
        f"Recall@{k}": recall,
        f"Hit@{k}": hit,
        f"RR@{k}": rr
    }


# ---------------------------------------------------------
# 3. Retriever Evaluation
# ---------------------------------------------------------
def evaluate_retriever(
    golden_path=Path(__file__).resolve().parent / "golden_set.jsonl",
    k_values=(1, 3, 5),
    fetch_k=fetch_k,
    lambda_mult=lambda_mult
):

    golden_set = load_golden_set(golden_path)
    vector_store = load_vectorstore()

    max_k = max(k_values)

    # 평가에 사용하는 청크 ID의 실제 존재 여부 확인
    stored_ids = set()
    offset = None
    while True:
        points, offset = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            offset=offset,
            limit=256,
            with_payload=[vector_store.metadata_payload_key],
            with_vectors=False,
        )
        for point in points:
            metadata = (point.payload or {}).get(vector_store.metadata_payload_key) or {}
            if "chunk_id" in metadata:
                stored_ids.add(metadata["chunk_id"])
        if offset is None:
            break

    for sample in golden_set:
        missing_ids = (
            set(sample["relevant_chunk_ids"]) - stored_ids
        )

        if missing_ids:
            raise ValueError(
                f"DB에 존재하지 않는 정답 ID: {missing_ids}\n"
                f"질문: {sample['question']}"
            )

    details = []

    # 질문별 검색 및 평가
    for i, sample in enumerate(golden_set, start=1):

        question = sample["question"]
        relevant_ids = sample["relevant_chunk_ids"]

        # MMR 검색
        if rerank:
            retrieved_docs = build_reranked_context(
                original_question=question,
                k=max_k,
                fetch_k=max(fetch_k, max_k),
                lambda_mult=lambda_mult
            )
        else:
            retrieved_docs = mmr_search(
                question=question,
                k=max_k,
                fetch_k=max(fetch_k, max_k),
                lambda_mult=lambda_mult
            )

        retrieved_ids = [
            doc.metadata["chunk_id"]
            for doc in retrieved_docs
        ]

        result = {
            "question": question,
            "relevant_chunk_ids": relevant_ids,
            "retrieved_chunk_ids": retrieved_ids
        }

        # 각 k별 지표 계산
        for k in k_values:
            result.update(
                calculate_metrics(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_ids,
                    k=k
                )
            )

        details.append(result)

        print(f"[{i}/{len(golden_set)}] 평가 완료")

    # -----------------------------------------------------
    # 4. 전체 평균 계산
    # -----------------------------------------------------
    summary = {
        "num_questions": len(golden_set),
        "fetch_k": fetch_k,
        "lambda_mult": lambda_mult
    }

    for k in k_values:
        summary[f"Recall@{k}"] = sum(
            row[f"Recall@{k}"] for row in details
        ) / len(details)

        summary[f"Hit@{k}"] = sum(
            row[f"Hit@{k}"] for row in details
        ) / len(details)

        summary[f"MRR@{k}"] = sum(
            row[f"RR@{k}"] for row in details
        ) / len(details)

    return summary, details


# ---------------------------------------------------------
# 5. 결과 저장
# ---------------------------------------------------------
def save_results(summary, details, output_dir="evaluation_results"):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 전체 평균 결과
    with open(
        output_dir / "summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 질문별 상세 결과
    with open(
        output_dir / "details.jsonl",
        "w",
        encoding="utf-8"
    ) as f:
        for row in details:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # CSV 저장
    csv_rows = []

    for row in details:
        csv_rows.append({
            key: json.dumps(value, ensure_ascii=False)
            if isinstance(value, list) else value
            for key, value in row.items()
        })

    with open(
        output_dir / "details.csv",
        "w",
        encoding="utf-8-sig",
        newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=csv_rows[0].keys()
        )
        writer.writeheader()
        writer.writerows(csv_rows)


# ---------------------------------------------------------
# 6. 실행
# ---------------------------------------------------------
if __name__ == "__main__":

    summary, details = evaluate_retriever(
        k_values=(1, 3, 5),
        fetch_k=20,
        lambda_mult=0.5
    )

    print("\n===== Retriever Evaluation =====")

    for metric, value in summary.items():
        if metric.startswith(("Recall@", "Hit@", "MRR@")):
            print(f"{metric}: {value:.4f}")

    save_results(summary, details)

    print("\n평가 결과 저장 완료")
