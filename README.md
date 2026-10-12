# UV 가상환경 설정 가이드

`notebook/`과 `app/`이 하나의 Python 가상환경을 공유하도록 구성한다.

## 1. 프로젝트 구조

```text
rag_minipjt/
│
├── .venv/
├── .env
├── .python-version
├── pyproject.toml
├── uv.lock
│
├── notebook/
│   └── test.ipynb
│
└── app/
    └── main.py
```

핵심은 다음과 같다.

- `.venv` : Python 가상환경
- `pyproject.toml` : 사용할 패키지 정의
- `uv.lock` : 실제 설치된 패키지 버전 고정
- `notebook/` : 실험 및 검증 코드
- `app/` : 실제 애플리케이션 코드

---

## 2. 프로젝트 초기화

프로젝트 루트에서 실행한다.

```bash
uv init --bare --python 3.12
```

---

## 3. 가상환경 생성

```bash
uv venv
```

가상환경은 프로젝트 루트의 `.venv/`에 생성된다.

필요한 경우 활성화한다.

```bash
source .venv/bin/activate
```

---

## 4. 필요한 패키지 설치

Qdrant, OpenAI Embedding, Notebook, FastAPI를 한 번에 설치한다.

```bash
uv add   openai   qdrant-client   python-dotenv   jupyter   ipykernel   fastapi   "uvicorn[standard]"
```

패키지를 설치하면 다음 파일이 자동으로 관리된다.

```text
pyproject.toml
uv.lock
```

---

## 5. 환경 복원

다른 PC 또는 수강생 환경에서는 다음 명령만 실행하면 동일한 환경을 만들 수 있다.

```bash
uv sync
```

`uv.lock`에 기록된 버전을 기준으로 패키지가 설치된다.

---

## 6. Notebook 실행

```bash
uv run jupyter lab
```

VS Code에서 `.ipynb`를 사용할 경우 `.venv`의 Python을 Kernel로 선택한다.

필요하면 Kernel을 등록한다.

```bash
uv run python -m ipykernel install --user --name rag-minipjt   --display-name "rag-minipjt"
```

---

## 7. FastAPI 실행

예를 들어 `app/main.py`를 다음과 같이 작성한다.

```python
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {
        "message": "RAG API"
    }
```

실행:

```bash
uv run uvicorn app.main:app --reload
```

브라우저:

```text
http://localhost:8000
```

API 문서:

```text
http://localhost:8000/docs
```

---

## 8. 패키지 추가

새로운 패키지가 필요하면 다음처럼 추가한다.

```bash
uv add 패키지명
```

예:

```bash
uv add langchain langchain-openai langchain-qdrant
```

---

## 9. Git 관리

`.gitignore`에는 다음 항목을 넣는다.

```gitignore
.venv/
.env
__pycache__/
.ipynb_checkpoints/
```

다음 파일은 Git에 포함한다.

```text
pyproject.toml
uv.lock
.python-version
```

---

## 핵심 정리

```text
프로젝트 루트
     │
     ├── pyproject.toml
     ├── uv.lock
     └── .venv
           │
      ┌────┴────┐
      │         │
 notebook/    app/
```

`notebook/`과 `app/`에 각각 별도의 가상환경을 만들지 않고  
**프로젝트 전체에서 하나의 `.venv`를 공유하는 방식이 가장 단순하다.**

---

## 10. common/ 공유 모듈 설정

`notebook/`과 `app/`에서 공통으로 쓰는 설정, Qdrant 클라이언트, 임베딩 로직은 `common/` 패키지로 분리한다.

```text
rag_minipjt/
└── common/
    ├── __init__.py
    ├── config.py      # 환경변수 로딩
    ├── qdrant.py       # Qdrant client 생성
    └── ai_model.py    # LLM / Embedding 모델 생성
```

`notebook/`은 프로젝트 루트가 아닌 `notebook/` 디렉토리에서 커널이 실행되기 때문에, `common`을 그냥 `.venv`에 패키지로 설치해두지 않으면 노트북에서 다음과 같은 오류가 발생한다.

```text
ModuleNotFoundError: No module named 'common'
```

이를 해결하기 위해 프로젝트 자체를 editable 패키지로 설치한다. `pyproject.toml`에 build-system을 추가한다.

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["common", "app"]
```

그리고 `common/__init__.py`, `app/__init__.py`를 빈 파일로 만들어 패키지로 인식시킨다.

```bash
uv sync
```

`uv sync`를 실행하면 `rag-minipjt` 프로젝트 자체가 `.venv`에 editable 모드로 설치되어, 노트북이 어느 위치에서 실행되든 다음처럼 공통 모듈을 바로 사용할 수 있다.

```python
import os
from dotenv import load_dotenv
from common.ai_model import get_llm_model, get_embedding_model
from common.qdrant import get_qdrant_client
```

> `common/`의 코드를 수정한 뒤에는 커널만 재시작하면 되고, `uv sync`를 다시 실행할 필요는 없다 (editable 설치이므로 소스 변경이 즉시 반영된다).

---

## 11. Retriever 평가 결과

[`eval/golden_set.jsonl`](eval/golden_set.jsonl)의 질문 25개를 대상으로 검색 결과를 평가했다. 각 질문에는 정답 청크 ID가 1개씩 지정되어 있다. `fetch_k=20`, `lambda_mult=0.5` 조건에서 상위 1·3·5개 결과의 Recall, Hit, MRR을 계산했다. 아래의 재정렬 적용 여부는 결과 폴더 이름(`rerankO`, `rerankX`)을 따른다.

| 검색 방식 | Recall@1 / Hit@1 | MRR@1 | Recall@3 / Hit@3 | MRR@3 | Recall@5 / Hit@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MMR (`rerankX`) | 0.72 | 0.72 | 0.76 | 0.74 | 0.92 | 0.778 |
| MMR + 재정렬 (`rerankO`) | **0.84** | **0.84** | **0.92** | **0.88** | 0.92 | **0.88** |

재정렬을 적용한 결과, 정답 청크가 1위에 나타난 비율은 72%에서 84%로, 상위 3개 안에 나타난 비율은 76%에서 92%로 높아졌다. 상위 5개 안에 정답이 나타난 비율은 두 방식 모두 92%다. 정답 청크가 질문마다 1개이므로 이 평가에서는 Recall@k와 Hit@k가 같다. MRR@k는 첫 번째 정답 청크의 순위가 높을수록 커진다.

이 수치는 **검색 단계의 성능**을 평가한 결과이며, 챗봇이 생성한 최종 답변의 정확도를 뜻하지는 않는다.

| 방식 | 요약 | 질문별 상세 |
| --- | --- | --- |
| 재정렬 적용 | [summary.json](<evaluation_results(rerankO)/summary.json>) | [details.jsonl](<evaluation_results(rerankO)/details.jsonl>) |
| 재정렬 미적용 | [summary.json](<evaluation_results(rerankX)/summary.json>) | [details.jsonl](<evaluation_results(rerankX)/details.jsonl>) |
