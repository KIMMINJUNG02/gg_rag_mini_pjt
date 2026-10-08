"""Docling HybridChunker 설정과 청크 메타데이터 정규화."""

from collections.abc import Iterable
import re

import tiktoken
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.openai import OpenAITokenizer
from langchain_core.documents import Document


EMBED_MODEL_NAME = "text-embedding-3-small"
MAX_TOKENS = 700
ARTICLE_HEADING_PATTERN = re.compile(r"제\d+조(?:의\d+)?\([^)\n]+\)")
STRUCTURE_HEADING_PATTERN = re.compile(r"^제\d+(?:장|절)\s")


def create_tokenizer(
    model_name: str = EMBED_MODEL_NAME,
    max_tokens: int = MAX_TOKENS,
) -> OpenAITokenizer:
    """임베딩 모델의 토큰 규칙을 사용하는 Docling tokenizer를 만든다."""
    encoding = tiktoken.encoding_for_model(model_name)
    return OpenAITokenizer(tokenizer=encoding, max_tokens=max_tokens)


def create_chunker(
    model_name: str = EMBED_MODEL_NAME,
    max_tokens: int = MAX_TOKENS,
) -> HybridChunker:
    """표 머리글과 문서 구조를 보존하는 HybridChunker를 만든다."""
    return HybridChunker(
        tokenizer=create_tokenizer(model_name=model_name, max_tokens=max_tokens),
        merge_peers=True,
        repeat_table_header=True,
    )


def extract_pages(dl_meta: dict) -> list[int]:
    """Docling provenance에서 청크가 포함된 PDF 페이지 번호를 추출한다."""
    pages: set[int] = set()
    for item in dl_meta.get("doc_items", []):
        for provenance in item.get("prov", []):
            page_no = provenance.get("page_no")
            if page_no is not None:
                pages.add(int(page_no))
    return sorted(pages)


def normalize_document(
    document: Document,
    chunk_id: int,
    *,
    tokenizer: OpenAITokenizer | None = None,
) -> Document:
    """중첩된 Docling metadata를 Qdrant payload에 적합하게 평탄화한다."""
    tokenizer = tokenizer or create_tokenizer()
    dl_meta = document.metadata.get("dl_meta", {})
    pages = extract_pages(dl_meta)
    headings = dl_meta.get("headings") or []
    captions = dl_meta.get("captions") or []

    return Document(
        page_content=document.page_content,
        metadata={
            "source": str(document.metadata.get("source", "")),
            "chunk_id": chunk_id,
            "page": pages[0] if pages else -1,
            "pages": ",".join(map(str, pages)),
            "heading": " > ".join(map(str, headings)),
            "caption": " | ".join(map(str, captions)),
            "num_tokens": tokenizer.count_tokens(document.page_content),
            "parser": "docling",
            "chunker": "HybridChunker",
        },
    )


def normalize_documents(documents: Iterable[Document]) -> list[Document]:
    """문서를 정규화하고 검색용 조문 문맥을 보강한다.

    원본 순서로 chunk_id를 먼저 부여하므로 표지·목차를 제외해도 본문
    chunk_id는 기존 Golden Set과 동일하게 유지된다.
    """
    document_tokenizer = create_tokenizer()
    normalized_documents = [
        normalize_document(document, chunk_id, tokenizer=document_tokenizer)
        for chunk_id, document in enumerate(documents, start=1)
    ]
    searchable_documents: list[Document] = []
    current_article = ""

    for document in normalized_documents:
        if is_front_matter_or_toc(document):
            continue

        text = document.page_content
        article_matches = list(ARTICLE_HEADING_PATTERN.finditer(text))
        explicit_articles = [match.group(0) for match in article_matches]

        should_repeat_article = bool(current_article) and (
            not article_matches
            or has_substantive_text_before(text, article_matches[0].start())
        )
        if should_repeat_article:
            text = f"[관련 조문: {current_article}]\n{text}"

        article_context = []
        if should_repeat_article:
            article_context.append(current_article)
        article_context.extend(explicit_articles)
        article_context = list(dict.fromkeys(article_context))

        metadata = {
            **document.metadata,
            "articles": " | ".join(article_context),
            "num_tokens": document_tokenizer.count_tokens(text),
        }
        searchable_documents.append(
            Document(page_content=text, metadata=metadata)
        )

        if explicit_articles:
            current_article = explicit_articles[-1]

    return searchable_documents


def has_substantive_text_before(text: str, first_article_start: int) -> bool:
    """첫 조문 제목 앞에 이전 조문의 본문이 이어지는지 확인한다."""
    prefix = text[:first_article_start]
    for line in prefix.splitlines():
        cleaned = line.strip().lstrip("-").strip()
        if not cleaned or STRUCTURE_HEADING_PATTERN.match(cleaned):
            continue
        return True
    return False


def is_front_matter_or_toc(document: Document) -> bool:
    """법률 표지와 본문 없는 목차 청크를 검색 대상에서 제외한다."""
    text = document.page_content.strip()
    article_headings = ARTICLE_HEADING_PATTERN.findall(text)

    is_cover = (
        "[시행 " in text
        and "법률 제" in text
        and not article_headings
        and "제1조(" not in text
    )

    # 목차는 여러 조문 제목을 나열하지만 제목 뒤에 같은 줄의 본문이 없다.
    has_article_body = bool(
        re.search(
            r"제\d+조(?:의\d+)?\([^)\n]+\)[ \t]+"
            r"(?!제\d+조(?:의\d+)?\()\S",
            text,
        )
    )
    is_toc = len(article_headings) >= 2 and not has_article_body
    return is_cover or is_toc


# 노트북처럼 설정 객체를 직접 가져다 쓸 수 있도록 기본 인스턴스를 제공한다.
tokenizer = create_tokenizer()
chunker = create_chunker()
