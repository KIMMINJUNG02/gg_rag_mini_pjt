import re
import xml.etree.ElementTree as ET
from langchain_core.documents import Document
from .loader import data_load
from common.config import EMBEDDING_MODEL
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------
# 1. XML 파싱
# ---------------------------------------------------------
response = data_load()
root = ET.fromstring(response.content)

def clean(value):
    return (value or "").strip()

def get_text(element, tag):
    return clean(element.findtext(tag))

law_name = get_text(root, ".//법령명_한글")
law_id = get_text(root, ".//법령ID")
effective_date = get_text(root, ".//시행일자")


# ---------------------------------------------------------
# 2. 장·절 판별 패턴
# ---------------------------------------------------------
chapter_pattern = re.compile(r"^제\s*\d+(?:의\d+)?장(?:\s|$)")
section_pattern = re.compile(r"^제\s*\d+(?:의\d+)?절(?:\s|$)")

current_chapter = ""
current_section = ""

raw_chunks = []


# ---------------------------------------------------------
# 3. 조문번호 구성
# 예: 1 + 가지번호 2 -> 제1조의2
# ---------------------------------------------------------
def make_article_label(article):
    no = get_text(article, "조문번호")
    sub_no = get_text(article, "조문가지번호")
    title = get_text(article, "조문제목")

    label = f"제{no}조"

    if sub_no and sub_no.isdigit() and int(sub_no) != 0:
        label += f"의{int(sub_no)}"

    if title:
        label += f"({title})"

    return label


# ---------------------------------------------------------
# 4. 조문내용에서 제목 중복 제거
# ---------------------------------------------------------
def remove_article_heading(text, article_label):
    text = clean(text)

    if text.startswith(article_label):
        return text[len(article_label):].strip()

    return text


# ---------------------------------------------------------
# 5. LangChain Document 생성 함수
# ---------------------------------------------------------
def add_document(
    article,
    content,
    paragraph_no="",
    item_no="",
    subitem_no=""
):
    content = clean(content)

    if not content:
        return

    article_label = make_article_label(article)

    hierarchy = [
        law_name,
        current_chapter,
        current_section,
        article_label,
        content
    ]

    page_content = "\n".join(
        value for value in hierarchy if value
    )

    raw_chunks.append(
        Document(
            page_content=page_content,
            metadata={
                "law_name": law_name,
                "law_id": law_id,
                "chapter": current_chapter,
                "section": current_section,
                "article_no": get_text(article, "조문번호"),
                "article_sub_no": get_text(article, "조문가지번호"),
                "article_title": get_text(article, "조문제목"),
                "paragraph_no": paragraph_no,
                "item_no": item_no,
                "subitem_no": subitem_no,
                "source": "law_api"
            }
        )
    )


# ---------------------------------------------------------
# 6. 조문단위 순서대로 처리
# ---------------------------------------------------------
def set_doc():
    for article in root.findall(".//조문단위"):

        article_type = get_text(article, "조문여부")
        article_content = get_text(article, "조문내용")

        # -----------------------------------------------------
        # 6-1. 장·절 정보 처리
        # -----------------------------------------------------
        if article_type == "전문":

            if chapter_pattern.match(article_content):
                current_chapter = article_content
                current_section = ""

            elif section_pattern.match(article_content):
                current_section = article_content

            continue

        # 실제 조문만 Document로 변환
        if article_type != "조문":
            continue

        article_label = make_article_label(article)

        # 제목을 제외한 실제 조문 본문
        article_intro = remove_article_heading(
            article_content,
            article_label
        )

        # -----------------------------------------------------
        # 6-2. 항 처리
        # -----------------------------------------------------
        paragraphs = article.findall("./항")

        # 항이 없으면 조 단위 Document 생성
        if not paragraphs:
            add_document(
                article=article,
                content=article_intro or article_content
            )
            continue

        for paragraph in paragraphs:

            paragraph_no = get_text(paragraph, "항번호")
            paragraph_content = get_text(paragraph, "항내용")

            # 항내용이 없는 경우 조문 본문을 상위 문맥으로 사용
            parent_content = "\n".join(
                part for part in [
                    article_intro,
                    paragraph_content
                ]
                if part
            )

            # -------------------------------------------------
            # 6-3. 호 처리
            # -------------------------------------------------
            items = paragraph.findall("./호")

            if not items:
                add_document(
                    article=article,
                    content=parent_content,
                    paragraph_no=paragraph_no
                )
                continue

            for item in items:

                item_no = get_text(item, "호번호")
                item_content = get_text(item, "호내용")

                item_context = "\n".join(
                    part for part in [
                        parent_content,
                        item_content
                    ]
                    if part
                )

                # ---------------------------------------------
                # 6-4. 목 처리
                # ---------------------------------------------
                subitems = item.findall("./목")

                if not subitems:
                    add_document(
                        article=article,
                        content=item_context,
                        paragraph_no=paragraph_no,
                        item_no=item_no
                    )
                    continue

                for subitem in subitems:

                    subitem_no = get_text(subitem, "목번호")
                    subitem_content = get_text(subitem, "목내용")

                    full_content = "\n".join(
                        part for part in [
                            item_context,
                            subitem_content
                        ]
                        if part
                    )

                    add_document(
                        article=article,
                        content=full_content,
                        paragraph_no=paragraph_no,
                        item_no=item_no,
                        subitem_no=subitem_no
                    )

def text_split():
    set_doc()
    tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name=EMBEDDING_MODEL,
        chunk_size=512,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    final_chunks = []

    for doc in raw_chunks:
        token_count = len(tokenizer.encode(doc.page_content))

        if token_count <= 512:
            # 기존 계층 구조 유지
            final_chunks.append(doc)
        else:
            # 긴 Document만 추가 분할
            split_docs = text_splitter.split_documents([doc])
            final_chunks.extend(split_docs)
    
    for i, doc in enumerate(final_chunks):
        doc.metadata["chunk_id"] = f"law_chunk_{i:05d}"
    
    return final_chunks