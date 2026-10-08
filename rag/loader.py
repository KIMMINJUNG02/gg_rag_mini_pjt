"""법률 PDF를 Docling 청크 문서로 읽는 loader."""

from pathlib import Path

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType

from rag.chunker import create_chunker


PDF_NAME = "인공지능 발전과 신뢰 기반 조성 등에 관한 기본법(법률)(제20676호)(20260122).pdf"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PDF_PATH = PROJECT_ROOT / "data" / PDF_NAME


def resolve_pdf_path(pdf_path: str | Path | None = None) -> Path:
    """지정 경로 또는 프로젝트 data 폴더에서 PDF를 찾는다."""
    if pdf_path is not None:
        path = Path(pdf_path).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {path}")
        return path.resolve()

    # 요청된 경로를 우선하고, 현재 저장소의 기존 하위 폴더도 지원한다.
    candidates = (
        DEFAULT_PDF_PATH,
        PROJECT_ROOT / "data" / "ai_basic_law" / PDF_NAME,
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()

    paths = "\n".join(f"- {path}" for path in candidates)
    raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다. 확인한 경로:\n{paths}")


def setup_loader(pdf_path: str | Path | None = None) -> DoclingLoader:
    """PDF를 HybridChunker 기반 LangChain Document로 내보내는 loader를 만든다."""
    # 순환 import를 피하면서 converter 설정은 pipeline.py 한 곳에서 관리한다.
    from rag.pipeline import setup_converter

    return DoclingLoader(
        file_path=str(resolve_pdf_path(pdf_path)),
        converter=setup_converter(),
        export_type=ExportType.DOC_CHUNKS,
        chunker=create_chunker(),
    )


def load_documents(pdf_path: str | Path | None = None):
    """PDF를 읽고 Qdrant 저장에 적합한 Document 목록을 반환한다."""
    from rag.chunker import normalize_documents

    return normalize_documents(setup_loader(pdf_path).load())


def set_loader(pdf_path: str | Path | None = None) -> DoclingLoader:
    """기존 호출부 호환용 별칭."""
    return setup_loader(pdf_path)
