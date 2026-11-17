from pathlib import Path

from app.services.extractors import extract_docx, extract_pdf, extract_pptx
from app.services.ocr_service import run_ocr

SUPPORTED_TYPES = {"pdf", "docx", "pptx"}


def extract_text(file_path: str, file_type: str) -> str:
    """
    Extract text, tables, and image OCR content from PDF, DOCX, or PPTX files.

    Uses high-fidelity native extraction first, layout-aware reading order
    reconstruction, table formatting, and OCR for embedded images or scanned pages.
    """
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError("Document file not found.")

    normalized_type = file_type.lower().lstrip(".")

    if normalized_type not in SUPPORTED_TYPES:
        raise ValueError("Unsupported document type.")

    if normalized_type == "pdf":
        return extract_pdf(path)

    if normalized_type == "docx":
        return extract_docx(path)

    if normalized_type == "pptx":
        return extract_pptx(path)

    raise ValueError("Unsupported document type.")


def _run_ocr(image) -> str:
    """
    Backward-compatible helper for running OCR with layout reconstruction.
    """
    return run_ocr(image)
