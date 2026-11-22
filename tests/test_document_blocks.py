"""
Step 2 tests: verify that extract_pdf_with_blocks, extract_docx_with_blocks,
and extract_pptx_with_blocks return correctly structured ExtractionResult objects
with proper block types, ordering, provenance, and heading context.
"""
import io
import struct
import zlib
from pathlib import Path

import pytest
from docx import Document as DocxDocument
from docx.shared import Pt, Inches
from pptx import Presentation
from pptx.util import Inches as PptxInches, Pt as PptxPt

from app.schemas.document_block import DocumentBlock, ExtractionResult
from app.services.extractors import (
    extract_docx_with_blocks,
    extract_pptx_with_blocks,
)
from app.services.text_extraction_service import extract_structured

FIXTURES_DIR = Path("tests/fixtures")


# ---------------------------------------------------------------------------
# Schema contract tests (no actual files required)
# ---------------------------------------------------------------------------

def test_document_block_schema_defaults():
    """DocumentBlock must accept only required fields with safe defaults."""
    b = DocumentBlock(block_index=0, block_type="paragraph", text="Hello")
    assert b.page_number is None
    assert b.slide_number is None
    assert b.heading_path == []
    assert b.metadata == {}


def test_document_block_roundtrip_json():
    """DocumentBlock must serialise and deserialise losslessly."""
    b = DocumentBlock(
        block_index=1,
        block_type="heading",
        text="Introduction",
        page_number=1,
        heading_path=["Introduction"],
        metadata={"style": "Heading 1"},
    )
    restored = DocumentBlock.model_validate(b.model_dump())
    assert restored == b


def test_extraction_result_schema():
    """ExtractionResult must expose text and blocks fields."""
    r = ExtractionResult(text="hello world", blocks=[])
    assert r.text == "hello world"
    assert r.blocks == []


def test_extract_structured_unsupported_type():
    with pytest.raises(ValueError, match="Unsupported document type"):
        extract_structured("pyproject.toml", "txt")


def test_extract_structured_missing_file():
    with pytest.raises(FileNotFoundError, match="Document file not found"):
        extract_structured("does_not_exist.pdf", "pdf")


# ---------------------------------------------------------------------------
# DOCX: block types, ordering, heading path, table detection
# ---------------------------------------------------------------------------

def _make_docx(tmp_path: Path) -> Path:
    """
    Build a deterministic DOCX with:
      - A Heading 1
      - A Heading 2 beneath it
      - A paragraph under the heading hierarchy
      - A list-style paragraph
      - A table (2×2)
    """
    doc = DocxDocument()
    doc.add_heading("Chapter One", level=1)
    doc.add_heading("Section 1.1", level=2)
    doc.add_paragraph("This is a regular paragraph under section 1.1.")
    # Bullet-style paragraph using List Bullet style
    p = doc.add_paragraph("First bullet point", style="List Bullet")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "A"
    table.cell(0, 1).text = "B"
    table.cell(1, 0).text = "C"
    table.cell(1, 1).text = "D"

    path = tmp_path / "structured.docx"
    doc.save(str(path))
    return path


def test_docx_block_count_and_text(tmp_path):
    path = _make_docx(tmp_path)
    result = extract_docx_with_blocks(path)

    assert isinstance(result, ExtractionResult)
    assert len(result.blocks) >= 4  # heading + heading + paragraph + table (minimum)
    # Flattened text must contain all key pieces
    assert "Chapter One" in result.text
    assert "Section 1.1" in result.text
    assert "regular paragraph" in result.text
    assert "|" in result.text  # table as markdown


def test_docx_heading_block_type(tmp_path):
    path = _make_docx(tmp_path)
    result = extract_docx_with_blocks(path)

    heading_blocks = [b for b in result.blocks if b.block_type == "heading"]
    assert len(heading_blocks) >= 2, "Expected at least 2 heading blocks"
    heading_texts = [b.text for b in heading_blocks]
    assert any("Chapter One" in t for t in heading_texts)
    assert any("Section 1.1" in t for t in heading_texts)


def test_docx_heading_path_propagation(tmp_path):
    path = _make_docx(tmp_path)
    result = extract_docx_with_blocks(path)

    # Regular paragraph under Section 1.1 must have both headings in its heading_path
    para_blocks = [
        b for b in result.blocks
        if b.block_type == "paragraph" and "regular paragraph" in b.text
    ]
    assert para_blocks, "Expected a paragraph block under Section 1.1"
    hp = para_blocks[0].heading_path
    assert any("Chapter One" in h for h in hp), f"heading_path missing Chapter One: {hp}"
    assert any("Section 1.1" in h for h in hp), f"heading_path missing Section 1.1: {hp}"


def test_docx_table_block_type(tmp_path):
    path = _make_docx(tmp_path)
    result = extract_docx_with_blocks(path)

    table_blocks = [b for b in result.blocks if b.block_type == "table"]
    assert table_blocks, "Expected at least one table block"
    # Table text should contain cell values
    assert "A" in table_blocks[0].text
    assert "D" in table_blocks[0].text


def test_docx_block_index_sequential(tmp_path):
    path = _make_docx(tmp_path)
    result = extract_docx_with_blocks(path)

    for i, block in enumerate(result.blocks):
        assert block.block_index == i, f"Block at position {i} has index {block.block_index}"


def test_docx_text_matches_blocks(tmp_path):
    """Flattened text should contain all block text."""
    path = _make_docx(tmp_path)
    result = extract_docx_with_blocks(path)

    for block in result.blocks:
        # Every block's text must appear somewhere in the full flattened text
        snippet = block.text[:30].strip()
        if snippet:
            assert snippet in result.text, f"Block text missing from full text: '{snippet}'"


# ---------------------------------------------------------------------------
# PPTX: slide number, block types, heading path from slide title
# ---------------------------------------------------------------------------

def _make_pptx(tmp_path: Path) -> Path:
    """
    Build a deterministic PPTX with:
      Slide 1: Title = "Introduction", body paragraph
      Slide 2: Title = "Data Table", body table (2×2)
    """
    prs = Presentation()
    blank_layout = prs.slide_layouts[1]  # title + content

    # Slide 1
    slide1 = prs.slides.add_slide(blank_layout)
    slide1.shapes.title.text = "Introduction"
    slide1.placeholders[1].text = "Machine learning is powerful."

    # Slide 2
    slide2 = prs.slides.add_slide(blank_layout)
    slide2.shapes.title.text = "Data Table"
    tf = slide2.placeholders[1].text_frame
    tf.text = ""  # clear
    # Add a table shape manually
    rows, cols = 2, 2
    left = PptxInches(1.0)
    top = PptxInches(2.0)
    width = PptxInches(6.0)
    height = PptxInches(1.5)
    tbl_shape = slide2.shapes.add_table(rows, cols, left, top, width, height)
    tbl_shape.table.cell(0, 0).text = "Col A"
    tbl_shape.table.cell(0, 1).text = "Col B"
    tbl_shape.table.cell(1, 0).text = "Val 1"
    tbl_shape.table.cell(1, 1).text = "Val 2"

    path = tmp_path / "structured.pptx"
    prs.save(str(path))
    return path


def test_pptx_slide_numbers_set(tmp_path):
    path = _make_pptx(tmp_path)
    result = extract_pptx_with_blocks(path)

    assert isinstance(result, ExtractionResult)
    slide_nums = {b.slide_number for b in result.blocks if b.slide_number is not None}
    assert 1 in slide_nums
    assert 2 in slide_nums


def test_pptx_slide1_heading_path(tmp_path):
    path = _make_pptx(tmp_path)
    result = extract_pptx_with_blocks(path)

    slide1_blocks = [b for b in result.blocks if b.slide_number == 1]
    assert slide1_blocks, "No blocks found for slide 1"
    # At least one block on slide 1 should have "Introduction" in its heading_path
    has_intro = any("Introduction" in b.heading_path for b in slide1_blocks)
    assert has_intro, f"No block has 'Introduction' in heading_path: {[b.heading_path for b in slide1_blocks]}"


def test_pptx_slide2_has_table_block(tmp_path):
    path = _make_pptx(tmp_path)
    result = extract_pptx_with_blocks(path)

    slide2_blocks = [b for b in result.blocks if b.slide_number == 2]
    table_blocks = [b for b in slide2_blocks if b.block_type == "table"]
    assert table_blocks, "Expected a table block on slide 2"
    assert "Col A" in table_blocks[0].text


def test_pptx_block_index_sequential(tmp_path):
    path = _make_pptx(tmp_path)
    result = extract_pptx_with_blocks(path)

    for i, block in enumerate(result.blocks):
        assert block.block_index == i, f"Block {i} has index {block.block_index}"


def test_pptx_blocks_ordered_by_slide(tmp_path):
    """All slide 1 blocks must appear before all slide 2 blocks in result.blocks."""
    path = _make_pptx(tmp_path)
    result = extract_pptx_with_blocks(path)

    last_slide1_idx = max(
        (i for i, b in enumerate(result.blocks) if b.slide_number == 1), default=-1
    )
    first_slide2_idx = min(
        (i for i, b in enumerate(result.blocks) if b.slide_number == 2), default=999999
    )
    assert last_slide1_idx < first_slide2_idx, "Slide 1 blocks should precede slide 2 blocks"


# ---------------------------------------------------------------------------
# PDF: via fixture (no OCR required — native digital text)
# ---------------------------------------------------------------------------

def test_pdf_structured_fixture():
    """PDF structured extraction using the existing mixed fixture."""
    pdf_path = FIXTURES_DIR / "sample_mixed.pdf"
    if not pdf_path.is_file():
        pytest.skip("sample_mixed.pdf fixture not found")

    result = extract_structured(str(pdf_path), "pdf")

    assert isinstance(result, ExtractionResult)
    assert result.text, "ExtractionResult.text must not be empty"
    assert result.blocks, "ExtractionResult.blocks must not be empty"

    for block in result.blocks:
        assert block.text.strip(), "No block should have empty text"
        assert block.block_type in (
            "heading", "paragraph", "list_item", "table", "code",
            "caption", "ocr_block", "chart",
        )


def test_pdf_blocks_have_page_numbers():
    """Every PDF block must have a 1-based page_number set."""
    pdf_path = FIXTURES_DIR / "sample_mixed.pdf"
    if not pdf_path.is_file():
        pytest.skip("sample_mixed.pdf fixture not found")

    result = extract_structured(str(pdf_path), "pdf")

    for block in result.blocks:
        assert block.page_number is not None, f"Block {block.block_index} missing page_number"
        assert block.page_number >= 1
