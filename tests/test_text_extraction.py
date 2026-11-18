from pathlib import Path

import pytest
from app.services.text_extraction_service import extract_text
from app.services.text_cleaning_service import clean_extracted_text

FIXTURES_DIR = Path("tests/fixtures")


def test_pdf_extraction_sample():
    pdf_path = FIXTURES_DIR / "sample_mixed.pdf"
    if not pdf_path.is_file():
        pytest.fail(f"Fixture file missing: {pdf_path}")

    text = extract_text(str(pdf_path), "pdf")
    cleaned = clean_extracted_text(text)

    assert "HELLO WORLD!!" in cleaned
    assert "STUDY-" in cleaned
    assert "APPLICATION!!" in cleaned
    # Ensure words were not merged together
    assert "HELLOWORLDIAMHERETOTEST" not in cleaned
    assert "COMPANIONAPPLICATION" not in cleaned


def test_docx_extraction_sample():
    docx_path = FIXTURES_DIR / "sample_mixed.docx"
    if not docx_path.is_file():
        pytest.fail(f"Fixture file missing: {docx_path}")

    text = extract_text(str(docx_path), "docx")
    cleaned = clean_extracted_text(text)

    assert "HELLO WORLD!!" in cleaned
    assert "STUDY-" in cleaned
    assert "APPLICATION!!" in cleaned
    assert "COMPANION APPLICATION!!" in cleaned


def test_docx_simple_sample():
    docx_path = FIXTURES_DIR / "sample_simple.docx"
    if not docx_path.is_file():
        pytest.fail(f"Fixture file missing: {docx_path}")

    text = extract_text(str(docx_path), "docx")
    cleaned = clean_extracted_text(text)
    assert cleaned == "HELLO WORLD!!"


def test_pptx_extraction_sample():
    pptx_path = FIXTURES_DIR / "sample_presentation.pptx"
    if not pptx_path.is_file():
        pytest.fail(f"Fixture file missing: {pptx_path}")

    text = extract_text(str(pptx_path), "pptx")
    cleaned = clean_extracted_text(text)

    assert "Hello" in cleaned
    assert "Machine learning is a branch of Artificial Intelligence" in cleaned
    assert "Machine Learning Basics" in cleaned
    assert "Data" in cleaned
    assert "Preprocessing" in cleaned
    assert "Feature" in cleaned
    assert "Selection" in cleaned
    assert "Model" in cleaned
    assert "Training" in cleaned
    assert "Evaluation" in cleaned

    # Check reading order: Data Preprocessing should precede Feature Selection
    data_idx = cleaned.find("Data")
    pre_idx = cleaned.find("Preprocessing")
    feat_idx = cleaned.find("Feature")
    train_idx = cleaned.find("Training")
    eval_idx = cleaned.find("Evaluation")

    assert data_idx != -1 and pre_idx != -1 and feat_idx != -1
    assert data_idx < pre_idx < feat_idx < train_idx < eval_idx


def test_unsupported_file_type():
    with pytest.raises(ValueError, match="Unsupported document type"):
        extract_text("pyproject.toml", "txt")


def test_file_not_found():
    with pytest.raises(FileNotFoundError, match="Document file not found"):
        extract_text("nonexistent_path_file.pdf", "pdf")


def test_docx_header_table_extraction(tmp_path):
    from docx import Document as DocxDocument
    from docx.shared import Inches

    doc = DocxDocument()
    section = doc.sections[0]
    header = section.header
    table = header.add_table(rows=1, cols=2, width=Inches(6.0))
    table.cell(0, 0).text = "Header Col 1"
    table.cell(0, 1).text = "Header Col 2"
    doc.add_paragraph("Main document body paragraph.")

    file_path = tmp_path / "header_table_test.docx"
    doc.save(str(file_path))

    text = extract_text(str(file_path), "docx")
    assert "Header Col 1" in text
    assert "Header Col 2" in text
    assert "Main document body paragraph." in text

