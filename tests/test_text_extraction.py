from pathlib import Path

import pytest
from app.services.text_extraction_service import extract_text
from app.services.text_cleaning_service import clean_extracted_text


UPLOADS_DIR = Path("uploads/documents")


def test_pdf_extraction_sample():
    pdf_files = list(UPLOADS_DIR.glob("*.pdf"))
    if not pdf_files:
        pytest.skip("No sample PDF files found")

    text = extract_text(str(pdf_files[0]), "pdf")
    cleaned = clean_extracted_text(text)

    assert "HELLO WORLD!!" in cleaned
    assert "STUDY-" in cleaned
    assert "APPLICATION!!" in cleaned
    # Ensure words were not merged together
    assert "HELLOWORLDIAMHERETOTEST" not in cleaned
    assert "COMPANIONAPPLICATION" not in cleaned


def test_docx_extraction_sample():
    docx_files = list(UPLOADS_DIR.glob("*.docx"))
    if not docx_files:
        pytest.skip("No sample DOCX files found")

    # Test the multi-part docx
    multi_docx = [f for f in docx_files if "a6d9" in f.name]
    target = multi_docx[0] if multi_docx else docx_files[0]

    text = extract_text(str(target), "docx")
    cleaned = clean_extracted_text(text)

    assert "HELLO WORLD!!" in cleaned
    if multi_docx:
        assert "STUDY-" in cleaned
        assert "APPLICATION!!" in cleaned
        assert "COMPANION APPLICATION!!" in cleaned


def test_pptx_extraction_sample():
    pptx_files = list(UPLOADS_DIR.glob("*.pptx"))
    if not pptx_files:
        pytest.skip("No sample PPTX files found")

    text = extract_text(str(pptx_files[0]), "pptx")
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
