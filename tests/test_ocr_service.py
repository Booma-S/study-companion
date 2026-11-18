import cv2
import numpy as np

from app.services.extractors.table_utils import format_table_as_markdown
from app.services.ocr_service import (
    fix_line_spacing,
    split_merged_token,
)


def test_split_merged_token_ocr_glued():
    # Ordinary words glued together by OCR should be separated
    assert split_merged_token("HELLOWORLD!!") == "HELLO WORLD!!"
    assert split_merged_token("HELLOWORLDIAMHERETOTESTTHESTUDY-") == "HELLO WORLD I AM HERE TO TEST THE STUDY-"
    assert split_merged_token("COMPANIONAPPLICATION!!") == "COMPANION APPLICATION!!"
    assert split_merged_token("fromthedata.") == "from the data."


def test_split_merged_token_prose_phrases():
    # Concatenated phrases in ordinary prose must be separated
    assert split_merged_token("MachineLearning") == "Machine Learning"
    assert split_merged_token("DeepLearning") == "Deep Learning"
    assert split_merged_token("StudyCompanion") == "Study Companion"


def test_split_merged_token_technical_brands_and_acronyms():
    # Actual technology names and uppercase acronyms must be preserved
    assert split_merged_token("TensorFlow") == "TensorFlow"
    assert split_merged_token("PostgreSQL") == "PostgreSQL"
    assert split_merged_token("FastAPI") == "FastAPI"
    assert split_merged_token("SQLAlchemy") == "SQLAlchemy"
    assert split_merged_token("PyTorch") == "PyTorch"
    assert split_merged_token("BERT") == "BERT"
    assert split_merged_token("CNN") == "CNN"
    assert split_merged_token("RNN") == "RNN"
    assert split_merged_token("API") == "API"
    assert split_merged_token("STUDY-COMPANION") == "STUDY-COMPANION"


def test_fix_line_spacing():
    raw = "task.In simple words,ML teaches systems to think and understand like humans by learning fromthedata."
    expected = "task. In simple words, ML teaches systems to think and understand like humans by learning from the data."
    assert fix_line_spacing(raw) == expected


def test_format_table_as_markdown():
    data = [
        ["Stage", "Description"],
        ["Feature Selection", "Select relevant features"],
        ["Model Training", "Train the model"],
    ]
    md = format_table_as_markdown(data)
    assert "| Stage | Description |" in md
    assert "| --- | --- |" in md
    assert "| Feature Selection | Select relevant features |" in md
    assert "| Model Training | Train the model |" in md


def test_affine_inverse_deskew_mapping():
    w, h = 400, 200
    angle = 10.0
    M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    inv_M = cv2.invertAffineTransform(M)

    orig_point = np.array([150.0, 80.0, 1.0])
    rot_point = M @ orig_point
    recovered_point = inv_M @ np.array([rot_point[0], rot_point[1], 1.0])

    assert np.allclose(orig_point[:2], recovered_point, atol=1e-4)
