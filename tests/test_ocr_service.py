from app.services.ocr_service import fix_line_spacing, split_merged_token
from app.services.extractors.table_utils import format_table_as_markdown


def test_split_merged_token():
    assert split_merged_token("HELLOWORLD!!") == "HELLO WORLD!!"
    assert split_merged_token("HELLOWORLDIAMHERETOTESTTHESTUDY-") == "HELLO WORLD I AM HERE TO TEST THE STUDY-"
    assert split_merged_token("COMPANIONAPPLICATION!!") == "COMPANION APPLICATION!!"
    assert split_merged_token("fromthedata.") == "from the data."
    assert split_merged_token("APPLICATION") == "APPLICATION"
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
