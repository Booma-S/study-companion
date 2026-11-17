from app.services.text_cleaning_service import clean_extracted_text


def test_clean_extracted_text():
    raw_text = "Hello     World\n\n\nThis is a test."

    result = clean_extracted_text(raw_text)

    assert result == "Hello World\n\nThis is a test."