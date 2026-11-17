import re


def clean_extracted_text(text: str) -> str:
    """
    Normalize extracted document text while preserving
    paragraph breaks and meaningful punctuation.
    """

    if not text:
        return ""

    # Normalize Windows and Mac line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove unwanted control characters
    text = "".join(
        char for char in text
        if char.isprintable() or char == "\n"
    )

    # Remove excessive spaces within lines
    lines = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line)
        lines.append(line.strip())

    text = "\n".join(lines)

    # Normalize excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()