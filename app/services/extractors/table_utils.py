def format_table_as_markdown(rows: list[list[str | None]]) -> str:
    """
    Format a 2D list of strings as a clean markdown table.
    """
    if not rows or not any(rows):
        return ""

    cleaned_rows: list[list[str]] = []
    for row in rows:
        cleaned_row = [(cell or "").strip().replace("\n", " ") for cell in row]
        cleaned_rows.append(cleaned_row)

    # Determine maximum number of columns
    max_cols = max(len(r) for r in cleaned_rows)
    if max_cols == 0:
        return ""

    # Pad all rows to max_cols
    padded_rows = [r + [""] * (max_cols - len(r)) for r in cleaned_rows]

    # If first row is completely empty, skip it
    if not any(padded_rows[0]):
        padded_rows = padded_rows[1:]
        if not padded_rows:
            return ""

    # Build header and separator
    header = "| " + " | ".join(padded_rows[0]) + " |"
    separator = "| " + " | ".join(["---"] * max_cols) + " |"

    data_lines = []
    for r in padded_rows[1:]:
        data_lines.append("| " + " | ".join(r) + " |")

    return "\n".join([header, separator] + data_lines)
