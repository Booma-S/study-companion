
import pytest
from app.schemas.document_block import DocumentBlock
from app.services.chunking_service import (
    _resolve_heading_path,
    _group_normal_blocks,
    _split_oversized_paragraph,
    _split_markdown_table,
)
from app.services.token_counting_service import count_tokens

def test_resolve_heading_path_uses_existing_path():
    block = DocumentBlock(
        block_index=1,
        block_type="paragraph",
        text="Some content",
        heading_path=["Chapter 1", "Section 1.1"],
    )

    result = _resolve_heading_path(block, [])

    assert result == ["Chapter 1", "Section 1.1"]


def test_resolve_heading_path_preserves_active_context():
    block = DocumentBlock(
        block_index=2,
        block_type="paragraph",
        text="More content",
    )

    active_path = ["Chapter 1", "Section 1.1"]

    result = _resolve_heading_path(block, active_path)

    assert result == active_path


def test_resolve_heading_path_fallback_for_heading():
    block = DocumentBlock(
        block_index=0,
        block_type="heading",
        text="Introduction",
    )

    result = _resolve_heading_path(block, ["Chapter 1"])

    assert result == ["Chapter 1", "Introduction"]


def test_resolve_heading_path_does_not_modify_active_path():
    active_path = ["Chapter 1"]

    block = DocumentBlock(
        block_index=1,
        block_type="paragraph",
        text="Content",
    )

    result = _resolve_heading_path(block, active_path)

    assert result == ["Chapter 1"]
    assert active_path == ["Chapter 1"]


def test_group_normal_blocks_combines_small_paragraphs():
    blocks = [
        DocumentBlock(
            block_index=i,
            block_type="paragraph",
            text=f"Paragraph {i}.",
            heading_path=["Chapter 1"],
        )
        for i in range(3)
    ]

    groups = _group_normal_blocks(
        blocks,
        target_tokens=600,
        max_tokens=900,
    )

    assert len(groups) == 1
    assert len(groups[0]) == 3


def test_group_normal_blocks_splits_at_heading_change():
    blocks = [
        DocumentBlock(
            block_index=0,
            block_type="paragraph",
            text="First section content.",
            heading_path=["Chapter 1"],
        ),
        DocumentBlock(
            block_index=1,
            block_type="paragraph",
            text="Second section content.",
            heading_path=["Chapter 2"],
        ),
    ]

    groups = _group_normal_blocks(blocks)

    assert len(groups) == 2
    assert groups[0][0].heading_path == ["Chapter 1"]
    assert groups[1][0].heading_path == ["Chapter 2"]


def test_group_normal_blocks_keeps_tables_separate():
    blocks = [
        DocumentBlock(
            block_index=0,
            block_type="paragraph",
            text="Before table.",
            heading_path=["Chapter 1"],
        ),
        DocumentBlock(
            block_index=1,
            block_type="table",
            text="| A | B |\n|---|---|\n| 1 | 2 |",
            heading_path=["Chapter 1"],
        ),
        DocumentBlock(
            block_index=2,
            block_type="paragraph",
            text="After table.",
            heading_path=["Chapter 1"],
        ),
    ]

    groups = _group_normal_blocks(blocks)

    assert len(groups) == 3
    assert groups[1][0].block_type == "table"


def test_group_normal_blocks_preserves_reading_order():
    blocks = [
        DocumentBlock(
            block_index=i,
            block_type="paragraph",
            text=f"Content {i}.",
            heading_path=["Chapter 1"],
        )
        for i in range(5)
    ]

    groups = _group_normal_blocks(blocks)

    grouped_indices = [
        block.block_index
        for group in groups
        for block in group
    ]

    assert grouped_indices == [0, 1, 2, 3, 4]


def test_group_normal_blocks_rejects_invalid_limits():
    with pytest.raises(ValueError):
        _group_normal_blocks([], target_tokens=0)

    with pytest.raises(ValueError):
        _group_normal_blocks([], target_tokens=600, max_tokens=500)

def test_split_oversized_paragraph_keeps_small_paragraph_unchanged():
    text = "This is a small paragraph."

    chunks = _split_oversized_paragraph(text)

    assert chunks == [text]


def test_split_oversized_paragraph_splits_at_sentence_boundaries():
    text = (
        "The first sentence explains the topic. "
        "The second sentence provides more details. "
        "The third sentence gives an example. "
        "The fourth sentence concludes the explanation."
    )

    chunks = _split_oversized_paragraph(
        text,
        target_tokens=10,
        max_tokens=20,
    )

    assert len(chunks) > 1
    assert " ".join(chunks) == text

    for chunk in chunks:
        assert count_tokens(chunk) <= 20


def test_split_oversized_paragraph_uses_token_fallback_for_long_sentence():
    text = "important " * 100

    chunks = _split_oversized_paragraph(
        text,
        target_tokens=10,
        max_tokens=15,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        assert count_tokens(chunk) <= 15


def test_split_oversized_paragraph_rejects_invalid_limits():
    with pytest.raises(ValueError):
        _split_oversized_paragraph(
            "Some text.",
            target_tokens=0,
        )

    with pytest.raises(ValueError):
        _split_oversized_paragraph(
            "Some text.",
            target_tokens=600,
            max_tokens=500,
        )


def test_split_oversized_paragraph_handles_empty_text():
    assert _split_oversized_paragraph("") == []

def test_split_markdown_table_keeps_small_table_unchanged():
    table = (
        "| Name | Age |\n"
        "|---|---|\n"
        "| Alice | 21 |\n"
        "| Bob | 22 |"
    )

    chunks = _split_markdown_table(table)

    assert chunks == [table]


def test_split_markdown_table_repeats_header():
    table = (
        "| Name | Description |\n"
        "|---|---|\n"
        + "".join(
            f"| Item {i} | This is a description for item {i}. |\n"
            for i in range(30)
        )
    ).strip()

    chunks = _split_markdown_table(
        table,
        max_tokens=40,
    )

    assert len(chunks) > 1

    for chunk in chunks:
        lines = chunk.splitlines()

        assert lines[0] == "| Name | Description |"
        assert lines[1] == "|---|---|"


def test_split_markdown_table_respects_token_limit():
    table = (
        "| Name | Description |\n"
        "|---|---|\n"
        + "".join(
            f"| Item {i} | Description for item {i}. |\n"
            for i in range(30)
        )
    ).strip()

    chunks = _split_markdown_table(table, max_tokens=40)

    for chunk in chunks:
        assert count_tokens(chunk) <= 40


def test_split_markdown_table_preserves_row_order():
    table = (
        "| Name | Value |\n"
        "|---|---|\n"
        + "".join(f"| Item {i} | {i} |\n" for i in range(20))
    ).strip()

    chunks = _split_markdown_table(table, max_tokens=30)

    extracted_rows = []

    for chunk in chunks:
        extracted_rows.extend(chunk.splitlines()[2:])

    expected_rows = [
        f"| Item {i} | {i} |"
        for i in range(20)
    ]

    assert extracted_rows == expected_rows


def test_split_markdown_table_rejects_invalid_table():
    with pytest.raises(ValueError):
        _split_markdown_table("| Name | Age |")


def test_split_markdown_table_rejects_invalid_limits():
    with pytest.raises(ValueError):
        _split_markdown_table(
            "| Name | Age |\n|---|---|",
            max_tokens=0,
        )