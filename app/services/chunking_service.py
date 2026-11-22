
from app.schemas.document_block import DocumentBlock
from app.services.token_counting_service import count_tokens
import re
from app.services.token_counting_service import (
    count_tokens,
    split_text_by_tokens,
)
TARGET_CHUNK_TOKENS = 600
MAX_CHUNK_TOKENS = 900

def _resolve_heading_path(
    block: DocumentBlock,
    active_heading_path: list[str],
    ) -> list[str]:
    """
    Resolve the heading context for a document block.
    Existing heading_path values from extractors are authoritative.
    If a heading block has no heading_path, use its text as a
    fallback heading without guessing its hierarchy level.
    """
    if block.heading_path:
        return list(block.heading_path)
    if block.block_type == "heading":
        return [*active_heading_path, block.text.strip()]
    return list(active_heading_path)


def _split_oversized_paragraph(
    text: str,
    target_tokens: int = TARGET_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
    encoding_name: str = "cl100k_base",
    ) -> list[str]:
    """
    Split an oversized paragraph at sentence boundaries.
    Consecutive sentences are grouped up to the target token limit.
    A sentence exceeding the hard limit is split using token boundaries.
    """

    if target_tokens <= 0:
        raise ValueError("target_tokens must be greater than 0")
    if max_tokens < target_tokens:
        raise ValueError("max_tokens must be >= target_tokens")
    text = text.strip()
    if not text:
        return []
    # No splitting is necessary if the paragraph fits the hard limit.
    if count_tokens(text, encoding_name) <= max_tokens:
        return [text]
    # Split at sentence boundaries while retaining sentence punctuation.
    sentences = [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]
    chunks: list[str] = []
    current_sentences: list[str] = []

    def flush_chunk() -> None:
        if current_sentences:
            chunks.append(" ".join(current_sentences))
            current_sentences.clear()

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence, encoding_name)

        # An individual sentence exceeds the hard limit.
        if sentence_tokens > max_tokens:
            flush_chunk()
            sentence_chunks = split_text_by_tokens(
                sentence,
                max_tokens=max_tokens,
                overlap_tokens=0,
                encoding_name=encoding_name,
            )
            chunks.extend(sentence_chunks)
            continue
        candidate = " ".join([*current_sentences, sentence])
        # Start a new chunk if adding this sentence exceeds the target.
        if (
            current_sentences
            and count_tokens(candidate, encoding_name) > target_tokens
        ):
            flush_chunk()
        current_sentences.append(sentence)
    flush_chunk()
    return chunks

def _split_markdown_table(
    table_text: str,
    max_tokens: int = MAX_CHUNK_TOKENS,
    encoding_name: str = "cl100k_base",
    ) -> list[str]:
    """
    Split an oversized Markdown table by data rows.
    Every resulting table retains the original header and separator rows.
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than 0")
    lines = [
        line.strip()
        for line in table_text.strip().splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        raise ValueError("A Markdown table must contain a header and separator")

    header = lines[0]
    separator = lines[1]
    data_rows = lines[2:]

    # Validate the Markdown separator row.
    if not re.fullmatch(r"\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?", separator):
        raise ValueError("Invalid Markdown table separator row")
    prefix = [header, separator]
    if count_tokens("\n".join(prefix), encoding_name) > max_tokens:
        raise ValueError("Table header exceeds max_tokens")
    if not data_rows:
        return ["\n".join(prefix)]
    chunks: list[str] = []
    current_rows: list[str] = []
    for row in data_rows:
        candidate_rows = [*current_rows, row]
        candidate_table = "\n".join([*prefix, *candidate_rows])

        if count_tokens(candidate_table, encoding_name) <= max_tokens:
            current_rows.append(row)
            continue
        # Save the current table before starting another one.
        if current_rows:
            chunks.append("\n".join([*prefix, *current_rows]))
            current_rows = []

        # A single row must fit with the repeated header.
        single_row_table = "\n".join([*prefix, row])
        if count_tokens(single_row_table, encoding_name) > max_tokens:
            raise ValueError("A table row exceeds max_tokens")
        current_rows.append(row)

    if current_rows:
        chunks.append("\n".join([*prefix, *current_rows]))

    return chunks

def _group_normal_blocks(
    blocks: list[DocumentBlock],
    target_tokens: int = TARGET_CHUNK_TOKENS,
    max_tokens: int = MAX_CHUNK_TOKENS,
    encoding_name: str = "cl100k_base",
) -> list[list[DocumentBlock]]:
    """
    Group consecutive normal document blocks while preserving reading order
    and heading context.
    Headings are included in their corresponding groups.
    Tables and charts are kept separate for specialized handling later.
    """

    if target_tokens <= 0:
        raise ValueError("target_tokens must be greater than 0")
    if max_tokens < target_tokens:
        raise ValueError("max_tokens must be >= target_tokens")
    groups: list[list[DocumentBlock]] = []
    current_group: list[DocumentBlock] = []
    current_tokens = 0
    current_heading_path: list[str] | None = None
    special_types = {"table", "chart"}
    def flush_group() -> None:
        nonlocal current_group, current_tokens, current_heading_path
        if current_group:
            groups.append(current_group)
        current_group = []
        current_tokens = 0
        current_heading_path = None

    for block in blocks:
        if not block.text.strip():
            continue
        heading_path = _resolve_heading_path(block,current_heading_path or [],)
        # Tables and charts will receive specialized chunking.
        if block.block_type in special_types:
            flush_group()
            groups.append([block])
            continue
        # A new heading context starts a new group.
        if (current_group and heading_path != current_heading_path):
            flush_group()

        block_tokens = count_tokens(
            block.text,
            encoding_name=encoding_name,
        )
        # Flush before adding a block if it would exceed the hard limit.
        if current_group and current_tokens + block_tokens > max_tokens:
            flush_group()
        if not current_group:
            current_heading_path = heading_path
        current_group.append(block)
        separator_tokens = 1 if len(current_group) > 1 else 0
        current_tokens += block_tokens + separator_tokens
        # Reaching the target ends the current group.
        if current_tokens >= target_tokens:
            flush_group()
    flush_group()
    return groups
