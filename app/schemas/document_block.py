from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

BlockType = Literal[
    "heading",
    "paragraph",
    "list_item",
    "table",
    "code",
    "caption",
    "ocr_block",
    "chart",
]


class DocumentBlock(BaseModel):
    """
    Common ordered representation of document elements before chunking.
    Preserves document structure, visual reading order, and source location.
    """

    model_config = ConfigDict(extra="ignore")

    block_index: int = Field(description="0-based sequential order of the block in the document.")
    block_type: BlockType = Field(description="Semantic type of the block (heading, paragraph, table, etc.).")
    text: str = Field(description="Cleaned text content of the block.")
    page_number: int | None = Field(default=None, description="1-based PDF/DOCX page number when known.")
    slide_number: int | None = Field(default=None, description="1-based PPTX slide number when known.")
    heading_path: list[str] = Field(
        default_factory=list,
        description="Active heading hierarchy leading up to this block.",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Format-specific details (e.g. bounding boxes, confidence, table dimensions).",
    )


class ExtractionResult(BaseModel):
    """
    Container returning both the canonical flattened cleaned text
    and the ordered list of structured DocumentBlocks.
    """

    text: str
    blocks: list[DocumentBlock] = Field(default_factory=list)
