# Study Companion

A document-based study companion built with **FastAPI** and **PostgreSQL**. It ingests learning materials, extracts their text and structure, and prepares the content as token-aware chunks for downstream study features.

> **Status:** In active development. Document extraction, token counting, and the core chunking helpers are implemented. The end-to-end chunking pipeline, chunk persistence, and API integration are in progress.

## Features

- Accepts PDF, DOCX, and PPTX files.
- Extracts text as structured blocks that retain page, slide, and heading provenance.
- Counts and splits text using model-aware tokenization.
- Chunks content within strict token budgets while preserving source context.

## Tech Stack

| Area                   | Technology        |
| ---------------------- | ----------------- |
| API                    | FastAPI           |
| Language               | Python 3.12       |
| Database               | PostgreSQL        |
| ORM                    | SQLAlchemy 2.0    |
| Migrations             | Alembic           |
| Validation and schemas | Pydantic          |
| Tokenization           | `tiktoken`        |
| Dependency management  | `uv`              |

## Implemented

### Structured document extraction

Extractors for PDF, DOCX, and PPTX return an `ExtractionResult`: the full document text plus an ordered list of `DocumentBlock` objects.

| Field          | Description                                                                              |
| -------------- | ---------------------------------------------------------------------------------------- |
| `block_index`  | Position of the block in the document                                                    |
| `block_type`   | Heading, paragraph, list item, table, code, caption, OCR block, or chart                 |
| `text`         | Extracted text                                                                           |
| `page_number`  | Source page, when available                                                              |
| `slide_number` | Source slide, when available                                                             |
| `heading_path` | Heading hierarchy, when the extractor provides one                                       |
| `metadata`     | Additional source-specific information                                                   |

Heading support varies by format: DOCX heading styles and PPTX slide titles provide heading context, while PDF heading hierarchy is not inferred.

### Token counting

A `tiktoken`-based service provides token counting and token-based text splitting. It supports encoding selection, falls back to `cl100k_base` when an encoding cannot be resolved, and caches encoders by name.

### Chunking helpers

Chunking is developed independently of the database and API layers, using these defaults:

| Setting            | Value      |
| ------------------ | ---------- |
| Target chunk size  | 600 tokens |
| Maximum chunk size | 900 tokens |
| Overlap target     | 80 tokens  |

Implemented helpers:

- Resolve and retain heading context.
- Group consecutive blocks in reading order.
- Split oversized paragraphs at sentence boundaries, falling back to token-based splitting.
- Split Markdown tables by data rows, repeating the header and separator.
- Build chunk results with page, slide, heading, and source-block provenance.
- Extract trailing overlap text, preferring complete sentences.

`ChunkResult` is the Pydantic schema for a generated chunk before persistence. It holds the chunk index, content, token count, page range, slide number, heading path, and metadata.

## Project Structure

```text
study-companion/
├── app/
│   ├── schemas/
│   │   ├── document_block.py
│   │   └── chunk.py
│   └── services/
│       ├── token_counting_service.py
│       └── chunking_service.py
├── tests/
│   ├── test_token_counting.py
│   └── test_chunking_service.py
├── alembic/
├── pyproject.toml
└── uv.lock
```

This shows the main areas only. Extractor modules for PDF, DOCX, and PPTX are also part of the application, and their locations may change.

## Getting Started

### Prerequisites

- Python 3.12
- PostgreSQL
- [`uv`](https://docs.astral.sh/uv/)

### Setup

```bash
# 1. Clone the repository
git clone <repository-url>
cd study-companion

# 2. Install dependencies
uv sync --all-groups
```

3. Create a local `.env` file from the repository's environment template (if provided) and set your PostgreSQL connection details. Never commit this file.

```bash
# 4. Apply database migrations
uv run alembic upgrade head

# 5. Start the API
uv run uvicorn app.main:app --reload
```

Interactive API docs are available at <http://127.0.0.1:8000/docs>. Adjust the import path if your FastAPI instance is not `app` in `app/main.py`.

## Testing

```bash
# Full suite
uv run pytest -v

# Individual modules
uv run pytest tests/test_token_counting.py -v
uv run pytest tests/test_chunking_service.py -v
```
------------------------------------------------------------------------

Study Companion is under active development.