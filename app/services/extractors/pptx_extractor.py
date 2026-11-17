from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from app.services.extractors.table_utils import format_table_as_markdown
from app.services.ocr_service import fix_line_spacing, run_ocr


def _flatten_shapes(shapes_collection) -> list[Any]:
    """
    Recursively extract all shapes, unpacking grouped shapes.
    """
    flat: list[Any] = []
    for shape in shapes_collection:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            flat.extend(_flatten_shapes(shape.shapes))
        else:
            flat.append(shape)
    return flat


def _extract_shape_content(shape) -> str:
    """
    Extract structured content from a single PowerPoint shape.
    Handles text frames, tables, charts, and pictures with OCR.
    """
    # 1. Table
    if shape.has_table:
        table_rows = []
        for row in shape.table.rows:
            row_cells = [cell.text.strip() for cell in row.cells]
            table_rows.append(row_cells)
        return format_table_as_markdown(table_rows).strip()

    # 2. Picture
    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
        if hasattr(shape, "image") and shape.image.blob:
            img = cv2.imdecode(
                np.frombuffer(shape.image.blob, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if img is not None:
                return run_ocr(img).strip()

    # 3. Chart
    if shape.has_chart:
        chart = shape.chart
        chart_parts = []
        if chart.has_title and chart.chart_title.has_text_frame:
            chart_parts.append(chart.chart_title.text_frame.text.strip())
        try:
            for plot in chart.plots:
                for series in plot.series:
                    series_name = series.name or "Series"
                    values = [str(v) for v in series.values]
                    chart_parts.append(f"{series_name}: {', '.join(values)}")
        except Exception:
            pass
        if chart_parts:
            return "\n".join(chart_parts)

    # 4. Text frame
    if shape.has_text_frame:
        lines = []
        for p in shape.text_frame.paragraphs:
            p_text = fix_line_spacing(p.text.strip())
            if p_text:
                lines.append(p_text)
        return "\n".join(lines).strip()

    # Fallback to text attribute if available
    if hasattr(shape, "text") and shape.text.strip():
        return fix_line_spacing(shape.text.strip())

    return ""


def extract_pptx(path: Path) -> str:
    """
    Extract text, tables, charts, and OCR image content from PPTX presentations.
    - Preserves visual reading order by sorting shapes spatially (top, left).
    - Unpacks grouped shapes recursively.
    - Formats tables as markdown.
    - Extracts speaker notes.
    """
    presentation = Presentation(str(path))
    slides_output: list[str] = []

    for slide_idx, slide in enumerate(presentation.slides):
        all_shapes = _flatten_shapes(slide.shapes)

        # Sort shapes by spatial position: top to bottom, then left to right
        all_shapes.sort(key=lambda s: (round(getattr(s, "top", 0) / 500000) * 500000, getattr(s, "left", 0)))

        slide_contents: list[str] = []

        for shape in all_shapes:
            content = _extract_shape_content(shape)
            if content:
                slide_contents.append(content)

        # Speaker notes (if present)
        if slide.has_notes_slide:
            notes_tf = slide.notes_slide.notes_text_frame
            if notes_tf and notes_tf.text.strip():
                notes_text = fix_line_spacing(notes_tf.text.strip())
                if notes_text:
                    slide_contents.append(f"Notes:\n{notes_text}")

        if slide_contents:
            slides_output.append("\n\n".join(slide_contents))

    return "\n\n".join(slides_output).strip()
