import logging
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE, PP_PLACEHOLDER

from app.schemas.document_block import DocumentBlock, ExtractionResult
from app.services.extractors.table_utils import format_table_as_markdown
from app.services.ocr_service import fix_line_spacing, run_ocr

logger = logging.getLogger(__name__)


class PositionedShape:
    """
    Wrapper for PowerPoint shapes that tracks absolute slide coordinates
    even when nested inside grouped shapes with coordinate scaling.
    """

    def __init__(self, shape: Any, left: int, top: int, width: int, height: int):
        self.shape = shape
        self.left = left
        self.top = top
        self.width = max(width, 1)
        self.height = max(height, 1)
        self.x_min = left
        self.x_max = left + width
        self.y_min = top
        self.y_max = top + height


class TransformContext:
    """
    Tracks coordinate frames and scaling transforms for PowerPoint groups.
    Ensures that nested groups do not double-count parent offsets.
    """

    def __init__(
        self,
        origin_x: int = 0,
        origin_y: int = 0,
        ch_off_x: int = 0,
        ch_off_y: int = 0,
        scale_x: float = 1.0,
        scale_y: float = 1.0,
    ):
        self.origin_x = origin_x
        self.origin_y = origin_y
        self.ch_off_x = ch_off_x
        self.ch_off_y = ch_off_y
        self.scale_x = scale_x
        self.scale_y = scale_y

    def transform_point(self, x: int, y: int) -> tuple[int, int]:
        abs_x = int(self.origin_x + (x - self.ch_off_x) * self.scale_x)
        abs_y = int(self.origin_y + (y - self.ch_off_y) * self.scale_y)
        return abs_x, abs_y

    def transform_size(self, w: int, h: int) -> tuple[int, int]:
        return int(w * self.scale_x), int(h * self.scale_y)


def _get_group_transform(group_shape: Any) -> tuple[int, int, float, float]:
    """
    Extract child offset (chOff) and child scaling factors (ext / chExt)
    from DrawingML group shape properties (p:grpSpPr/a:xfrm).
    """
    ch_off_x, ch_off_y = 0, 0
    scale_x, scale_y = 1.0, 1.0

    try:
        if hasattr(group_shape, "_element"):
            xfrm_list = group_shape._element.xpath(".//p:grpSpPr/a:xfrm")
            if xfrm_list:
                xf = xfrm_list[0]
                ch_off = xf.find("{http://schemas.openxmlformats.org/drawingml/2006/main}chOff")
                ch_ext = xf.find("{http://schemas.openxmlformats.org/drawingml/2006/main}chExt")
                ext = xf.find("{http://schemas.openxmlformats.org/drawingml/2006/main}ext")

                if ch_off is not None:
                    ch_off_x = int(ch_off.get("x", 0))
                    ch_off_y = int(ch_off.get("y", 0))

                if ch_ext is not None and ext is not None:
                    ch_cx = int(ch_ext.get("cx", 1))
                    ch_cy = int(ch_ext.get("cy", 1))
                    p_cx = int(ext.get("cx", ch_cx))
                    p_cy = int(ext.get("cy", ch_cy))
                    if ch_cx > 0:
                        scale_x = p_cx / ch_cx
                    if ch_cy > 0:
                        scale_y = p_cy / ch_cy
    except Exception as exc:
        logger.debug("Could not parse group transform XML: %s", exc)

    return ch_off_x, ch_off_y, scale_x, scale_y


def _flatten_shapes(shapes_collection, ctx: TransformContext | None = None) -> list[PositionedShape]:
    """
    Recursively extract all shapes, properly composing parent group coordinate systems
    and scaling transformations so nested shapes have accurate slide coordinates
    without double-counting parent offsets.
    """
    if ctx is None:
        ctx = TransformContext()

    flat: list[PositionedShape] = []

    for shape in shapes_collection:
        shape_raw_left = getattr(shape, "left", 0)
        shape_raw_top = getattr(shape, "top", 0)
        shape_raw_w = getattr(shape, "width", 0)
        shape_raw_h = getattr(shape, "height", 0)

        # Calculate absolute position in the current context
        abs_l, abs_t = ctx.transform_point(shape_raw_left, shape_raw_top)
        abs_w, abs_h = ctx.transform_size(shape_raw_w, shape_raw_h)

        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            ch_off_x, ch_off_y, scale_x, scale_y = _get_group_transform(shape)
            # Create a new child context rooted at the group's absolute slide position
            child_ctx = TransformContext(
                origin_x=abs_l,
                origin_y=abs_t,
                ch_off_x=ch_off_x,
                ch_off_y=ch_off_y,
                scale_x=ctx.scale_x * scale_x,
                scale_y=ctx.scale_y * scale_y,
            )
            flat.extend(_flatten_shapes(shape.shapes, ctx=child_ctx))
        else:
            flat.append(PositionedShape(shape, abs_l, abs_t, abs_w, abs_h))

    return flat


def _extract_chart_content(chart: Any) -> str:
    """
    Extract structured content from PowerPoint charts, preserving titles,
    categories, series names, and category-to-value relationships.
    """
    parts = []
    if chart.has_title and chart.chart_title.has_text_frame:
        title = chart.chart_title.text_frame.text.strip()
        if title:
            parts.append(f"**Chart: {title}**")

    try:
        categories: list[str] = []
        for plot in chart.plots:
            if hasattr(plot, "categories") and plot.categories:
                categories = [str(c) for c in plot.categories]
                break

        series_list: list[tuple[str, list[str]]] = []
        for plot in chart.plots:
            for s in plot.series:
                s_name = s.name or f"Series {len(series_list) + 1}"
                s_vals = [str(v) for v in s.values]
                series_list.append((s_name, s_vals))

        if categories and series_list:
            headers = ["Category"] + [s[0] for s in series_list]
            table_rows = [headers]
            for idx, cat in enumerate(categories):
                row = [cat]
                for s_name, s_vals in series_list:
                    row.append(s_vals[idx] if idx < len(s_vals) else "")
                table_rows.append(row)
            parts.append(format_table_as_markdown(table_rows))
        elif series_list:
            for s_name, s_vals in series_list:
                parts.append(f"{s_name}: {', '.join(s_vals)}")
    except Exception as exc:
        logger.warning("Failed to extract chart data: %s", exc)

    return "\n\n".join(parts)


def _extract_shape_content_typed(shape: Any) -> tuple[str, str, dict[str, Any]]:
    """
    Extract structured content from a single PowerPoint shape,
    identifying semantic block type and shape metadata.
    """
    # 1. Table
    if shape.has_table:
        table_rows = []
        for row in shape.table.rows:
            row_cells = [cell.text.strip() for cell in row.cells]
            table_rows.append(row_cells)
        return (
            format_table_as_markdown(table_rows).strip(),
            "table",
            {"rows": len(shape.table.rows), "cols": len(shape.table.columns)},
        )

    # 2. Picture
    if shape.shape_type == MSO_SHAPE_TYPE.PICTURE:
        if hasattr(shape, "image") and shape.image.blob:
            img = cv2.imdecode(
                np.frombuffer(shape.image.blob, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if img is not None:
                return run_ocr(img).strip(), "ocr_block", {"source": "picture_ocr"}

    # 3. Chart
    if shape.has_chart:
        return _extract_chart_content(shape.chart).strip(), "chart", {"source": "chart"}

    # 4. Text frame
    if shape.has_text_frame:
        lines = []
        is_bullet = False
        for p in shape.text_frame.paragraphs:
            p_text = fix_line_spacing(p.text.strip())
            if p_text:
                lines.append(p_text)
                if p_text.startswith(("- ", "* ", "• ")) or (hasattr(p, "level") and p.level and p.level > 0):
                    is_bullet = True

        joined = "\n".join(lines).strip()
        is_title = False
        try:
            if shape.is_placeholder and shape.placeholder_format.type in (
                PP_PLACEHOLDER.TITLE,
                PP_PLACEHOLDER.CENTER_TITLE,
            ):
                is_title = True
        except Exception:
            pass

        if is_title:
            b_type = "heading"
        elif is_bullet or (lines and (lines[0].startswith(("- ", "* ", "• ")) or re.match(r"^\d+\.\s", lines[0]))):
            b_type = "list_item"
        else:
            b_type = "paragraph"

        return joined, b_type, {"source": "text_frame"}

    # Fallback to text attribute if available
    if hasattr(shape, "text") and shape.text.strip():
        return fix_line_spacing(shape.text.strip()), "paragraph", {}

    return "", "paragraph", {}


def _extract_shape_content(shape: Any) -> str:
    """
    Extract structured content from a single PowerPoint shape.
    Handles text frames, tables, charts, and pictures with OCR.
    """
    content, _, _ = _extract_shape_content_typed(shape)
    return content


def _sort_shapes_reading_order(items: list[PositionedShape]) -> list[PositionedShape]:
    """
    Sort slide shapes in visual reading order using dynamic vertical-overlap band grouping
    rather than coarse static coordinate rounding.
    """
    if not items:
        return []

    items.sort(key=lambda it: it.y_min)
    rows: list[list[PositionedShape]] = []

    for it in items:
        placed = False
        for row in rows:
            r_ymin = min(r.y_min for r in row)
            r_ymax = max(r.y_max for r in row)
            overlap = min(it.y_max, r_ymax) - max(it.y_min, r_ymin)
            min_h = min(it.height, r_ymax - r_ymin)
            if overlap >= min_h * 0.4:
                row.append(it)
                placed = True
                break
        if not placed:
            rows.append([it])

    sorted_shapes: list[PositionedShape] = []
    for row in rows:
        row.sort(key=lambda it: it.x_min)
        sorted_shapes.extend(row)

    return sorted_shapes


def extract_pptx_with_blocks(path: Path) -> ExtractionResult:
    """
    Extract text, tables, charts, and OCR image content from PPTX presentations
    as structured DocumentBlocks along with canonical flattened text.
    - Preserves visual reading order using dynamic vertical-overlap grouping.
    - Unpacks grouped shapes recursively with exact coordinate transform composition.
    - Formats tables and charts as structured markdown tables.
    - Extracts speaker notes.
    """
    presentation = Presentation(str(path))
    slides_output: list[str] = []
    all_blocks: list[DocumentBlock] = []

    for slide_idx, slide in enumerate(presentation.slides):
        slide_num = slide_idx + 1

        # Detect slide title if available
        slide_title = ""
        try:
            if slide.shapes.title and slide.shapes.title.has_text_frame:
                slide_title = fix_line_spacing(slide.shapes.title.text_frame.text.strip())
        except Exception:
            pass

        positioned_shapes = _flatten_shapes(slide.shapes)
        sorted_shapes = _sort_shapes_reading_order(positioned_shapes)

        slide_contents: list[str] = []
        for pos_shape in sorted_shapes:
            content, b_type, meta = _extract_shape_content_typed(pos_shape.shape)
            if content:
                slide_contents.append(content)
                if not slide_title and b_type == "heading":
                    slide_title = content.split("\n")[0].strip()

                heading_path = [slide_title] if slide_title else []
                all_blocks.append(
                    DocumentBlock(
                        block_index=len(all_blocks),
                        block_type=b_type,
                        text=content,
                        slide_number=slide_num,
                        heading_path=heading_path,
                        metadata={
                            **meta,
                            "bbox": [pos_shape.x_min, pos_shape.y_min, pos_shape.x_max, pos_shape.y_max],
                        },
                    )
                )

        # Speaker notes (if present)
        if slide.has_notes_slide:
            notes_tf = slide.notes_slide.notes_text_frame
            if notes_tf and notes_tf.text.strip():
                notes_text = fix_line_spacing(notes_tf.text.strip())
                if notes_text:
                    note_content = f"Notes:\n{notes_text}"
                    slide_contents.append(note_content)
                    all_blocks.append(
                        DocumentBlock(
                            block_index=len(all_blocks),
                            block_type="paragraph",
                            text=note_content,
                            slide_number=slide_num,
                            heading_path=[slide_title] if slide_title else [],
                            metadata={"type": "speaker_notes"},
                        )
                    )

        if slide_contents:
            slides_output.append("\n\n".join(slide_contents))

    full_text = "\n\n".join(slides_output).strip()
    return ExtractionResult(text=full_text, blocks=all_blocks)


def extract_pptx(path: Path) -> str:
    """
    Backward-compatible entry point returning plain extracted text.
    """
    return extract_pptx_with_blocks(path).text

