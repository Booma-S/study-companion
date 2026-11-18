import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pymupdf

from app.services.extractors.table_utils import format_table_as_markdown
from app.services.ocr_service import (
    fix_line_spacing,
    reconstruct_layout_blocks,
    run_ocr,
    run_ocr_raw,
)

logger = logging.getLogger(__name__)


def _bbox_overlap_ratio(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    """
    Calculate the overlap ratio of box1 relative to its own area.
    box: (x_min, y_min, x_max, y_max)
    """
    inter_x = max(0.0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    inter_y = max(0.0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    inter_area = inter_x * inter_y
    area1 = max((box1[2] - box1[0]) * (box1[3] - box1[1]), 1.0)
    return inter_area / area1


def extract_pdf(path: Path) -> str:
    """
    Extract text, tables, and image content from PDF files.
    - Native digital text is preferred for high accuracy.
    - Structured tables are extracted as markdown tables.
    - Scanned pages are processed with OCR.
    - Mixed pages reconcile native text and image OCR by coordinates to avoid duplicates.
    - Scanned/image-based tables fallback to OCR if native table extraction is empty.
    - Preserves individual OCR block bounding boxes so image text interleaves properly with native text.
    """
    pdf_document = pymupdf.open(str(path))
    pages_output: list[str] = []

    scale = 2.0  # Rendering scale for OCR

    for page_index in range(len(pdf_document)):
        page = pdf_document[page_index]

        # 1. Extract structured tables
        table_bboxes_with_content: list[tuple[float, float, float, float]] = []
        table_elements: list[dict[str, Any]] = []

        try:
            tabs = page.find_tables()
            for tab in tabs:
                extracted_data = tab.extract()
                md = format_table_as_markdown(extracted_data)
                has_content = any(
                    any((cell or "").strip() for cell in row)
                    for row in extracted_data
                )
                if has_content and md.strip():
                    table_bboxes_with_content.append(tab.bbox)
                    table_elements.append(
                        {
                            "type": "table",
                            "y_min": tab.bbox[1],
                            "y_max": tab.bbox[3],
                            "x_min": tab.bbox[0],
                            "x_max": tab.bbox[2],
                            "content": md.strip(),
                        }
                    )
        except Exception as exc:
            logger.warning("Table extraction failed on page %d: %s", page_index + 1, exc)

        # 2. Extract native text blocks
        raw_blocks = page.get_text("blocks")
        native_blocks: list[dict[str, Any]] = []

        for b in raw_blocks:
            # b: (x0, y0, x1, y1, text, block_no, block_type)
            if b[6] != 0:
                continue
            text = b[4].strip()
            if not text:
                continue

            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
            in_table = any(
                tb[0] <= cx <= tb[2] and tb[1] <= cy <= tb[3]
                for tb in table_bboxes_with_content
            )
            if in_table:
                continue

            fixed_lines = [fix_line_spacing(l) for l in text.split("\n") if l.strip()]
            fixed_text = "\n".join(fixed_lines)

            native_blocks.append(
                {
                    "type": "native_text",
                    "bbox": (b[0], b[1], b[2], b[3]),
                    "x_min": b[0],
                    "y_min": b[1],
                    "x_max": b[2],
                    "y_max": b[3],
                    "height": max(b[3] - b[1], 1.0),
                    "content": fixed_text,
                }
            )

        # 3. Check if page has images / drawings
        has_images = len(page.get_images()) > 0
        has_drawings = len(page.get_drawings()) > 0

        # Case A: Scanned page (no native text)
        if not native_blocks and not table_elements:
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
            img = cv2.imdecode(
                np.frombuffer(pixmap.tobytes("png"), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            ocr_text = run_ocr(img).strip()
            if ocr_text:
                pages_output.append(ocr_text)
            continue

        # Case B: Mixed content or potentially has image text
        page_elements: list[dict[str, Any]] = []
        page_elements.extend(table_elements)
        page_elements.extend(native_blocks)

        if has_images or has_drawings or len(page_elements) < 3:
            pixmap = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale))
            img = cv2.imdecode(
                np.frombuffer(pixmap.tobytes("png"), dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            detections = run_ocr_raw(img, min_score=0.35)

            image_detections: list[dict[str, Any]] = []
            for d in detections:
                box_page = (
                    d["x_min"] / scale,
                    d["y_min"] / scale,
                    d["x_max"] / scale,
                    d["y_max"] / scale,
                )

                is_duplicate = False
                for nb in native_blocks:
                    if _bbox_overlap_ratio(box_page, nb["bbox"]) > 0.35:
                        is_duplicate = True
                        break
                    norm_det = d["text"].lower().replace(" ", "")
                    norm_nat = nb["content"].lower().replace(" ", "")
                    if norm_det in norm_nat and abs(box_page[1] - nb["y_min"]) < 40.0:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    for tb in table_bboxes_with_content:
                        if _bbox_overlap_ratio(box_page, tb) > 0.35:
                            is_duplicate = True
                            break

                if not is_duplicate:
                    image_detections.append(d)

            # Preserve individual OCR blocks so localized image regions interleave correctly
            if image_detections:
                ocr_blocks = reconstruct_layout_blocks(image_detections)
                for blk in ocr_blocks:
                    content = blk["content"].strip()
                    if content:
                        page_elements.append(
                            {
                                "type": "ocr_text",
                                "y_min": blk["y_min"] / scale,
                                "y_max": blk["y_max"] / scale,
                                "x_min": blk["x_min"] / scale,
                                "x_max": blk["x_max"] / scale,
                                "content": content,
                            }
                        )

        # 4. Sort page elements into reading order
        page_elements.sort(key=lambda el: el["y_min"])
        rows: list[list[dict[str, Any]]] = []
        for el in page_elements:
            placed = False
            el_h = max(el["y_max"] - el["y_min"], 1.0)
            for row in rows:
                r_ymin = min(it["y_min"] for it in row)
                r_ymax = max(it["y_max"] for it in row)
                overlap = min(el["y_max"], r_ymax) - max(el["y_min"], r_ymin)
                min_h = min(el_h, r_ymax - r_ymin)
                if overlap >= min_h * 0.4:
                    row.append(el)
                    placed = True
                    break
            if not placed:
                rows.append([el])

        ordered_contents: list[str] = []
        for row in rows:
            row.sort(key=lambda it: it["x_min"])
            for it in row:
                if it["content"]:
                    ordered_contents.append(it["content"])

        if ordered_contents:
            pages_output.append("\n\n".join(ordered_contents))

    pdf_document.close()

    return "\n\n".join(pages_output).strip()
