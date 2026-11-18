from pathlib import Path
from typing import Any

import cv2
import numpy as np
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.services.extractors.table_utils import format_table_as_markdown
from app.services.ocr_service import fix_line_spacing, run_ocr


def _extract_images_from_element(element: Any, part: Any, processed_rids: set[tuple[str, str]]) -> list[str]:
    """
    Find embedded images within a Word XML element and run OCR.
    Tracks images by (partname, r_id) so relationship IDs in different
    document parts (e.g. header vs body vs footer) do not collide.
    """
    image_texts: list[str] = []
    part_name = str(getattr(part, "partname", "body"))

    for el in element.iter():
        tag = el.tag.split("}")[-1]
        r_id = None

        if tag == "blip":
            # DrawingML image
            for k, v in el.attrib.items():
                if k.endswith("embed"):
                    r_id = v
                    break
        elif tag == "imagedata":
            # Legacy VML image
            for k, v in el.attrib.items():
                if k.endswith("id") or k.endswith("href"):
                    r_id = v
                    break

        if r_id:
            rid_key = (part_name, r_id)
            if rid_key not in processed_rids:
                processed_rids.add(rid_key)
                target = getattr(part, "related_parts", {}).get(r_id)
                if target and hasattr(target, "blob"):
                    img = cv2.imdecode(
                        np.frombuffer(target.blob, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if img is not None:
                        ocr_res = run_ocr(img).strip()
                        if ocr_res:
                            image_texts.append(ocr_res)

    return image_texts


def _extract_textboxes_from_element(element: Any) -> list[str]:
    """
    Find textboxes inside Word XML elements (e.g. w:txbxContent).
    Preserves paragraph structure and run boundaries.
    """
    tb_texts: list[str] = []
    for el in element.iter():
        tag = el.tag.split("}")[-1]
        if tag == "txbxContent":
            para_lines = []
            for child in el:
                if child.tag.split("}")[-1] == "p":
                    t_nodes = [t.text for t in child.iter() if t.tag.split("}")[-1] == "t" and t.text]
                    p_text = "".join(t_nodes).strip()
                    if p_text:
                        para_lines.append(fix_line_spacing(p_text))
            if para_lines:
                tb_texts.append("\n".join(para_lines))
    return tb_texts


def _extract_table_content(table: Table, part: Any, processed_rids: set[tuple[str, str]]) -> str:
    """
    Extract table rows and cells, applying line spacing correction,
    and placing in-cell images directly inside their originating cell.
    """
    table_rows = []

    for row in table.rows:
        row_cells = []
        for cell in row.cells:
            cell_text = fix_line_spacing(cell.text.strip())
            # Extract any embedded images directly within this cell
            cell_imgs = _extract_images_from_element(cell._tc, part, processed_rids)
            if cell_imgs:
                cell_text = (cell_text + " " + " ".join(cell_imgs)).strip()
            row_cells.append(cell_text)

        table_rows.append(row_cells)

    return format_table_as_markdown(table_rows).strip()


def extract_docx(path: Path) -> str:
    """
    Extract text, tables, and OCR image content from DOCX files in document order.
    - Preserves all multi-section headers and footers (deduplicating linked/shared ones).
    - Extracts header and footer images and tables using their respective part relationships.
    - Preserves in-order document flow of paragraphs, tables, and images.
    - Normalizes word spacing inside table cells and embeds in-cell OCR text.
    - Preserves paragraph structure in textboxes.
    """
    doc = Document(str(path))
    content_blocks: list[str] = []
    processed_rids: set[tuple[str, str]] = set()
    seen_headers_footers: set[str] = set()

    # 1. Section headers (across all sections, with header image & table extraction)
    for section in doc.sections:
        if section.header:
            header_lines = []
            if section.header.paragraphs:
                for p in section.header.paragraphs:
                    p_text = fix_line_spacing(p.text.strip())
                    img_texts = _extract_images_from_element(p._p, section.header.part, processed_rids)
                    combined = " ".join(filter(None, [p_text] + img_texts)).strip()
                    if combined:
                        header_lines.append(combined)

            if section.header.tables:
                for tbl in section.header.tables:
                    t_str = _extract_table_content(tbl, section.header.part, processed_rids)
                    if t_str:
                        header_lines.append(t_str)

            header_str = "\n".join(header_lines).strip()
            if header_str and header_str not in seen_headers_footers:
                seen_headers_footers.add(header_str)
                content_blocks.append(header_str)

    # 2. Document body in natural reading order
    body_part = doc.part
    for child in doc.element.body:
        tag = child.tag.split("}")[-1]

        # Paragraphs & headings
        if tag == "p":
            para = Paragraph(child, doc)
            p_text = fix_line_spacing(para.text.strip())
            if p_text:
                content_blocks.append(p_text)

            # Check for embedded textboxes
            tb_texts = _extract_textboxes_from_element(child)
            for tbt in tb_texts:
                content_blocks.append(tbt)

            # Check for inline or anchored images in this paragraph
            img_texts = _extract_images_from_element(child, body_part, processed_rids)
            for it in img_texts:
                content_blocks.append(it)

        # Tables
        elif tag == "tbl":
            table = Table(child, doc)
            table_str = _extract_table_content(table, body_part, processed_rids)
            if table_str.strip():
                content_blocks.append(table_str.strip())

    # 3. Any remaining images in relationships that weren't inside the body or tables
    for rel_id, rel in doc.part.rels.items():
        part_name = str(getattr(doc.part, "partname", "body"))
        rid_key = (part_name, rel_id)
        if "image" in rel.reltype and rid_key not in processed_rids:
            processed_rids.add(rid_key)
            if hasattr(rel.target_part, "blob"):
                img = cv2.imdecode(
                    np.frombuffer(rel.target_part.blob, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if img is not None:
                    ocr_res = run_ocr(img).strip()
                    if ocr_res:
                        content_blocks.append(ocr_res)

    # 4. Section footers (across all sections, with footer image & table extraction)
    for section in doc.sections:
        if section.footer:
            footer_lines = []
            if section.footer.paragraphs:
                for p in section.footer.paragraphs:
                    p_text = fix_line_spacing(p.text.strip())
                    img_texts = _extract_images_from_element(p._p, section.footer.part, processed_rids)
                    combined = " ".join(filter(None, [p_text] + img_texts)).strip()
                    if combined:
                        footer_lines.append(combined)

            if section.footer.tables:
                for tbl in section.footer.tables:
                    t_str = _extract_table_content(tbl, section.footer.part, processed_rids)
                    if t_str:
                        footer_lines.append(t_str)

            footer_str = "\n".join(footer_lines).strip()
            if footer_str and footer_str not in seen_headers_footers:
                seen_headers_footers.add(footer_str)
                content_blocks.append(footer_str)

    return "\n\n".join(content_blocks).strip()
