from pathlib import Path

import cv2
import numpy as np
from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph

from app.services.extractors.table_utils import format_table_as_markdown
from app.services.ocr_service import fix_line_spacing, run_ocr


def _extract_images_from_element(element, doc: Document, processed_rids: set[str]) -> list[str]:
    """
    Find embedded images within a Word XML element using .iter() and run OCR.
    """
    image_texts: list[str] = []

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

        if r_id and r_id not in processed_rids:
            processed_rids.add(r_id)
            part = doc.part.related_parts.get(r_id)
            if part and hasattr(part, "blob"):
                img = cv2.imdecode(
                    np.frombuffer(part.blob, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if img is not None:
                    ocr_res = run_ocr(img).strip()
                    if ocr_res:
                        image_texts.append(ocr_res)

    return image_texts


def _extract_textboxes_from_element(element) -> list[str]:
    """
    Find textboxes inside Word XML elements (e.g. w:txbxContent).
    """
    tb_texts: list[str] = []
    for el in element.iter():
        tag = el.tag.split("}")[-1]
        if tag == "txbxContent":
            # Extract all text inside this textbox
            t_nodes = [t.text for t in el.iter() if t.tag.endswith("t") and t.text]
            full_t = "".join(t_nodes).strip()
            if full_t:
                tb_texts.append(fix_line_spacing(full_t))
    return tb_texts


def extract_docx(path: Path) -> str:
    """
    Extract text, tables, and OCR image content from DOCX files in document order.
    """
    doc = Document(str(path))
    content_blocks: list[str] = []
    processed_rids: set[str] = set()

    # 1. Primary header (if any)
    for section in doc.sections:
        if section.header and section.header.paragraphs:
            header_lines = [
                fix_line_spacing(p.text.strip())
                for p in section.header.paragraphs
                if p.text.strip()
            ]
            if header_lines:
                content_blocks.append("\n".join(header_lines))
        break

    # 2. Document body in natural reading order
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
            img_texts = _extract_images_from_element(child, doc, processed_rids)
            for it in img_texts:
                content_blocks.append(it)

        # Tables
        elif tag == "tbl":
            table = Table(child, doc)
            table_rows = []
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells]
                table_rows.append(row_cells)
            md_table = format_table_as_markdown(table_rows)
            if md_table.strip():
                content_blocks.append(md_table.strip())

    # 3. Any remaining images in relationships that weren't inside the body
    for rel_id, rel in doc.part.rels.items():
        if "image" in rel.reltype and rel_id not in processed_rids:
            processed_rids.add(rel_id)
            if hasattr(rel.target_part, "blob"):
                img = cv2.imdecode(
                    np.frombuffer(rel.target_part.blob, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                if img is not None:
                    ocr_res = run_ocr(img).strip()
                    if ocr_res:
                        content_blocks.append(ocr_res)

    # 4. Primary footer (if any)
    for section in doc.sections:
        if section.footer and section.footer.paragraphs:
            footer_lines = [
                fix_line_spacing(p.text.strip())
                for p in section.footer.paragraphs
                if p.text.strip()
            ]
            if footer_lines:
                content_blocks.append("\n".join(footer_lines))
        break

    return "\n\n".join(content_blocks).strip()
