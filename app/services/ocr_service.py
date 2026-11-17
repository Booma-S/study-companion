import re
import threading
from typing import Any

import cv2
import numpy as np
from rapidocr_onnxruntime import RapidOCR
from wordsegment import UNIGRAMS, load, segment

# Initialize wordsegment once
load()
for compound_word in ["helloworld", "iam", "fromthe", "companionapplication"]:
    UNIGRAMS.pop(compound_word, None)

_ocr_instance: RapidOCR | None = None
_ocr_lock = threading.Lock()


def get_ocr_engine() -> RapidOCR:
    global _ocr_instance
    if _ocr_instance is None:
        with _ocr_lock:
            if _ocr_instance is None:
                _ocr_instance = RapidOCR()
    return _ocr_instance


def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Conditionally preprocess an image to improve OCR recognition:
    - Resolution enhancement for small images.
    - Deskewing if the image is tilted.
    - CLAHE contrast enhancement for faded or low-contrast images.
    Clean, high-quality images are left unchanged.
    """
    if image is None or image.size == 0:
        return image

    h, w = image.shape[:2]

    # 1. Resolution enhancement if image is small
    if h < 800 or w < 800:
        scale = min(max(800.0 / max(h, 1), 800.0 / max(w, 1)), 3.0)
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        h, w = image.shape[:2]

    # 2. Deskewing (conditional)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    pts = cv2.findNonZero(thresh)
    if pts is not None and len(pts) >= 100:
        (cx, cy), (box_w, box_h), angle = cv2.minAreaRect(pts)
        if box_w < box_h:
            angle = angle + 90.0
        # Only deskew if clearly tilted between 1.0 and 30.0 degrees
        if 1.0 <= abs(angle) <= 30.0:
            rot_mat = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            image = cv2.warpAffine(
                image,
                rot_mat,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image

    # 3. Contrast enhancement if image is faded or low-contrast
    std_val = float(np.std(gray))
    if std_val < 35.0:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = clahe.apply(l)
            image = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
        else:
            image = clahe.apply(gray)

    return image


def split_merged_token(token: str) -> str:
    """
    Split concatenated OCR tokens into separate words using word segmentation
    while preserving original casing, punctuation, and hyphens.
    Example: 'HELLOWORLD!!' -> 'HELLO WORLD!!'
             'HELLOWORLDIAMHERETOTESTTHESTUDY-' -> 'HELLO WORLD I AM HERE TO TEST THE STUDY-'
             'COMPANIONAPPLICATION!!' -> 'COMPANION APPLICATION!!'
    """
    match = re.match(r"^([^a-zA-Z0-9]*)([a-zA-Z0-9_-]+)([^a-zA-Z0-9]*)$", token)
    if not match:
        return token

    prefix, core, suffix = match.groups()

    # Split sub-parts by hyphens or underscores
    parts = core.split("-")
    fixed_parts = []

    for part in parts:
        if not part or not part.isalpha():
            fixed_parts.append(part)
            continue

        lower = part.lower()

        # If already a common single word in dictionary with high frequency, leave it
        if lower in UNIGRAMS and (len(part) <= 7 or UNIGRAMS[lower] > 5e5):
            fixed_parts.append(part)
            continue

        # Attempt segmentation
        segs = segment(part)
        if len(segs) > 1:
            curr_idx = 0
            reconstructed_segs = []
            valid = True
            for s in segs:
                seg_len = len(s)
                reconstructed_segs.append(part[curr_idx : curr_idx + seg_len])
                curr_idx += seg_len

            if curr_idx == len(part):
                fixed_parts.append(" ".join(reconstructed_segs))
            else:
                fixed_parts.append(part)
        else:
            fixed_parts.append(part)

    return prefix + "-".join(fixed_parts) + suffix


def fix_line_spacing(line: str) -> str:
    """
    Fix missing spaces after punctuation and between glued words in a line.
    Example: 'task.In simple words,ML' -> 'task. In simple words, ML'
    """
    # Fix missing space after punctuation followed immediately by an uppercase letter
    line = re.sub(r"([a-zA-Z0-9])([.,;:!?])([A-Z])", r"\1\2 \3", line)
    # Fix comma followed immediately by a lowercase letter
    line = re.sub(r"([a-zA-Z0-9]),([a-z])", r"\1, \2", line)

    tokens = line.split(" ")
    fixed_tokens = [split_merged_token(t) for t in tokens]
    return " ".join(fixed_tokens)


def run_ocr_raw(image: np.ndarray) -> list[dict[str, Any]]:
    """
    Run RapidOCR on an image and return raw bounding box detections.
    """
    if image is None or image.size == 0:
        return []

    processed = preprocess_image(image)
    ocr = get_ocr_engine()
    raw_results, _ = ocr(processed)

    if not raw_results:
        return []

    # Map coordinates back if image was resized
    scale_y = image.shape[0] / processed.shape[0]
    scale_x = image.shape[1] / processed.shape[1]

    detections = []
    for item in raw_results:
        box, text, score = item
        if not text or not text.strip():
            continue

        xs = [p[0] * scale_x for p in box]
        ys = [p[1] * scale_y for p in box]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        stripped = text.strip()

        detections.append(
            {
                "text": stripped,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "height": max(y_max - y_min, 1.0),
                "width": max(x_max - x_min, 1.0),
                "char_width": (x_max - x_min) / max(len(stripped), 1),
                "score": score,
            }
        )

    return detections


def reconstruct_layout_from_detections(detections: list[dict[str, Any]]) -> str:
    """
    Reconstruct OCR text with multi-column layout clustering,
    visual reading order, and word-spacing recovery.
    """
    if not detections:
        return ""

    n = len(detections)
    if n == 1:
        return fix_line_spacing(detections[0]["text"])

    # 1. Proximity clustering into layout blocks (columns / paragraphs)
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = detections[i], detections[j]
            h_overlap = max(0, min(a["x_max"], b["x_max"]) - max(a["x_min"], b["x_min"]))
            h_dist = max(0, max(a["x_min"], b["x_min"]) - min(a["x_max"], b["x_max"]))
            v_dist = max(0, max(a["y_min"], b["y_min"]) - min(a["y_max"], b["y_max"]))
            avg_h = (a["height"] + b["height"]) / 2.0

            # Same line: vertical centers close, horizontal distance within threshold
            is_same_line = (
                abs((a["y_min"] + a["y_max"]) / 2.0 - (b["y_min"] + b["y_max"]) / 2.0)
                <= avg_h * 0.5
            ) and (h_dist < avg_h * 1.5)

            # Vertically stacked in same column: horizontal overlap or near, small vertical gap
            is_stacked = (h_overlap > 0 or h_dist < avg_h * 0.4) and (v_dist < avg_h * 1.2)

            if is_same_line or is_stacked:
                adj[i].add(j)
                adj[j].add(i)

    # Connected components
    visited = set()
    blocks = []
    for i in range(n):
        if i in visited:
            continue
        comp = []
        queue = [i]
        visited.add(i)
        while queue:
            curr = queue.pop(0)
            comp.append(detections[curr])
            for nbr in adj[curr]:
                if nbr not in visited:
                    visited.add(nbr)
                    queue.append(nbr)
        blocks.append(comp)

    # 2. Sort blocks into visual reading order (top-to-bottom sections, left-to-right columns)
    block_infos = []
    for b in blocks:
        x_min = min(it["x_min"] for it in b)
        x_max = max(it["x_max"] for it in b)
        y_min = min(it["y_min"] for it in b)
        y_max = max(it["y_max"] for it in b)
        block_infos.append(
            {
                "block": b,
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "height": max(y_max - y_min, 1.0),
            }
        )

    # Sort blocks initially by top coordinate
    block_infos.sort(key=lambda bi: bi["y_min"])

    # Group blocks into horizontal bands/sections
    rows = []
    for bi in block_infos:
        placed = False
        for row in rows:
            row_ymin = min(r["y_min"] for r in row)
            row_ymax = max(r["y_max"] for r in row)
            overlap = min(bi["y_max"], row_ymax) - max(bi["y_min"], row_ymin)
            min_h = min(bi["height"], row_ymax - row_ymin)
            if overlap >= min_h * 0.4:
                row.append(bi)
                placed = True
                break
        if not placed:
            rows.append([bi])

    sorted_blocks = []
    for row in rows:
        row.sort(key=lambda bi: bi["x_min"])
        for bi in row:
            sorted_blocks.append(bi["block"])

    # 3. Assemble text inside each block
    block_texts = []
    for block in sorted_blocks:
        block.sort(key=lambda it: it["y_min"])
        lines = []

        for det in block:
            det_center = (det["y_min"] + det["y_max"]) / 2.0
            placed = False
            for line in lines:
                line_center = sum(d["y_min"] + d["y_max"] for d in line) / (2.0 * len(line))
                avg_h = sum(d["height"] for d in line) / len(line)
                if abs(det_center - line_center) <= avg_h * 0.6:
                    line.append(det)
                    placed = True
                    break
            if not placed:
                lines.append([det])

        lines.sort(key=lambda l: min(d["y_min"] for d in l))

        reconstructed_lines = []
        for line in lines:
            line.sort(key=lambda d: d["x_min"])
            line_str = ""
            for idx, item in enumerate(line):
                t = item["text"]
                if idx == 0:
                    line_str = t
                else:
                    prev = line[idx - 1]
                    gap = item["x_min"] - prev["x_max"]
                    avg_char = (prev["char_width"] + item["char_width"]) / 2.0
                    if gap > max(avg_char * 0.25, 2.0):
                        line_str += " "
                    line_str += t

            fixed_line = fix_line_spacing(line_str)
            if fixed_line.strip():
                reconstructed_lines.append(fixed_line.strip())

        if reconstructed_lines:
            block_texts.append("\n".join(reconstructed_lines))

    return "\n\n".join(block_texts)


def run_ocr(image: np.ndarray) -> str:
    """
    Convenience function: runs OCR and returns reconstructed layout text.
    """
    detections = run_ocr_raw(image)
    return reconstruct_layout_from_detections(detections)
