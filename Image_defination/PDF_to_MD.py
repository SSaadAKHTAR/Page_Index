from __future__ import annotations

import argparse
import os
import re
import difflib
import unicodedata
import sys
from typing import Dict, List, Optional, Tuple

import fitz          # PyMuPDF  – image / block detection
import pdfplumber    # table + text extraction


# ─── 1. Hierarchy / TOC Helpers ──────────────────────────────────────────────────

_SYMBOL_RE = re.compile(r"[™©®†‡§¶°•·]")
_SECTION_NUM_RE = re.compile(
    r"^(?:[0-9]+(?:\.[0-9]+)*\.?\s*|[A-Z]+\s+[IVXLC]+\.\s*|[A-Z]\.\s+)",
    re.IGNORECASE,
)

def _clean(text: str) -> str:
    """Strip trademark symbols, collapse whitespace, to lowercase."""
    text = _SYMBOL_RE.sub("", text)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text

def _strip_section_num(text: str) -> str:
    return _SECTION_NUM_RE.sub("", text).strip()

def _is_table_caption(text: str) -> bool:
    return bool(re.match(r"^table\s+\d+[\.\s]", text.lower()))

def _build_toc_map(pdf_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()
    full_map: Dict[str, int] = {}
    bare_map: Dict[str, int] = {}

    for level, title, _page in toc:
        norm = _clean(title)
        bare = _strip_section_num(norm)
        if norm and (norm not in full_map or full_map[norm] > level):
            full_map[norm] = level
        if bare and len(bare) > 3 and (bare not in bare_map or bare_map[bare] > level):
            bare_map[bare] = level

    return full_map, bare_map

def _lookup_depth(heading: str, full_map: Dict[str, int], bare_map: Dict[str, int]) -> Tuple[int, bool]:
    norm = _clean(heading)
    bare = _strip_section_num(norm)

    if norm in full_map: return full_map[norm], True
    if bare and bare in bare_map: return bare_map[bare], True

    for key, depth in full_map.items():
        if norm.startswith(key) and len(key) > 4: return depth, True
        if key.startswith(norm) and len(norm) > 4: return depth, True

    for key, depth in bare_map.items():
        if len(key) <= 4: continue
        if bare.startswith(key) or key.startswith(bare): return depth, True

    candidates = list(full_map.keys())
    matches = difflib.get_close_matches(norm, candidates, n=1, cutoff=0.72)
    if matches: return full_map[matches[0]], True

    bare_candidates = list(bare_map.keys())
    matches = difflib.get_close_matches(bare, bare_candidates, n=1, cutoff=0.72)
    if matches: return bare_map[matches[0]], True

    return 2, False


# ─── 2. PDF Extraction Helpers ───────────────────────────────────────────────────

CAPTION_RE = re.compile(
    r"\b(Figure|Fig\.?|Table)\s+([0-9]+(?:[.\-][0-9]+)*)\b",
    re.IGNORECASE,
)

def _caption_in_text(text: str) -> str:
    m = CAPTION_RE.search(text)
    return f"{m.group(1)} {m.group(2)}" if m else ""

def _rects_overlap(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float], threshold: float = 0.30) -> bool:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0: return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1)
    return inter / area_a >= threshold

def _table_to_markdown(rows: List[List[Optional[str]]]) -> str:
    if not rows: return ""
    def clean(cell: Optional[str]) -> str:
        if not cell: return ""
        cell = re.sub(r"\r?\n", " ", cell)
        cell = re.sub(r" {2,}", " ", cell)
        return cell.replace("|", "\\|").strip()
    cleaned = [[clean(c) for c in row] for row in rows]
    num_cols = max(len(r) for r in cleaned)
    for row in cleaned:
        while len(row) < num_cols:
            row.append("")
    widths = [max(max(len(r[c]) for r in cleaned), 3) for c in range(num_cols)]
    def fmt_row(row: List[str]) -> str: return "| " + " | ".join(row[c].ljust(widths[c]) for c in range(num_cols)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    lines = [fmt_row(cleaned[0]), sep] + [fmt_row(r) for r in cleaned[1:]]
    return "\n".join(lines)

def _find_figure_label_near(img_rect: Tuple[float, float, float, float], page_fitz: fitz.Page, margin: float = 80.0) -> str:
    x0, y0, x1, y1 = img_rect
    search = fitz.Rect(0, max(0, y0 - margin), page_fitz.rect.width, min(page_fitz.rect.height, y1 + margin))
    nearby = page_fitz.get_textbox(search)
    return _caption_in_text(nearby)


# ─── 3. Single Page Rendering ────────────────────────────────────────────────────

def _render_page(plumber_page, fitz_page: fitz.Page, full_map: Dict[str, int], bare_map: Dict[str, int]) -> str:
    pl_table_bboxes: List[Tuple[float, float, float, float]] = []
    table_items: List[Tuple[float, str]] = []

    for tbl_obj in plumber_page.find_tables():
        bbox = tbl_obj.bbox
        data = tbl_obj.extract()
        md = _table_to_markdown(data)
        if md:
            pl_table_bboxes.append(bbox)
            table_items.append((bbox[1], md))

    raw = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    blocks = sorted(raw.get("blocks", []), key=lambda b: b["bbox"][1])

    output_items: List[Tuple[float, str]] = []
    emitted_table_indices: set = set()

    for block in blocks:
        btype = block.get("type")
        bx0, by0, bx1, by1 = block["bbox"]
        b_rect = (bx0, by0, bx1, by1)

        overlapping_table_idx: Optional[int] = None
        for idx, tbl_bbox in enumerate(pl_table_bboxes):
            if _rects_overlap(b_rect, tbl_bbox):
                overlapping_table_idx = idx
                break

        if overlapping_table_idx is not None:
            if overlapping_table_idx not in emitted_table_indices:
                ty0, tmd = table_items[overlapping_table_idx]
                output_items.append((ty0, f"\n{tmd}\n"))
                emitted_table_indices.add(overlapping_table_idx)
            continue

        if btype == 0:
            # Combine all lines in block to catch split titles like "1.0" \n "Introduction"
            raw_lines = ["".join(span.get("text", "") for span in line.get("spans", [])).strip() for line in block.get("lines", [])]
            raw_lines = [rl for rl in raw_lines if rl]
            if not raw_lines:
                continue
                
            block_text = " ".join(raw_lines)
            
            # Check numerical prefix on the combined block text
            depth_override = None
            # e.g., matches "1.0", "1.1.1", "1. "
            m_sec = re.match(r"^([0-9]+(?:\.[0-9]+)*)\.?\s+(.+)$", block_text)
            if m_sec:
                parts = m_sec.group(1).split('.')
                # if last part is '0' (like 1.0), it's treated as same level as 1
                if parts[-1] == '0':
                    calc_depth = max(1, len(parts) - 1)
                else:
                    calc_depth = max(1, len(parts))
                depth_override = min(calc_depth, 6)

            depth, matched = _lookup_depth(block_text, full_map, bare_map)
            is_aggressive_heading = (m_sec and len(block_text) < 150)
            
            if (matched or is_aggressive_heading) and not _is_table_caption(_clean(block_text)):
                final_depth = depth_override if depth_override is not None else depth
                hashes = "#" * max(1, final_depth)
                output_items.append((by0, f"\n{hashes} {block_text}\n"))
            else:
                # Output as separate lines if it's not a heading
                output_items.append((by0, "\n".join(raw_lines)))

        elif btype == 1:
            label = _find_figure_label_near(b_rect, fitz_page)
            if not label:
                label = "Image"
            output_items.append((by0, f"\nimage: {label}\n<!-- image -->\n"))

    for idx, (ty0, tmd) in enumerate(table_items):
        if idx not in emitted_table_indices:
            output_items.append((ty0, f"\n{tmd}\n"))

    output_items.sort(key=lambda x: x[0])
    return "\n".join(chunk for _, chunk in output_items)


# ─── 4. Main Process Flow ────────────────────────────────────────────────────────

def pdf_to_md(pdf_path: str, out_path: str, overwrite: bool = False) -> str:
    if not overwrite and os.path.exists(out_path):
        raise FileExistsError(f"Output already exists: {out_path} (use --overwrite to replace)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print("Building TOC Map for Hierarchy Extraction...")
    full_map, bare_map = _build_toc_map(pdf_path)

    fitz_doc = fitz.open(pdf_path)
    page_chunks: List[str] = []

    with pdfplumber.open(pdf_path) as plumber_doc:
        total = len(fitz_doc)
        for i in range(total):
            print(f"  Processing page {i + 1}/{total} …", end="\r", flush=True)
            fitz_page = fitz_doc.load_page(i)
            plumber_page = plumber_doc.pages[i]
            chunk = _render_page(plumber_page, fitz_page, full_map, bare_map)
            page_chunks.append(chunk)

    print()

    md = "\n\n<!-- page-break -->\n\n".join(page_chunks).strip() + "\n"
    md = re.sub(r"\n{4,}", "\n\n\n", md)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Convert a PDF to Hierarchical Markdown natively.")
    ap.add_argument("--pdf", required=True, help="Input PDF path")
    ap.add_argument("--out", required=True, help="Output Markdown path")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite the output file if it already exists")
    args = ap.parse_args()

    print(f"Converting: {args.pdf}")
    out = pdf_to_md(pdf_path=args.pdf, out_path=args.out, overwrite=args.overwrite)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()