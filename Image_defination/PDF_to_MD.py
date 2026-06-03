from __future__ import annotations

import argparse
import os
import re
from typing import List, Optional, Tuple

import fitz          # PyMuPDF  – image / block detection
import pdfplumber    # table + text extraction


# Regex for figure / table caption labels, e.g. "Figure 1", "Fig. 2-3"
CAPTION_RE = re.compile(
    r"\b(Figure|Fig\.?|Table)\s+([0-9]+(?:[.\-][0-9]+)*)\b",
    re.IGNORECASE,
)


# Helpers

def _caption_in_text(text: str) -> str:
    """Return the first caption label found in *text*, or '' if none."""
    m = CAPTION_RE.search(text)
    return f"{m.group(1)} {m.group(2)}" if m else ""


def _rects_overlap(a: Tuple[float, float, float, float],
                   b: Tuple[float, float, float, float],
                   threshold: float = 0.30) -> bool:
    """True when rect *a* overlaps rect *b* by at least *threshold* of a's area."""
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = max((ax1 - ax0) * (ay1 - ay0), 1)
    return inter / area_a >= threshold


def _table_to_markdown(rows: List[List[Optional[str]]]) -> str:
    """Convert a list-of-rows (from pdfplumber) into a Markdown table string."""
    if not rows:
        return ""

    def clean(cell: Optional[str]) -> str:
        if not cell:
            return ""
        cell = re.sub(r"\r?\n", " ", cell)
        cell = re.sub(r" {2,}", " ", cell)
        return cell.replace("|", "\\|").strip()

    cleaned = [[clean(c) for c in row] for row in rows]
    num_cols = max(len(r) for r in cleaned)
    for row in cleaned:
        while len(row) < num_cols:
            row.append("")

    widths = [max(max(len(r[c]) for r in cleaned), 3) for c in range(num_cols)]

    def fmt_row(row: List[str]) -> str:
        return "| " + " | ".join(row[c].ljust(widths[c]) for c in range(num_cols)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    lines = [fmt_row(cleaned[0]), sep] + [fmt_row(r) for r in cleaned[1:]]
    return "\n".join(lines)


def _find_figure_label_near(img_rect: Tuple[float, float, float, float],
                             page_fitz: fitz.Page,
                             margin: float = 80.0) -> str:
    """
    Search text near *img_rect* (above/below within *margin* points) for a
    figure label.  Returns e.g. 'Figure 1-2' or '' if nothing found.
    """
    x0, y0, x1, y1 = img_rect
    search = fitz.Rect(0,
                       max(0, y0 - margin),
                       page_fitz.rect.width,
                       min(page_fitz.rect.height, y1 + margin))
    nearby = page_fitz.get_textbox(search)
    return _caption_in_text(nearby)


# Per-page rendering

def _render_page(plumber_page, fitz_page: fitz.Page) -> str:
    """
    Build the Markdown string for a single page.

    Strategy
    --------
    We collect all "items" as (y_position, markdown_text) tuples, then sort
    by y and join.

    Item types:
      • text block  → plain text line(s)
      • image block → 3-line gap: blank / figure-label / blank
      • table       → structured Markdown table
    """

    # 1. Detect tables via pdfplumber 
    pl_tables = plumber_page.extract_tables() or []
    pl_table_bboxes: List[Tuple[float, float, float, float]] = []
    table_items: List[Tuple[float, str]] = []

    for tbl_obj in plumber_page.find_tables():
        bbox = tbl_obj.bbox          # (x0, top, x1, bottom) in pdfplumber coords
        data = tbl_obj.extract()
        md = _table_to_markdown(data)
        if md:
            pl_table_bboxes.append(bbox)
            table_items.append((bbox[1], md))   # y0 = bbox[1] (top)

    #  2. Collect fitz blocks sorted by y 
    raw = fitz_page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
    blocks = sorted(raw.get("blocks", []), key=lambda b: b["bbox"][1])

    output_items: List[Tuple[float, str]] = []
    emitted_table_indices: set = set()

    for block in blocks:
        btype = block.get("type")
        bx0, by0, bx1, by1 = block["bbox"]
        b_rect = (bx0, by0, bx1, by1)

        # Check if this block overlaps with a table region
        overlapping_table_idx: Optional[int] = None
        for idx, tbl_bbox in enumerate(pl_table_bboxes):
            if _rects_overlap(b_rect, tbl_bbox):
                overlapping_table_idx = idx
                break

        if overlapping_table_idx is not None:
            # Emit the table once, keyed on first overlapping block
            if overlapping_table_idx not in emitted_table_indices:
                ty0, tmd = table_items[overlapping_table_idx]
                output_items.append((ty0, f"\n{tmd}\n"))
                emitted_table_indices.add(overlapping_table_idx)
            continue   # skip raw text inside table region

        #  3. Text block 
        if btype == 0:
            lines: List[str] = []
            for line in block.get("lines", []):
                line_text = "".join(
                    span.get("text", "") for span in line.get("spans", [])
                ).rstrip()
                lines.append(line_text)
            if lines:
                output_items.append((by0, "\n".join(lines)))

        #  4. Image block 
        elif btype == 1:
            label = _find_figure_label_near(b_rect, fitz_page)
            if not label:
                label = "Image"   # fallback when PDF has no caption nearby
            # 3-line gap: blank line, label, blank line
            output_items.append((by0, f"\nimage: {label}\n"))

    # ── 3. Emit tables that were never touched by a text/image block ─────────
    for idx, (ty0, tmd) in enumerate(table_items):
        if idx not in emitted_table_indices:
            output_items.append((ty0, f"\n{tmd}\n"))

    #  4. Sort by vertical position and concatenate 
    output_items.sort(key=lambda x: x[0])
    return "\n".join(chunk for _, chunk in output_items)


# Main conversion

def pdf_to_md(pdf_path: str, out_path: str, overwrite: bool = False) -> str:
    """Convert *pdf_path* → Markdown file at *out_path*. Returns *out_path*."""
    if not overwrite and os.path.exists(out_path):
        raise FileExistsError(
            f"Output already exists: {out_path}  (use --overwrite to replace)"
        )

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    fitz_doc = fitz.open(pdf_path)
    page_chunks: List[str] = []

    with pdfplumber.open(pdf_path) as plumber_doc:
        total = len(fitz_doc)
        for i in range(total):
            print(f"  Processing page {i + 1}/{total} …", end="\r", flush=True)
            fitz_page    = fitz_doc.load_page(i)
            plumber_page = plumber_doc.pages[i]
            chunk = _render_page(plumber_page, fitz_page)
            page_chunks.append(chunk)

    print()   # newline after progress indicator

    md = "\n\n---\n\n".join(page_chunks).strip() + "\n"
    # Collapse runs of 3+ blank lines → 2 blank lines
    md = re.sub(r"\n{4,}", "\n\n\n", md)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md)

    return out_path


# CLI

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Convert a PDF to Markdown.\n"
            "  • Text  → extracted as-is\n"
            "  • Images → 3-line gap with figure label on the middle line\n"
            "  • Tables → structured Markdown tables"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--pdf",       required=True, help="Input PDF path")
    ap.add_argument("--out",       required=True, help="Output Markdown path")
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite the output file if it already exists")
    args = ap.parse_args()

    print(f"Converting: {args.pdf}")
    out = pdf_to_md(pdf_path=args.pdf, out_path=args.out, overwrite=args.overwrite)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()