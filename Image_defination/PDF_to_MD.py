from __future__ import annotations

import argparse
import os
import re
import unicodedata
from collections import Counter
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

def _is_figure_caption(text: str) -> bool:
    """True if text starts with a Figure/Fig label (standalone caption, not heading)."""
    return bool(re.match(r"^\s*(?:Figure|Fig\.?)\s+[0-9A-Z]", text, re.IGNORECASE))

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
        if len(key) <= 4:
            continue
        if norm == key:
            return depth, True

    for key, depth in bare_map.items():
        if len(key) <= 4:
            continue
        if bare == key:
            return depth, True

    return 2, False


# ─── 2. Font / Visual Analysis Helpers ──────────────────────────────────────────

def _get_doc_body_font_size(fitz_doc: fitz.Document, sample_pages: int = 10) -> float:
    size_counts: Counter = Counter()
    total_pages = len(fitz_doc)
    step = max(1, total_pages // sample_pages)

    for i in range(0, min(total_pages, sample_pages * step), step):
        page = fitz_doc.load_page(i)
        raw = page.get_text("dict")
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    size = span.get("size", 0)
                    if size > 4:
                        size_counts[round(size * 2) / 2] += len(span.get("text", "").strip())

    if not size_counts:
        return 10.0
    return size_counts.most_common(1)[0][0]

def _block_font_info(block: dict) -> Tuple[float, bool]:
    sizes = []
    is_bold = False
    for line in block.get("lines", []):
        for span in line.get("spans", []):
            text = span.get("text", "").strip()
            if not text:
                continue
            size = span.get("size", 0)
            if size > 0:
                sizes.append(size)
            flags = span.get("flags", 0)
            font_name = span.get("font", "").lower()
            if (flags & 16) or any(b in font_name for b in ("bold", "-b", "bd", "heavy", "black")):
                is_bold = True

    avg_size = sum(sizes) / len(sizes) if sizes else 0.0
    return avg_size, is_bold

def _is_visually_heading(block: dict, body_size: float) -> bool:
    avg_size, is_bold = _block_font_info(block)
    if avg_size <= 0:
        return False

    is_larger = avg_size >= body_size * 1.05
    all_text = " ".join(
        "".join(sp.get("text", "") for sp in ln.get("spans", []))
        for ln in block.get("lines", [])
    ).strip()

    is_bold_heading = is_bold and len(all_text) < 200
    return is_larger or is_bold_heading


# ─── 3. Figure List Parser ───────────────────────────────────────────────────────

_FIG_LIST_NUM_RE = re.compile(r"^([A-Z]?\d+[-\.]\d+)$", re.IGNORECASE)
_FIG_LIST_NUM_BARE_RE = re.compile(r"^(\d+)$")
_FIG_LIST_PAGE_RE = re.compile(r"[.\s]{2,}(\d+)\s*$")

def _build_figure_page_map(pdf_path: str) -> Dict[str, int]:
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()

    figures_page_0 = None
    end_page_0 = None

    for i, (level, title, page) in enumerate(toc):
        t = title.strip().lower()
        if t in ("figures", "list of figures"):
            figures_page_0 = page - 1
        elif figures_page_0 is not None and t not in ("figures", "list of figures"):
            end_page_0 = page - 1
            break

    if figures_page_0 is None:
        print("  [Warning] 'Figures' section not found in TOC; figure detection will be limited.")
        return {}

    if end_page_0 is None:
        end_page_0 = figures_page_0 + 15

    figure_map: Dict[str, int] = {}

    for page_idx in range(figures_page_0, min(end_page_0, len(doc))):
        page = doc.load_page(page_idx)
        raw = page.get_text("dict")

        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue

            lines = [
                "".join(sp.get("text", "") for sp in ln.get("spans", [])).strip()
                for ln in block.get("lines", [])
            ]
            lines = [l for l in lines if l]

            if not lines:
                continue

            m_num = _FIG_LIST_NUM_RE.match(lines[0]) or _FIG_LIST_NUM_BARE_RE.match(lines[0])
            if m_num:
                fig_num = m_num.group(1)
                for ln in reversed(lines):
                    m_pg = _FIG_LIST_PAGE_RE.search(ln)
                    if m_pg:
                        pg = int(m_pg.group(1))
                        figure_map[f"figure {fig_num.lower()}"] = pg
                        break
                continue

            block_text = " ".join(lines)
            for m in re.finditer(r"\b([A-Z]?\d+[-\.]\d+)\s+[^\n]{3,}?[.\s]{2,}(\d+)(?=\s|$)", block_text, re.IGNORECASE):
                fig_num = m.group(1)
                pg = int(m.group(2))
                figure_map[f"figure {fig_num.lower()}"] = pg

            for m in re.finditer(r"(?:Figure|Fig\.?)\s+([A-Z]?\d+[-\.]\d+)[^\n]*?[.\s]{2,}(\d+)(?=\s|$)", block_text, re.IGNORECASE):
                fig_num = m.group(1)
                pg = int(m.group(2))
                figure_map[f"figure {fig_num.lower()}"] = pg

    print(f"  [Info] Parsed {len(figure_map)} figure entries from List of Figures.")
    return figure_map


# ─── 4. PDF Extraction Helpers ───────────────────────────────────────────────────

CAPTION_RE = re.compile(
    r"\b(Figure|Fig\.?|Table)\s+([0-9A-Z]+(?:[.\-][0-9]+)*)\b",
    re.IGNORECASE,
)

def _normalise_fig_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip()).lower()

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


# ─── 5. Section-number depth calculator ─────────────────────────────────────────

def _section_num_depth(num_str: str) -> int:
    parts = num_str.split(".")
    if parts[-1] == "0":
        return max(1, len(parts) - 1)
    return max(1, len(parts))


# ─── 6. Single Page Rendering ────────────────────────────────────────────────────

def _render_page(
    plumber_page,
    fitz_page: fitz.Page,
    full_map: Dict[str, int],
    bare_map: Dict[str, int],
    figure_page_map: Dict[str, int],
    page_num: int,
    body_size: float,
) -> str:
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
    emitted_figure_labels: set = set()

    for block in blocks:
        btype = block.get("type")
        bx0, by0, bx1, by1 = block["bbox"]
        b_rect = (bx0, by0, bx1, by1)

        # ── Table overlap check ──────────────────────────────────────────────────
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

        # ── Text block ───────────────────────────────────────────────────────────
        if btype == 0:
            raw_lines = [
                "".join(span.get("text", "") for span in line.get("spans", [])).strip()
                for line in block.get("lines", [])
            ]
            raw_lines = [rl for rl in raw_lines if rl]
            if not raw_lines:
                continue

            block_text = " ".join(raw_lines)
            is_fig_caption = _is_figure_caption(block_text)

            # ── Heading detection ────────────────────────────────────────────
            if not is_fig_caption:
                visually_heading = _is_visually_heading(block, body_size)
                m_sec = re.match(r"^([0-9]+(?:\.[0-9]+)*)\.?\s+(.+)$", block_text)
                has_section_prefix = m_sec is not None and len(block_text) < 200
                toc_depth, toc_matched = _lookup_depth(block_text, full_map, bare_map)

                is_tbl_caption = _is_table_caption(_clean(block_text))
                mid_period = bool(re.search(r"\w\.\s+\w", block_text))
                is_long_sentence = mid_period and len(block_text) > 60

                if (
                    visually_heading
                    and (has_section_prefix or toc_matched)
                    and not is_tbl_caption
                    and not is_long_sentence
                ):
                    if m_sec and has_section_prefix:
                        depth = min(_section_num_depth(m_sec.group(1)), 6)
                    else:
                        depth = min(toc_depth, 6)
                    hashes = "#" * max(1, depth)
                    output_items.append((by0, f"\n\n{hashes} {block_text}\n\n"))
                    continue  # heading emitted

            # ── Safe Vector Figure Fallback ──────────────────────────────────
            # Only trigger text-based image tags if the entire block is a short caption
            if is_fig_caption and len(block_text) < 150:
                label_match = re.search(r"(Figure|Fig\.?)\s+([0-9A-Z]+(?:[.\-][0-9]+)*)", block_text, re.IGNORECASE)
                if label_match:
                    label_raw = f"{label_match.group(1)} {label_match.group(2)}"
                    label_key = _normalise_fig_label(label_raw)
                    true_page = figure_page_map.get(label_key)
                    on_correct_page = (true_page is None) or (true_page == page_num)
                    
                    if on_correct_page and label_key not in emitted_figure_labels:
                        emitted_figure_labels.add(label_key)
                        # We use 'by0 - 0.1' so it sorts immediately before the text caption
                        output_items.append((by0 - 0.1, f"\n\nImage found: {label_raw}.\n\n"))

            # Regardless of what happened above, output the raw text payload normally
            output_items.append((by0, f"{block_text}\n"))

        # ── Raster image block (btype == 1) ──────────────────────────────────────
        elif btype == 1:
            label = _find_figure_label_near(b_rect, fitz_page)
            if label:
                label_key = _normalise_fig_label(label)
                true_page = figure_page_map.get(label_key)
                on_correct_page = (true_page is None) or (true_page == page_num)
                if on_correct_page and label_key not in emitted_figure_labels:
                    emitted_figure_labels.add(label_key)
                    output_items.append((by0, f"\n\nImage found: {label}.\n\n"))
            else:
                output_items.append((by0, f"\n\nImage found: Unknown Figure.\n\n"))

    for idx, (ty0, tmd) in enumerate(table_items):
        if idx not in emitted_table_indices:
            output_items.append((ty0, f"\n{tmd}\n"))

    # Sorting exactly by the Y-coordinate correctly stacks text and images logically
    output_items.sort(key=lambda x: x[0])
    return "\n".join(chunk for _, chunk in output_items)


# ─── 7. Main Process Flow ────────────────────────────────────────────────────────

def pdf_to_md(pdf_path: str, out_path: str, overwrite: bool = False) -> str:
    if not overwrite and os.path.exists(out_path):
        raise FileExistsError(f"Output already exists: {out_path} (use --overwrite to replace)")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print("Building TOC Map for Hierarchy Extraction...")
    full_map, bare_map = _build_toc_map(pdf_path)

    print("Building Figure→Page Map from List of Figures...")
    figure_page_map = _build_figure_page_map(pdf_path)

    fitz_doc = fitz.open(pdf_path)

    print("Estimating body font size...")
    body_size = _get_doc_body_font_size(fitz_doc)
    print(f"  [Info] Estimated body font size: {body_size}pt")

    page_chunks: List[str] = []

    with pdfplumber.open(pdf_path) as plumber_doc:
        total = len(fitz_doc)
        for i in range(total):
            print(f"  Processing page {i + 1}/{total} …", end="\r", flush=True)
            fitz_page = fitz_doc.load_page(i)
            plumber_page = plumber_doc.pages[i]
            chunk = _render_page(
                plumber_page, fitz_page,
                full_map, bare_map,
                figure_page_map, page_num=i + 1,
                body_size=body_size,
            )
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