#!/usr/bin/env python3
"""
md_hierarchy.py – Convert a flat-heading Markdown file into a proper
hierarchical heading structure, using the PDF table-of-contents (TOC)
to determine the depth of each section.

Usage:
    python md_hierarchy.py --md  output.md \
                            --pdf source.pdf \
                            --out hierarchical.md

Requirements:
    PyMuPDF  (pip install pymupdf)

Why some headings stay as ##:
    The PDF TOC (bookmarks) only contains entries the document author chose
    to bookmark.  Common un-bookmarked items include:
      • Cover / title page text
      • DISCLAIMER, REVISION HISTORY
      • "Table Of Contents", "Table Of Tables" navigational pages
      • Table captions  (e.g. "Table 3. Foo Bar")
    Those headings have no depth information available and are kept at ##.

Matching strategy (in priority order):
    1. Exact normalised match  (whitespace-collapsed, lower-cased, symbols stripped)
    2. Section-number-free exact match  ("1.1. Scope"  →  looks for "Scope" too)
    3. Heading starts with TOC title or vice-versa  (prefix match)
    4. difflib closest match  (cutoff 0.72 – loose enough for Unicode/punctuation diffs)
    5. Fall back to ##  (original depth)
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
import unicodedata
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF


# ─── normalisation helpers ──────────────────────────────────────────────────────

# Symbols that appear in document titles but not TOC entries
_SYMBOL_RE = re.compile(r"[™©®†‡§¶°•·]")
# Leading section-number prefix like "1.", "1.1.", "APPENDIX I.", "A."
_SECTION_NUM_RE = re.compile(
    r"^(?:[0-9]+(?:\.[0-9]+)*\.?\s*|[A-Z]+\s+[IVXLC]+\.\s*|[A-Z]\.\s+)",
    re.IGNORECASE,
)


def _clean(text: str) -> str:
    """Strip trademark symbols, collapse whitespace, to lowercase."""
    text = _SYMBOL_RE.sub("", text)
    # Normalise Unicode (e.g. curly quotes → straight, accented → base)
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text


def _strip_section_num(text: str) -> str:
    """Remove leading section number from a normalised heading, e.g. '1.1. foo' → 'foo'."""
    return _SECTION_NUM_RE.sub("", text).strip()


def _is_table_caption(text: str) -> bool:
    """True for headings that are really table captions: 'Table 3. Foo'."""
    return bool(re.match(r"^table\s+\d+[\.\s]", text.lower()))


# ─── TOC map builder ────────────────────────────────────────────────────────────

def _build_toc_map(pdf_path: str) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Returns two dicts keyed by normalised title:
      full_map  – normalised full title  → depth
      bare_map  – title with section-num stripped → depth
    """
    doc = fitz.open(pdf_path)
    toc: List[List] = doc.get_toc()
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


# ─── depth lookup ───────────────────────────────────────────────────────────────

def _lookup_depth(heading: str,
                  full_map: Dict[str, int],
                  bare_map: Dict[str, int]) -> Tuple[int, bool]:
    """Return (depth, matched)."""

    norm = _clean(heading)
    bare = _strip_section_num(norm)

    # 1 – exact (full normalised)
    if norm in full_map:
        return full_map[norm], True

    # 2 – exact on bare (no section number)
    if bare and bare in bare_map:
        return bare_map[bare], True

    # 3 – prefix / containment  (both directions, using normalised form)
    for key, depth in full_map.items():
        if norm.startswith(key) and len(key) > 4:
            return depth, True
        if key.startswith(norm) and len(norm) > 4:
            return depth, True

    # 4 – prefix / containment on bare keys
    for key, depth in bare_map.items():
        if len(key) <= 4:
            continue
        if bare.startswith(key) or key.startswith(bare):
            return depth, True

    # 5 – fuzzy on full normalised
    candidates = list(full_map.keys())
    matches = difflib.get_close_matches(norm, candidates, n=1, cutoff=0.72)
    if matches:
        return full_map[matches[0]], True

    # 6 – fuzzy on bare
    bare_candidates = list(bare_map.keys())
    matches = difflib.get_close_matches(bare, bare_candidates, n=1, cutoff=0.72)
    if matches:
        return bare_map[matches[0]], True

    return 2, False   # fallback: keep original ##


# ─── main conversion ────────────────────────────────────────────────────────────

def convert(md_path: str, pdf_path: str, out_path: str) -> None:
    full_map, bare_map = _build_toc_map(pdf_path)

    unmatched_genuine: list[str] = []   # genuinely not in TOC
    unmatched_tables:  list[str] = []   # table captions (expected to be unmatched)

    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    output_lines: list[str] = []
    total_headings = 0

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", line)
        if not heading_match:
            output_lines.append(raw_line)
            continue

        total_headings += 1
        heading_text = heading_match.group(2).strip()
        norm_text = _clean(heading_text)

        # Table captions are definitely not section headings – keep ##
        if _is_table_caption(norm_text):
            unmatched_tables.append(heading_text)
            output_lines.append(raw_line)   # unchanged
            continue

        depth, matched = _lookup_depth(heading_text, full_map, bare_map)

        if not matched:
            unmatched_genuine.append(heading_text)

        hashes = "#" * max(1, depth)
        output_lines.append(f"{hashes} {heading_text}\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)

    print(f"✓ Written → {out_path}")
    print(f"  TOC entries   : {len(full_map)}")
    print(f"  Total headings: {total_headings}")
    print(f"  Matched       : {total_headings - len(unmatched_genuine) - len(unmatched_tables)}")
    print(f"  Table captions: {len(unmatched_tables)}  (kept as-is, not section headings)")

    if unmatched_genuine:
        print(f"\n⚠ {len(unmatched_genuine)} heading(s) NOT found in PDF TOC (kept as ##):",
              file=sys.stderr)
        for h in unmatched_genuine[:20]:
            print(f"    • {h}", file=sys.stderr)
        if len(unmatched_genuine) > 20:
            print(f"    … and {len(unmatched_genuine) - 20} more.", file=sys.stderr)
        print("\n  These are typically: cover page text, DISCLAIMER, REVISION HISTORY,",
              file=sys.stderr)
        print("  'Table Of Contents' pages — items the PDF author did not bookmark.",
              file=sys.stderr)


# ─── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Re-level flat '##' Markdown headings into a proper hierarchy\n"
            "using the PDF table-of-contents for depth information.\n\n"
            "Works with any PDF that has a TOC (bookmarks)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--md",  required=True, help="Input flat Markdown file")
    ap.add_argument("--pdf", required=True, help="Source PDF (for TOC)")
    ap.add_argument("--out", required=True, help="Output Markdown file path")
    args = ap.parse_args()

    convert(md_path=args.md, pdf_path=args.pdf, out_path=args.out)


if __name__ == "__main__":
    main()
