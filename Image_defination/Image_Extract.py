from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import fitz  # PyMuPDF


@dataclass(frozen=True)
class ImageRef:
    image_id: str
    page: int
    bbox: List[float]
    caption: str
    relationship_id: Any = None
    source_file: Optional[str] = None


def _parse_image_refs(chunk: Dict[str, Any]) -> List[ImageRef]:
    if not chunk.get("has_image_refs"):
        return []
    refs = chunk.get("image_refs") or []
    out: List[ImageRef] = []
    for r in refs:
        out.append(
            ImageRef(
                image_id=r["image_id"],
                page=int(r["page"]),  # stored as 1-indexed
                bbox=list(r["bbox"]),
                caption=r.get("caption", ""),
                relationship_id=r.get("relationship_id"),
                source_file=r.get("source_file"),
            )
        )
    return out


def extract_images(
    chunks_path: str,
    pdf_path: str,
    out_dir: str,
    dpi: int = 200,
    bbox_pad: float = 2.0,
    overwrite: bool = False,
    limit: Optional[int] = None,
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    image_refs: List[ImageRef] = []
    for chunk in chunks:
        image_refs.extend(_parse_image_refs(chunk))

    if limit is not None:
        image_refs = image_refs[:limit]

    if not image_refs:
        print("No image refs found (has_image_refs=false for all chunks).")
        return

    doc = fitz.open(pdf_path)
    # PyMuPDF pages are 0-indexed; bbox/page are stored as 1-indexed in your JSON.

    zoom = dpi / 72.0

    extracted = 0
    for ref in image_refs:
        page_index = ref.page - 1
        if page_index < 0 or page_index >= len(doc):
            print(f"[skip] image_id={ref.image_id} invalid page={ref.page}")
            continue

        x0, y0, x1, y1 = ref.bbox
        # Add a small padding in PDF coordinates.
        x0 -= bbox_pad
        y0 -= bbox_pad
        x1 += bbox_pad
        y1 += bbox_pad
        # Ensure positive width/height.
        if x1 <= x0 or y1 <= y0:
            print(f"[skip] image_id={ref.image_id} invalid bbox={ref.bbox}")
            continue

        page = doc.load_page(page_index)

        rect = fitz.Rect(x0, y0, x1, y1)

        # Render only the clipped region.
        # Note: use matrix for dpi and still pass clip in page coordinates.
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)

        out_img = os.path.join(out_dir, f"{ref.image_id}.png")
        out_meta = os.path.join(out_dir, f"{ref.image_id}.json")

        if (not overwrite) and os.path.exists(out_img):
            continue

        pix.save(out_img)

        with open(out_meta, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "image_id": ref.image_id,
                    "page": ref.page,
                    "bbox": ref.bbox,
                    "caption": ref.caption,
                    "relationship_id": ref.relationship_id,
                    "source_file": ref.source_file,
                    "pdf_path": pdf_path,
                    "chunks_path": chunks_path,
                    "dpi": dpi,
                    "bbox_pad": bbox_pad,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        extracted += 1
        if extracted % 25 == 0:
            print(f"Extracted {extracted}/{len(image_refs)}...")

    print(f"Done. Extracted {extracted} images to: {out_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chunks",
        default="Image_defination/chunks_be1581d8.json",
        help="Path to chunks JSON containing image_refs.",
    )
    ap.add_argument(
        "--pdf",
        default="UCIE_1.1.pdf",
        help="Path to the PDF to crop from.",
    )
    ap.add_argument(    
        "--out",
        default="Image_defination/Images",
        help="Output directory for extracted images.",
    )
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--bbox-pad", type=float, default=2.0)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)

    args = ap.parse_args()

    extract_images(
        chunks_path=args.chunks,
        pdf_path=args.pdf,
        out_dir=args.out,
        dpi=args.dpi,
        bbox_pad=args.bbox_pad,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()

