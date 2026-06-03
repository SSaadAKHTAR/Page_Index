from __future__ import annotations

import argparse
import json
import os
import re

import fitz


def sanitize_filename(name: str) -> str:
    """
    Remove characters that are invalid in filenames on Windows/Linux/macOS.
    """
    name = re.sub(r'[<>:"/\\|?*]', '_', name)
    name = re.sub(r'\s+', ' ', name)
    return name.strip()


def extract_images(
    figures_json: str,
    pdf_path: str,
    out_dir: str,
    dpi: int = 200,
    bbox_pad: float = 2.0,
    overwrite: bool = False,
    limit: int | None = None,
):
    os.makedirs(out_dir, exist_ok=True)

    with open(figures_json, "r", encoding="utf-8") as f:
        figures = json.load(f)

    image_entries = [
        item
        for item in figures
        if item.get("chunk_type") == "image"
    ]

    if limit is not None:
        image_entries = image_entries[:limit]

    if not image_entries:
        print("No image entries found.")
        return

    doc = fitz.open(pdf_path)

    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    extracted = 0

    # Track filenames to avoid duplicates
    used_filenames = set()

    for item in image_entries:

        image_id = item["image_id"]
        page_num = item["page"]
        bbox = item["bbox"]

        page_index = page_num - 1

        if page_index < 0 or page_index >= len(doc):
            print(
                f"[skip] {image_id}: invalid page {page_num}"
            )
            continue

        x0, y0, x1, y1 = bbox

        x0 -= bbox_pad
        y0 -= bbox_pad
        x1 += bbox_pad
        y1 += bbox_pad

        if x1 <= x0 or y1 <= y0:
            print(
                f"[skip] {image_id}: invalid bbox"
            )
            continue

        page = doc.load_page(page_index)

        clip_rect = fitz.Rect(x0, y0, x1, y1)

        pix = page.get_pixmap(
            matrix=matrix,
            clip=clip_rect,
            alpha=False,
        )

        # ----------------------------
        # Generate filename from caption
        # ----------------------------
        caption = item.get("caption", "").strip()

        if caption:
            filename = sanitize_filename(caption)
        else:
            filename = image_id

        # Handle duplicate captions
        original_filename = filename
        counter = 1

        while filename in used_filenames:
            filename = f"{original_filename}_{counter}"
            counter += 1

        used_filenames.add(filename)

        out_img = os.path.join(
            out_dir,
            f"{filename}.png"
        )

        out_meta = os.path.join(
            out_dir,
            f"{filename}.json"
        )

        if (
            not overwrite
            and os.path.exists(out_img)
        ):
            continue

        pix.save(out_img)

        with open(out_meta, "w", encoding="utf-8") as mf:
            json.dump(
                {
                    "image_id": image_id,
                    "page": page_num,
                    "bbox": bbox,
                    "caption": caption,
                    "caption_bbox": item.get(
                        "caption_bbox"
                    ),
                    "subtype": item.get(
                        "subtype"
                    ),
                    "source_file": item.get(
                        "source_file"
                    ),
                    "dpi": dpi,
                    "bbox_pad": bbox_pad,
                },
                mf,
                indent=2,
                ensure_ascii=False,
            )

        extracted += 1

        print(
            f"[{extracted}] Saved: {filename}.png"
        )

        if extracted % 25 == 0:
            print(
                f"Extracted {extracted}/{len(image_entries)}"
            )

    print(
        f"\nDone. Extracted {extracted} images to '{out_dir}'."
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--figures",
        default="figures_1803e192.json",
        help="Path to figures JSON file."
    )

    parser.add_argument(
        "--pdf",
        default="UCIE_1.1.pdf",
        help="Path to PDF file."
    )

    parser.add_argument(
        "--out",
        default="Images",
        help="Output directory."
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
    )

    parser.add_argument(
        "--bbox-pad",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
    )

    args = parser.parse_args()

    extract_images(
        figures_json=args.figures,
        pdf_path=args.pdf,
        out_dir=args.out,
        dpi=args.dpi,
        bbox_pad=args.bbox_pad,
        overwrite=args.overwrite,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()