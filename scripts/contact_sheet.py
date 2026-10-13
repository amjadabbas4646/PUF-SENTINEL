"""Tile rendered PDF pages into contact sheets for visual review.

Usage:  python scripts/contact_sheet.py <folder with page PNGs> <prefix> [per_sheet]
Writes <folder>/sheet_XX.png with the pages in reading order.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image


def main() -> None:
    folder, prefix = Path(sys.argv[1]), sys.argv[2]
    per = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    pages = sorted(folder.glob(f"{prefix}-*.png"))
    cols = 4
    for s in range(0, len(pages), per):
        imgs = [Image.open(p) for p in pages[s:s + per]]
        w, h = imgs[0].size
        rows = (len(imgs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * w, rows * h), "white")
        for i, im in enumerate(imgs):
            sheet.paste(im, ((i % cols) * w, (i // cols) * h))
        out = folder / f"sheet_{s // per + 1:02d}.png"
        sheet.save(out)
        print(out)


if __name__ == "__main__":
    main()
