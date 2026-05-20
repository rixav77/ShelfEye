"""Download shelf images using Google image crawler (different index than Bing)."""

import logging
from pathlib import Path

from icrawler.builtin import GoogleImageCrawler

logging.getLogger("icrawler").setLevel(logging.WARNING)

SHELF_DIR = Path(__file__).parent.parent / "shelf_images"

QUERIES = [
    "Indian store biscuit shelf organized Bourbon Oreo Marie Gold",
    "supermarket biscuit rack India multiple brands shelf",
    "Indian hypermarket biscuit cookies aisle front view",
]


def main():
    SHELF_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(SHELF_DIR.glob("shelf_*.jpg"))
    next_idx = max((int(f.stem.split("_")[1]) for f in existing), default=0) + 1

    for query in QUERIES:
        temp_dir = SHELF_DIR / "_temp"
        temp_dir.mkdir(exist_ok=True)

        print(f"[SEARCH] {query}")
        crawler = GoogleImageCrawler(
            storage={"root_dir": str(temp_dir)},
            log_level=logging.WARNING,
        )
        crawler.crawl(keyword=query, max_num=1, min_size=(400, 300))

        for f in sorted(temp_dir.glob("*")):
            target = SHELF_DIR / f"shelf_{next_idx:02d}.jpg"
            f.rename(target)
            print(f"  [OK] {target.name}")
            next_idx += 1

        for f in temp_dir.glob("*"):
            f.unlink()
        temp_dir.rmdir()

    total = len(list(SHELF_DIR.glob("shelf_*.jpg")))
    print(f"\nTotal shelf images: {total}")


if __name__ == "__main__":
    main()
