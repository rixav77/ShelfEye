"""Download Indian retail biscuit shelf images for testing."""

import logging
import urllib.request
from pathlib import Path

from icrawler.builtin import BingImageCrawler, FlickrImageCrawler

logging.getLogger("icrawler").setLevel(logging.WARNING)

SHELF_DIR = Path(__file__).parent.parent / "shelf_images"

FLICKR_API_KEY = "f7a65c6e0821d0447e66cb6df83b8a4e"


def download_url(url: str, save_path: Path) -> bool:
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"
    }
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = resp.read()
            if len(data) < 5000:
                return False
            with open(save_path, "wb") as f:
                f.write(data)
            return True
    except Exception:
        return False


def main():
    SHELF_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(SHELF_DIR.glob("shelf_*.jpg"))
    next_idx = max((int(f.stem.split("_")[1]) for f in existing), default=0) + 1

    # Bing queries — more varied to avoid duplicates
    bing_queries = [
        "biscuit shelf rack Indian supermarket Parle Britannia packets",
        "Indian store shelf display cookies biscuits multiple brands",
        "grocery biscuit shelf India organized rows front",
    ]

    for query in bing_queries:
        temp_dir = SHELF_DIR / "_temp"
        temp_dir.mkdir(exist_ok=True)

        print(f"[BING] {query}")
        crawler = BingImageCrawler(
            storage={"root_dir": str(temp_dir)},
            log_level=logging.WARNING,
        )
        crawler.crawl(keyword=query, max_num=3, min_size=(400, 300))

        for f in sorted(temp_dir.glob("*")):
            target = SHELF_DIR / f"shelf_{next_idx:02d}.jpg"
            f.rename(target)
            print(f"  [OK] {target.name}")
            next_idx += 1

        for f in temp_dir.glob("*"):
            f.unlink()
        temp_dir.rmdir()

    # Flickr search
    flickr_queries = [
        "Indian supermarket biscuit shelf",
        "India grocery store shelf products",
    ]

    for query in flickr_queries:
        temp_dir = SHELF_DIR / "_temp"
        temp_dir.mkdir(exist_ok=True)

        print(f"[FLICKR] {query}")
        crawler = FlickrImageCrawler(
            apikey=FLICKR_API_KEY,
            storage={"root_dir": str(temp_dir)},
            log_level=logging.WARNING,
        )
        crawler.crawl(keyword=query, max_num=2, min_size=(400, 300))

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
    print("Review images and delete bad ones manually.")


if __name__ == "__main__":
    main()
