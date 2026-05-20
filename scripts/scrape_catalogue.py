"""
Download product reference images for the biscuit catalogue using Bing Image search.
Each product gets 1 clean front-facing packet image saved to catalogue/images/.
"""

import json
import logging
import time
from pathlib import Path

from icrawler.builtin import BingImageCrawler

logging.getLogger("icrawler").setLevel(logging.WARNING)

CATALOGUE_PATH = Path(__file__).parent.parent / "catalogue" / "catalogue.json"
IMAGES_DIR = Path(__file__).parent.parent / "catalogue" / "images"

SEARCH_QUERIES = {
    "parle_g_250g": "Parle-G Gold biscuit packet product image",
    "parle_monaco_200g": "Parle Monaco Classic biscuit packet product",
    "parle_hide_seek_100g": "Parle Hide and Seek chocolate chip cookies packet",
    "britannia_good_day_butter_150g": "Britannia Good Day butter cookies packet",
    "britannia_good_day_cashew_150g": "Britannia Good Day cashew cookies packet",
    "britannia_bourbon_150g": "Britannia Bourbon chocolate cream biscuit packet",
    "britannia_marie_gold_250g": "Britannia Marie Gold biscuit packet",
    "britannia_5050_150g": "Britannia 50-50 Maska Chaska biscuit packet",
    "oreo_vanilla_120g": "Cadbury Oreo vanilla creme biscuit packet india",
    "oreo_chocolate_120g": "Cadbury Oreo chocolate creme biscuit packet india",
    "sunfeast_dark_fantasy_150g": "Sunfeast Dark Fantasy Choco Fills packet",
    "sunfeast_marie_light_200g": "Sunfeast Marie Light biscuit packet",
    "mcvities_digestive_250g": "McVities Digestive biscuit packet",
    "parle_krackjack_200g": "Parle Krackjack biscuit packet",
    "britannia_jimjam_150g": "Britannia Jim Jam cream biscuit packet",
    "britannia_milk_bikis_200g": "Britannia Milk Bikis biscuit packet",
    "sunfeast_mom_magic_200g": "Sunfeast Moms Magic butter biscuit packet",
    "priyagold_butter_bite_200g": "Priyagold Butter Bite biscuit packet",
    "britannia_tiger_glucose_250g": "Britannia Tiger glucose biscuit packet",
    "unibic_choco_chip_150g": "Unibic Choco Chip cookies packet",
}


def download_product_image(sku_id: str, query: str) -> bool:
    """Download a single product image using Bing Image search."""
    temp_dir = IMAGES_DIR / f"_temp_{sku_id}"
    temp_dir.mkdir(parents=True, exist_ok=True)

    crawler = BingImageCrawler(
        storage={"root_dir": str(temp_dir)},
        log_level=logging.WARNING,
    )
    crawler.crawl(
        keyword=query,
        max_num=3,
        min_size=(100, 100),
        filters={"type": "photo"},
    )

    downloaded = list(temp_dir.glob("*"))
    if downloaded:
        best = downloaded[0]
        target = IMAGES_DIR / f"{sku_id}.jpg"
        best.rename(target)
        # Clean up temp dir and remaining files
        for f in temp_dir.glob("*"):
            f.unlink()
        temp_dir.rmdir()
        print(f"  [OK] {sku_id} -> {target.name}")
        return True
    else:
        temp_dir.rmdir()
        print(f"  [FAIL] No images found for {sku_id}")
        return False


def main():
    with open(CATALOGUE_PATH) as f:
        catalogue = json.load(f)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    success = 0
    failed = []

    for product in catalogue:
        sku_id = product["sku_id"]
        target = IMAGES_DIR / f"{sku_id}.jpg"

        if target.exists():
            print(f"[EXISTS] {sku_id}")
            success += 1
            continue

        query = SEARCH_QUERIES.get(sku_id, product["product_name"] + " biscuit packet")
        print(f"[SEARCH] {product['product_name']}")

        if download_product_image(sku_id, query):
            success += 1
        else:
            failed.append(sku_id)

        time.sleep(1)

    print(f"\n{'='*50}")
    print(f"Downloaded: {success}/{len(catalogue)}")
    if failed:
        print(f"Failed: {failed}")
        print(f"Manually save images to: {IMAGES_DIR}/<sku_id>.jpg")
    else:
        print("All catalogue images ready!")


if __name__ == "__main__":
    main()
