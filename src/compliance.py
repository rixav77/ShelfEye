"""Compare detected shelf state against expected planogram."""

import json
from pathlib import Path


def check_compliance(
    pipeline_result: dict,
    planogram_path: str = "catalogue/planogram.json",
) -> dict:
    """Compare pipeline result against planogram and flag violations.

    Returns a compliance report with:
      - compliant: list of products in expected position
      - out_of_stock: expected products not found on shelf
      - wrong_position: expected products found but in wrong row
      - unexpected: detected products not in planogram
      - score: compliance percentage
    """
    with open(planogram_path) as f:
        planogram = json.load(f)

    expected_rows = planogram["rows"]

    # Build set of all expected SKUs and their expected rows
    expected = {}
    for row_str, skus in expected_rows.items():
        row = int(row_str)
        for sku in skus:
            if sku not in expected:
                expected[sku] = []
            expected[sku].append(row)

    # Build detected SKU -> rows mapping
    detected = {}
    for p in pipeline_result["products"]:
        if p["matched"] and p["sku_id"]:
            sku = p["sku_id"]
            if sku not in detected:
                detected[sku] = []
            detected[sku].append(p["row"])

    compliant = []
    out_of_stock = []
    wrong_position = []
    unexpected = []

    # Check each expected SKU
    for sku, expected_row_list in expected.items():
        if sku not in detected:
            out_of_stock.append({
                "sku_id": sku,
                "expected_rows": expected_row_list,
                "violation": "out_of_stock",
            })
        else:
            detected_rows = detected[sku]
            # Check if at least one detection is in an expected row
            in_correct_row = any(r in expected_row_list for r in detected_rows)
            if in_correct_row:
                compliant.append({
                    "sku_id": sku,
                    "expected_rows": expected_row_list,
                    "detected_rows": detected_rows,
                })
            else:
                wrong_position.append({
                    "sku_id": sku,
                    "expected_rows": expected_row_list,
                    "detected_rows": detected_rows,
                    "violation": "wrong_position",
                })

    # Check for products on shelf but not in planogram
    expected_skus = set(expected.keys())
    for sku, rows in detected.items():
        if sku not in expected_skus:
            unexpected.append({
                "sku_id": sku,
                "detected_rows": rows,
                "violation": "not_in_planogram",
            })

    # Also flag unmatched detections
    unmatched = [
        {"bbox": p["bbox"], "row": p["row"], "column": p["column"], "violation": "unrecognized"}
        for p in pipeline_result["products"]
        if not p["matched"]
    ]

    total_expected = len(expected)
    compliant_count = len(compliant)
    score = (compliant_count / total_expected * 100) if total_expected > 0 else 0

    report = {
        "planogram": planogram["name"],
        "image": pipeline_result["image"],
        "compliance_score": round(score, 1),
        "total_expected": total_expected,
        "summary": {
            "compliant": compliant_count,
            "out_of_stock": len(out_of_stock),
            "wrong_position": len(wrong_position),
            "unexpected": len(unexpected),
            "unrecognized": len(unmatched),
        },
        "compliant": compliant,
        "violations": {
            "out_of_stock": out_of_stock,
            "wrong_position": wrong_position,
            "unexpected": unexpected,
            "unrecognized": unmatched,
        },
    }

    print(f"[compliance] {pipeline_result['image']}: score={score:.1f}% "
          f"({compliant_count}/{total_expected} compliant, "
          f"{len(out_of_stock)} OOS, {len(wrong_position)} misplaced, "
          f"{len(unexpected)} unexpected)")

    return report


if __name__ == "__main__":
    import sys

    result_path = sys.argv[1] if len(sys.argv) > 1 else "outputs/result_shelf_03.json"
    planogram_path = sys.argv[2] if len(sys.argv) > 2 else "catalogue/planogram.json"

    with open(result_path) as f:
        result = json.load(f)

    report = check_compliance(result, planogram_path)

    out_path = result_path.replace("result_", "compliance_")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nCompliance report saved to {out_path}")
    print(json.dumps(report, indent=2))
