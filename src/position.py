"""Assign row and column positions to detected products based on bounding box layout."""

import numpy as np


def _cluster_rows(y_centers: list[float], min_gap_ratio: float = 0.4) -> list[int]:
    """Cluster y-center values into shelf rows using gap-based splitting.

    Sorts detections by y-center, then splits into a new row whenever
    the gap between consecutive detections exceeds min_gap_ratio * median_box_height.
    """
    if not y_centers:
        return []

    indices = np.argsort(y_centers)
    sorted_y = np.array(y_centers)[indices]

    if len(sorted_y) == 1:
        row_labels = np.array([1])
        result = np.empty_like(row_labels)
        result[indices] = row_labels
        return result.tolist()

    gaps = np.diff(sorted_y)
    median_gap = np.median(gaps) if len(gaps) > 0 else 1.0
    threshold = max(median_gap * 1.5, 20)

    row_labels = np.ones(len(sorted_y), dtype=int)
    current_row = 1
    for i in range(1, len(sorted_y)):
        if gaps[i - 1] > threshold:
            current_row += 1
        row_labels[i] = current_row

    result = np.empty_like(row_labels)
    result[indices] = row_labels
    return result.tolist()


def assign_positions(detections: list[dict]) -> list[dict]:
    """Assign row and column positions to each detection.

    Rows are numbered top-to-bottom (row 1 = top shelf).
    Columns are numbered left-to-right within each row.
    """
    if not detections:
        return detections

    y_centers = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        y_centers.append((y1 + y2) / 2)

    rows = _cluster_rows(y_centers)

    for det, row in zip(detections, rows):
        det["row"] = row

    max_row = max(rows)
    for r in range(1, max_row + 1):
        row_dets = [(i, det) for i, det in enumerate(detections) if det["row"] == r]
        row_dets.sort(key=lambda x: x[1]["bbox"][0])
        for col, (i, det) in enumerate(row_dets, start=1):
            detections[i]["column"] = col

    row_counts = {}
    for r in rows:
        row_counts[r] = row_counts.get(r, 0) + 1

    print(f"[position] {len(detections)} detections -> {max_row} rows: {dict(sorted(row_counts.items()))}")
    return detections


if __name__ == "__main__":
    import sys
    import json
    sys.path.insert(0, "src")
    from detector import detect_products

    path = sys.argv[1] if len(sys.argv) > 1 else "shelf_images/shelf_01.jpg"
    dets = detect_products(path)
    dets = assign_positions(dets)

    for d in dets:
        print(f"  row={d['row']} col={d['column']}  bbox={d['bbox']}  conf={d['confidence']:.4f}")
