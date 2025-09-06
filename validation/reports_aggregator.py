#!/usr/bin/env python3
"""
Aggregate validation_report_on_*.json files into one CSV.

- Recursively scans INPUT_DIR for files: validation_report_on_*.json
- Reads each JSON, adds `algorithm` (segment after 'algorithms/' in file path)
- Writes a single CSV with all fields + `algorithm` + `source_json_path`
"""

# ====== Config (edit these) ======
INPUT_DIR = r"../algorithms"   # پوشه‌ای که داخلش این JSONها هست (یا ریشهٔ بزرگ‌تر)
OUTPUT_CSV = r"all_validation_reports.csv"  # جای خروجی CSV نهایی

# ====== Imports ======
import os
import json
import pandas as pd
from glob import glob

# ====== Helpers ======
def extract_algorithm_from_path(path: str) -> str:
    """Return the segment right after 'algorithms' in a normalized path, else ''."""
    parts = os.path.normpath(path).split(os.sep)
    try:
        i = parts.index('algorithms')
        if i + 1 < len(parts):
            return parts[i + 1]
    except ValueError:
        pass
    return ''

def safe_read_json(fp: str):
    try:
        with open(fp, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"[skip] Could not read JSON: {fp} ({e})")
        return None

def ensure_parent_dir(p: str):
    d = os.path.dirname(os.path.abspath(p))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

# ====== Main ======
def main():
    # find all summary jsons
    pattern = os.path.join(os.path.abspath(INPUT_DIR), "**", "validation_report_on_*.json")
    files = glob(pattern, recursive=True)

    rows = []
    for fp in files:
        data = safe_read_json(fp)
        if not isinstance(data, dict):
            continue

        row = dict(data)  # copy all fields as-is
        row["algorithm"] = extract_algorithm_from_path(fp)

        # if algorithm not found in file path, try csv_path field
        if not row["algorithm"]:
            csv_path = data.get("csv_path", "")
            if csv_path:
                row["algorithm"] = extract_algorithm_from_path(csv_path)

        row["source_json_path"] = os.path.abspath(fp)
        rows.append(row)

    if not rows:
        print("No matching summary JSON files found.")
        return

    df = pd.DataFrame(rows)

    # choose a friendly column order (present columns only)
    preferred = [
        "algorithm",
        "json_file",
        "augmented_gml_file",
        "num_nodes",
        "num_edges",
        "num_communities_with_data",
        "overlap_nodes",
        "average_dominant_state_share_percent",
        "average_outside_dominant_share_percent",
        "csv_path",
        "source_json_path",
    ]
    cols = [c for c in preferred if c in df.columns] + [c for c in df.columns if c not in preferred]

    df = df[cols].sort_values(by=[c for c in ["algorithm", "json_file"] if c in df.columns]).reset_index(drop=True)

    ensure_parent_dir(OUTPUT_CSV)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Saved: {os.path.abspath(OUTPUT_CSV)}")
    print(f"Rows: {len(df)} | Columns: {len(df.columns)}")

if __name__ == "__main__":
    main()
