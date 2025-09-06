#!/usr/bin/env python3
"""
Plots from aggregate CSV (no CLI) + map-collages:

- Per augmented_gml_file: charts + a collage of found map PNGs
- Per algorithm: charts + collages (overall and per augmented)
- Map images are discovered by glob pattern near source_json_path/csv_path:
    community_map_based-on-<json_file>_and-*.png

Required columns in CSV (others are kept/ignored for plotting):
    algorithm, json_file, augmented_gml_file, average_dominant_state_share_percent
Recommended extra columns for map discovery:
    source_json_path, csv_path
"""

# ====== CONFIG ======
INPUT_CSV  = r"./all_validation_reports.csv"   # CSV تجمیعی
OUTPUT_DIR = r"./figures"                          # ریشه‌ی خروجی نمودارها
DPI        = 150
COLLAGE_NCOLS = 3        # تعداد ستون‌های کلاژ
THUMB_W = 600            # عرض هدف هر تصویر در کلاژ (px)
PADDING = 10             # فاصله‌ها در کلاژ (px)
BG_COLOR = (255, 255, 255)  # پس‌زمینه سفید

# ====== Imports ======
import os, re, math
import pandas as pd
import numpy as np
from glob import glob
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ====== Helpers ======
def ensure_dir(p: str):
    if p and not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def sanitize(name: str) -> str:
    if not name: return "unknown"
    return re.sub(r"[^\w\-.]+", "_", str(name)).strip("_")

def parse_config_value(json_file: str):
    if not json_file: return None
    base = os.path.basename(str(json_file))
    m = re.search(r"\.gml[_\-]?([0-9]+(?:\.[0-9]+)?)", base, flags=re.IGNORECASE)
    if m:
        try: return float(m.group(1))
        except: return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", base)
    if m:
        try: return float(m.group(1))
        except: return None
    return None

def find_map_images_for_row(row) -> list[str]:
    """
    Try to locate PNG maps near the validation report/csv.
    Pattern: community_map_based-on-<json_file>_and-*.png
    """
    json_base = os.path.basename(str(row.get("json_file", "")))
    dirs = []

    # prefer directory of source_json_path
    src = row.get("source_json_path")
    if isinstance(src, str) and src:
        dirs.append(os.path.dirname(src))

    # fallback: directory of csv_path
    csp = row.get("csv_path")
    if isinstance(csp, str) and csp:
        dirs.append(os.path.dirname(csp))

    # unique, existing dirs
    dirs = [d for i, d in enumerate(dirs) if d and os.path.isdir(d) and d not in dirs[:i]]

    results = []
    for d in dirs:
        pat = os.path.join(d, f"community_map_based-on-{json_base}_and-*.png")
        matches = glob(pat)
        for m in matches:
            if os.path.isfile(m) and m not in results:
                results.append(m)
    return results

def collect_maps_for_group(sub_df: pd.DataFrame) -> list[str]:
    paths = []
    for _, row in sub_df.iterrows():
        paths.extend(find_map_images_for_row(row))
    # unique preserve order
    uniq = []
    for p in paths:
        if p not in uniq:
            uniq.append(p)
    return uniq

def build_collage(image_paths: list[str], out_path: str,
                  ncols: int = COLLAGE_NCOLS, thumb_w: int = THUMB_W,
                  padding: int = PADDING, bg=BG_COLOR) -> bool:
    """Create a simple grid collage. Returns True if saved, False if no images."""
    if not image_paths:
        return False

    # load images & resize to width=thumb_w (keep aspect)
    thumbs = []
    for p in image_paths:
        try:
            im = Image.open(p).convert("RGB")
            w, h = im.size
            scale = thumb_w / float(w)
            new_h = max(1, int(round(h * scale)))
            im_resized = im.resize((thumb_w, new_h), Image.LANCZOS)
            thumbs.append(im_resized)
        except Exception:
            continue

    if not thumbs:
        return False

    n = len(thumbs)
    rows = math.ceil(n / ncols)

    # compute row heights (max of each row)
    row_heights = []
    for r in range(rows):
        start = r * ncols
        end = min(start + ncols, n)
        hmax = max(im.size[1] for im in thumbs[start:end])
        row_heights.append(hmax)

    total_w = ncols * thumb_w + (ncols + 1) * padding
    total_h = sum(row_heights) + (rows + 1) * padding

    canvas = Image.new("RGB", (total_w, total_h), color=bg)

    y = padding
    idx = 0
    for r in range(rows):
        x = padding
        hmax = row_heights[r]
        for c in range(ncols):
            if idx >= n: break
            im = thumbs[idx]
            # center vertically within the row
            y_offset = (hmax - im.size[1]) // 2
            canvas.paste(im, (x, y + y_offset))
            x += thumb_w + padding
            idx += 1
        y += hmax + padding

    ensure_dir(os.path.dirname(out_path))
    canvas.save(out_path, format="PNG")
    return True

# ====== Plot primitives ======
def bar_config_vs_alg(ax, sub_df, title):
    pivot = (sub_df
             .groupby(["json_file", "algorithm"], as_index=False)["average_dominant_state_share_percent"]
             .mean()
             .pivot(index="json_file", columns="algorithm",
                    values="average_dominant_state_share_percent"))
    if pivot.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return
    pivot = pivot.sort_index()
    x = np.arange(len(pivot.index))
    width = 0.8 / max(1, pivot.shape[1])
    for i, col in enumerate(pivot.columns):
        vals = pivot[col].values
        ax.bar(x + i*width - (pivot.shape[1]-1)*width/2, vals, width=width, label=str(col))
    ax.set_xticks(x)
    ax.set_xticklabels([os.path.basename(s) for s in pivot.index], rotation=45, ha="right")
    ax.set_ylabel("Average dominant-state share (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2, frameon=True)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

def box_by_algorithm(ax, sub_df, title):
    groups, labels = [], []
    for alg, g in sub_df.groupby("algorithm"):
        vals = pd.to_numeric(g["average_dominant_state_share_percent"], errors="coerce").dropna().values
        if len(vals):
            groups.append(vals); labels.append(str(alg))
    if not groups:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return
    ax.boxplot(groups, labels=labels, showmeans=True)
    ax.set_ylabel("Average dominant-state share (%)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

def line_by_config_value(ax, sub_df, title):
    if not sub_df["config_value"].notna().any():
        ax.text(0.5, 0.5, "No numeric config value", ha="center", va="center"); return
    for alg, g in sub_df.groupby("algorithm"):
        gg = g.dropna(subset=["config_value"]).sort_values("config_value")
        if gg.empty: continue
        ax.plot(gg["config_value"], gg["average_dominant_state_share_percent"], marker="o", label=str(alg))
    ax.set_xlabel("Config value")
    ax.set_ylabel("Average dominant-state share (%)")
    ax.set_title(title)
    ax.legend(fontsize=8, frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

def hbar_algorithm_means(ax, sub_df, title):
    means = (sub_df.groupby("algorithm")["average_dominant_state_share_percent"]
             .mean().sort_values(ascending=True))
    if means.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return
    y = np.arange(len(means))
    ax.barh(y, means.values)
    ax.set_yticks(y); ax.set_yticklabels(means.index)
    ax.set_xlabel("Mean dominant-state share (%)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.6)

def bar_within_algorithm(ax, sub_df, title):
    means = (sub_df.groupby("json_file")["average_dominant_state_share_percent"]
             .mean().sort_values(ascending=False))
    if means.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center"); return
    x = np.arange(len(means))
    ax.bar(x, means.values)
    ax.set_xticks(x)
    ax.set_xticklabels([os.path.basename(s) for s in means.index], rotation=45, ha="right")
    ax.set_ylabel("Average dominant-state share (%)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

def line_within_algorithm(ax, sub_df, title):
    gg = sub_df.dropna(subset=["config_value"]).copy()
    if gg.empty:
        ax.text(0.5, 0.5, "No numeric config value", ha="center", va="center"); return
    gg = (gg.groupby("config_value", as_index=False)["average_dominant_state_share_percent"]
          .mean().sort_values("config_value"))
    ax.plot(gg["config_value"], gg["average_dominant_state_share_percent"], marker="o")
    ax.set_xlabel("Config value")
    ax.set_ylabel("Average dominant-state share (%)")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.6)

# ====== Main ======
def main():
    ensure_dir(OUTPUT_DIR)
    df = pd.read_csv(INPUT_CSV)

    # check required cols but keep all
    required = ["algorithm","json_file","augmented_gml_file","average_dominant_state_share_percent"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    df["average_dominant_state_share_percent"] = pd.to_numeric(
        df["average_dominant_state_share_percent"], errors="coerce")
    df = df.dropna(subset=["average_dominant_state_share_percent"])
    df["config_value"] = df["json_file"].apply(parse_config_value)

    # ---------- A) Per augmented_gml_file ----------
    for aug_name, sub in df.groupby("augmented_gml_file"):
        if sub.empty: continue
        tag = sanitize(os.path.splitext(os.path.basename(str(aug_name)))[0])
        base_title = f"Augmented: {os.path.basename(str(aug_name))}"
        outdir = os.path.join(OUTPUT_DIR, f"augmented__{tag}")
        ensure_dir(outdir)

        # Charts
        fig, ax = plt.subplots(figsize=(12,6))
        bar_config_vs_alg(ax, sub, f"{base_title} — Grouped Bar (config × algorithm)")
        fig.savefig(os.path.join(outdir, f"{tag}_bar_grouped.png"), dpi=DPI, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(9,6))
        box_by_algorithm(ax, sub, f"{base_title} — Box by Algorithm")
        fig.savefig(os.path.join(outdir, f"{tag}_box_alg.png"), dpi=DPI, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10,6))
        line_by_config_value(ax, sub, f"{base_title} — Line by Config Value")
        fig.savefig(os.path.join(outdir, f"{tag}_line_config.png"), dpi=DPI, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(8,6))
        hbar_algorithm_means(ax, sub, f"{base_title} — Algorithm Means")
        fig.savefig(os.path.join(outdir, f"{tag}_hbar_alg_means.png"), dpi=DPI, bbox_inches="tight"); plt.close(fig)

        # Collage of maps for this augmented group
        maps = collect_maps_for_group(sub)
        collage_path = os.path.join(outdir, f"{tag}_maps_collage.png")
        if build_collage(maps, collage_path):
            print(f"[augmented] collage saved: {collage_path}")
        else:
            print(f"[augmented] no maps found for collage: {tag}")

    # ---------- B) Per algorithm ----------
    for alg, sub_alg in df.groupby("algorithm"):
        if sub_alg.empty: continue
        alg_tag = sanitize(alg)
        alg_dir = os.path.join(OUTPUT_DIR, f"algorithm__{alg_tag}")
        ensure_dir(alg_dir)

        # overall (all augmented together) — charts
        fig, ax = plt.subplots(figsize=(12,6))
        bar_within_algorithm(ax, sub_alg, f"Algorithm: {alg} — Bar by Config (all augmented)")
        fig.savefig(os.path.join(alg_dir, f"{alg_tag}_bar_by_config_all_aug.png"), dpi=DPI, bbox_inches="tight"); plt.close(fig)

        fig, ax = plt.subplots(figsize=(10,6))
        line_within_algorithm(ax, sub_alg, f"Algorithm: {alg} — Line by Config Value (all augmented)")
        fig.savefig(os.path.join(alg_dir, f"{alg_tag}_line_by_config_all_aug.png"), dpi=DPI, bbox_inches="tight"); plt.close(fig)

        # overall collage of maps for this algorithm
        maps_alg_all = collect_maps_for_group(sub_alg)
        collage_alg_all = os.path.join(alg_dir, f"{alg_tag}_maps_collage_all_augmented.png")
        if build_collage(maps_alg_all, collage_alg_all):
            print(f"[algorithm] collage saved: {collage_alg_all}")
        else:
            print(f"[algorithm] no maps found for collage: {alg}")

        # per augmented within this algorithm
        for aug_name, sub_pair in sub_alg.groupby("augmented_gml_file"):
            tag = sanitize(os.path.splitext(os.path.basename(str(aug_name)))[0])
            pair_dir = os.path.join(alg_dir, f"augmented__{tag}")
            ensure_dir(pair_dir)

            fig, ax = plt.subplots(figsize=(12,6))
            bar_within_algorithm(ax, sub_pair, f"Algorithm: {alg} — Augmented: {os.path.basename(str(aug_name))} — Bar by Config")
            fig.savefig(os.path.join(pair_dir, f"{alg_tag}__{tag}_bar_by_config.png"), dpi=DPI, bbox_inches="tight"); plt.close(fig)

            fig, ax = plt.subplots(figsize=(10,6))
            line_within_algorithm(ax, sub_pair, f"Algorithm: {alg} — Augmented: {os.path.basename(str(aug_name))} — Line by Config Value")
            fig.savefig(os.path.join(pair_dir, f"{alg_tag}__{tag}_line_by_config.png"), dpi=DPI, bbox_inches="tight"); plt.close(fig)

            # collage for this algorithm+augmented pair
            maps_pair = collect_maps_for_group(sub_pair)
            collage_pair = os.path.join(pair_dir, f"{alg_tag}__{tag}_maps_collage.png")
            if build_collage(maps_pair, collage_pair):
                print(f"[algorithm-aug] collage saved: {collage_pair}")
            else:
                print(f"[algorithm-aug] no maps found for collage: {alg}/{tag}")

    print(f"All figures & collages under: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()
