#!/usr/bin/env python3
"""
Community–State Top-5 (CSV-only) — Minimal & Correct

Reads:
- JSON: node -> community (single int or list[int] for overlaps)
- Augmented GML: graph with node attribute 'state' (and edges)

Outputs:
- CSV: one row per community with Top-5 states (state + percent + count),
        plus dominant vs outside shares and total nodes
- TXT: tiny summary that says the per-community details are in the CSV

No maps. No Markdown tables. Super simple.
"""

# ========= Config =========
algorithm_path = "../algorithms/leiden/community_detection_outputs/communication"
JSON_PATH = algorithm_path + "/Communication_Network.gml_2_map.json"
AUGMENTED_GML_PATH = r"..\data_enrichment\Communication_Augmented.gml"
# AUGMENTED_GML_PATH = r"..\data_enrichment\Mobility_Augmented.gml"

REPORT_OUTDIR = algorithm_path
CSV_OUTDIR = algorithm_path

# choose your dataset!
# communication
# mobility

# ========= Imports =========
import os
import json
import pandas as pd
import networkx as nx

# ========= Helpers =========
def _basename(p: str) -> str:
    return os.path.basename(p) if p else ""

def _safe_name(p: str) -> str:
    # for filenames: replace spaces with underscores
    return _basename(p).replace(" ", "_") if p else ""

def load_community_json(json_path: str) -> dict:
    """Return dict[str, list[int]] mapping node_id -> list of community ids."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = {}
    if isinstance(data, dict):
        # {"1": 3, "2": [7,8], ...}
        for k, v in data.items():
            if isinstance(v, list):
                mapping[str(k)] = [int(x) for x in v]
            else:
                mapping[str(k)] = [int(v)]
    elif isinstance(data, list):
        # [{"id": "1", "community": 3}, {"id": "2", "community": [7,8]}, ...]
        for item in data:
            if not isinstance(item, dict):
                continue
            nid = item.get("id", item.get("node", item.get("name")))
            if nid is None:
                continue
            comm = item.get("community", item.get("communities"))
            if comm is None:
                continue
            if isinstance(comm, list):
                mapping[str(nid)] = [int(x) for x in comm]
            else:
                mapping[str(nid)] = [int(comm)]
    else:
        raise ValueError("Unsupported JSON format for community mapping.")
    return mapping

def load_augmented_gml(gml_path: str) -> nx.Graph:
    """Read GML (label as id), return graph with string node ids."""
    G = nx.read_gml(os.path.normpath(gml_path), label="label")
    return nx.relabel_nodes(G, {n: str(n) for n in G.nodes})

# ========= Core =========
def main():
    # Prepare output paths
    os.makedirs(CSV_OUTDIR, exist_ok=True)
    os.makedirs(REPORT_OUTDIR, exist_ok=True)
    json_name = _safe_name(JSON_PATH)
    gml_name = _safe_name(AUGMENTED_GML_PATH)
    csv_path = os.path.join(
        CSV_OUTDIR,
        f"validation_report_on_{json_name}_and-{gml_name}.csv",
    )
    summary_json_path = os.path.join(
        REPORT_OUTDIR,
        f"validation_report_on_{json_name}_and-{gml_name}.json",
    )

    # Load inputs
    if not os.path.exists(JSON_PATH):
        raise FileNotFoundError(f"JSON not found: {JSON_PATH}")
    if not os.path.exists(AUGMENTED_GML_PATH):
        raise FileNotFoundError(f"Augmented GML not found: {AUGMENTED_GML_PATH}")

    comm_map = load_community_json(JSON_PATH)
    G = load_augmented_gml(AUGMENTED_GML_PATH)

    # Build node DataFrame (only need id + state)
    node_rows = []
    for n, d in G.nodes(data=True):
        node_rows.append({"id": str(n), "state": d.get("state")})
    nodes_df = pd.DataFrame(node_rows)

    # Community DataFrame (primary community + overlap flag)
    def primary_comm(lst): return lst[0] if lst else None
    def is_overlap(lst): return isinstance(lst, list) and len(lst) > 1

    comm_df = pd.DataFrame({
        "id": list(comm_map.keys()),
        "communities": list(comm_map.values()),
    })
    comm_df["community"] = comm_df["communities"].apply(primary_comm)
    comm_df["overlap"] = comm_df["communities"].apply(is_overlap)

    # Merge & keep rows with a valid community
    df = nodes_df.merge(comm_df[["id", "community", "overlap"]], on="id", how="left")
    df = df.loc[df["community"].notna()].copy()
    df["community"] = df["community"].astype(int)
    df["state"] = df["state"].fillna("Unknown")

    rows = []
    for comm, sub in df.groupby("community", sort=True):
        total = len(sub)
        vc = sub["state"].value_counts()
        top5 = (vc / total).head(5)  # state -> share (0..1)

        # percents first (پنج عدد)، سپس پنج اسم ایالت
        percents = [round(float(x * 100), 2) for x in top5.values.tolist()]
        states = top5.index.tolist()

        # پَد کردن تا پنج‌تا
        while len(percents) < 5: percents.append(0.0)
        while len(states) < 5: states.append("")

        rows.append({
            "community": int(comm),
            "p1": percents[0], "p2": percents[1], "p3": percents[2], "p4": percents[3], "p5": percents[4],
            "s1": states[0], "s2": states[1], "s3": states[2], "s4": states[3], "s5": states[4],
        })

    out_df = pd.DataFrame(rows).sort_values("community").reset_index(drop=True)
    out_df.to_csv(csv_path, index=False, encoding="utf-8")

    # Tiny JSON summary
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_communities = out_df["community"].nunique()
    overlap_nodes = int(comm_df["overlap"].fillna(False).sum())
    avg_dom = round(out_df["p1"].mean(), 2) if not out_df.empty else 0.0
    avg_out = round(100.0 - avg_dom, 2)

    summary = {
        "json_file": _basename(JSON_PATH),
        "augmented_gml_file": _basename(AUGMENTED_GML_PATH),
        "num_nodes": int(num_nodes),
        "num_edges": int(num_edges),
        "num_communities_with_data": int(num_communities),
        "overlap_nodes": int(overlap_nodes),
        "average_dominant_state_share_percent": float(avg_dom),
        "average_outside_dominant_share_percent": float(avg_out),
        "csv_path": os.path.abspath(csv_path),  # دیتاهای per-community داخل این CSV است
    }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Console
    print("Done.")
    print(f"CSV saved to: {csv_path}")
    print(f"Summary JSON saved to: {summary_json_path}")


if __name__ == "__main__":
    main()
