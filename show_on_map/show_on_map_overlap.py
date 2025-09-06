import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap

algorithm_directory = "../algorithms/bigCLAM/community_detection_outputs/communication"
JSON_PATH = algorithm_directory + "/comm_bigclam_like_overlap_20.json"

GML_PATH  = "../dataset/Communication_Network.gml/Communication_Network.gml"


SHP_STATE  = "downloadedMap/cb_2018_us_state_20m/cb_2018_us_state_20m.shp"
SHP_COUNTY = "downloadedMap/cb_2018_us_county_20m/cb_2018_us_county_20m.shp"

with open(JSON_PATH, encoding="utf-8") as f:
    raw = json.load(f)
membership = ({list(d.keys())[0]: list(d.values())[0] for d in raw}
              if isinstance(raw, list) else raw)

G = nx.read_gml(GML_PATH)
rows = []
for nid, data in G.nodes(data=True):
    key = str(nid)
    val = membership.get(key, data.get("class", 0))
    if isinstance(val, list):
        vals = [int(v) for v in val]
        overlap_len = len(vals)

        community = int(pd.Series(vals).value_counts().index[0]) if vals else int(data.get("class", 0))
    else:
        community = int(val); overlap_len = 1
    rows.append({"id": key, "lon": float(data["lon"]), "lat": float(data["lat"]),
                 "community": community, "overlap_len": overlap_len})
df = pd.DataFrame(rows)

states  = gpd.read_file(SHP_STATE).to_crs("EPSG:4326")
counties = gpd.read_file(SHP_COUNTY).to_crs("EPSG:4326")

pts = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df["lon"], df["lat"]), crs="EPSG:4326")
joined = gpd.sjoin(pts, counties[["GEOID","geometry"]], how="left", predicate="within")


dom = (joined.dropna(subset=["GEOID"])
              .groupby("GEOID")["community"]
              .agg(lambda s: int(s.value_counts().index[0]))
              .rename("dom_comm"))


ovr = (joined.dropna(subset=["GEOID"])
               .assign(is_multi = joined["overlap_len"] > 1)
               .groupby("GEOID")
               .agg(multi_rate=("is_multi","mean"),
                    n_points=("is_multi","size"))
               )

counties = counties.set_index("GEOID").join(dom).join(ovr).reset_index()


TOP_N = 20
top_vals = counties["dom_comm"].value_counts().sort_values(ascending=False).index[:TOP_N]
counties["dom_plot"] = counties["dom_comm"].where(counties["dom_comm"].isin(top_vals), -1)

n_colors = len(top_vals)
cmap_comm = (plt.cm.get_cmap("tab20", max(n_colors,1))
             if n_colors <= 20 else ListedColormap(plt.cm.gist_ncar(np.linspace(0,1,n_colors))))
val2idx = {v:i for i,v in enumerate(top_vals)}

fig, axes = plt.subplots(1,2, figsize=(17,9), constrained_layout=True)
minx, miny, maxx, maxy = -130, 23, -65, 50
for ax in axes:
    ax.set_facecolor("#cfd3d8")
    states.boundary.plot(ax=ax, color="#444", linewidth=0.7, zorder=2)
    ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_edgecolor("#222"); s.set_linewidth(1.3)

ax = axes[0]

counties[counties["dom_plot"]==-1].plot(ax=ax, facecolor="#d0d0d0", edgecolor="#777", linewidth=0.2)

for v in top_vals:
    sub = counties[counties["dom_plot"]==v]
    color = cmap_comm(val2idx[v])
    sub.plot(ax=ax, facecolor=color, edgecolor="#777", linewidth=0.2, label=f"Community {v}")
ax.set_title("Dominant community per county", fontsize=12)

handles, labels = ax.get_legend_handles_labels()
if len(handles)>12:
    handles, labels = handles[:12], labels[:12]
leg = ax.legend(handles, labels, fontsize=8, loc="lower left", frameon=True)
leg.get_frame().set_facecolor("#f4f4f4"); leg.get_frame().set_edgecolor("#666")


ax = axes[1]

C = counties.dropna(subset=["multi_rate"])

im = C.plot(ax=ax, column="multi_rate", cmap="magma", vmin=0, vmax=min(0.7, C["multi_rate"].max()),
            edgecolor="#777", linewidth=0.2, legend=False)
ax.set_title("Multi-membership rate per county", fontsize=12)

from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
sm = ScalarMappable(norm=Normalize(vmin=0, vmax=min(0.7, C["multi_rate"].max())), cmap="magma")
sm._A = []
cbar = fig.colorbar(sm, ax=ax, shrink=0.75, pad=0.01)
cbar.set_label("fraction of nodes with |membership| > 1", fontsize=9)

cap = f"based on {Path(JSON_PATH).name} & {Path(GML_PATH).name}  |  Top-{TOP_N} communities"
fig.suptitle(cap, fontsize=10)

out = f"{algorithm_directory}/county_level_summary.png"
Path(algorithm_directory).mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=280, facecolor=fig.get_facecolor(), bbox_inches="tight")
print("Saved:", out)
plt.show()