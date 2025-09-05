import json
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.colors import ListedColormap
from shapely.geometry import box

# ---------- تنظیم مسیر فایل‌ها ----------
algorithm_directory = "../algorithms/bigCLAM/community_detection_outputs/communication"
JSON_PATH = algorithm_directory + "/comm_bigclam_like_single_40.json"

# choose dataset!
# communication
GML_PATH  = "../dataset/Communication_Network.gml/Communication_Network.gml"

# mobility
# GML_PATH  = "../dataset/Mobility_Network.gml/Mobility_Network.gml"

# States
SHP_PATH  = "downloadedMap/cb_2018_us_state_20m/cb_2018_us_state_20m.shp"
# Counties
COUNTY_SHP   = "downloadedMap/cb_2018_us_county_20m/cb_2018_us_county_20m.shp"

# Natural Earth
NE_LAND      = "downloadedMap/ne_10m_land/ne_10m_land.shp"
NE_OCEAN     = "downloadedMap/ne_10m_ocean/ne_10m_ocean.shp"
NE_COAST     = "downloadedMap/ne_10m_coastline/ne_10m_coastline.shp"
NE_LAKES     = "downloadedMap/ne_10m_lakes/ne_10m_lakes.shp"
NE_ADMIN0    = "downloadedMap/ne_10m_admin_0_boundary_lines_land/ne_10m_admin_0_boundary_lines_land.shp"
NE_ADMIN1    = "downloadedMap/ne_10m_admin_1_states_provinces_lines/ne_10m_admin_1_states_provinces_lines.shp"

def _read_opt(path: str):
    p = Path(path)
    if not p.exists():
        print(f"[MISS] {path}")
        return None
    try:
        gdf = gpd.read_file(p).to_crs("EPSG:4326")
        print(f"[OK]   {path}")
        return gdf
    except Exception as e:
        print(f"[WARN] {p.name} load error: {e}")
        return None

# ---------- 1) نگاشت اجتماع ----------
with open(JSON_PATH, encoding="utf-8") as f:
    raw = json.load(f)
membership = ({list(d.keys())[0]: list(d.values())[0] for d in raw}
              if isinstance(raw, list) else raw)

# ---------- 2) گراف ----------
G = nx.read_gml(GML_PATH)
records = []
for n_id, data in G.nodes(data=True):
    records.append({
        "id": n_id,
        "lat": float(data["lat"]),
        "lon": float(data["lon"]),
        "community": int(membership.get(str(n_id), data.get("class", 0))),
    })
df = pd.DataFrame(records)

# ---------- 3) لایهٔ ایالت‌ها ----------
if not Path(SHP_PATH).exists():
    raise FileNotFoundError("Shapefile states پیدا نشد. مسیر SHP_PATH را چک کنید.")
states = gpd.read_file(SHP_PATH).to_crs("EPSG:4326")

# ---------- 4) پالت رنگ مناسب تعداد زیاد ----------
n_colors = df["community"].nunique()
cmap = (plt.cm.get_cmap("tab20", n_colors)
        if n_colors <= 20 else
        ListedColormap(plt.cm.gist_ncar(np.linspace(0, 1, n_colors))))

# ---------- 5) رسم به سبک مرجع با جزئیات بیشتر ----------
fig, ax = plt.subplots(figsize=(14, 9))

# رنگ‌ها
bg      = "#bdbdbd"   # پس‌زمینه (اقیانوس)
land_c  = "#7a7a7a"   # خشکی
border  = "#111111"   # خطوط مرزی تیره

fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

# برای بهبود سرعت، محدودهٔ نقشه را آماده می‌کنیم
minx, miny, maxx, maxy = -130, 23, -65, 50
extent_poly = box(minx, miny, maxx, maxy)

# 0) (اختیاری) اقیانوس‌ها
oceans = _read_opt(NE_OCEAN)
if oceans is not None:
    gpd.clip(oceans, extent_poly).plot(ax=ax, facecolor=bg, edgecolor="none", zorder=0)

# 1) خشکی جهانی (Land) – پایهٔ صحیح برای قاره
ne_land = _read_opt(NE_LAND)
if ne_land is not None:
    gpd.clip(ne_land, extent_poly).plot(ax=ax, facecolor=land_c, edgecolor="none", alpha=0.97, zorder=0.2)

# 2) مرز ایالت‌های آمریکا (روی Land)
states.boundary.plot(ax=ax, linewidth=1.2, edgecolor=border, zorder=1)

# 3) کانتی‌های آمریکا (خطوط ریز داخل کشور)
counties = _read_opt(COUNTY_SHP)
if counties is not None:
    gpd.clip(counties, extent_poly).boundary.plot(ax=ax, linewidth=0.35, edgecolor="#2e2e2e", zorder=1.05)

# 4) مرز بین‌المللی (admin_0)
adm0 = _read_opt(NE_ADMIN0)
if adm0 is not None:
    gpd.clip(adm0, extent_poly).plot(ax=ax, linewidth=1.0, edgecolor=border, zorder=1.1)

# 5) مرز داخلی استان/ایالت‌های جهان (admin_1)
adm1 = _read_opt(NE_ADMIN1)
if adm1 is not None:
    gpd.clip(adm1, extent_poly).plot(ax=ax, linewidth=0.6, edgecolor=border, zorder=1.1)

# 6) خط ساحلی دقیق
coast = _read_opt(NE_COAST)
if coast is not None:
    gpd.clip(coast, extent_poly).plot(ax=ax, linewidth=1.0, edgecolor=border, zorder=1.2)

# 7) دریاچه‌ها (روی خشکی با رنگ پس‌زمینه)
lakes = _read_opt(NE_LAKES)
if lakes is not None:
    gpd.clip(lakes, extent_poly).plot(ax=ax, facecolor=bg, edgecolor=border, linewidth=0.4, zorder=1.15)

# 8) نقاط شبکه (روی همه چیز)
for i, (comm, grp) in enumerate(df.groupby("community")):
    ax.scatter(grp["lon"], grp["lat"], s=6, color=cmap(i), alpha=0.65, linewidth=0,
               zorder=3, label=f"Community {comm}")

# ظاهر نهایی مثل مرجع
ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_edgecolor(border); s.set_linewidth(1.5)

# (اختیاری) لِجند
leg = ax.legend(fontsize=8, markerscale=1.6, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
leg.get_frame().set_facecolor("#d7d7d7"); leg.get_frame().set_edgecolor(border)

plt.tight_layout()

# ---------- کپشن + ذخیره با timestamp ----------
json_name = Path(JSON_PATH).name
gml_name  = Path(GML_PATH).name
caption = f"based on {json_name} and {gml_name}"
fig.text(0.5, -0.02, caption, ha="center", va="top", fontsize=10, color="black")

safe_json = json_name.replace(" ", "_")
safe_gml  = gml_name.replace(" ", "_")
out_name = f"{algorithm_directory}/community_map_based-on-{safe_json}_and-{safe_gml}.png"

plt.savefig(out_name, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_name}")
plt.show()
