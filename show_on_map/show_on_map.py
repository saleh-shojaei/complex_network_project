import json
import networkx as nx
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib.colors import ListedColormap
from datetime import datetime

# ---------- تنظیم مسیر فایل‌ها ----------
JSON_PATH = "community_membership_map.json"
GML_PATH  = "Communication_Network.gml/Communication_Network.gml"
SHP_PATH  = "cb_2018_us_state_20m/cb_2018_us_state_20m.shp"

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
if n_colors <= 20:
    cmap = plt.cm.get_cmap("tab20", n_colors)
else:
    # پالت پیوسته و پرکنتراست روی پس‌زمینهٔ تیره
    cmap = ListedColormap(plt.cm.gist_ncar(np.linspace(0, 1, n_colors)))



# ---------- مسیر لایه‌های اضافی (اختیاری) ----------
COUNTY_SHP   = "cb_2018_us_county_20m/cb_2018_us_county_20m.shp"
NE_ADMIN0    = "ne_10m_admin_0_boundary_lines_land/ne_10m_admin_0_boundary_lines_land.shp"
NE_ADMIN1    = "ne_10m_admin_1_states_provinces_lines/ne_10m_admin_1_states_provinces_lines.shp"
NE_COAST     = "ne_10m_coastline/ne_10m_coastline.shp"
NE_LAKES     = "ne_10m_lakes/ne_10m_lakes.shp"
NE_LAND  = "ne_10m_land/ne_10m_land.shp"
NE_OCEAN = "ne_10m_ocean/ne_10m_ocean.shp"  # اختیاری

def _read_opt(path):
    p = Path(path)
    if p.exists():
        try:
            return gpd.read_file(p).to_crs("EPSG:4326")
        except Exception as e:
            print(f"[WARN] نتوانستم {p.name} را بخوانم:", e)
    return None

# ---------- 5) رسم به سبک مرجع با جزئیات بیشتر ----------
fig, ax = plt.subplots(figsize=(14, 9))

# رنگ‌ها
bg     = "#bdbdbd"   # پس‌زمینه (اقیانوس)
land   = "#7a7a7a"   # خشکی
border = "#111111"   # خطوط مرزی تیره

fig.patch.set_facecolor(bg)
ax.set_facecolor(bg)

# برای بهبود سرعت، محدودهٔ نقشه را آماده می‌کنیم
minx, miny, maxx, maxy = -130, 23, -65, 50
extent_poly = box(minx, miny, maxx, maxy)

# 0) (اختیاری) اقیانوس‌ها، اگر فایل ocean داری
oceans = _read_opt(NE_OCEAN)
if oceans is not None:
    oceans_clip = gpd.clip(oceans, extent_poly)
    oceans_clip.plot(ax=ax, facecolor=bg, edgecolor="none", zorder=0)

# 1) لایهٔ Land جهانی: این «خشکی» درست را می‌دهد
ne_land = _read_opt(NE_LAND)
if ne_land is not None:
    land_clip = gpd.clip(ne_land, extent_poly)
    land_clip.plot(ax=ax, facecolor=land_c, edgecolor="none", alpha=0.97, zorder=0.2)

# 2) ایالت‌های آمریکا روی Land
states.plot(ax=ax, facecolor="none", edgecolor=border, linewidth=1.2, zorder=1)

# 3) کانتی‌ها و مرزها/ساحل/دریاچه‌ها (اگر حاضرند)
counties = _read_opt("cb_2018_us_county_20m/cb_2018_us_county_20m.shp")
if counties is not None:
    counties_clip = gpd.clip(counties, extent_poly)
    counties_clip.boundary.plot(ax=ax, linewidth=0.35, edgecolor="#2e2e2e", zorder=1.05)

adm0 = _read_opt("ne_10m_admin_0_boundary_lines_land.shp")
if adm0 is not None:
    gpd.clip(adm0, extent_poly).plot(ax=ax, linewidth=1.0, edgecolor=border, zorder=1.1)

adm1 = _read_opt("ne_10m_admin_1_states_provinces_lines.shp")
if adm1 is not None:
    gpd.clip(adm1, extent_poly).plot(ax=ax, linewidth=0.6, edgecolor=border, zorder=1.1)

coast = _read_opt("ne_10m_coastline.shp")
if coast is not None:
    gpd.clip(coast, extent_poly).plot(ax=ax, linewidth=1.0, edgecolor=border, zorder=1.2)

lakes = _read_opt("ne_10m_lakes.shp")
if lakes is not None:
    gpd.clip(lakes, extent_poly).plot(ax=ax, facecolor=bg, edgecolor=border, linewidth=0.4, zorder=1.15)

# 4) نقاط شبکه (روی همه چیز)
for i, (comm, grp) in enumerate(df.groupby("community")):
    ax.scatter(grp["lon"], grp["lat"], s=6, color=cmap(i), alpha=0.65, linewidth=0, zorder=3,
               label=f"Community {comm}")

# ظاهر نهایی مثل مرجع
ax.set_xlim(minx, maxx); ax.set_ylim(miny, maxy)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_edgecolor(border); s.set_linewidth(1.5)

# (اختیاری) لِجند
leg = ax.legend(fontsize=8, markerscale=1.6, bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True)
leg.get_frame().set_facecolor("#d7d7d7"); leg.get_frame().set_edgecolor(border)

plt.tight_layout()

# ذخیره با timestamp (مثل قبل)
from datetime import datetime
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
out_name = f"community_map_{ts}.png"
plt.savefig(out_name, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Saved: {out_name}")