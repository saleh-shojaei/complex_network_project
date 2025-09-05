import networkx as nx
import community as community_louvain  # package: python-louvain
from collections import Counter
import json

# This code runs the Louvain (python-louvain) community detection algorithm.
# 1) pip install -r requirements.txt  (must include: networkx, python-louvain)
# 2) Configure settings (GML path, resolution gamma, trials, seed)
# 3) Run
# Output format: {"node_id": community, ...}   (JSON map)

# Settings
OUTPUT_TITLE = "Communication_Network"
ALGO_TAG = "louvain"
GML_PATH = "../../dataset/Communication_Network.gml/Communication_Network.gml"

RESOLUTION = 2       # γ (Larger ---> Smaller Communities)
N_TRIALS = 3             # Number of executions
SEED = 37                # Base seed

G = nx.read_gml(GML_PATH, label="id")
for u, v, d in G.edges(data=True):
    d["weight"] = float(d.get("weight", 1.0))

# Run Louvain multiple times (different random_state) and keep best modularity
best_partition = None
best_q = float("-inf")

for t in range(N_TRIALS):
    rs = SEED + t
    part = community_louvain.best_partition(
        G, weight="weight", resolution=RESOLUTION, random_state=rs
    )
    q = community_louvain.modularity(part, G, weight="weight")
    if q > best_q:
        best_q = q
        best_partition = part

node2comm = best_partition

# Reporting
sizes = Counter(node2comm.values())
print(f"Best modularity (Q): {best_q:.6f}")
print(f"#Communities: {len(sizes)}")
print("Top 10 community sizes:", sizes.most_common(10))
if 0 in node2comm:
    print("node 0 → community", node2comm[0])
if 0 in sizes:
    comm0_nodes = [n for n, c in node2comm.items() if c == 0]
    print("Sample community[0] (first 20 nodes):", sorted(comm0_nodes)[:20])

# Save outputs (JSON)
# map: {"node_id": community, ...}
dict_out = {str(n): int(c) for n, c in node2comm.items()}
fname_map = f"{OUTPUT_TITLE}_{ALGO_TAG}_{RESOLUTION}_map.json"
with open(fname_map, "w", encoding="utf-8") as f:
    json.dump(dict_out, f, ensure_ascii=False, indent=2)
print("Saved:", fname_map)
