import networkx as nx
import igraph as ig
import leidenalg as la
from collections import Counter
import random
import json

# This code is developed for running the Leiden community detection algorithm
# You should follow the instructions below:
# 1. Install the needed dependencies (pip install -re requirements.txt)
# 2. Configure the setting (the gml file path, the required resolution, number of trials, etc)
# 3. Run the python file
# The output format is a json file like [{"node_id": X, "community": C}, ...]

# Settings
OUTPUT_TITLE = "Mobility_Network"
GML_PATH = "../../dataset/Mobility_Network.gml/Mobility_Network.gml"
RESOLUTION = 1.5  # γ (Larger ---> Smaller Communities)
N_TRIALS = 3  # Number of executions
SEED = 37  # Randomness seed

G = nx.read_gml(GML_PATH, label="id")
for u, v, d in G.edges(data=True):
    d["weight"] = float(d.get("weight", 1.0))

# Convert NetworkX to igraph
nodes = list(G.nodes())
node_index = {n: i for i, n in enumerate(nodes)}
edges = [(node_index[u], node_index[v]) for u, v in G.edges()]
g = ig.Graph(n=len(nodes), edges=edges, directed=False)
g.es["weight"] = [float(G[u][v].get("weight", 1.0)) for u, v in G.edges()]

# Run Leiden algorithm multiple times and select the best one
random.seed(SEED)
best_part = None
best_q = float("-inf")

for t in range(N_TRIALS):
    seed_t = SEED + t
    part = la.find_partition(
        g,
        la.RBConfigurationVertexPartition,  # The target function that should be optimized
        weights=g.es["weight"],
        resolution_parameter=RESOLUTION,
        seed=seed_t,
        n_iterations=-1,
    )
    if part.q > best_q:
        best_q = part.q
        best_part = part

membership = best_part.membership
node2comm = {node: membership[node_index[node]] for node in nodes}

# Printing the results and some samples
sizes = Counter(membership)
print(f"Best Q (objective): {best_q:.6f}")
print(f"#Communities: {len(sizes)}")
print("Top 10 community sizes:", sizes.most_common(10))

if 0 in node_index:
    print("node 0 → community", node2comm[0])

communities = {}
for n, cid in node2comm.items():
    communities.setdefault(cid, []).append(n)
communities = list(communities.values())
print("Sample community[0] (first 20 nodes):", sorted(communities[0])[:20])

# Saving the results to the file with format: [{"node_id": X, "community": C}, ...]
list_out = [
    {"node_id": int(n), "community": int(c)}
    for n, c in node2comm.items()
]

# Save file like format: {"node_id": community, ...}
dict_out = {str(n): int(c) for n, c in node2comm.items()}
with open(f"{OUTPUT_TITLE}_{RESOLUTION}_map.json", "w", encoding="utf-8") as f:
    json.dump(dict_out, f, ensure_ascii=False, indent=2)
