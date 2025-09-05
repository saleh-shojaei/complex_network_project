import json, math, random, numpy as np
import collections


INPUT_GML  = "../../dataset/Communication_Network.gml/Communication_Network.gml"
OUTPUT_EDG = "./community_detection_outputs/comm_for_bigclam.edgelist"
WEIGHT_KEY = "weight"
SYM_MODE   = "sum"
WT_TRANS   = "log1p"
PCTL_KEEP  = 70.0
KEEP_GCC   = True

EDGE_PATH  = OUTPUT_EDG
K_LIST     = [40]
N_ITER     = 80
SEED       = 42
HELDOUT_FRAC = 0.1
NEG_MULT   = 1
TAU        = 0.1
LOG_EVERY  = 10

OUTPUT_JSON_SINGLE   = "./community_detection_outputs/communication/comm_bigclam_like_single_40.json"
OUTPUT_JSON_OVERLAP  = "./community_detection_outputs/communication/comm_bigclam_like_overlap_40.json"

random.seed(SEED)
np.random.seed(SEED)

def _transform(w, mode='log1p'):
    if mode == 'log1p': return math.log1p(w)
    if mode == 'sqrt':  return math.sqrt(w)
    return float(w)

def stream_gml_edges(path, weight_key='weight'):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        in_edge = False
        src = tgt = None
        w = 1.0
        for line in f:
            s = line.strip()
            if s.startswith('edge'):
                in_edge = True; src = tgt = None; w = 1.0
                continue
            if in_edge and s == ']':
                if src is not None and tgt is not None:
                    yield (src, tgt, w)
                in_edge = False; src = tgt = None; w = 1.0
                continue
            if in_edge and s:
                parts = s.split(' ', 1)
                if len(parts) == 2:
                    k, v = parts[0], parts[1].strip('"')
                    if k in ('source','id','from','u'):
                        if src is None: src = v
                    elif k in ('target','to','v'):
                        if tgt is None: tgt = v
                    elif k == weight_key:
                        try: w = float(v)
                        except: w = 1.0

def _percentile(values, p):
    vals = sorted(values)
    if not vals: return float('inf')
    idx = int((p/100.0) * (len(vals)-1))
    return vals[idx]

def _giant_component(edges):
    adj = collections.defaultdict(list)
    for u,v in edges:
        adj[u].append(v); adj[v].append(u)
    visited, best = set(), set()
    for s in adj:
        if s in visited: continue
        stack = [s]; comp = {s}; visited.add(s)
        while stack:
            x = stack.pop()
            for y in adj[x]:
                if y not in visited:
                    visited.add(y); comp.add(y); stack.append(y)
        if len(comp) > len(best): best = comp
    return best

def convert_gml_for_bigclam(
    input_gml,
    output_edgelist,
    weight_key='weight',
    sym_mode='sum',
    weight_transform='log1p',
    threshold_percentile=70.0,
    keep_gcc=True
):
    fwd = collections.defaultdict(float)
    for u,v,w in stream_gml_edges(input_gml, weight_key):
        fwd[(u,v)] += w

    und = collections.defaultdict(float)
    seen = set()
    for (u,v), w_uv in fwd.items():
        if (u,v) in seen: continue
        w_vu = fwd.get((v,u), 0.0)
        a,b = (u,v) if u <= v else (v,u)
        if sym_mode == 'sum':
            w = w_uv + w_vu
            if w > 0: und[(a,b)] += w
        elif sym_mode == 'max':
            w = max(w_uv, w_vu)
            if w > 0: und[(a,b)] += w
        elif sym_mode == 'reciprocal':
            if w_uv > 0 and w_vu > 0:
                und[(a,b)] += min(w_uv, w_vu)
        seen.add((u,v)); seen.add((v,u))

    if not und:
        raise RuntimeError("No undirected edges after symmetrization.")

    transformed, vals = {}, []
    for e, w in und.items():
        tw = _transform(w, weight_transform)
        transformed[e] = tw; vals.append(tw)
    thr = _percentile(vals, threshold_percentile)
    edges = [e for e,t in transformed.items() if t >= thr and e[0] != e[1]]
    if not edges:
        raise RuntimeError("No edges after thresholding.")

    if keep_gcc:
        gcc = _giant_component(edges)
        edges = [(u,v) for (u,v) in edges if u in gcc and v in gcc]

    with open(output_edgelist, 'w', encoding='utf-8') as out:
        for u,v in edges:
            out.write(f"{u} {v}\n")
    print(f"[OK] wrote {len(edges)} edges to {output_edgelist}")


def load_undirected_edgelist(path):
    nodes = {}
    edges = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip().split()
            if len(s) < 2: continue
            u, v = s[0], s[1]
            if u == v: continue
            if u not in nodes: nodes[u] = len(nodes)
            if v not in nodes: nodes[v] = len(nodes)
            iu, iv = nodes[u], nodes[v]
            a, b = (iu, iv) if iu < iv else (iv, iu)
            edges.append((a, b))
    edges = list(set(edges))
    n = len(nodes)
    id2node = {i: u for u, i in nodes.items()}
    return n, edges, id2node

def split_train_test(edges, heldout_frac=0.1):
    deg = collections.defaultdict(int)
    for u,v in edges:
        deg[u]+=1; deg[v]+=1
    edges_shuffled = edges[:]
    random.shuffle(edges_shuffled)
    test, train = [], []
    target_test = int(len(edges) * heldout_frac)
    for (u,v) in edges_shuffled:
        if len(test) < target_test and deg[u] > 1 and deg[v] > 1:
            test.append((u,v)); deg[u]-=1; deg[v]-=1
        else:
            train.append((u,v))
    return train, test, set(train)

def sample_negatives(n, forbid_set, m):
    neg = set(); tries = 0
    while len(neg) < m and tries < m*50:
        i = random.randrange(n); j = random.randrange(n)
        if i==j: tries += 1; continue
        a,b = (i,j) if i<j else (j,i)
        if (a,b) in forbid_set or (a,b) in neg:
            tries += 1; continue
        neg.add((a,b))
    if len(neg) < m:
        for i in range(n):
            if len(neg) >= m: break
            for j in range(i+1, n):
                if (i,j) not in forbid_set:
                    neg.add((i,j))
                    if len(neg) >= m: break
    return list(neg)

def poisson_nmf_symmetric(n, edges, k=100, n_iter=100, seed=1, eps=1e-9, print_every=10):
    rng = np.random.default_rng(seed)
    F = rng.random((n, k)) + 1e-2
    I = np.array([i for i,j in edges], dtype=np.int64)
    J = np.array([j for i,j in edges], dtype=np.int64)
    for it in range(1, n_iter+1):
        lam = np.sum(F[I] * F[J], axis=1) + eps
        num = np.zeros_like(F)
        den = np.sum(F, axis=0) + eps
        contrib = (1.0 / lam)[:, None] * F[J, :]
        np.add.at(num, I, contrib)
        np.add.at(num, J, (1.0 / lam)[:, None] * F[I, :])
        F = F * (num / den)
        scale = np.sum(F, axis=0) + eps
        F = F / scale
        if print_every and it % print_every == 0:
            lam = np.sum(F[I] * F[J], axis=1) + eps
            loss = np.sum(lam - np.log(lam))
            print(f"[SymNMF k={k}] iter {it}/{n_iter}  loss≈{loss:.4e}")
    return F

def roc_auc_from_scores(pos_scores, neg_scores):
    m = len(pos_scores); n_neg = len(neg_scores)
    scores = np.array(pos_scores + neg_scores)
    order = scores.argsort()
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores)+1)
    uniq, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    start = 0
    for c in counts:
        end = start + c
        ranks[order][start:end] = np.mean(ranks[order][start:end])
        start = end
    R1 = ranks[:m].sum()
    return float((R1 - m*(m+1)/2) / (m*n_neg))

def average_precision_from_scores(pos_scores, neg_scores):
    y = np.array([1]*len(pos_scores) + [0]*len(neg_scores))
    s = np.array(pos_scores + neg_scores)
    order = (-s).argsort()
    y_sorted = y[order]
    tp = 0; precisions = []
    total_pos = y.sum()
    for i,lab in enumerate(y_sorted, start=1):
        if lab == 1:
            tp += 1
            precisions.append(tp / i)
    return 0.0 if total_pos == 0 else float(np.mean(precisions))

def eval_for_K(n, edges_train, edges_test, K, n_iter, neg_mult=1):
    F = poisson_nmf_symmetric(n, edges_train, k=K, n_iter=n_iter, seed=SEED, print_every=LOG_EVERY)
    def score_pairs(pairs):
        I = np.array([i for i,j in pairs]); J = np.array([j for i,j in pairs])
        return list(np.sum(F[I] * F[J], axis=1))
    neg = sample_negatives(n, forbid_set=set(edges_train), m=len(edges_test)*neg_mult)
    pos_scores = score_pairs(edges_test); neg_scores = score_pairs(neg)
    auc = roc_auc_from_scores(pos_scores, neg_scores)
    ap  = average_precision_from_scores(pos_scores, neg_scores)
    return auc, ap

def memberships_from_F(F, tau=0.1):
    row_sum = F.sum(axis=1, keepdims=True) + 1e-12
    Fn = F / row_sum
    overlap, single = [], []
    for i in range(F.shape[0]):
        comms = np.where(Fn[i] >= tau)[0]
        if comms.size == 0:
            cstar = int(np.argmax(Fn[i]))
            overlap.append([cstar]); single.append(cstar)
        else:
            overlap.append(list(map(int, comms))); single.append(int(np.argmax(Fn[i])))
    return overlap, single

if __name__ == "__main__":
    try:
        convert_gml_for_bigclam(
            input_gml=INPUT_GML,
            output_edgelist=OUTPUT_EDG,
            weight_key=WEIGHT_KEY,
            sym_mode=SYM_MODE,
            weight_transform=WT_TRANS,
            threshold_percentile=PCTL_KEEP,
            keep_gcc=KEEP_GCC
        )
    except FileNotFoundError:
        pass


    n, edges_all, id2node = load_undirected_edgelist(EDGE_PATH)
    print(f"Graph: |V|={n} |E|={len(edges_all)}")
    edges_train, edges_test, _ = split_train_test(edges_all, HELDOUT_FRAC)
    print(f"Split: |E_train|={len(edges_train)} |E_test|={len(edges_test)}")

    results = []
    for K in K_LIST:
        print(f"\n=== Evaluating K={K} ===")
        auc, ap = eval_for_K(n, edges_train, edges_test, K, N_ITER, NEG_MULT)
        print(f"K={K}  AUROC={auc:.4f}  AP={ap:.4f}")
        results.append((K, auc, ap))
    bestK, bestAUC, bestAP = sorted(results, key=lambda x: (x[2], x[1]), reverse=True)[0]
    print("\n>>> BEST K by (AP, then AUROC):", bestK, f"(AP={bestAP:.4f}, AUROC={bestAUC:.4f})")


    print(f"\nRefitting final model on ALL edges with K={bestK} ...")
    F_final = poisson_nmf_symmetric(n, edges_all, k=bestK, n_iter=max(N_ITER, 120), seed=SEED, print_every=LOG_EVERY)
    overlap, single = memberships_from_F(F_final, tau=TAU)

    single_dict = {str(id2node[i]): int(c + 1) for i, c in enumerate(single)}
    overlap_dict = {str(id2node[i]): [int(cc + 1) for cc in comms] for i, comms in enumerate(overlap)}

    with open(OUTPUT_JSON_SINGLE, "w", encoding="utf-8") as f:
        json.dump(single_dict, f, ensure_ascii=False)
    with open(OUTPUT_JSON_OVERLAP, "w", encoding="utf-8") as f:
        json.dump(overlap_dict, f, ensure_ascii=False)

    print("\nSaved:")
    print(" - Metrics:", results)
    print(" - Best K :", bestK)
    print(" - Overlap JSON :", OUTPUT_JSON_OVERLAP)
    print(" - Single  JSON :", OUTPUT_JSON_SINGLE)