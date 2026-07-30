import json
import os
import time

import faiss
import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
LATENT = os.path.join(CACHE, "hnoca_latent.npz")
ATLAS = os.path.join(CACHE, "hnoca_atlas.npz")
OUT = os.path.join(CACHE, "hnoca_metrics.json")

K = 15
N_ANCHOR = int(os.environ.get("N_ANCHOR", "50000"))
N_PAIRS = int(os.environ.get("N_PAIRS", "400000"))
SEED = 777

# name -> (cache file, key in it, the space that cache was fit on)
LAYOUTS = {
    "flodr": ("hnoca_atlas.npz", "Y", "raw"),
    "flodr_w1": ("hnoca_atlas_w1.npz", "Y", "raw"),
    "flodr_std": ("hnoca_atlas_std.npz", "Y", "standardised"),
    "umap_published": ("hnoca_atlas.npz", "umap", "raw"),
    "umap_std": ("hnoca_umap_std.npz", "umap_std", "standardised"),
}


def knn(ref, query, k=K):
    ref = np.ascontiguousarray(ref, np.float32)
    index = faiss.IndexFlatL2(ref.shape[1])
    index.add(ref)
    # k+1 then drop self, which is at distance 0 but not always first under ties
    _, nbr = index.search(np.ascontiguousarray(query, np.float32), k + 1)
    return nbr


def recall_at_k(ref_hd, ref_ld, anchors, k=K):
    hd = knn(ref_hd, ref_hd[anchors], k)
    ld = knn(ref_ld, ref_ld[anchors], k)
    hits = 0
    for row, (a, b) in enumerate(zip(hd, ld)):
        i = anchors[row]
        hits += len(set(a[a != i][:k]) & set(b[b != i][:k]))
    return float(hits / (len(anchors) * k))


def cpd(ref_hd, ref_ld, rng, n_pairs=N_PAIRS):
    n = len(ref_hd)
    i = rng.integers(0, n, n_pairs)
    j = rng.integers(0, n, n_pairs)
    keep = i != j
    i, j = i[keep], j[keep]
    d_hd = np.linalg.norm(ref_hd[i] - ref_hd[j], axis=1)
    d_ld = np.linalg.norm(ref_ld[i] - ref_ld[j], axis=1)
    return float(spearmanr(d_hd, d_ld).statistic)


def main():
    lat = np.load(LATENT, allow_pickle=True)
    rows = np.asarray(np.load(ATLAS, allow_pickle=True)["rows"])
    Z = np.asarray(lat["Z"], np.float32)[rows]
    spaces = {
        "raw": np.ascontiguousarray(Z),
        "standardised": np.ascontiguousarray(
            ((Z - Z.mean(0)) / (Z.std(0) + 1e-9)).astype(np.float32)),
    }
    n = len(Z)

    found = {}
    for name, (fname, key, fit_space) in LAYOUTS.items():
        path = os.path.join(CACHE, fname)
        if not os.path.exists(path):
            print(f"skip {name}: no {fname}", flush=True)
            continue
        z = np.load(path, allow_pickle=True)
        if key not in z.files:
            print(f"skip {name}: no '{key}' in {fname}", flush=True)
            continue
        found[name] = (np.ascontiguousarray(np.asarray(z[key], np.float32)), fit_space)

    rng = np.random.default_rng(SEED)
    # one anchor and pair set for every layout, so nothing differs but the layout
    anchors = np.sort(rng.permutation(n)[: min(N_ANCHOR, n)])

    res = {"n": int(n), "d_latent": int(Z.shape[1]), "k": K,
           "n_anchors": int(len(anchors)), "n_pairs": N_PAIRS,
           "space_agreement_spearman": None, "layouts": {}}

    # how far apart the two candidate spaces actually are, which is why this matters
    pr = np.random.default_rng(SEED + 2)
    i, j = pr.integers(0, n, 200000), pr.integers(0, n, 200000)
    keep = i != j
    res["space_agreement_spearman"] = round(float(spearmanr(
        np.linalg.norm(spaces["raw"][i[keep]] - spaces["raw"][j[keep]], axis=1),
        np.linalg.norm(spaces["standardised"][i[keep]]
                       - spaces["standardised"][j[keep]], axis=1)).statistic), 4)
    print(f"raw vs standardised pair-distance Spearman: "
          f"{res['space_agreement_spearman']}\n", flush=True)

    for name, (Y, fit_space) in found.items():
        cell = {"fit_space": fit_space, "scores": {}}
        for space, P in spaces.items():
            t0 = time.perf_counter()
            cell["scores"][space] = {
                "recall15": round(recall_at_k(P, Y, anchors), 4),
                "cpd": round(cpd(P, Y, np.random.default_rng(SEED + 1)), 4),
            }
            print(f"{name:15s} scored in {space:13s} "
                  f"recall@15 {cell['scores'][space]['recall15']:.4f}  "
                  f"CPD {cell['scores'][space]['cpd']:.4f}  "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
        res["layouts"][name] = cell

    print()
    for space in spaces:
        matched = [(k, v) for k, v in res["layouts"].items() if v["fit_space"] == space]
        if not matched:
            continue
        print(f"protocol '{space}': input and scoring space agree")
        for name, cell in matched:
            s = cell["scores"][space]
            print(f"  {name:15s} recall@15 {s['recall15']:.4f}   CPD {s['cpd']:.4f}")

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
