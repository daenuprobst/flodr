import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from hnoca_metrics import cpd, recall_at_k  # noqa: E402

CACHE = os.path.join(HERE, "cache")
PCA50 = os.path.join(CACHE, "zebrafish_pca50.npz")
OUT = os.path.join(CACHE, "zebrafish_metrics.json")
LAYOUTS = {"flodr": ("zebrafish_atlas.npz", "Y"),
           "flodr_w1": ("zebrafish_atlas_w1.npz", "Y"),
           # 2x the default training budget, which at this n is 1.6 passes over the
           # kNN graph against about 490 at the n the recipe was tuned on
           "flodr_i1600": ("zebrafish_atlas_i1600.npz", "Y"),
           "flodr_i8000": ("zebrafish_atlas_i8000.npz", "Y"),
           "flodr_L8": ("zebrafish_atlas_L8_H256.npz", "Y"),
           "flodr_H512": ("zebrafish_atlas_L4_H512.npz", "Y"),
           "flodr_L8H512": ("zebrafish_atlas_L8_H512.npz", "Y"),
           "flodr_s1": ("zebrafish_atlas_s1.npz", "Y"),
           "flodr_s2": ("zebrafish_atlas_s2.npz", "Y"),
           "umap": ("zebrafish_umap.npz", "umap")}
N_ANCHOR = int(os.environ.get("N_ANCHOR", "50000"))
SEED = 777


def main():
    P = np.ascontiguousarray(np.load(PCA50, allow_pickle=True)["P"], np.float32)
    n = len(P)
    rng = np.random.default_rng(SEED)
    anchors = np.sort(rng.permutation(n)[: min(N_ANCHOR, n)])

    res = {"n": int(n), "d": int(P.shape[1]), "k": 15,
           "n_anchors": int(len(anchors)),
           "reference_space": "standardised PCA-50, as handed to both methods",
           "layouts": {}}

    for name, (fname, key) in LAYOUTS.items():
        path = os.path.join(CACHE, fname)
        if not os.path.exists(path):
            print(f"skip {name}: no {fname}", flush=True)
            continue
        z = np.load(path, allow_pickle=True)
        Y = np.ascontiguousarray(np.asarray(z[key], np.float32))
        t0 = time.perf_counter()
        cell = {"recall15": round(recall_at_k(P, Y, anchors), 4),
                "cpd": round(cpd(P, Y, np.random.default_rng(SEED + 1)), 4)}
        if "fit_secs" in z.files:
            cell["fit_secs"] = round(float(z["fit_secs"]), 1)
        elif "secs" in z.files:
            cell["fit_secs"] = round(float(z["secs"]), 1)
        res["layouts"][name] = cell
        print(f"{name:6s} recall@15 {cell['recall15']:.4f}   CPD {cell['cpd']:.4f}   "
              f"({time.perf_counter() - t0:.0f}s)", flush=True)

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
