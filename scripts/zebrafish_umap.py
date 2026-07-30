import os
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "cache")
PCA50 = os.path.join(CACHE, "zebrafish_pca50.npz")
OUT = os.path.join(CACHE, "zebrafish_umap.npz")
N_COMPONENTS = int(os.environ.get("N_COMPONENTS", "2"))


def main():
    if not os.path.exists(PCA50):
        raise SystemExit(f"no {PCA50}; run scripts/zebrafish_atlas.py first")

    P = np.ascontiguousarray(np.load(PCA50, allow_pickle=True)["P"], np.float32)
    print(f"UMAP on n={len(P):,}, d={P.shape[1]}, "
          f"n_components={N_COMPONENTS}", flush=True)

    import umap

    t0 = time.perf_counter()
    Y = umap.UMAP(n_neighbors=15, n_components=N_COMPONENTS).fit_transform(P)
    secs = time.perf_counter() - t0

    key = "umap" if N_COMPONENTS == 2 else f"umap{N_COMPONENTS}d"
    np.savez_compressed(OUT, **{key: np.asarray(Y, np.float32), "secs": secs})
    print(f"fit {secs:.0f}s, wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
