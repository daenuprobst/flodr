import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from flodr import FloDR, default_device  # noqa: E402

CACHE = os.path.join(HERE, "cache")
LATENT = os.path.join(CACHE, "hnoca_latent.npz")
OUT = os.path.join(CACHE, "hnoca_3d.npz")
N_SUB = int(os.environ.get("N_SUB", "250000"))
SEED = 0
DEVICE = os.environ.get("DEVICE") or default_device()


def main():
    lat = np.load(LATENT, allow_pickle=True)
    Z_all = np.asarray(lat["Z"], np.float32)
    n_all = len(Z_all)

    rows = np.arange(n_all)
    if N_SUB and n_all > N_SUB:
        rows = np.sort(np.random.default_rng(SEED).permutation(n_all)[:N_SUB])
    Z = np.ascontiguousarray(Z_all[rows])
    print(f"n={len(Z):,} of {n_all:,}, d={Z.shape[1]}", flush=True)

    out = {"rows": rows, "n_all": n_all,
           "cell_type": np.asarray(lat["annot_level_2"]).astype(str)[rows],
           "age": np.asarray(lat["organoid_age_days"], np.float64)[rows]}

    # the latent as it ships, matching hnoca_atlas.py. Standardising it here would move
    # the kNN graph and the ordinal stress target, not just the flow's parameterisation,
    # and on this latent that costs about 0.11 CPD
    print(f"fitting FloDR (k=3) on {DEVICE}", flush=True)
    t0 = time.perf_counter()
    model = FloDR(w=2.0, random_state=SEED, device=DEVICE, n_components=3,
                  advanced=dict(gate_max=0.5, edge_batch=32768)).fit(Z)
    out["flodr"] = np.asarray(model.embedding_, np.float32)
    out["flodr_secs"] = time.perf_counter() - t0
    print(f"  fit {out['flodr_secs']:.0f}s, roundtrip {model.roundtrip_:.2e}", flush=True)
    del model

    import umap

    # left unpinned on purpose. random_state serialises UMAP onto one thread, which at
    # this n is the difference between minutes and most of a day
    print("fitting UMAP (n_components=3)", flush=True)
    t0 = time.perf_counter()
    out["umap"] = np.asarray(
        umap.UMAP(n_neighbors=15, n_components=3).fit_transform(Z), np.float32)
    out["umap_secs"] = time.perf_counter() - t0
    print(f"  fit {out['umap_secs']:.0f}s", flush=True)

    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
