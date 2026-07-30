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
# STANDARDISE=1 rescales each latent dim to unit variance first. off by default because
# the atlas ships this latent and its own UMAP consumed it as shipped, so feeding it
# unchanged is what makes the two comparable. It changes less than it looks like it
# should either way, because raw_coords whitens its PCA head regardless.
STANDARDISE = os.environ.get("STANDARDISE", "0") == "1"
# the one exposed knob, the local-to-global dial. Non-default values get their own cache
W = float(os.environ.get("W", "2.0"))
TAG = ("_std" if STANDARDISE else "") + ("" if W == 2.0 else f"_w{W:g}")
OUT = os.path.join(CACHE, f"hnoca_atlas{TAG}.npz")
URL = ("https://zenodo.org/records/14161275/files/hnoca_cleanedmeta.h5ad?download=1")
# Fig 1c is the coarse cell-type annotation, Fig 1e the organoid age
COLS = ("annot_level_2", "annot_level_1", "annot_region_rev2", "organoid_age_days")
# 0 keeps every cell, a cap subsamples uniformly at random for a cheaper trial run
N_CAP = int(os.environ.get("N_CAP", "0"))
SEED = 0
DEVICE = os.environ.get("DEVICE") or default_device()


def fetch_latent():
    if os.path.exists(LATENT):
        return dict(np.load(LATENT, allow_pickle=True))

    import fsspec
    import h5py
    from anndata.io import read_elem

    os.makedirs(CACHE, exist_ok=True)
    out = {}
    # a block size well above the HDF5 chunk size keeps the request count sane
    fs = fsspec.filesystem("http", block_size=16 * 1024 * 1024)
    t0 = time.perf_counter()

    with fs.open(URL, "rb") as fh:
        with h5py.File(fh, "r") as f:
            out["Z"] = np.asarray(read_elem(f["obsm"]["X_scpoli"]), np.float32)
            print(f"  X_scpoli {out['Z'].shape} in {time.perf_counter() - t0:.0f}s",
                  flush=True)
            out["umap"] = np.asarray(read_elem(f["obsm"]["X_umap_scpoli"]), np.float32)

            for col in COLS:
                # read_elem resolves anndata's categorical encoding to real labels
                out[col] = np.asarray(read_elem(f["obs"][col]))
                print(f"  {col} {out[col].dtype}", flush=True)

    np.savez_compressed(LATENT, **out)
    print(f"wrote {LATENT} ({os.path.getsize(LATENT) / 1e6:.0f} MB, "
          f"{time.perf_counter() - t0:.0f}s total)", flush=True)
    return out


def main():
    data = fetch_latent()
    Z = np.ascontiguousarray(data["Z"], np.float32)
    n = len(Z)
    rows = np.arange(n)

    if N_CAP and n > N_CAP:
        rows = np.sort(np.random.default_rng(SEED).permutation(n)[:N_CAP])
        Z = np.ascontiguousarray(Z[rows], np.float32)
        print(f"subsampled to n={len(Z)}", flush=True)

    if STANDARDISE:
        Z = ((Z - Z.mean(0)) / (Z.std(0) + 1e-9)).astype(np.float32)

    print(f"fitting FloDR on n={len(Z):,}, d={Z.shape[1]}, device={DEVICE}, "
          f"w={W:g}, standardise={STANDARDISE}", flush=True)
    t0 = time.perf_counter()
    # edge_batch keeps the kNN term off the full edge list, which is what makes a
    # million-plus points fit in 12 GB, and gate_max is the paper's recipe
    model = FloDR(w=W, random_state=SEED, device=DEVICE,
                  advanced=dict(gate_max=0.5, edge_batch=32768)).fit(Z)
    fit_secs = time.perf_counter() - t0
    print(f"fit {fit_secs:.0f}s, roundtrip {model.roundtrip_:.2e}", flush=True)

    age = np.asarray(data["organoid_age_days"], np.float64)[rows]
    np.savez_compressed(
        OUT,
        Y=np.asarray(model.embedding_, np.float32),
        umap=np.asarray(data["umap"], np.float32)[rows],
        age=age,
        cell_type=np.asarray(data["annot_level_2"]).astype(str)[rows],
        cell_type_coarse=np.asarray(data["annot_level_1"]).astype(str)[rows],
        region=np.asarray(data["annot_region_rev2"]).astype(str)[rows],
        rows=rows,
        fit_secs=fit_secs,
        roundtrip=model.roundtrip_,
        standardised=STANDARDISE,
        w=W,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
