import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from flodr import FloDR, default_device  # noqa: E402

CACHE = os.path.join(HERE, "cache")
H5AD = os.path.join(ROOT, "data", "zebrafish", "dev_zebrafish.h5ad")
PCA50 = os.path.join(CACHE, "zebrafish_pca50.npz")
N_HVG = 2000
N_PCS = 50
SEED = int(os.environ.get("SEED", "0"))
# the one exposed knob, the local-to-global dial. non-default values get own caches
W = float(os.environ.get("W", "2.0"))
# raw_recipe fixes iters at 800 and edge_batch at 32768, so the number of passes over
# the kNN graph falls linearly with n, ~490 at the paper's n=4000 and ~1.6 here. ITERS
# buys those passes back.
ITERS = int(os.environ.get("ITERS", "0"))
N_LAYERS = int(os.environ.get("N_LAYERS", "0"))
HID = int(os.environ.get("HID", "0"))
TAG = (("" if W == 2.0 else f"_w{W:g}") + ("" if not ITERS else f"_i{ITERS}")
       + ("" if SEED == 0 else f"_s{SEED}")
       + ("" if not N_LAYERS else f"_L{N_LAYERS}")
       + ("" if not HID else f"_H{HID}"))
OUT = os.path.join(CACHE, f"zebrafish_atlas{TAG}.npz")
DEVICE = os.environ.get("DEVICE") or default_device()


def build_pca50():
    if os.path.exists(PCA50):
        return dict(np.load(PCA50, allow_pickle=True))

    import scanpy as sc
    from sklearn.decomposition import PCA

    if not os.path.exists(H5AD):
        raise SystemExit(f"no {H5AD}; download the CELLxGENE h5ad first")

    sc.settings.verbosity = 1
    t0 = time.perf_counter()
    adata = sc.read_h5ad(H5AD)
    print(f"read {adata.shape} in {time.perf_counter() - t0:.0f}s", flush=True)

    obs = {"cell_type": np.asarray(adata.obs["cell_type"]).astype(str),
           "tissue": np.asarray(adata.obs["tissue"]).astype(str),
           "timepoint": np.asarray(adata.obs["timepoint"]).astype(float)}

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=N_HVG)
    adata = adata[:, adata.var.highly_variable].copy()
    print(f"HVG subset {adata.shape}, {time.perf_counter() - t0:.0f}s", flush=True)

    sc.pp.scale(adata, max_value=10)
    pcs = PCA(n_components=N_PCS, svd_solver="randomized",
              random_state=SEED).fit_transform(adata.X)
    del adata
    # per-component standardisation, so the input handed to the flow is isotropic
    P = ((pcs - pcs.mean(0)) / (pcs.std(0) + 1e-12)).astype(np.float32)
    print(f"PCA-{N_PCS} done, {time.perf_counter() - t0:.0f}s", flush=True)

    out = dict(P=P, **obs)
    np.savez_compressed(PCA50, **out)
    print(f"wrote {PCA50} ({os.path.getsize(PCA50) / 1e6:.0f} MB)", flush=True)
    return out


def main():
    data = build_pca50()
    P = np.ascontiguousarray(data["P"], np.float32)

    print(f"fitting FloDR on n={len(P):,}, d={P.shape[1]}, device={DEVICE}, "
          f"w={W:g}, iters={ITERS or 800}, n_layers={N_LAYERS or 4}, "
          f"hid={HID or 256}", flush=True)
    t0 = time.perf_counter()
    adv = dict(gate_max=0.5, edge_batch=32768)
    if ITERS:
        adv["iters"] = ITERS
    if N_LAYERS:
        adv["n_layers"] = N_LAYERS
    if HID:
        adv["hid"] = HID
    model = FloDR(w=W, random_state=SEED, device=DEVICE, advanced=adv).fit(P)
    fit_secs = time.perf_counter() - t0
    print(f"fit {fit_secs:.0f}s, roundtrip {model.roundtrip_:.2e}", flush=True)

    np.savez_compressed(
        OUT,
        Y=np.asarray(model.embedding_, np.float32),
        cell_type=data["cell_type"],
        tissue=data["tissue"],
        timepoint=np.asarray(data["timepoint"], np.float64),
        fit_secs=fit_secs,
        roundtrip=model.roundtrip_,
        w=W,
        iters=ITERS or 800,
        n_layers=N_LAYERS or 4,
        hid=HID or 256,
    )
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
