import json
import os
import sys
import time

import numpy as np
from sklearn.decomposition import PCA

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(HERE, "ablations"))

from donor_split import preprocess, donor_split, recall_at_15, cpd  # noqa: E402
from flodr import FloDR  # noqa: E402
from heldout_benchmark import run_transform_opt, knn_train  # noqa: E402

CACHE = os.path.join(HERE, "cache")
OUT = os.path.join(CACHE, "atlas_frontier.json")
EMBEDS = os.path.join(CACHE, "atlas_frontier_embeds.npz")
ATLASES = ("bmarrow", "cerebellum", "plant")
REPS = (0, 1, 2)
# train cells kept per rep in the embeddings archive
N_SUB = 2000
# The plant h5ad has no donor or batch column, so it is split on the Seurat-merge barcode
# suffix instead: a sample-level split over 6 ids. Reps 0 and 1 draw the same test set by
# chance, and rep 1 is kept as a seed-noise duplicate of rep 0.

DEVICE = "cuda"
try:
    import torch
    if not torch.cuda.is_available():
        DEVICE = "cpu"
    torch._dynamo.config.cache_size_limit = 128
except ImportError:
    torch = None
    DEVICE = "cpu"


def load_preproc(atlas):
    if atlas == "plant":
        z = np.load(os.path.join(CACHE, "atlas_frontier_preproc_plant.npz"),
                    allow_pickle=True)
        return z["P"].astype(np.float32), z["donors"].astype(str)
    return preprocess(atlas)


def fit_flodr(X_train, rep):
    for dev in ((DEVICE, "cpu") if DEVICE != "cpu" else ("cpu",)):
        try:
            est = FloDR(w=2.0, random_state=rep, device=dev,
                        advanced=dict(gate_max=0.5, edge_batch=32768))
            t0 = time.perf_counter()
            est.fit(X_train)
            return est, time.perf_counter() - t0
        except RuntimeError as e:
            if dev == "cpu" or "memory" not in str(e).lower():
                raise
            print(f"    cuda failed ({e}); retrying on cpu", flush=True)
            torch.cuda.empty_cache()


def base_embeddings(atlas, rep, X_train, X_test):
    out = {}
    for method in ("flodr", "umap"):
        path = os.path.join(CACHE, f"atlas_frontier_{method}_{atlas}_{rep}.npz")
        if os.path.exists(path):
            z = np.load(path)
            out[method] = (z["Ytr"], z["Yte"])
            continue
        if method == "flodr":
            est, fit_secs = fit_flodr(X_train, rep)
            t0 = time.perf_counter()
            Yte = est.transform(X_test)
            xform_secs = time.perf_counter() - t0
            Ytr = np.asarray(est.embedding_, np.float32)
            del est
            if torch is not None and DEVICE == "cuda":
                torch.cuda.empty_cache()
        else:
            import umap
            t0 = time.perf_counter()
            est = umap.UMAP(n_neighbors=15, random_state=rep).fit(X_train)
            fit_secs = time.perf_counter() - t0
            t0 = time.perf_counter()
            Yte = est.transform(X_test)
            xform_secs = time.perf_counter() - t0
            Ytr = np.asarray(est.embedding_, np.float32)
        Yte = np.asarray(Yte, np.float32)
        np.savez(path, Ytr=Ytr, Yte=Yte, fit_secs=fit_secs, transform_secs=xform_secs)
        out[method] = (Ytr, Yte)
        print(f"  [{atlas} rep{rep}] {method}: embeddings computed", flush=True)
    return out


def opentsne_cell(atlas, P, train_mask, test_mask):
    path = os.path.join(CACHE, f"atlas_opentsne_{atlas}_rep0.npz")
    if os.path.exists(path):
        return
    if atlas == "bmarrow":
        z = np.load(os.path.join(CACHE, "showcase_bmarrow.npz"), allow_pickle=True)
        np.savez(path, Ytr=z["Ytr_opentsne"], Yte=z["Yte_opentsne"],
                 fit_secs=np.nan, transform_secs=np.nan)
        print("  [bmarrow rep0] opentsne: reused showcase_bmarrow.npz", flush=True)
        return
    import openTSNE
    X_train, X_test = P[train_mask], P[test_mask]
    est = openTSNE.TSNE(random_state=0)
    t0 = time.perf_counter()
    emb = est.fit(X_train)
    fit_secs = time.perf_counter() - t0
    t0 = time.perf_counter()
    Yte = np.asarray(emb.transform(X_test), np.float32)
    transform_secs = time.perf_counter() - t0
    np.savez(path, Ytr=np.asarray(emb, np.float32), Yte=Yte,
             fit_secs=fit_secs, transform_secs=transform_secs)
    print(f"  [{atlas} rep0] opentsne: fit {fit_secs:.0f}s, transform {transform_secs:.0f}s",
          flush=True)


def metrics(P_all, emb_all, n_train):
    test_rows = np.arange(n_train, len(emb_all))
    return dict(recall15=round(recall_at_15(P_all, emb_all, test_rows), 4),
                cpd=round(cpd(P_all, emb_all, test_rows), 4))


def main():
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    ds_metrics = json.load(open(os.path.join(CACHE, "donor_split.json")))
    embeds = dict(np.load(EMBEDS).items()) if os.path.exists(EMBEDS) else {}

    for atlas in ATLASES:
        P, donors = load_preproc(atlas)
        for rep in REPS:
            train_mask, test_mask, test_donors = donor_split(donors, rep)
            X_train, X_test = P[train_mask], P[test_mask]
            n_train = len(X_train)
            P_all = np.concatenate([X_train, X_test])
            print(f"[{atlas} rep{rep}] n_train={n_train} n_test={len(X_test)} "
                  f"test_donors={test_donors}", flush=True)
            base = base_embeddings(atlas, rep, X_train, X_test)
            sub = np.random.default_rng(100 + rep).permutation(n_train)[:N_SUB]
            embeds[f"sub_{atlas}_{rep}"] = sub

            # bmarrow and cerebellum come from donor_split.json; plant is scored here
            for method in ("flodr", "umap"):
                row = res.setdefault(atlas, {}).setdefault(method, {}).setdefault(
                    "per_rep", {})
                key = f"{atlas}/{method}/{rep}"
                Ytr, Yte = base[method]
                if key in ds_metrics:
                    c = ds_metrics[key]
                    row[str(rep)] = dict(recall15=c["recall@15"], cpd=c["cpd"],
                                         fit_secs=c["fit_secs"],
                                         transform_secs=c["transform_secs"])
                elif str(rep) not in row:
                    m = metrics(P_all, np.concatenate([Ytr, Yte]), n_train)
                    m.update(fit_secs=base_fit_secs(atlas, rep, method, ds_metrics),
                             transform_secs=base_xform_secs(atlas, rep, method))
                    row[str(rep)] = m
                    print(f"  [{atlas} rep{rep}] {method}: {m}", flush=True)
                embeds[f"Ytr_{atlas}_{method}_{rep}"] = Ytr[sub]
                embeds[f"Yte_{atlas}_{method}_{rep}"] = Yte

            # attraction-only placement, initialised at the flodr forward pass
            if str(rep) not in res[atlas].setdefault("opta", {}).setdefault("per_rep", {}):
                Ytr_f, Yte_f = base["flodr"]
                t0 = time.perf_counter()
                _, idx_nn = knn_train(X_train, X_test, "euclidean")
                # one forward pass, per donor_split
                fwd_secs = 0.02
                Yte_o = run_transform_opt(Ytr_f, Yte_f, idx_nn, seed=rep, repulsion=False)
                place_secs = time.perf_counter() - t0 + fwd_secs
                emb_all = np.concatenate([Ytr_f, Yte_o])
                m = metrics(P_all, emb_all, n_train)
                m.update(fit_secs=base_fit_secs(atlas, rep, "flodr", ds_metrics),
                         transform_secs=round(place_secs, 2))
                res[atlas]["opta"]["per_rep"][str(rep)] = m
                embeds[f"Ytr_{atlas}_opta_{rep}"] = Ytr_f[sub]
                embeds[f"Yte_{atlas}_opta_{rep}"] = Yte_o
                print(f"  [{atlas} rep{rep}] opta: {m}", flush=True)

            if str(rep) not in res[atlas].setdefault("pca2", {}).setdefault("per_rep", {}):
                t0 = time.perf_counter()
                pca = PCA(n_components=2, random_state=rep).fit(X_train)
                fit_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                Ytr_p = pca.transform(X_train).astype(np.float32)
                Yte_p = pca.transform(X_test).astype(np.float32)
                x_s = time.perf_counter() - t0
                m = metrics(P_all, np.concatenate([Ytr_p, Yte_p]), n_train)
                m.update(fit_secs=round(fit_s, 2), transform_secs=round(x_s, 2))
                res[atlas]["pca2"]["per_rep"][str(rep)] = m
                embeds[f"Ytr_{atlas}_pca2_{rep}"] = Ytr_p[sub]
                embeds[f"Yte_{atlas}_pca2_{rep}"] = Yte_p
                print(f"  [{atlas} rep{rep}] pca2: {m}", flush=True)

            # knnmap, anchored on the umap train embedding
            if str(rep) not in res[atlas].setdefault("knnmap", {}).setdefault("per_rep", {}):
                Ytr_u, _ = base["umap"]
                t0 = time.perf_counter()
                d_nn, idx_nn = knn_train(X_train, X_test, "euclidean")
                fit_s = time.perf_counter() - t0
                t0 = time.perf_counter()
                w = 1.0 / (d_nn + 1e-9)
                w /= w.sum(1, keepdims=True)
                Yte_k = (w[..., None] * Ytr_u[idx_nn]).sum(1).astype(np.float32)
                x_s = time.perf_counter() - t0
                m = metrics(P_all, np.concatenate([Ytr_u, Yte_k]), n_train)
                m.update(fit_secs=round(fit_s, 2), transform_secs=round(x_s, 2))
                res[atlas]["knnmap"]["per_rep"][str(rep)] = m
                embeds[f"Ytr_{atlas}_knnmap_{rep}"] = Ytr_u[sub]
                embeds[f"Yte_{atlas}_knnmap_{rep}"] = Yte_k
                print(f"  [{atlas} rep{rep}] knnmap: {m}", flush=True)

            if rep == 0 and str(rep) not in res[atlas].setdefault("opentsne", {}) \
                    .setdefault("per_rep", {}):
                path = os.path.join(CACHE, f"atlas_opentsne_{atlas}_rep0.npz")
                if os.path.exists(path):
                    z = np.load(path)
                    Ytr_t, Yte_t = z["Ytr"], z["Yte"]
                    m = metrics(P_all, np.concatenate([Ytr_t, Yte_t]), n_train)
                    m.update(fit_secs=None if np.isnan(z["fit_secs"]) else
                             round(float(z["fit_secs"]), 2),
                             transform_secs=None if np.isnan(z["transform_secs"]) else
                             round(float(z["transform_secs"]), 2))
                    res[atlas]["opentsne"]["per_rep"][str(rep)] = m
                    embeds[f"Ytr_{atlas}_opentsne_{rep}"] = Ytr_t[sub]
                    embeds[f"Yte_{atlas}_opentsne_{rep}"] = Yte_t
                    print(f"  [{atlas} rep0] opentsne: {m}", flush=True)

            json.dump(res, open(OUT, "w"), indent=1, sort_keys=True)
            np.savez(EMBEDS, **embeds)

    # mean/std over reps; opentsne has a single rep
    for atlas in ATLASES:
        for method, cell in res.get(atlas, {}).items():
            reps = cell.get("per_rep", {})
            ms = {}
            for f in ("recall15", "cpd", "fit_secs", "transform_secs"):
                v = [r[f] for r in reps.values() if r.get(f) is not None]
                if v:
                    ms[f] = [float(np.mean(v)), float(np.std(v))]
            cell["mean_std"] = ms
    res["_notes"] = [
        "flodr/umap metric rows (bmarrow, cerebellum, reps 0-2) reused from donor_split.json",
        "bmarrow rep0 opentsne reused from showcase_bmarrow.npz (fit/transform secs unknown)",
        "plant split is SAMPLE-level (Seurat-merge barcode suffix _N, 6 ids), not "
        "donor-level: the h5ad has no donor/batch column (orig.ident constant 'seed_125d'). "
        "With 6 samples, reps 0 and 1 draw the same test set {2, 6} by chance",
        "showcase_bmarrow.npz carries no flodr forward-pass keys, so bmarrow rep0 "
        "flodr/optA were computed like the other reps",
    ]
    json.dump(res, open(OUT, "w"), indent=1, sort_keys=True)
    np.savez(EMBEDS, **embeds)
    print("done", flush=True)


def base_fit_secs(atlas, rep, method, ds_metrics):
    key = f"{atlas}/{method}/{rep}"
    if key in ds_metrics:
        return ds_metrics[key]["fit_secs"]
    z = np.load(os.path.join(CACHE, f"atlas_frontier_{method}_{atlas}_{rep}.npz"))
    return round(float(z["fit_secs"]), 2)


def base_xform_secs(atlas, rep, method):
    z = np.load(os.path.join(CACHE, f"atlas_frontier_{method}_{atlas}_{rep}.npz"))
    return round(float(z["transform_secs"]), 2) if "transform_secs" in z else None


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "opentsne":
        atlas = sys.argv[2]
        P, donors = load_preproc(atlas)
        train_mask, test_mask, _ = donor_split(donors, 0)
        opentsne_cell(atlas, P, train_mask, test_mask)
    else:
        main()
