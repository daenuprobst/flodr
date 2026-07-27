"""Validate the two diagnostic fields against synthetic ground truth, three seeds per cell.

Spread: conditional_spread on make_hetero with per-cluster spread ranges of 3/10/30/100x,
on make_ring, and on the make_blob flat control; truth is the generator's per-point spread.
Hidden: hidden_contrast on make_hidden for lam in {0, .25, .5, .75, 1}, plus a stressed
layout (the fitted embedding with its rows permuted, which leaks none of the contrast by
construction). The hidden truth is I(G;X) - I(G;Y): the leak I(G;Y) is measured with an
oracle classifier on the embedding rather than assumed away, and the per-position truth
field is hidden_truth. Writes cache/synthetic_validation.json and
figures/synthetic_validation.pdf; finished cells are skipped unless RECOMPUTE=1.

Run: [DEVICE=cuda] [RECOMPUTE=1] .venv/bin/python scripts/ablations/synthetic_validation.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, SCRIPTS)

from datasets import make_blob, make_hetero, make_hidden, make_ring              # noqa: E402
from datasets import hidden_total_mi, hidden_truth                              # noqa: E402
from flodr import FloDR, viz                                                    # noqa: E402

OUT = os.path.join(SCRIPTS, "cache", "synthetic_validation.json")
PANEL1 = os.path.join(SCRIPTS, "cache", "synthetic_validation_30x.npz")
FIG = os.path.join(ROOT, "figures")
SEEDS = (0, 1, 2)
RANGES = (3, 10, 30, 100)
LAMS = (0.0, 0.25, 0.5, 0.75, 1.0)
RECOMPUTE = os.environ.get("RECOMPUTE") == "1"
DEVICE = os.environ.get("DEVICE")
if DEVICE is None:
    import torch
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def lin_stats(est, truth):
    """OLS slope (with intercept), Pearson and Spearman of est against truth, per point."""
    from scipy.stats import pearsonr, spearmanr
    design = np.column_stack([truth, np.ones_like(truth)])
    slope = float(np.linalg.lstsq(design, est, rcond=None)[0][0])
    return slope, float(pearsonr(est, truth).statistic), \
        float(spearmanr(est, truth).statistic)


def spread_seed(kind, spread_range, seed):
    """One (condition, seed) spread cell: fit, estimate sigma(y), score against truth."""
    if kind == "hetero":
        X, truth, lab = make_hetero(seed=seed, s_lo=3.0 / spread_range, s_hi=3.0)
    elif kind == "ring":
        X, truth = make_ring(seed=seed)
        lab = None
    else:
        X, truth, lab = make_blob(seed=seed), None, None
    m = FloDR(w=2.0, random_state=seed, device=DEVICE, density=True).fit(X)
    est = np.asarray(m.conditional_spread(), np.float64)
    cert = m.spread_calibration()
    cell = {"seed": seed, "field_cv": float(est.std() / est.mean()),
            "cert_passed": bool(cert["passed"]), "certifies": cert["certifies"],
            "cert_slope": round(float(cert["slope"]), 3),
            "cert_level": round(float(cert["ratio_quartiles"][1]), 3)}
    if truth is not None:
        slope, pear, spear = lin_stats(est, truth)
        cell.update(slope=slope, pearson=pear, spearman=spear,
                    median_ratio=float(np.median(est / truth)))
    if lab is not None:
        med = [np.median(est[lab == c]) for c in np.unique(lab)]
        cell["est_range"] = float(max(med) / min(med))
    if kind == "hetero" and spread_range == 30 and seed == SEEDS[0] and (
            RECOMPUTE or not os.path.exists(PANEL1)):
        np.savez_compressed(PANEL1, est=est.astype(np.float32),
                            truth=truth.astype(np.float32), lab=lab)
    return cell


def oracle_mi(m, view, g, seed, iters=3000):
    """I(G;view) in nats: H(G) plus the held-out log p(true class) of the estimator's own
    classifier head, fit on one half and scored on the other."""
    gi = np.unique(g, return_inverse=True)[1]
    n = len(gi)
    rng = np.random.default_rng(seed + 13)
    perm = rng.permutation(n)
    rows_fit, rows_eval = perm[: n // 2], perm[n // 2:]
    pk = np.bincount(gi) / n
    ent = float(-(pk * np.log(pk)).sum())
    logp = m._class_logp(np.asarray(view, np.float64), gi, rows_fit, rows_eval,
                         len(pk), 128, iters, seed + 41)
    return ent + float(logp.mean())


def hidden_seed(lam, stress, seed):
    """One (lam, seed) hidden-contrast cell. The stress arm permutes the embedding rows of
    the lam=1 fit, so the display carries the contrast neither locally nor globally."""
    X, u, g, truth = make_hidden(lam=lam, seed=seed)
    m = FloDR(w=2.0, random_state=seed, device=DEVICE, density=True).fit(X)
    if stress:
        m.embedding_ = m.embedding_[np.random.default_rng(seed + 31).permutation(len(X))]
    field, cert = m.hidden_contrast(g)
    field = np.asarray(field, np.float64)
    true_field = hidden_truth(u, truth)
    mi_u, mi_x = hidden_total_mi(truth)
    mi_y = oracle_mi(m, m.embedding_, g, seed)
    mi_u_or = oracle_mi(m, u, g, seed)
    pear = spear = float("nan")
    if true_field.std() > 1e-12 and field.std() > 1e-12:
        from scipy.stats import pearsonr, spearmanr
        pear = float(pearsonr(field, true_field).statistic)
        spear = float(spearmanr(field, true_field).statistic)
    mean_field = float(field.mean())
    true_level = mi_x - mi_y
    return {"seed": seed, "mean_field": mean_field, "true_level": true_level,
            "level_ratio": mean_field / max(true_level, 1e-12),
            "pearson": pear, "spearman": spear,
            "I_G_u": mi_u, "I_G_X": mi_x, "I_G_Y_oracle": mi_y, "I_G_u_oracle": mi_u_or,
            "cert_passed": bool(cert.get("passed")), "shape_ok": cert.get("shape_ok"),
            "p_level": cert.get("p_level"), "mean_gap": round(float(cert["mean_gap"]), 4)}


def mean_sd(vals):
    a = np.asarray(vals, float)
    return float(a.mean()), float(a.std())


def aggregate(res):
    """Seed means (and population sds) next to the per-seed cells, matching the shipped JSON."""
    for key, arm in res.items():
        cells = arm["per_seed"]
        arm["cert_passed"] = [c["cert_passed"] for c in cells]
        if arm["kind"] == "spread":
            if "true_range" not in arm:     # ring/blob stay per-seed only, as shipped
                continue
            for fld in ("slope", "pearson", "spearman", "median_ratio", "est_range"):
                arm[fld] = mean_sd([c[fld] for c in cells])[0]
            arm["sd"] = {fld: mean_sd([c[fld] for c in cells])[1]
                         for fld in ("slope", "pearson", "median_ratio")}
        else:
            arm["mean_field"], arm["mean_field_sd"] = mean_sd(
                [c["mean_field"] for c in cells])
            arm["true_level"] = mean_sd([c["true_level"] for c in cells])[0]
            for fld in ("pearson", "spearman", "I_G_X", "I_G_Y_oracle"):
                vals = [c[fld] for c in cells]
                arm[fld] = float("nan") if np.isnan(vals).all() else \
                    float(np.nanmean(vals))
    return res


def make_figure(res):
    """Three panels: est vs true spread at 30x, range tracking, hidden contrast vs truth."""
    z = np.load(PANEL1)
    est, truth, lab = z["est"], z["truth"], z["lab"]
    with plt.rc_context(viz.RC_PAPER):
        fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.3))
        ax = axes[0]
        pal = plt.get_cmap("viridis")(np.linspace(0, 1, len(np.unique(lab))))
        for c_i, c in enumerate(np.unique(lab)):
            sel = lab == c
            ax.scatter(truth[sel], est[sel], s=1.2, color=pal[c_i], linewidths=0,
                       rasterized=True)
        lo, hi = min(truth.min(), est.min()), max(truth.max(), est.max())
        ax.loglog([lo, hi], [lo, hi], color="0.25", lw=0.9)
        ax.set_xlabel(r"true $\sigma(y)$")
        ax.set_ylabel(r"estimated $\sigma(y)$")
        ax.set_title("conditional spread (30x)", fontsize=8)

        ax = axes[1]
        xs = [res[f"hetero_{r}x"]["true_range"] for r in RANGES]
        ys = [res[f"hetero_{r}x"]["est_range"] for r in RANGES]
        sd = [np.std([c["est_range"] for c in res[f"hetero_{r}x"]["per_seed"]])
              for r in RANGES]
        ax.errorbar(xs, ys, yerr=sd, fmt="o", ms=3.2, color="#0072B2", elinewidth=0.7,
                    capsize=1.6)
        ax.loglog([2, 130], [2, 130], color="0.25", lw=0.9)
        ax.set_xlabel("true spread range")
        ax.set_ylabel("estimated range")
        ax.set_title("range tracking", fontsize=8)

        ax = axes[2]
        keys = [f"lam_{lam}" for lam in LAMS]
        est_m = [res[k]["mean_field"] for k in keys]
        est_s = [res[k]["mean_field_sd"] for k in keys]
        true_m = [res[k]["true_level"] for k in keys]
        ax.errorbar(LAMS, est_m, yerr=est_s, fmt="s--", ms=3.0, color="#0072B2", lw=0.9,
                    elinewidth=0.7, capsize=1.6, label="estimated field")
        ax.plot(LAMS, true_m, "o-", ms=3.0, color="0.10", lw=1.0,
                label=r"true $I(G;X) - \hat{I}(G;Y)$")
        ax.plot([1.0], [res["stress_lam_1"]["mean_field"]], "s", ms=4.5, color="#D55E00",
                label="stress (no leak)")
        ax.set_xlabel(r"$\lambda$  (0 visible, 1 hidden)")
        ax.set_ylabel("nats")
        ax.set_title("hidden contrast vs truth", fontsize=8)
        ax.legend(fontsize=5.6, frameon=False, loc="upper left")
        for a in axes:
            a.grid(True, lw=0.3, color="0.88")
            a.set_axisbelow(True)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        fig.tight_layout(w_pad=1.4)
        os.makedirs(FIG, exist_ok=True)
        for ext in ("pdf", "png"):
            fig.savefig(os.path.join(FIG, f"synthetic_validation.{ext}"), dpi=300,
                        bbox_inches="tight")
        plt.close(fig)
    print(f"-> {FIG}/synthetic_validation.pdf / .png", flush=True)


def main():
    res = {} if RECOMPUTE else (json.load(open(OUT)) if os.path.exists(OUT) else {})
    todo = [(f"hetero_{r}x", "spread", r, False) for r in RANGES]
    todo += [("ring", "spread", None, False), ("blob", "spread", None, False)]
    todo += [(f"lam_{lam}", "hidden", lam, False) for lam in LAMS]
    todo += [("stress_lam_1", "hidden", 1.0, True)]
    for key, kind, param, stress in todo:
        arm = res.setdefault(key, {"kind": kind, "per_seed": []})
        if kind == "spread" and param is not None:
            arm["true_range"] = param
        if kind == "hidden":
            arm["lam"], arm["stress"] = param, stress
        done = {c["seed"] for c in arm["per_seed"]}
        for seed in SEEDS:
            if seed in done:
                continue
            print(f"[{key} seed {seed}]", flush=True)
            cell = spread_seed(key.split("_")[0], param, seed) if kind == "spread" \
                else hidden_seed(param, stress, seed)
            arm["per_seed"].append(cell)
            arm["per_seed"].sort(key=lambda c: c["seed"])
            json.dump(res, open(OUT, "w"), indent=1)
            print(f"  {cell}", flush=True)
    res = aggregate(res)
    json.dump(res, open(OUT, "w"), indent=1)
    make_figure(res)
    print(f"-> {OUT}", flush=True)


if __name__ == "__main__":
    main()
