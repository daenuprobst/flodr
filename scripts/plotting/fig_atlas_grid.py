"""2x3 paper figure: three cell atlases in columns, cell type on the top row and the merged
sigma x h diagnostic on the bottom row. Reads the 2D caches; no refit.

Run: [SCHEME=DkViolet2] [STEPS=4] .venv/bin/python scripts/plotting/fig_atlas_grid.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src"))
from flodr import viz  # noqa: E402

ATLASES = [("plant", "Arabidopsis germinating seed"),
           ("bmarrow", "Human fetal bone marrow"),
           ("cerebellum", "Developing mouse cerebellum")]
SCHEME = os.environ.get("SCHEME", "DkViolet2")
_sv = os.environ.get("STEPS", "4").lower()
STEPS = None if _sv in ("continuous", "cont", "none", "0") else int(_sv)
_hv = os.environ.get("HIGHLIGHT", "80")
HIGHLIGHT = None if _hv in ("", "none", "off") else float(_hv)
PAL = plt.get_cmap("tab20")(np.linspace(0, 1, 20))


def passed(z, key):
    if key not in z.files:
        return True
    try:
        return bool(json.loads(str(z[key])).get("passed", True))
    except Exception:
        return True


def square(ax, Y):
    ctr = 0.5 * (Y.min(0) + Y.max(0))
    half = 0.53 * float(max(np.ptp(Y[:, 0]), np.ptp(Y[:, 1])))
    ax.set_xlim(ctr[0] - half, ctr[0] + half); ax.set_ylim(ctr[1] - half, ctr[1] + half)
    ax.set_autoscale_on(False); ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color("0.75")


with plt.rc_context(viz.RC_PAPER):
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 6.6), constrained_layout=True)
    for col, (ds, title) in enumerate(ATLASES):
        npz = np.load(os.path.join(ROOT, "scripts", "cache", f"hidden_contrast_{ds}.npz"),
                      allow_pickle=True)
        emb = np.asarray(npz["Y"], float)
        labs = np.asarray(npz["labels"]).astype(str)
        sigma = np.asarray(npz["sigma"], float)
        hidden = np.asarray(npz["field"], float)
        lab_idx = np.unique(labs, return_inverse=True)[1]
        sig_pass, hid_pass = passed(npz, "sigma_cert"), passed(npz, "cert")
        btitle = (r"$\sigma \times h$" if sig_pass and hid_pass else
                  r"$\sigma$  ($h$ refused)" if sig_pass else
                  r"$h$  ($\sigma$ refused)" if hid_pass else
                  r"$\sigma \times h$  (both refused)")

        ax_map, ax_diag = axes[0, col], axes[1, col]
        square(ax_map, emb); square(ax_diag, emb)
        shuf = np.random.default_rng(0).permutation(len(emb))
        ax_map.scatter(emb[shuf, 0], emb[shuf, 1], s=2, c=PAL[lab_idx[shuf] % 20], alpha=0.75,
                       linewidths=0, rasterized=True)
        ax_map.set_title(title, fontsize=10.5)
        viz.bivariate_field(emb, sigma, hidden, ax=ax_diag, scheme=SCHEME, steps=STEPS,
                            x_pass=sig_pass, y_pass=hid_pass, highlight=HIGHLIGHT, s=2)
        ax_diag.set_title(btitle, fontsize=10.5)
        square(ax_map, emb); square(ax_diag, emb)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(ROOT, "figures", f"atlas_grid.{ext}"), dpi=300)
    plt.close(fig)
print("-> figures/atlas_grid.pdf / .png", flush=True)
