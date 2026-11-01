import json
import os
import subprocess
import sys
import time
from resource import RUSAGE_SELF, getrusage

import matplotlib.pyplot as plt
import numpy as np

import common

SIZES = (1000, 5000, 10_000, 50_000, 100_000, 500_000, 1_000_000)
METHODS = ("FloDR (w=2)", "UMAP", "openTSNE", "TriMap", "PaCMAP", "LocalMAP")
METHODS += ("PHATE", "PyMDE", "PCUMAP", "SQuadMDS", "PCA-2")
GPU = ("FloDR (w=2)", "PyMDE", "SQuadMDS")
LABEL = {"FloDR (w=2)": "FloDR", "PCUMAP": "Parametric UMAP"}


def cell(m, n):
    # a fresh interpreter, so the peak memory is the fit's own (labels left unread)
    X = np.load(os.path.join(common.CACHE, "data", "zebrafish.npz"))["P"]
    X = np.ascontiguousarray(X[np.random.default_rng(0).permutation(len(X))[:n]])
    data = getrusage(RUSAGE_SELF).ru_maxrss
    if m in GPU:
        import torch

        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    common.FIT[m](X, 0, "euclid")
    secs, peak = time.perf_counter() - t0, getrusage(RUSAGE_SELF).ru_maxrss
    rec = dict(secs=secs, host_mb=(peak - data) / 1024)
    if m in GPU:
        rec["gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20
    print(json.dumps(rec))


def run(m, n):
    print(f"fitting {m} on {n} cells", flush=True)
    common.load("zebrafish")  # built before any measured fit
    cmd = [sys.executable, os.path.abspath(__file__), m, str(n)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        return {"failed": r.stderr[-400:]}
    return json.loads(r.stdout.splitlines()[-1])


if len(sys.argv) == 3:
    sys.exit(cell(sys.argv[1], int(sys.argv[2])))
path = os.path.join(common.CACHE, "ed_fig8", "scaling.json")
thin = {"axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6}
plt.rcParams.update({**common.RC, **thin, "axes.axisbelow": True})
fig, axes = plt.subplots(1, 3, figsize=(common.TEXT_W, 2.6))
for m in METHODS:
    ok = []
    for n in SIZES:
        rec = common.memo(path, f"{m}@{n}", lambda: run(m, n))
        # a method that fails is not tried at larger n
        if "failed" in rec:
            break
        ok.append((n, rec))
    ns, big = [n for n, _ in ok], m == "FloDR (w=2)"
    colour, marker = common.METHOD_STYLE[LABEL.get(m, m)]
    kw = dict(color=colour, marker=marker, label=LABEL.get(m, m), mec="w", mew=0.4)
    kw.update(lw=1.6 if big else 0.9, ms=4.5 if big else 3, zorder=6 if big else 3)
    axes[0].loglog(ns, [r["secs"] for _, r in ok], **kw)
    # Parametric UMAP fits in a worker process whose memory this one cannot see
    if m != "PCUMAP":
        axes[1].loglog(ns, [max(r["host_mb"], 1) for _, r in ok], **kw)
    if m in GPU:
        axes[2].loglog(ns, [max(r["gpu_mb"], 1) for _, r in ok], **kw)
ylabels = ("wall time (s)", "peak host memory above input (MB)", "peak GPU memory (MB)")
for ax, ylabel, letter in zip(axes, ylabels, "abc"):
    ax.set(xlabel="$n$ cells", ylabel=ylabel)
    ax.grid(True, lw=0.4, color="0.88")
    common.letter(ax, letter)
kw = dict(ncol=6, frameon=False, fontsize=7, handletextpad=0.4, columnspacing=1.0)
handles = axes[0].get_legend_handles_labels()
fig.legend(*handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), **kw)
fig.tight_layout(rect=(0, 0.12, 1, 1), w_pad=1.2)
common.save(fig, "ed_fig8.pdf")
