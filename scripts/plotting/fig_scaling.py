"""Speed and memory against n, from cache/scaling.json.

Left  wall clock. Right  the method's own peak RSS above the loaded input, with FloDR's
GPU reservation drawn separately. Points stop where a method died, and the failure is
marked at the size that killed it.

Run: .venv/bin/python scripts/fig_scaling.py
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))
from flodr import viz  # noqa: E402

CACHE = os.path.join(SCRIPTS, "cache", "scaling.json")
FIG = os.path.join(ROOT, "figures", "scaling")
STYLE = {
    # full-batch FloDR is deliberately not drawn: it dies at 5e5 and its OOM marker
    # dominated the panel. The failure is reported in the text instead.
    "flodr_eb": ("FloDR (GPU)", "#D55E00", "o", 2.2),
    "umap": ("UMAP", "#009E73", "s", 1.1),
    "opentsne": ("openTSNE", "#0072B2", "o", 1.1),
    "trimap": ("TriMap", "#A6761D", "^", 1.1),
    "pacmap": ("PaCMAP", "#7570B3", "D", 1.1),
    "localmap": ("LocalMAP", "#CC79A7", "v", 1.1),
    "phate": ("PHATE", "#56B4E9", "*", 1.1),
    "pymde": ("PyMDE (GPU)", "#8C564B", "h", 1.1),
    "pcumap": ("PCUMAP (GPU)", "#E69F00", "<", 1.1),
    "squadmds": ("SQuadMDS", "#2F4F4F", ">", 1.1),
    "pca": ("PCA-2", "#7a7a7a", "x", 1.0),
}
FAIL_MARK = {"oom": "OOM", "gpu_oom": "GPU OOM", "timeout": "timeout", "error": "error"}


def main():
    res = json.load(open(CACHE))
    rows = {}
    for rec in res.values():
        rows.setdefault(rec["method"], []).append(rec)
    for recs in rows.values():
        recs.sort(key=lambda rec: rec["n"])

    with plt.rc_context(viz.RC_PAPER):
        fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.5))
        for meth, recs in rows.items():
            if meth not in STYLE:
                continue
            label, color, mark, lw = STYLE[meth]
            hero = meth == "flodr_eb"
            kw = dict(marker=mark, color=color, label=label, lw=lw,
                      ms=4.6 if hero else 3.0, alpha=1.0 if hero else 0.55,
                      zorder=6 if hero else 3)
            ok = [rec for rec in recs if rec["status"] == "ok"]
            if ok:
                ns = [rec["n"] for rec in ok]
                axes[0].loglog(ns, [rec["secs"] for rec in ok], **kw)
                axes[1].loglog(ns, [max(rec["fit_mb"], 1) for rec in ok], **kw)
                # GPU gets its own axis: host RSS and device memory are different
                # quantities, and the CUDA runtime puts a flat ~1.6GB floor under every
                # torch method's RSS that has nothing to do with n
                gpu = [rec for rec in ok if "gpu_mb" in rec]
                if gpu:
                    axes[2].loglog([rec["n"] for rec in gpu],
                                   [max(rec["gpu_mb"], 1) for rec in gpu], **kw)
        axes[0].set_ylabel("wall clock (s)", fontsize=7.5, labelpad=2)
        axes[0].set_title("speed", fontsize=8.5)
        axes[1].set_ylabel("peak host RSS above input (MB)", fontsize=7.5, labelpad=2)
        axes[1].set_title("host memory", fontsize=8.5)
        axes[2].set_ylabel("peak GPU reserved (MB)", fontsize=7.5, labelpad=2)
        axes[2].set_title("GPU memory", fontsize=8.5)
        for ax in axes:
            ax.set_xlabel("$n$ points", fontsize=7.5, labelpad=2)
            ax.grid(True, which="major", lw=0.3, alpha=0.4)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=6.8)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
        handles, labels = axes[0].get_legend_handles_labels()
        handles2, labels2 = axes[1].get_legend_handles_labels()
        for h2, l2 in zip(handles2, labels2):
            if l2 not in labels:
                handles.append(h2); labels.append(l2)
        fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=6.2,
                   bbox_to_anchor=(0.5, -0.10), handletextpad=0.35, columnspacing=1.1)
        fig.tight_layout(w_pad=1.6)
        for ext in ("png", "pdf"):
            fig.savefig(f"{FIG}.{ext}", dpi=300, bbox_inches="tight")
    print(f"-> {FIG}.png / .pdf", flush=True)

    print("\nfailures:")
    for rec in sorted(res.values(), key=lambda rec: (rec["method"], rec["n"])):
        if rec["status"] != "ok":
            print(f"  {rec['method']:10s} n={rec['n']:>8,}  {rec['status']:8s} "
                  f"{rec.get('note', '')[:90]}")
    batched_table()


def batched_table():
    """paper/batched_table.tex: full batch against edge batched, from cache/batched.json."""
    path = os.path.join(SCRIPTS, "cache", "batched.json")
    if not os.path.exists(path):
        return
    res = json.load(open(path))
    pretty = {"full": "full batch", "eb32k": "batched, $2^{15}$ edges",
              "eb128k": "batched, $2^{17}$ edges"}
    rows = []
    for n in (50000, 100000, 500000, 1000000):
        for mode in ("full", "eb32k", "eb128k"):
            rec = res.get(f"{n}@{mode}")
            if rec is None:
                continue
            n_tex = f"{n:,}".replace(",", "{,}")
            if rec["status"] != "ok":
                rows.append(rf"${n_tex}$ & {pretty[mode]} & \multicolumn{{4}}{{c}}{{"
                            rf"\emph{{out of GPU memory}}}} \\")
            else:
                rows.append(rf"${n_tex}$ & {pretty[mode]} & ${rec['secs']:.0f}$ & "
                            rf"${rec['gpu_mb']:.0f}$ & ${rec['recall']:.4f}$ & "
                            rf"${rec['cpd']:.3f}$ \\")
    tex = ("\\begin{table}[t]\n\\centering\n\\small\n"
           "\\caption{Edge-batched against full-batch training on the synthetic family "
           "($d=50$, one seed). Quality is scored on a common $20{,}000$-point subsample, "
           "so the absolute recall is not the full-scale value but the comparison within a "
           "row block is like-for-like. Batching is faster, roughly halves GPU memory, and "
           "reaches sizes the full-batch path cannot, at $97$--$99\\%$ of its distance "
           "correlation.}\n\\label{tab:batched}\n"
           "\\begin{tabular}{rlrrrr}\n\\toprule\n"
           "$n$ & training & s & GPU (MB) & recall@15 & CPD \\\\\n\\midrule\n"
           + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    out = os.path.join(ROOT, "tables", "batched_table.tex")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    open(out, "w").write(tex)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
