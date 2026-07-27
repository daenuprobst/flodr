"""Main comparison figures and LaTeX table, from the cache/eval caches written by
evaluate.py, so the figures can be retuned without recomputing anything.

Layout: one row per method, one column per dataset; equal-aspect square panels are placed
explicitly because constrained_layout would shrink them to the tightest cell. `main` draws
FloDR (w=2, no ordinal), UMAP, openTSNE and the PCA-2 control; `appendix` the rest.
The table carries every method regardless of which figure draws it.

Run: [DATASETS=mnist,fmnist,paul15,drfp] .venv/bin/python scripts/fig_main_comparison.py
"""
import json
import os
import sys
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, SCRIPTS)
sys.path.insert(0, HERE)
from datasets import CACHE as DCACHE  # noqa: E402

CACHE = os.path.join(DCACHE, "eval")
FIG = os.path.join(ROOT, "figures")
PAPER = os.path.join(ROOT, "tables")
os.makedirs(PAPER, exist_ok=True)
ORDER = os.environ.get("DATASETS", "mnist,fmnist,paul15,drfp").split(",")

MAIN = ["FloDR (w=2)", "FloDR (no ordinal)", "UMAP", "openTSNE", "PCA-2"]
APPENDIX = ["TriMap", "PaCMAP", "LocalMAP", "PHATE", "PyMDE", "PCUMAP", "SQuadMDS"]
METHODS = MAIN + APPENDIX                       # table order; every method is tabulated
SHORT = {"fmnist": "Fashion-MNIST", "paul15": "paul15 (scRNA)", "drfp": "Schneider 50k"}
INK, GRID, ALERT = "#0b0b0b", "#e4e3df", "#c1442f"
CLASS_CMAP = "tab10"
PT = float(os.environ.get("PT", "1.6"))
ALPHA = float(os.environ.get("ALPHA", "0.75"))


def panel(ax, Y, labels):
    lo, hi = np.percentile(Y, [0.5, 99.5], axis=0)        # robust frame, ignore stragglers
    # square frame at equal aspect: one half-range for both axes, so panels are directly
    # comparable in size and no layout is stretched to fill its cell
    center = (lo + hi) / 2
    half = 0.53 * (hi - lo).max()
    cmap = plt.get_cmap("tab20" if len(np.unique(labels)) > 10 else CLASS_CMAP)
    ax.scatter(Y[:, 0], Y[:, 1], s=PT, c=[cmap(int(lb) % cmap.N) for lb in labels],
               linewidths=0, alpha=ALPHA, rasterized=True)
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(0.6)


def draw(ds, meta, rows, stem):
    """rows x ds panels, sized to sit on A4 portrait at 100% so point sizes are literal."""
    if not rows:
        return
    nrow, ncol = len(rows), len(ds)
    LEFT, TOP, GAP = 0.60, 0.62, 0.05           # row labels, column headers, panel gap
    edge = min((6.69 - LEFT - GAP * (ncol - 1)) / ncol, (10.12 - TOP - GAP * (nrow - 1)) / nrow)
    fig_w = LEFT + ncol * edge + (ncol - 1) * GAP
    fig_h = TOP + nrow * edge + (nrow - 1) * GAP
    fig = plt.figure(figsize=(fig_w, fig_h))
    axes = [[fig.add_axes([(LEFT + col * (edge + GAP)) / fig_w,
                           (fig_h - TOP - (row + 1) * edge - row * GAP) / fig_h,
                           edge / fig_w, edge / fig_h]) for col in range(ncol)]
            for row in range(nrow)]
    coords = {d: np.load(f"{CACHE}/{d}_coords.npz") for d in ds}
    for row, mname in enumerate(rows):
        for col, d in enumerate(ds):
            ax, npz = axes[row][col], coords[d]
            if mname not in npz.files or mname not in meta[d]["methods"]:
                ax.set_axis_off()
                continue
            panel(ax, npz[mname], npz["labels"])
            ms = meta[d]["methods"][mname]["mean_std"]
            # a method that cannot consume this row's metric is fit on Euclidean but scored
            # against the metric the field reads: say so on the panel, not in a footnote
            blind = mname in meta[d].get("euclid_only", [])
            ax.text(0.03, 0.03, f"recall {ms['recall'][0]:.3f}\nglobal {ms['gspear'][0]:.3f}"
                    + ("\nEuclidean-only" if blind else ""),
                    transform=ax.transAxes, fontsize=6.4, va="bottom", ha="left",
                    color=ALERT if blind else INK,
                    bbox=dict(fc="white", ec="none", alpha=0.72, pad=1.4))
            if row == 0:
                met = meta[d].get("metric", "euclid")
                ax.set_title(f"{SHORT.get(d, meta[d]['dataset'])}\n"
                             f"$n$={meta[d]['n']}, $d$={meta[d]['d']}\n"
                             f"{meta[d]['space'].replace(' -> ', r'$\to$')}\n"
                             f"{'Jaccard' if met == 'jaccard' else 'Euclidean'}",
                             fontsize=7.0, linespacing=1.3)
        # every FloDR arm is ours, not just the headline one; two lines because the variant
        # alone would not fit along a ~1.2in panel edge as one string
        ours = mname.startswith("FloDR")
        label = f"FloDR (ours)\n{mname[7:-1]}" if ours else mname
        axes[row][0].set_ylabel(label, fontsize=7.4, fontweight="bold" if ours else "normal",
                                labelpad=4, linespacing=1.2)
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.png / .pdf  ({nrow}x{ncol}, panel {edge:.2f}in)", flush=True)


# hi=True -> higher is better. SNS is a stress: lower is better. hi=None -> never ranked.
FIELDS = [("recall", "recall@15", True), ("trust", "trust", True), ("cont", "cont.", True),
          ("gspear", r"CPD ($\rho$)", True), ("pearson", "Shepard $r$", True),
          ("sns", r"SNS $\downarrow$", False), ("centroid", "centroid", True),
          ("secs", "time (s)", None)]
QUALITY = [(fld, hdr, higher) for fld, hdr, higher in FIELDS if higher is not None]
FIT_OPEN = r"  \resizebox{\ifdim\width>\textwidth\textwidth\else\width\fi}{!}{%"


def _vals(meta, d, fld, present):
    out = {}
    for mname in present:
        ms = meta[d]["methods"][mname]["mean_std"].get(fld)
        if ms and ms[0] is not None:
            out[mname] = ms[0]
    return out


def _ranks(vals, higher):
    """{method: rank}, 1 = best. Ties share the average rank."""
    order = sorted(vals, key=lambda mname: vals[mname], reverse=higher)
    ranks, i = {}, 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for mname in order[i:j + 1]:
            ranks[mname] = avg
        i = j + 1
    return ranks


def _mark(inner, rank):
    """Best bold, second underlined. `inner` is math content without the $ $ -- \\textbf is a
    text-mode command and cannot reach material inside $ $, so it must be \\mathbf inside."""
    if rank == 1:
        return rf"$\mathbf{{{inner}}}$"
    if rank == 2:
        return rf"$\underline{{{inner}}}$"
    return f"${inner}$"


def _tex(s):
    for old, new in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("->", r"$\to$")):
        s = s.replace(old, new)
    return s


def _name(mname, blind):
    return (f"{mname[:-1]}, ours)" if mname.startswith("FloDR") else mname) + (
        r"$^{\dagger}$" if blind else "")


def _notes(extra=()):
    """Left-aligned notes: a bare \\footnotesize under \\centering comes out centred, the
    minipage restores normal justification."""
    return ([r"  \vspace{3pt}", r"  \begin{minipage}{\textwidth}", r"  \footnotesize"]
            + list(extra) + [r"  \end{minipage}"])


def write_latex(ds, meta, rows):
    """booktabs table: best per column per dataset bold, second best underlined."""
    lines = [r"% generated by scripts/fig_main_comparison.py -- do not hand-edit",
             r"\begin{table*}[t]", r"  \centering",
             r"  \caption{FloDR against neighbour-embedding baselines and a linear control across four",
             r"    domains. Each dataset is measured in the metric its own field reads; all methods see",
             r"    the identical matrix. Mean $\pm$ s.d.\ over 3 seeds; \textbf{best} and",
             r"    \underline{second best} per column within each dataset.}",
             r"  \label{tab:main-comparison}",
             r"  \setlength{\tabcolsep}{4pt}",
             FIT_OPEN,
             r"  \begin{tabular}{@{}l" + "r" * len(FIELDS) + r"@{}}", r"    \toprule",
             r"    Method & " + " & ".join(hdr for _, hdr, _ in FIELDS) + r" \\"]
    for d in ds:
        met = "Jaccard" if meta[d].get("metric") == "jaccard" else "Euclidean"
        lines += [r"    \midrule",
                  rf"    \multicolumn{{{len(FIELDS) + 1}}}{{@{{}}l}}{{\emph{{{_tex(meta[d]['dataset'])}}}"
                  rf" ($n={meta[d]['n']}$, $d={meta[d]['d']}$, {_tex(meta[d]['space'])};"
                  rf" {met})}} \\"]
        present = [mname for mname in rows if mname in meta[d]["methods"]]
        ranks = {fld: _ranks(_vals(meta, d, fld, present), higher)
                 for fld, _, higher in QUALITY}
        for mname in present:
            ms = meta[d]["methods"][mname]["mean_std"]
            cells = []
            for fld, _, higher in FIELDS:
                val = ms.get(fld)
                if not val or val[0] is None:
                    cells.append("{--}")
                    continue
                inner = (rf"{val[0]:.1f}" if fld == "secs" else
                         rf"{val[0]:.3f} \pm {val[1]:.3f}" if fld in ("recall", "gspear", "sns") else
                         rf"{val[0]:.3f}")
                cells.append(f"${inner}$" if higher is None else _mark(inner, ranks[fld].get(mname)))
            lines.append(rf"    {_name(mname, mname in meta[d].get('euclid_only', []))} & "
                         + " & ".join(cells) + r" \\")
    lines += [r"    \bottomrule", r"  \end{tabular}}"]
    lines += _notes([
        r"  $^{\dagger}$Euclidean-only: fit on Euclidean distance but scored in the metric the row",
        r"  is measured in, which it cannot consume.",
        r"  \emph{recall@15} $\equiv Q_{NX}(15)$ (Lee \& Verleysen, Neurocomputing 2009).",
        r"  \emph{cont.} = continuity, the missed-neighbour dual of trustworthiness (Venna \&",
        r"  Kaski, ICANN 2001); the two are only meaningful as a pair.",
        r"  \emph{CPD} (Kobak \& Berens, Nat Commun 2019) = Shepard goodness (Espadoto et al.,",
        r"  IEEE TVCG 2021): the Spearman of pairwise distances. \emph{Shepard $r$} is the Pearson",
        r"  of the same pairs.",
        r"  \emph{SNS} = scale-normalized stress (Smelser et al., IEEE TVCG 32(7), 2026),",
        r"  minimised over isotropic scale and therefore scale-invariant. It is the global metric",
        r"  FloDR does not optimise: the ordinal term hinges on the SIGN of a measured distance",
        r"  difference and carries no magnitude information, whereas SNS is pure magnitude",
        r"  fidelity. CPD and triplet accuracies are rank-based and so are the same family as that",
        r"  term; SNS is not.",
        r"  PCA-2 is a linear control, not a competitor: PCA is classical MDS, so it is the",
        r"  least-squares-optimal linear distance-preserver and tops a global metric by",
        r"  construction; it is the bar to clear, not an anomaly.",
        r"  \emph{time} is not like-for-like and is never ranked: FloDR runs on an RTX~4070~Ti,",
        r"  every baseline is a CPU-only library, and UMAP is pinned with \texttt{random\_state},",
        r"  which serialises it onto one thread ($\sim$7$\times$ slower than its default).",
    ])
    lines.append(r"\end{table*}")
    out = os.path.join(PAPER, "main_comparison_table.tex")
    open(out, "w").write("\n".join(lines) + "\n")
    print(f"wrote {out}", flush=True)


def main():
    ds = [d for d in ORDER if os.path.exists(f"{CACHE}/{d}_coords.npz")]
    if not ds:
        raise SystemExit(f"no cache in {CACHE}/ -- benchmark caches were removed with the generators")
    meta = {d: json.load(open(f"{CACHE}/{d}_metrics.json")) for d in ds}
    have = [mname for mname in METHODS if any(mname in meta[d]["methods"] for d in ds)]
    plt.rcParams.update({"font.size": 8, "figure.dpi": 300})
    os.makedirs(FIG, exist_ok=True)
    draw(ds, meta, [mname for mname in MAIN if mname in have], f"{FIG}/main_comparison")
    draw(ds, meta, [mname for mname in APPENDIX if mname in have],
         f"{FIG}/main_comparison_appendix")
    write_latex(ds, meta, have)


if __name__ == "__main__":
    main()
