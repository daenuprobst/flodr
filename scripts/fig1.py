import os
import shutil
import subprocess
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

import common as C
import openproblems
from fig4 import LIM, density_lines

H = 4.05
# the letters a, b and d end at the same distance from the left edge, in inches
EDGE = 1.2 - 74 / 72
DATA = (*C.BENCHMARKS, *C.ATLASES)
NAMES = "MNIST|Fashion-MNIST|paul15|Schneider50k|bone marrow|cerebellum|seed".split("|")
# the other methods of Fig. 3, in grey
REST = "DREAMS TriMap PaCMAP LocalMAP UMATO ParamRepulsor PHATE PyMDE PCUMAP SQuadMDS"
REST += " PCA-2"
GREY = "#C8C8C8"
# a, the architecture, drawn by TikZ above the panels
SCHEMATIC = r"""
\documentclass[tikz,border=0pt]{standalone}PREAMBLE\usepackage{amssymb}
\usetikzlibrary{arrows.meta,calc,fit}
\definecolor{ours}{HTML}{AA4499}
\begin{document}
\begin{tikzpicture}[font=\fontsize{8}{9.4}\selectfont,
  box/.style={draw, semithick, rounded corners=1pt, fill=white, align=center,
    inner sep=1.2mm, minimum height=7.5mm},
  dens/.style={box, draw=black!60, fill=black!3, minimum height=6.5mm, minimum width=44mm},
  grp/.style={draw=black!45, dashed, rounded corners=2pt, inner sep=1.4mm},
  sub/.style={font=\fontsize{7}{8.4}\selectfont, text=black!60, align=center},
  arr/.style={-{Latex[length=1.8mm,width=1.6mm]}, semithick, black!80}]
\useasboundingbox (0,BOTTOM) rectangle (18.288,4.826);
\node[anchor=north east, inner sep=0pt, font=\fontsize{9}{10}\selectfont\bfseries]
  at (EDGE,4.8) {a};
\node[box, minimum width=17mm] (x) at (1.15,2.55)
  {$x \in \mathbb{R}^{D}$\\[-0.2mm]{\fontsize{7}{8}\selectfont\color{black!60}cell, $D$ PCs}};
\node[box, minimum width=13mm] (z) at (3.75,2.55) {$z \in \mathbb{R}^{D}$};
\draw[arr] (x) -- node[midway, above=0.2mm] {$V$} node[sub, midway, below=0.3mm] {whiten} (z);
\foreach \i/\p in {1/5.30, 2/6.08, 3/6.86, 4/7.64} \node[box, minimum width=6.6mm,
  minimum height=6.6mm, inner sep=0.5mm, fill=black!5] (c\i) at (\p,2.55) {$c_{\i}$};
\node[grp, fit=(c1)(c4)] (cgrp) {};
\node[anchor=south] at ($(cgrp.north)+(0,0.02)$) {$f_\theta$: affine couplings};
\draw[arr] (z) -- (cgrp);
\node[box, very thick, minimum width=21mm] (y) at (10.55,3.05) {map $y \in \mathbb{R}^{2}$};
\node[box, draw=black!55, fill=black!8, minimum width=21mm] (r) at (10.55,2.05)
  {residual $r \in \mathbb{R}^{D-2}$};
\draw[semithick, black!80] (cgrp.east) -- (8.95,2.55);
\draw[arr, rounded corners=1.2mm] (8.95,2.55) |- (y.west);
\draw[arr, rounded corners=1.2mm] (8.95,2.55) |- (r.west);
\node[box, draw=ours!80, text=black!85, fill=ours!5, anchor=south] (L) at (10.55,3.98)
  {$\mathcal{L}_{\mathrm{attr}} + w_{\mathrm{rep}}\mathcal{L}_{\mathrm{rep}}
  + w_{\mathrm{glob}}\mathcal{L}_{\mathrm{glob}} + w\,\mathcal{L}_{\mathrm{ord}}$};
\draw[arr, thick, ours] (L.south) -- (y.north);
\node[sub, anchor=west, text=ours] at ($(L.south)+(0.12,-0.27)$) {layout loss, on $y$ only};
\node[dens] (py) at (15.85,3.05) {$p(y)$: Gaussian mixture on the map};
\node[dens] (pr) at (15.85,2.05) {$p(r \mid y)$: Gaussian scale mixture};
\node[grp, fit=(py)(pr)] (dgrp) {};
\node[sub, anchor=south] at ($(dgrp.north)+(0,0.02)$) {fitted afterwards, map frozen};
\draw[arr] (y.east) -- (dgrp.west |- y.east);
\draw[arr] (r.east) -- (dgrp.west |- r.east);
\draw[arr, rounded corners=1.2mm] (10.55,1.62) -- (10.55,1.0) -- (1.15,1.0) -- (x.south);
\node[anchor=south] at (5.85,1.05) {$x = V^{-1} f_\theta^{-1}(y, r)$, exact};
\node[anchor=north west, inner sep=0pt] at (0,0.648) {\includegraphics{panels.pdf}};
\end{tikzpicture}
\end{document}
"""


def style(ax, grid=True):
    ax.tick_params(labelsize=6.5, length=2, pad=1.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    if grid:
        ax.grid(True, axis="y", lw=0.4, color="0.92")
        ax.set_axisbelow(True)


def main():
    plt.rcParams.update(C.RC)
    fig = plt.figure(figsize=(C.TEXT_W, H))

    def at(x, y, w, h):
        return fig.add_axes([x / C.TEXT_W, y / H, w / C.TEXT_W, h / H])

    def dot(c, m="o", **kw):
        return plt.Line2D([], [], ls="", marker=m, color=c, **kw)

    # b, seed means on every data set: FloDR, UMAP and openTSNE against the others
    hi = {"FloDR (w=2)": (C.OURS, "o", 26), "UMAP": (*C.METHOD_STYLE["UMAP"], 16)}
    hi["openTSNE"] = (C.METHOD_STYLE["openTSNE"][0], "o", 16)
    rest = REST.split()
    sm = {(d, m): C.scores(d, m).mean(0) for d in DATA for m in (*hi, *rest)}
    axes = [at(0.86, 2.18, 1.18, 1.62), at(2.18, 2.18, 1.18, 1.62)]
    for j, ax in enumerate(axes):
        for i, d in enumerate(DATA):
            x = [sm[d, m][j] for m in rest]
            ax.scatter(x, [i] * len(rest), s=7, color=GREY, lw=0)
            for z, (m, (c, mk, s)) in enumerate(hi.items()):
                kw = dict(s=s, marker=mk, color=c, lw=0.4, edgecolor="white")
                ax.scatter(sm[d, m][j], i, zorder=3 + (z == 0), **kw)
        ax.set_yticks(range(len(DATA)), NAMES if j == 0 else [])
        ax.set_ylim(len(DATA) - 0.4, -0.6)
        ax.spines[["top", "right"]].set_visible(False)
        style(ax)
    axes[0].set_xlim(0, 0.56)
    axes[1].set_xlim(-0.05, 0.95)
    axes[0].set_xlabel("neighbours kept\n(recall@15)", fontsize=7, labelpad=2)
    axes[1].set_xlabel("order of distances\n(CPD)", fontsize=7, labelpad=2)
    C.letter(axes[0], "b", (EDGE - 0.86) * 72)
    h = [dot(c, mk, ms=np.sqrt(s) * 0.9, mec="w", mew=0.4) for c, mk, s in hi.values()]
    h.append(dot(GREY, ms=2.6))
    labels = ["FloDR", "UMAP", "openTSNE", f"{len(rest)} other methods"]
    kw = dict(handletextpad=0.15, columnspacing=0.8, borderaxespad=0.1, ncol=4)
    kw.update(loc="lower left", bbox_to_anchor=(-0.05, 1), frameon=False, fontsize=6.5)
    axes[0].legend(h, labels, **kw)
    # c, 60 000 random pairs of bone marrow cells, distances over their 99th centile
    X = C.load("bmarrow")[0]
    rng = np.random.default_rng(0)
    i, j = rng.integers(0, len(X), 60_000), rng.integers(0, len(X), 60_000)
    i, j = i[i != j], j[i != j]
    dh = np.linalg.norm(X[i] - X[j], axis=1)
    for k, m in enumerate(("FloDR (w=2)", "UMAP")):
        ax, Y = at(3.98 + k * 1.52, 2.18, 1.38, 1.38), C.fit("bmarrow", m, 0)[0]
        dl = np.linalg.norm(Y[i] - Y[j], axis=1)
        c = (C.OURS, C.METHOD_STYLE["UMAP"][0])[k]
        density_lines(ax, dh / np.percentile(dh, 99), dl / np.percentile(dl, 99), c)
        ax.plot([0, LIM[1]], [0, LIM[1]], ls=":", lw=0.7, color="0.3")
        ax.set(xlim=LIM, ylim=LIM, xticks=[0, 0.5, 1], yticks=[0, 0.5, 1], box_aspect=1)
        ax.set_xticklabels(["0", "0.5", "1"])
        ax.set_yticklabels(["0", "0.5", "1"] if k == 0 else [])
        name = (r"\textbf{FloDR}", "UMAP")[k]
        rho = spearmanr(dh, dl).statistic
        ax.set_title(f"{name}, $\\rho = {rho:.2f}$", fontsize=7)
        ax.set_xlabel("input distance", fontsize=7, labelpad=1)
        style(ax, False)
        if k == 0:
            ax.set_ylabel("map distance", fontsize=7, labelpad=1)
            C.letter(ax, "c", -20)
    s = "human fetal bone marrow, 60\\,000 random pairs"
    fig.text(5.37 / C.TEXT_W, 3.86 / H, s, ha="center", fontsize=6.5, color="0.35")
    # d, every method of the OpenProblems task by metric family, FloDR and UMAP as in b
    scores = {m: openproblems.official(S) for m, S in openproblems.scores().items()}
    rows = {"all 14 metrics": openproblems.METRICS, **openproblems.FAMILIES}
    ax, green = at(1.2, 0.32, 4.15, 1.18), C.METHOD_STYLE["UMAP"][0]
    for i, family in enumerate(rows.values()):
        cols = [openproblems.METRICS.index(k) for k in family]
        x = {m: S[:, cols].mean() for m, S in scores.items()}
        ours, dens, umap = (x.pop(m) for m in ("FloDR", "FloDR, density term", "UMAP"))
        ax.scatter(list(x.values()), [i] * len(x), s=9, color=GREY, lw=0, zorder=2)
        kw = dict(lw=0.4, edgecolor="white")
        ax.scatter(umap, i, s=14, marker="s", color=green, zorder=3, **kw)
        ax.scatter(ours, i, s=30, color=C.OURS, zorder=5, **kw)
        kw = dict(facecolor="white", edgecolor=C.OURS, lw=0.9)
        ax.scatter(dens, i, s=26, zorder=4, **kw)
        # the rank of FloDR at its defaults among the published methods
        r = 1 + sum(v > ours for v in (umap, *x.values()))
        text = {1: "1st", 2: "2nd", 3: "3rd"}.get(r, f"{r}th")
        c, tr = C.OURS_DARK if r == 1 else "0.35", ax.get_yaxis_transform()
        ax.text(1.03, i, text, transform=tr, va="center", fontsize=6.5, color=c)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_yticks(range(len(rows)), list(rows), fontsize=6.5)
    ax.set_xlim(0, 1)
    s = "OpenProblems score, scaled between the controls (mean over "
    s += f"{len(openproblems.DATASETS)} data sets)"
    ax.set_xlabel(s, fontsize=7, labelpad=2)
    style(ax)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines[["top", "right", "left"]].set_visible(False)
    C.letter(ax, "d", (EDGE - 1.2) * 72)
    h = [dot(C.OURS, ms=4.5, mec="w", mew=0.4), dot(C.OURS, ms=4.2, mfc="w", mew=0.9)]
    h += [dot(green, "s", ms=3.4, mec="w", mew=0.4), dot(GREY, ms=3)]
    labels = ["FloDR", "FloDR, density term", "UMAP", f"{len(x)} other methods"]
    kw = dict(handletextpad=0.3, labelspacing=0.5, frameon=False, fontsize=6.5)
    ax.legend(h, labels, loc="upper left", bbox_to_anchor=(1.12, 1.0), **kw)
    # the schematic over the panels, both at print size
    with tempfile.TemporaryDirectory() as tmp:
        fig.savefig(os.path.join(tmp, "panels.pdf"), dpi=300)
        tex = SCHEMATIC.replace("BOTTOM", f"{0.648 - H * 2.54:.3f}")
        tex = tex.replace("EDGE", f"{EDGE * 2.54:.4f}")
        tex = tex.replace("PREAMBLE", C.RC["text.latex.preamble"])
        open(os.path.join(tmp, "fig1.tex"), "w").write(tex)
        run = dict(cwd=tmp, check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["pdflatex", "-interaction=nonstopmode", "fig1.tex"], **run)
        shutil.copy(os.path.join(tmp, "fig1.pdf"), C.out("fig1.pdf"))


if __name__ == "__main__":
    main()
