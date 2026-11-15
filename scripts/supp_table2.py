import sys

import numpy as np

from common import ATLASES, BENCHMARKS, K, NAMES, SEEDS, fit, out, scores

TITLE = {**NAMES, "drfp": "Schneider50k", "bmarrow": "Bone marrow"}
TITLE.update(cerebellum="Cerebellum", plant=r"\textit{Arabidopsis} seed")
SETS = BENCHMARKS + ATLASES


def tuned(method, **kw):
    # a baseline of common.FIT at other settings
    def go(X, seed, metric):
        m = "jaccard" if metric == "jaccard" else "euclidean"
        if method == "UMAP":
            import umap

            return umap.UMAP(
                n_neighbors=K, random_state=seed, init="pca", metric=m, **kw
            ).fit_transform(X)
        if method == "openTSNE":
            from openTSNE import TSNE

            return np.asarray(
                TSNE(perplexity=30, random_state=seed, n_jobs=8, metric=m, **kw).fit(X)
            )
        import pacmap

        return pacmap.PaCMAP(
            n_neighbors=K, random_state=seed, verbose=False, **kw
        ).fit_transform(X)

    return go


# the defaults are UMAP at (0.1, 5), openTSNE at an exaggeration of 1, PaCMAP at (0.5, 2)
GRID = (
    [
        ("UMAP", dict(min_dist=md, negative_sample_rate=nsr))
        for md in (0.001, 0.01, 0.1, 0.5)
        for nsr in (5, 15)
        if (md, nsr) != (0.1, 5)
    ]
    + [("openTSNE", dict(exaggeration=e)) for e in (2, 4, 8, 12)]
    + [
        ("PaCMAP", dict(MN_ratio=mn, FP_ratio=fp))
        for mn in (0.5, 2)
        for fp in (1, 2, 4)
        if (mn, fp) != (0.5, 2)
    ]
)
TUNED = {
    m + " " + " ".join(f"{k}={v:g}" for k, v in kw.items()): tuned(m, **kw)
    for m, kw in GRID
}


def label(m):
    v = [p.split("=")[1] for p in m.split()[1:]]
    if m.startswith("openTSNE"):
        return f"openTSNE, exaggeration {v[0]}"
    return f"{m.split()[0]}, {v[0]}, {v[1]}"


ROWS = [("FloDR, $w=2$", "FloDR (w=2)"), ("UMAP, default ($0.1$, 5)", "UMAP")]
ROWS += [
    ("openTSNE, default (1)", "openTSNE"),
    ("PaCMAP, default ($0.5$, 2)", "PaCMAP"),
]
ROWS += [(label(m), m) for m in TUNED]

if __name__ == "__main__":
    # data sets may be named on the command line; all seven by default
    sets = [ds for ds in SETS if ds in sys.argv[1:]] or SETS
    for ds in sets:
        for m, fn in TUNED.items():
            for s in SEEDS:
                fit(ds, m, s, fn=fn)
    val = {m: np.array([scores(ds, m)[:, :2].mean(0) for ds in sets]) for _, m in ROWS}
    flodr = val["FloDR (w=2)"]
    lines = [" & " + " & ".join(TITLE[ds] for ds in sets) + r" \\"]
    for j, part in enumerate(("Recall", "CPD")):
        n = len(sets) + 1
        lines += [
            r"\midrule",
            rf"\multicolumn{{{n}}}{{@{{}}l}}{{\textit{{{part}}}}} \\",
        ]
        for i, (name, m) in enumerate(ROWS):
            # bold where a baseline is above FloDR
            cells = [
                rf"$\mathbf{{{v:.3f}}}$" if i and v > f else f"${v:.3f}$"
                for v, f in zip(val[m][:, j], flodr[:, j])
            ]
            lines.append(" & ".join([name] + cells) + r" \\")
    open(out("supp_table2.tex"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
