import os
from functools import cache, partial

import common


def umap_arm(X, seed, metric, md=0.1, nsr=5):
    import umap

    kw = dict(min_dist=md, negative_sample_rate=nsr, init="pca", random_state=seed)
    return umap.UMAP(n_neighbors=common.K, **kw).fit_transform(X)


@cache
def scorer(ds):
    return common.Scorer(common.load(ds)[0], large=True)


ROWS = (
    ("FloDR", "FloDR (w=2)", None),
    ("FloDR, no ordinal term", "FloDR (w=0)", None),
    ("UMAP", "UMAP", None),
    (r"UMAP, min\_dist 0.01", "UMAP md0.01", partial(umap_arm, md=0.01)),
    (r"UMAP, negative\_sample\_rate 15", "UMAP nsr15", partial(umap_arm, nsr=15)),
    ("UMAP, both", "UMAP md0.01 nsr15", partial(umap_arm, md=0.01, nsr=15)),
)
path, lines = os.path.join(common.CACHE, "supp_table6", "scores.json"), []
for ds, title in (("hnoca", "HNOCA, 1.77M"), ("zebrafish", "Zebrafish, 1.22M")):
    lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{title} cells}}}} \\")
    for row, method, fn in ROWS:
        # without the ordinal term the full HNOCA latent gave non-finite coordinates
        if (ds, method) == ("hnoca", "FloDR (w=0)"):
            continue
        Y, secs = common.fit(ds, method, 0, fn)
        r = common.memo(path, f"{ds}/{method}", lambda: scorer(ds)(Y))
        lines.append(
            rf"\quad {row} & {r[0]:.4f} & {r[1]:.3f} & {r[2]:.3f} & {secs / 60:.0f} \\"
        )
open(common.out("supp_table6.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
