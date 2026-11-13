import numpy as np

from common import out
from openproblems import completed, official, scores

lines = [
    r"Method & Myeloid & HSPC & Pancreas & Cortex & Zebrafish & All & Completed \\"
]
lines.append(r"\midrule")
for name, S in scores().items():
    means = official(S).mean(1)
    # a dash where the published run has no score of the method on the data set
    cells = ["--" if np.isnan(r).all() else f"{x:.3f}" for r, x in zip(S, means)]
    cells += [f"{official(S).mean():.3f}", f"{completed(S):.3f}"]
    lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    if name == "FloDR, density term":
        lines.append(r"\midrule")

open(out("table1.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
