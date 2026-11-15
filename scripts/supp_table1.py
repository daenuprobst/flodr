from common import ATLASES, BENCHMARKS, NAMES, out, scores

TITLE = {**NAMES, "drfp": "Schneider50k", "bmarrow": "Bone marrow"}
TITLE.update(cerebellum="Cerebellum", plant=r"\textit{Arabidopsis} seed")
# the regularisation strength of DREAMS: 0 is t-SNE, 0.15 the default
LAMBDA = (0, 0.05, 0.15, 0.3, 0.5, 0.8)
METHODS = ["DREAMS" if v == 0.15 else f"DREAMS (lambda={v})" for v in LAMBDA]

lines = [" & FloDR & " + " & ".join(map(str, LAMBDA)) + r" \\"]
for j, part in enumerate(("Recall", "CPD")):
    lines += [r"\midrule", rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{part}}}}} \\"]
    for ds in BENCHMARKS + ATLASES:
        means = [scores(ds, m)[:, j].mean() for m in ["FloDR (w=2)"] + METHODS]
        lines.append(f"{TITLE[ds]} & " + " & ".join(f"{x:.3f}" for x in means) + r" \\")

open(out("supp_table1.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
