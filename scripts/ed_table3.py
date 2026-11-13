from common import out
from openproblems import official, scores

HEAD = r"Method & ARI & NMI & Cont. & Trust. & QNX & LCMC & AUC & $Q_\mathrm{loc}$ & "
HEAD += r"$Q_\mathrm{glob}$ & Dens. & Centr. & Label & Waypt. & Spec. \\"

lines = [HEAD, r"\midrule"]
for name, S in scores().items():
    cells = [f"{x:.2f}" for x in official(S).mean(0)]
    lines.append(f"{name} & " + " & ".join(cells) + r" \\")
    if name == "FloDR, density term":
        lines.append(r"\midrule")

open(out("ed_table3.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
