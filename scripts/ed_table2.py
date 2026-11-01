from flodr import FloDR

from common import BENCHMARKS, NAMES, SEEDS, fit, load, out, scores

TITLE = {**NAMES, "paul15": "paul15 (scRNA-seq)", "drfp": "Schneider 50k (reactions)"}
# each arm changes one setting of the released default
ARMS = {
    "base": {},
    "$L=2$": {"n_layers": 2},
    "$L=8$": {"n_layers": 8},
    "no sketch": {"sketch": 0},
    "+NLL": {"w_nll": 0.5},
    "no Fourier branch": {"fine_eps": 0.0},
    "scalar gate": {"gate_display": 0.0},
    "800 steps": {"iters": 800},
}

lines = [r"Data set & Arm & recall@15 & CPD ($\rho$) \\", r"\midrule"]
for ds in BENCHMARKS:
    if ds != BENCHMARKS[0]:
        lines.append(r"\midrule")
    head = TITLE[ds]
    for label, advanced in ARMS.items():
        # the sketch conditioner reads inputs of up to 64 dimensions directly
        if "sketch" in advanced and load(ds)[0].shape[1] <= 64:
            continue
        name = "".join(f"{k}{v}" for k, v in advanced.items())
        method = f"FloDR {name}" if advanced else "FloDR (w=2)"
        for s in SEEDS:
            model = FloDR(w=2.0, random_state=s, verbose=False, advanced=advanced)
            fit(ds, method, s, fn=lambda X, seed, metric: model.fit(X).embedding_)
        rec, cpd = scores(ds, method)[:, :2].T
        lines.append(
            f"{head} & {label} & ${rec.mean():.3f} \\pm {rec.std():.3f}$ & "
            f"${cpd.mean():.3f} \\pm {cpd.std():.3f}$ \\\\"
        )
        head = ""

open(out("ed_table2.tex"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
