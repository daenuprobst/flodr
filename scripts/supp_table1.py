import common
from supp_table2 import SUBSETS, E, e1, e4, e6, floor, welch
from supp_table3 import no_collapse, placement_test


def credit(function, objective, flow):
    # the rules written before the runs: fitting a function when FloDR and the encoder
    # both beat free coordinates and FloDR does not beat the encoder, the objective when
    # free coordinates match FloDR, the flow when FloDR beats the encoder (E1 and E4,
    # beyond the tie margin)
    c = ["function"] if function else ["objective"] if objective else []
    c += ["flow"] if flow["verdict"] == "a" and not flow["within"] else []
    return " and ".join(c) or "unresolved"


def bench(name, name_free=None):
    # E6 per subset: a function beats free coordinates when the interval of free minus
    # it lies below zero, even inside the tie margin; the rule asks two of three
    n_fn = n_free = n_eq = 0
    for ds in SUBSETS:
        f, m = (e6(ds, name, a, "free", name_free)["ci"] for a in ("flodr", "mlp"))
        n_fn += f[1] < 0 and m[1] < 0
        n_free += f[0] > 0 or m[0] > 0
        n_eq += -floor(name) < f[0] and f[1] < floor(name)
    return credit(n_fn >= 2 and n_free == 0, n_eq >= 2 and n_fn == 0, e1(name))


def atlas(name):
    # E4: FloDR and the encoder on both atlases, free coordinates on the bone marrow
    ff, ef = (e4(name, a, "free")["verdict"] for a in ("flodr", "mlp"))
    return credit(ff == ef == "a", ff == "equivalent", e4(name, "flodr", "mlp"))


def pair(*comparisons):
    v = {
        {"a": "flow", "b": "encoder"}.get(c["verdict"], "unresolved")
        for c in comparisons
    }
    return v.pop() if len(v) == 1 else "unresolved"


def rows():
    b = {
        m: bench(m)
        for m in ("M1 recall", "M1 cpd", "M1 geo", "M1 far5", "M2r opt recall")
    }
    a = {
        m: atlas(m)
        for m in ("M1 recall", "M1 cpd", "M1 geo", "M1 far5", "M2r opt recall")
    }
    for k in ("knn", "dist"):
        b[f"M3 {k}"], b[f"M4x {k}"] = bench(f"M3 {k}"), bench(f"M4x {k}")
        b[f"M5x {k}"] = bench(f"M5x forward {k}", f"M5x interp {k}")
    fwd = bench("M2r forward recall", "M2r interp recall")
    stable = "unresolved" if placement_test()[1] else "function"
    donors = pair(welch(E("E3"), "M2r forward recall", "flodr", "mlp"))
    narrow = pair(
        *(welch(E("E2"), "M1 recall", "flodr", m) for m in ("mlp", "mlp_pad"))
    )
    held = no_collapse(E("E5"))
    held_atlas = no_collapse(
        {"donors": E("E3")["donors"], "bmarrow": E("E4")["bmarrow"]}
    )
    group = r"\multicolumn{{4}}{{@{{}}l}}{{\emph{{{}}}}} \\"
    text = rf"""
{group.format("Layout")}
Neighbours kept (recall) & {b["M1 recall"]} & {a["M1 recall"]} & E1, E4, E6 \\
Global structure (CPD) & {b["M1 cpd"]} & {a["M1 cpd"]} & E1, E4, E6 \\
Distances along the neighbour graph & {b["M1 geo"]} & {a["M1 geo"]} & E1, E4, E6 \\
Order of the largest distances & {b["M1 far5"]} & {a["M1 far5"]} & E1, E4, E6 \\
{group.format("Refits")}
Same neighbours across seeds & {b["M3 knn"]} & -- & E1, E6 \\
Same neighbours, all points against 80\% & {b["M4x knn"]} & -- & E1, E6 \\
Same neighbours, disjoint halves & {b["M5x knn"]} & -- & E1, E6 \\
Map distances across seeds & {b["M3 dist"]} & -- & E1, E6 \\
Map distances, all points against 80\% & {b["M4x dist"]} & -- & E1, E6 \\
Map distances, disjoint halves & {b["M5x dist"]} & -- & E1, E6 \\
{group.format("New points")}
Placement by one forward pass & {fwd} & -- & E1, E6 \\
Placement by the optimised transform$^\ast$ & {b["M2r opt recall"]} &
    {a["M2r opt recall"]} & E1, E4, E6 \\
Same placement alone, in any batch, thread count or device$^\ast$ & {stable} & -- &
    follow-up \\
Cells from held-out donors, forward pass & -- & {donors} & E3 \\
No collapse of new inputs & {held} & {held_atlas} & E5 \\
Narrow inputs, $D=10$ (two latents) & {narrow} & -- & E2 \\
{group.format("By construction")}
Exact inverse and normalised density & flow & flow & all fits \\
Inverse, density and placement agree$^\ast$ & flow & flow &
    Supp.\ Note~\ref{{sec:flowonly}} \\
Hidden-information audit$^\ast$ & none & none & -- \\
"""
    return [
        r.strip() + r" \\" for r in " ".join(text.split()).split(r" \\") if r.strip()
    ]


if __name__ == "__main__":
    text = "\n".join(rows())
    open(common.out("supp_table1.tex"), "w").write(text + "\n")
    print(text)
