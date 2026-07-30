import io
import os
import sys

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

matplotlib.use("Agg")

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPTS = os.path.dirname(HERE)
ROOT = os.path.dirname(SCRIPTS)
sys.path.insert(0, os.path.join(ROOT, "src"))

from flodr import viz  # noqa: E402

CACHE = os.path.join(SCRIPTS, "cache", "hnoca_3d.npz")
# only for the cell-type colour order, so it agrees with the 2D figure
FULL = os.path.join(SCRIPTS, "cache", "hnoca_atlas.npz")
OUT = os.path.join(ROOT, "figures", "hnoca_3d.gif")
FRAMES = int(os.environ.get("FRAMES", "72"))
MAX_PTS = int(os.environ.get("MAX_PTS", "120000"))
ELEV = float(os.environ.get("ELEV", "18"))
FPS = int(os.environ.get("FPS", "15"))
METHODS = (("flodr", "FloDR"), ("umap", "UMAP"))


def cube_limits(Y, pad=0.03):
    # one cube per method, sized on the largest axis so the aspect stays true and every
    # cell stays inside the box
    lo, hi = Y.min(0), Y.max(0)
    ctr = 0.5 * (lo + hi)
    half = (0.5 + pad) * float((hi - lo).max())
    return ctr - half, ctr + half


def main():
    if not os.path.exists(CACHE):
        raise SystemExit(f"no {CACHE}; run scripts/hnoca_3d.py first")

    z = np.load(CACHE, allow_pickle=True)
    all_kinds = np.asarray(z["cell_type"]).astype(str)
    n_have = len(all_kinds)

    cats, all_code = np.unique(all_kinds, return_inverse=True)
    # rank the types on the full atlas where that cache exists, so a type keeps the same
    # colour here as in the 2D figure, since this subsample alone reorders the middling ones
    counts = np.bincount(all_code, minlength=len(cats))
    if os.path.exists(FULL):
        full = np.asarray(np.load(FULL, allow_pickle=True)["cell_type"]).astype(str)
        lookup = {c: i for i, c in enumerate(cats)}
        counts = np.bincount([lookup[c] for c in full if c in lookup],
                             minlength=len(cats))
    by_size = np.argsort(counts)[::-1]
    pal = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    rank = np.empty(len(cats), int)
    rank[by_size] = np.arange(len(cats))

    rng = np.random.default_rng(0)
    sel = rng.permutation(n_have)[: min(MAX_PTS, n_have)]
    kinds = all_kinds[sel]
    layouts = {k: np.asarray(z[k], np.float64)[sel] for k, _ in METHODS}
    base_col = pal[rank[all_code[sel]] % 20][:, :3]

    boxes = {k: cube_limits(layouts[k]) for k, _ in METHODS}
    n_leg_col = 3
    frames = []

    for f in range(FRAMES):
        azim = -60.0 + 360.0 * f / FRAMES

        with plt.rc_context(viz.RC):
            fig = plt.figure(figsize=(9.6, 5.6), dpi=100)
            for i, (key, name) in enumerate(METHODS):
                Y = layouts[key]
                ax = fig.add_subplot(1, 2, i + 1, projection="3d")
                # the repo's own depth cue, with back-to-front order, colour faded into the
                # background and a size gradient, all from one camera direction
                order, weight, near = viz.depth_fog(Y, elev=ELEV, azim=azim,
                                                    strength=0.55)
                col = base_col * weight + 1.0 * (1.0 - weight)
                size = 2.1 * (0.35 + 1.3 * near)
                ax.scatter(Y[order, 0], Y[order, 1], Y[order, 2], s=size[order],
                           c=np.clip(col[order], 0, 1), linewidths=0, depthshade=False)
                lo, hi = boxes[key]
                ax.set_xlim(lo[0], hi[0])
                ax.set_ylim(lo[1], hi[1])
                ax.set_zlim(lo[2], hi[2])
                # unzoomed, a 3D axes leaves most of its rect empty with panes off
                ax.set_box_aspect((1, 1, 1), zoom=1.7)
                ax.view_init(elev=ELEV, azim=azim)
                ax.set_axis_off()

            fig.text(0.5, 0.975, f"HNOCA in 3D, {len(kinds):,} cells, by cell type",
                     fontsize=12.5, ha="center", va="center")
            # figure coordinates, so the labels cannot drift with the zoomed 3D rect
            for x, (_, name) in zip((0.26, 0.76), METHODS):
                fig.text(x, 0.915, name, fontsize=13, ha="center", va="center")
            handles = [Line2D([0], [0], marker="o", linestyle="", markersize=4.0,
                              markerfacecolor=pal[j % 20], markeredgewidth=0,
                              label=cats[by_size[j]]) for j in range(len(cats))]
            fig.legend(handles=handles, loc="lower center", ncol=n_leg_col,
                       frameon=False, fontsize=6.6, handletextpad=0.3,
                       columnspacing=0.9, labelspacing=0.28,
                       bbox_to_anchor=(0.5, -0.005))
            fig.subplots_adjust(left=0.0, right=1.0, top=0.99, bottom=0.19, wspace=0.0)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100)
            plt.close(fig)

        buf.seek(0)
        # quantise per frame, then let PIL build one shared palette on save
        frames.append(Image.open(buf).convert("RGB"))
        if (f + 1) % 12 == 0:
            print(f"  {f + 1}/{FRAMES} frames", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    quant = [im.quantize(colors=128, method=Image.MEDIANCUT) for im in frames]
    quant[0].save(OUT, save_all=True, append_images=quant[1:],
                  duration=int(1000 / FPS), loop=0, optimize=True, disposal=2)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB, {FRAMES} frames, "
          f"{len(kinds):,} cells per panel)", flush=True)


if __name__ == "__main__":
    main()
