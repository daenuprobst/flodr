# Reproducing the paper

**NOTE: A local LLM was used to clean up and document the scripts.**

Every figure and number in the paper is produced here. Stages cache to `scripts/cache/`, so re-running only fills gaps. `RECOMPUTE=1` forces a rebuild. Commands assume the repo root and `pip install -e .` done.

Run everything:

```bash
bash scripts/run_all.sh          # numbers then figures; log in scripts/cache/pipeline.log
```

Or a single artifact:

| Paper item | Command |
|---|---|
| Initialisation ablation | `python scripts/ablations/pca_init_confound.py` |
| Diagnostic certificates | `python scripts/ablations/hidden_contrast.py` |
| Synthetic validation | `python scripts/ablations/synthetic_validation.py` |
| Component ablations | `python scripts/ablations/component_ablation.py` |
| Donor-split atlases | `python scripts/ablations/donor_split.py` |
| Held-out embedding benchmark | `python scripts/heldout_benchmark.py && python scripts/heldout_score.py` |
| Frontier figure | `python scripts/plotting/figures.py` |
| Held-out frontier figure | `python scripts/plotting/fig_heldout_frontier.py` |
| Main comparison figures | `python scripts/plotting/fig_main_comparison.py` |
| Diagnostics figure | `python scripts/plotting/fig_diagnostics_fields.py` |
| Atlas grid figure | `python scripts/plotting/fig_atlas_grid.py` |
| Scaling figure | `python scripts/plotting/fig_scaling.py` |
| Atlas figures | `DATASET=bmarrow python scripts/plotting/fig_scrna_combined.py` |
| Architecture figure | `pdflatex -output-directory=figures scripts/architecture.tex` |

The benchmark, certificate-band, scaling, and atlas-metric generators were removed; their results remain cached in `scripts/cache/` and are what the tables and figures read.

`heldout_benchmark.py` writes per-split embeddings to `scripts/cache/heldout_embeds/` (including the Parametric UMAP cells, produced separately with `umap-learn`'s `umap.parametric_umap.ParametricUMAP` in a TensorFlow environment, on identical splits); `heldout_score.py` scores whatever embeddings are present and writes `scripts/cache/heldout_benchmark.json`. The split for rep `r` is `np.random.default_rng(10_000 + r).permutation(n)` with the first `n // 5` indices held out.

## Data files

MNIST and Fashion-MNIST are fetched from OpenML and the Schneider 50k reactions from the DRFP repository automatically on first use (the DRFP frame is cached at `scripts/cache/drfp_schneider50k.npz`). The single-cell loaders expect local files, not shipped with the repo:

- `data/paul15.h5` — scanpy downloads it there itself on the first `sc.datasets.paul15()` call
- `data/scrna/fetal_bone_marrow.h5ad` and `data/scrna/cerebellum.h5ad` — read by `scripts/ablations/donor_split.py`
- `data/scrna/arabidopsis_seed.h5ad` — source of the cached `scripts/cache/scrna/plant.npz`

The atlas stages take a key in `DATASET`, one of `bmarrow`, `cerebellum`, `plant`. Training uses the GPU when one is available.
