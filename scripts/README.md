# Reproducing the paper

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
| Held-out optA frontier + table | `python scripts/plotting/fig_heldout_opta.py && python scripts/plotting/tab_heldout_opta.py` |
| Atlas held-out (donor-split) metrics | `python scripts/atlas_frontier.py` |
| Main comparison figures | `python scripts/plotting/fig_main_comparison.py` |
| Diagnostics figure | `python scripts/plotting/fig_diagnostics_fields.py` |
| Atlas grid figure | `python scripts/plotting/fig_atlas_grid.py` |
| Scaling figure | `python scripts/plotting/fig_scaling.py` |
| Atlas figures | `DATASET=bmarrow python scripts/plotting/fig_scrna_combined.py` |
| Architecture figure | `pdflatex -output-directory=figures scripts/architecture.tex` |
| HNOCA figure (README) | `python scripts/hnoca_atlas.py && python scripts/hnoca_metrics.py && python scripts/plotting/fig_hnoca.py` |
| HNOCA 3D animation | `python scripts/hnoca_3d.py && python scripts/plotting/fig_hnoca_3d.py` |
| Zebrafish figure | `python scripts/zebrafish_atlas.py && python scripts/zebrafish_umap.py && python scripts/zebrafish_metrics.py && python scripts/plotting/fig_zebrafish.py` |

The benchmark, certificate-band, scaling, and atlas-metric generators were removed; their results remain cached in `scripts/cache/` and are what the tables and figures read.

`heldout_benchmark.py` writes per-split embeddings to `scripts/cache/heldout_embeds/` (including the Parametric UMAP cells, produced separately with `umap-learn`'s `umap.parametric_umap.ParametricUMAP` in a TensorFlow environment, on identical splits); `heldout_score.py` scores whatever embeddings are present and writes `scripts/cache/heldout_benchmark.json`. The split for rep `r` is `np.random.default_rng(10_000 + r).permutation(n)` with the first `n // 5` indices held out.

## Data files

MNIST and Fashion-MNIST are fetched from OpenML and the Schneider 50k reactions from the DRFP repository automatically on first use (the DRFP frame is cached at `scripts/cache/drfp_schneider50k.npz`). The single-cell loaders expect local files, not shipped with the repo:

- `data/paul15.h5`: scanpy downloads it there itself on the first `sc.datasets.paul15()` call
- `data/scrna/fetal_bone_marrow.h5ad` and `data/scrna/cerebellum.h5ad`: read by `scripts/ablations/donor_split.py`
- `data/scrna/arabidopsis_seed.h5ad`: source of the cached `scripts/cache/scrna/plant.npz`

The atlas stages take a key in `DATASET`, one of `bmarrow`, `cerebellum`, `plant`. Training uses the GPU when one is available.

## HNOCA

The Human Neural Organoid Cell Atlas (He et al., Nature 635, 690-698, 2024). Nothing needs downloading by hand. `hnoca_atlas.py` range-reads `obsm/X_scpoli` and the four `obs` columns it needs out of the 18.7 GB h5ad on Zenodo, about 80 MB of transfer, and caches them in `cache/hnoca_latent.npz`. It then fits the full 1.77M cells into `cache/hnoca_atlas.npz`. The figure scripts read only the caches, so a drawing can be restyled without refitting.

The input scaling matters here in a way it does not at PCA-50. The latent is 10-dimensional, so `raw_coords` takes a PCA head of `min(50, d - 1) = 9` dims and whitens all of it. The scPoli dimensions span 3.6x in scale, so raw and whitened pairwise distances rank-correlate only about 0.5, and a layout scored in a space it was never given is graded on a geometry it never saw. `hnoca_metrics.py` scores every cached layout in both spaces and records which one it was fit on. That gives two self-consistent protocols.

- **raw** is the default. FloDR and the atlas's published UMAP both get the latent exactly as it ships, each preprocesses internally however it likes, and both are scored in it.
- **standardised** comes from `STANDARDISE=1 python scripts/hnoca_atlas.py`, which fits FloDR on the per-dimension standardised latent into `cache/hnoca_atlas_std.npz`, to be read beside a UMAP refit on the same matrix.

Standardising costs about 0.11 CPD on this latent, far more than the whitening itself is worth. `fit` hands the array it is given to both objectives, so `fuzzy_knn_graph` builds the graph on it and `stress_X` is what the ordinal term measures against. Rescaling the input moves the targets rather than just the flow's parameterisation. Hand a pre-integrated latent over unchanged.

## Developmental Zebrafish Atlas

The complement to HNOCA. 1.22M cells over 18 to 96 hours post-fertilisation, and the PCA-50 is built here rather than shipped, so FloDR runs in the regime the paper's other atlases use. `zebrafish_atlas.py` expects `data/zebrafish/dev_zebrafish.h5ad`, the CELLxGENE Discover h5ad for dataset `bcd4103c-afdc-4233-af6e-554589d70cd3` (1.3 GB). It applies the `donor_split.py` recipe of normalise, log1p, 2000 highly variable genes, scale, and PCA-50 standardised per component. Read to PCA-50 takes about 70 s.

PCA components are uncorrelated, so standardising them leaves the input isotropic and `raw_coords` reduces to a rotation. There is only one sensible reference space here and `zebrafish_metrics.py` scores against it. `zebrafish_umap.py` is separate because UMAP at this n wants the machine to itself. A run sharing it with a FloDR fit segfaulted once.

`hnoca_3d.py` refits at `n_components=3` on a seeded 250k-cell subsample and runs UMAP at `n_components=3` on the same cells. UMAP on the full atlas costs hours and a rotating scatter cannot resolve that many points anyway. `N_SUB=0` runs the whole atlas.
