# Ablations

Knob sweeps and component ablations. The w-sweep and no-ordinal ablation were arms of the
benchmark evaluator (since removed); their results remain cached in
`cache/eval/*_metrics.json`, across all four benchmark datasets and three seeds:

- `FloDR (w=1)`, `FloDR (w=2)`, `FloDR (w=3)`: the sweep of the one exposed knob
- `FloDR (no ordinal)`: the ordinal global term switched off

These are rendered into the main comparison table, but the paper has no section that
frames them as an ablation, so they currently read as extra baselines.

Anchor paths on `SCRIPTS` for the cache and `ROOT` for `figures/` and `tables/`, as
described in `../README.md`; a script one level deep that uses `HERE/..` writes into the
group folder.

- `synthetic_validation.py`: validates the two diagnostics against generator ground
  truth: conditional spread vs the known per-point sigma of `make_hetero`/`make_ring`
  (plus the `make_blob` flat control), and hidden contrast vs `make_hidden`'s lambda
  sweep, where the true field I(G;X|U=u) is Monte-Carlo-computable
  (`datasets.hidden_truth`). Writes `cache/synthetic_validation.json` and
  `figures/synthetic_validation.pdf`.
- `component_ablation.py`: one component varied per arm (flow depth, sketch conditioner,
  density NLL) against the reference recipe, cached cell by cell in
  `cache/component_ablation.json`.
- `donor_split.py`: donor-level held-out splits for the bone marrow and cerebellum
  atlases; reads `data/scrna/*.h5ad`, caches in `cache/donor_split.json`.
- `hidden_contrast.py`: the hidden-contrast field with its permutation null on the three
  atlases; writes the `cache/hidden_contrast_*.npz` the diagnostics figures read.
- `pca_init_confound.py`: rotates only the whitened head by a random orthogonal Q to test
  whether the global-structure advantage is inherited from the PCA initialisation.
