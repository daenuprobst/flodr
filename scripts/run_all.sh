#!/usr/bin/env bash
# Reconstructs every figure and number in the paper. Figures land in figures/.
# Every stage caches to scripts/cache/, so re-running only fills gaps; RECOMPUTE=1 forces a rebuild.
# Usage: bash scripts/run_all.sh
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
LOG=scripts/cache/pipeline.log
mkdir -p scripts/cache
echo "pipeline start $(date -Is)" | tee -a $LOG

stage () {                       # stage <name> <command...>
  local name=$1; shift
  echo "$name $(date -Is)" | tee -a $LOG
  if "$@" >> $LOG 2>&1; then
    echo "    $name OK" | tee -a $LOG
  else
    echo "    $name FAILED (continuing)" | tee -a $LOG
  fi
}

# numbers: diagnostics certificates, ablations, held-out benchmark
# (benchmark/atlas/scaling generators were removed; their caches in scripts/cache/ remain)
stage hidden_contrast $PY scripts/ablations/hidden_contrast.py
stage init_ablation   $PY scripts/ablations/pca_init_confound.py
stage synth_valid     $PY scripts/ablations/synthetic_validation.py
stage components      $PY scripts/ablations/component_ablation.py
stage donor_split     $PY scripts/ablations/donor_split.py
stage heldout_fit     $PY scripts/heldout_benchmark.py
stage heldout_score   $PY scripts/heldout_score.py

# figures (land in figures/)
stage main_comparison $PY scripts/plotting/fig_main_comparison.py
stage frontier        $PY scripts/plotting/figures.py
stage diagnostics     $PY scripts/plotting/fig_diagnostics_fields.py
stage atlas_grid      $PY scripts/plotting/fig_atlas_grid.py
stage scaling_fig     $PY scripts/plotting/fig_scaling.py
for ds in bmarrow cerebellum plant; do
  stage combined_$ds  env DATASET=$ds $PY scripts/plotting/fig_scrna_combined.py
done
stage architecture    pdflatex -interaction=nonstopmode -output-directory=figures scripts/architecture.tex

echo "pipeline done $(date -Is)" | tee -a $LOG
