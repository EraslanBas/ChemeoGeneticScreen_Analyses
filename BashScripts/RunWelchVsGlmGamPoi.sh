#!/usr/bin/env bash
# End-to-end: pick 500 perturbations from DMSO batch1 (stratified by n_cells),
# compute our Welch SEs, and run glmGamPoi against non-targeting controls.
# Both result tables land under Notes/welch_vs_glmgampoi/.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/../.." && pwd)"
OUT_DIR="${OUT_DIR:-$PROJ/Notes/welch_vs_glmgampoi}"

PYTHON="${PYTHON:-/home/beraslan/miniconda/envs/py312/bin/python}"
RSCRIPT="${RSCRIPT:-Rscript}"
N_PERTS="${N_PERTS:-500}"
N_BINS="${N_BINS:-10}"
N_CTRL="${N_CTRL:-20000}"
N_CORES="${N_CORES:-8}"

mkdir -p "$OUT_DIR"

echo "=== Step 1/2: pick subset + Welch SEs (python) ==="
"$PYTHON" "$PROJ/SRC/PythonScripts/PrepareDMSOb1SubsetAndWelchSE.py" \
    --out-dir "$OUT_DIR" \
    --n-perts "$N_PERTS" \
    --n-bins "$N_BINS" \
    --n-ctrl "$N_CTRL"

echo
echo "=== Step 2/2: glmGamPoi on the subset (R) ==="
"$RSCRIPT" "$PROJ/SRC/RScripts/RunGlmGamPoi_subset.R" \
    --adata     "$OUT_DIR/subset_DMSO_b1.h5ad" \
    --perts-csv "$OUT_DIR/selected_perts.csv" \
    --out       "$OUT_DIR/glmgampoi_long.csv" \
    --n-cores   "$N_CORES"

echo
echo "Done. Outputs under $OUT_DIR:"
ls -la "$OUT_DIR"
