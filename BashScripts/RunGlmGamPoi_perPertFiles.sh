#!/usr/bin/env bash
# Run glmGamPoi in parallel across the per-pert h5ad files produced by
# PrepareWelchVsGlmGamPoi_v2.py. Each Rscript invocation handles one
# (pert + shared NT controls) h5ad and writes a CSV. After all are done
# we concatenate them into a single long CSV for the comparison notebook.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/../.." && pwd)"

# Use the py312 conda R; that's where glmGamPoi + zellkonverter are installed.
RSCRIPT="${RSCRIPT:-/home/beraslan/miniconda/envs/py312/bin/Rscript}"
R_SCRIPT="${R_SCRIPT:-$PROJ/SRC/RScripts/RunGlmGamPoi_onePert.R}"
IN_DIR="${IN_DIR:-/processed_datasets/VCI/ChemoGenetic_H1_Basak/DMSO_round2/per_pert_counts_500}"
OUT_DIR_CSV="${OUT_DIR_CSV:-$PROJ/Notes/welch_vs_glmgampoi/per_pert_glm_results}"
COMBINED="${COMBINED:-$PROJ/Notes/welch_vs_glmgampoi/glmgampoi_long.csv}"
N_JOBS="${N_JOBS:-8}"

[ -d "$IN_DIR" ] || { echo "[error] missing input dir: $IN_DIR" >&2; exit 1; }
chmod +x "$R_SCRIPT" 2>/dev/null || true
mkdir -p "$OUT_DIR_CSV"

n_in=$(find "$IN_DIR" -maxdepth 1 -name '*_counts.h5ad' | wc -l)
echo "Input h5ad files       : $n_in   (in $IN_DIR)"
echo "Per-pert CSV output dir: $OUT_DIR_CSV"
echo "Combined CSV           : $COMBINED"
echo "Parallel jobs          : $N_JOBS"
echo "Rscript                : $RSCRIPT"
echo

# Parallel run via xargs -P + -I to pass each h5ad as the first arg to the
# R script and a fixed second arg (output dir) per invocation. The R
# script is idempotent (skips files whose CSV already exists), so partial
# reruns are safe.
find "$IN_DIR" -maxdepth 1 -name '*_counts.h5ad' -print0 \
    | sort -z \
    | xargs -0 -P "$N_JOBS" -I '{}' "$RSCRIPT" "$R_SCRIPT" '{}' "$OUT_DIR_CSV"

# Concatenate per-pert CSVs into one long table
echo
echo "Concatenating per-pert CSVs → $COMBINED"
/home/beraslan/miniconda/envs/py312/bin/python - <<PY
from pathlib import Path
import pandas as pd
out_dir = Path("$OUT_DIR_CSV")
parts = sorted(p for p in out_dir.glob("*.csv") if not p.name.startswith("."))
print(f"  concatenating {len(parts)} CSVs")
df = pd.concat((pd.read_csv(p) for p in parts), ignore_index=True)
df.to_csv("$COMBINED", index=False)
print(f"  wrote {len(df):,} rows → $COMBINED")
PY
