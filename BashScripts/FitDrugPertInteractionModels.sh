#!/usr/bin/env bash
# Wrapper to manually invoke FitDrugPertInteractionModels.py.
# Edit the variables below and run.  Useful patterns:
#
#   • Smoke-test (one chunk):                 ./FitDrugPertInteractionModels.sh smoke
#   • Full run (all 14 chunks):               ./FitDrugPertInteractionModels.sh full
#   • Custom args:                            ./FitDrugPertInteractionModels.sh -- --chunk-size 50 --n-ctrl 5000
#
# Output (defaults from the underlying Python script):
#   AnnData cache: /processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways/scored_chunk_NNN.h5ad
#   Results:       /home/beraslan/Projects/ChemoGeneticScreens/PathwayORA/KEGG/interaction_models.parquet
#
# A timestamped log of stdout+stderr is written to ./logs/.

set -euo pipefail

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
PYTHON=/home/beraslan/miniconda/envs/py312/bin/python
SCRIPT=/home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/FitDrugPertInteractionModels.py
LOG_DIR=/home/beraslan/Projects/ChemoGeneticScreens/SRC/BashScripts/logs
mkdir -p "$LOG_DIR"

# Tunables — override on the command line via `-- --flag value`
CHUNK_SIZE=100      # perturbations per chunk
N_CTRL=10000        # non-targeting cells subsampled per drug per chunk
N_WORKERS=$(($(nproc) - 2))   # OLS workers; leave a couple of cores free
SEED=0

# --------------------------------------------------------------------------
# Mode selection
# --------------------------------------------------------------------------
MODE="${1:-smoke}"
shift || true   # consume the mode arg if present

EXTRA_ARGS=()
if [[ "${1:-}" == "--" ]]; then
    shift
    EXTRA_ARGS=("$@")
fi

case "$MODE" in
    smoke)
        # Process only the first chunk (perturbations 1–100) — quick sanity run.
        MAX_CHUNKS=1
        TAG="smoke"
        ;;
    full)
        # All 14 chunks (1,378 perturbations).
        MAX_CHUNKS=""
        TAG="full"
        ;;
    *)
        echo "Unknown mode: $MODE" >&2
        echo "Usage: $0 [smoke|full] [-- --extra --flags]" >&2
        exit 1
        ;;
esac

# --------------------------------------------------------------------------
# Build the command
# --------------------------------------------------------------------------
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S)_${TAG}.log"
CMD=(
    "$PYTHON" -u "$SCRIPT"
    --chunk-size "$CHUNK_SIZE"
    --n-ctrl     "$N_CTRL"
    --n-workers  "$N_WORKERS"
    --seed       "$SEED"
)
[[ -n "$MAX_CHUNKS" ]] && CMD+=(--max-chunks "$MAX_CHUNKS")
CMD+=("${EXTRA_ARGS[@]}")

# --------------------------------------------------------------------------
# Run
# --------------------------------------------------------------------------
echo "----------------------------------------------------------------"
echo "Mode:    $MODE"
echo "Logging: $LOG_FILE"
echo "Command:"
printf '  %q ' "${CMD[@]}"; echo
echo "----------------------------------------------------------------"

# `tee` so the user can watch live AND a log file is preserved
"${CMD[@]}" 2>&1 | tee "$LOG_FILE"

echo
echo "Done.  Log: $LOG_FILE"
