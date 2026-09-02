#!/usr/bin/env bash
# Generate volcano plots for all drug contexts.
#  x : shrunken logFC (PosteriorMeanMatrices/PosteriorMean_matrix_<drug>.csv)
#  y : -log10(p_value) from DATA/<drug>/<drug>_res.csv (min across source files)
#
# Per-drug PNGs land in Figures/Volcano/.
# A per-drug parquet of (target, feature, p_value, n_obs) is cached under
# Figures/Volcano/minp_cache/ so re-runs avoid the expensive _res.csv pass.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/../.." && pwd)"

PYTHON="${PYTHON:-/home/beraslan/miniconda/envs/py312/bin/python}"
SCRIPT="$PROJ/SRC/PythonScripts/GenerateVolcanoPlots.py"

P_THR="${P_THR:-0.05}"
LOGFC_THR="${LOGFC_THR:-0.5}"
SAMPLE_N="${SAMPLE_N:-400000}"
CHUNKSIZE="${CHUNKSIZE:-2000000}"

# Mode: "individual" → one PNG per drug. "grid" → single 4×4 grid figure.
# "both" → both. Default: both.
MODE="${MODE:-both}"

EXTRA_ARGS=()
[ "${REFRESH:-0}" = "1" ] && EXTRA_ARGS+=(--refresh)

case "$MODE" in
    individual)
        "$PYTHON" "$SCRIPT" \
            --p-threshold "$P_THR" \
            --logfc-threshold "$LOGFC_THR" \
            --sample-n "$SAMPLE_N" \
            --chunksize "$CHUNKSIZE" \
            "${EXTRA_ARGS[@]}"
        ;;
    grid)
        "$PYTHON" "$SCRIPT" --grid \
            --p-threshold "$P_THR" \
            --logfc-threshold "$LOGFC_THR" \
            --sample-n "$SAMPLE_N" \
            --chunksize "$CHUNKSIZE" \
            "${EXTRA_ARGS[@]}"
        ;;
    both)
        "$PYTHON" "$SCRIPT" \
            --p-threshold "$P_THR" \
            --logfc-threshold "$LOGFC_THR" \
            --sample-n "$SAMPLE_N" \
            --chunksize "$CHUNKSIZE" \
            "${EXTRA_ARGS[@]}"
        "$PYTHON" "$SCRIPT" --grid \
            --p-threshold "$P_THR" \
            --logfc-threshold "$LOGFC_THR" \
            --sample-n "$SAMPLE_N" \
            --chunksize "$CHUNKSIZE" \
            "${EXTRA_ARGS[@]}"
        ;;
    *)
        echo "MODE must be one of: individual | grid | both" >&2
        exit 1
        ;;
esac

echo "Done. Outputs under: $PROJ/Figures/Volcano/"
