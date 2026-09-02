#!/usr/bin/env bash
set -euo pipefail

OUTDIR="./PosteriorMeanMatrices"
mkdir -p "${OUTDIR}"

find . -type f -name "PosteriorMean_matrix.csv" | while read -r filepath; do
    # get parent directory name
    dir_name=$(basename "$(dirname "$filepath")")

    out_file="${OUTDIR}/PosteriorMean_matrix_${dir_name}.csv"

    echo "Copying $filepath -> $out_file"
    mv "$filepath" "$out_file"
done

echo "[done] All PosteriorMean_matrix.csv files collected."
