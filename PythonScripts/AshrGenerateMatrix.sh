#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_build_posterior_matrices_inplace.sh /base/dir
#
# Example:
#   ./run_build_posterior_matrices_inplace.sh /processed_datasets/VCI/ChemoGenetic_H1_Basak

BASE_DIR="${1:?Usage: $0 BASE_DIR}"

PY_SCRIPT="./AshrGenerateMatrix.py"

FORMAT="csv"        # change to csv if needed
DELETE_INPUTS="false"   # set to true only if you want to delete AshrResult_chunk_*.csv

if [[ ! -x "${PY_SCRIPT}" ]]; then
  echo "[error] Python script not found or not executable: ${PY_SCRIPT}" >&2
  exit 1
fi

drugConditions_round2=(
     "LDN-193189"
)



ALL_CONDS=(
  "${drugConditions_round2[@]}"
)

for cond in "${ALL_CONDS[@]}"; do
  COND_DIR="${BASE_DIR}/${cond}"
  OUT_FILE="${COND_DIR}/PosteriorMean_matrix.${FORMAT}"

  if [[ ! -d "${COND_DIR}" ]]; then
    echo "[skip] missing directory: ${COND_DIR}"
    continue
  fi

  echo "======================================"
  echo "Condition: ${cond}"
  echo "Directory: ${COND_DIR}"
  echo "Output:    ${OUT_FILE}"
  echo "======================================"

  cmd=(python "${PY_SCRIPT}"
       --dir "${COND_DIR}"
       --out "${OUT_FILE}"
       --format "${FORMAT}")

  if [[ "${DELETE_INPUTS}" == "true" ]]; then
    cmd+=(--delete-inputs)
  fi

  "${cmd[@]}"

done

echo "[done] All condition matrices generated (in place)."
