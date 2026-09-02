#!/usr/bin/env bash
set -euo pipefail

PYTHON_SCRIPT="ComputeSE.py"

BASE_IN="/processed_datasets/VCI/ChemoGenetic_H1_Basak"
OUT_BASE="./"

# Analysis params (adjust once, applies to all conds)
GROUP_KEY="target_gene"
CONTROL_LABEL="non-targeting"
LAYER=""          # set to "" to use adata.X instead
MIN_CELLS=1
CHUNK_PERTS=100

# Parallelism: set to >1 to process multiple conditions concurrently
# (requires GNU parallel or will fall back to xargs -P)
N_JOBS=1

CONDS=(
  "LDN-193189"
)

run_one () {
  local cond="$1"
  local adata="${BASE_IN}/${cond}/${cond}.h5ad"
  local out_dir="${OUT_BASE}/${cond}"

  if [[ ! -f "${adata}" ]]; then
    echo "[skip] missing input: ${adata}" >&2
    return 0
  fi

  # simple resume: if manifest exists we assume it finished
  if [[ -f "${out_dir}/manifest.tsv" ]]; then
    echo "[skip] already processed (manifest exists): ${cond}"
    return 0
  fi

  mkdir -p "${out_dir}"

  echo "======================================"
  echo "cond:      ${cond}"
  echo "adata:     ${adata}"
  echo "out_dir:   ${out_dir}"
  echo "======================================"

  # Build command
  cmd=(python "${PYTHON_SCRIPT}"
    --adata "${adata}"
    --out-dir "${out_dir}"
    --group-key "${GROUP_KEY}"
    --control-label "${CONTROL_LABEL}"
    --min-cells "${MIN_CELLS}"
    --chunk-perts "${CHUNK_PERTS}"
  )

  # Only pass --layer if non-empty
  if [[ -n "${LAYER}" ]]; then
    cmd+=(--layer "${LAYER}")
  fi

  "${cmd[@]}"
}

export -f run_one
export PYTHON_SCRIPT BASE_IN OUT_BASE GROUP_KEY CONTROL_LABEL LAYER MIN_CELLS CHUNK_PERTS

mkdir -p "${OUT_BASE}"

if [[ "${N_JOBS}" -le 1 ]]; then
  for cond in "${CONDS[@]}"; do
    run_one "${cond}"
  done
else
  # Prefer GNU parallel if present; otherwise use xargs -P
  if command -v parallel >/dev/null 2>&1; then
    printf "%s\n" "${CONDS[@]}" | parallel -j "${N_JOBS}" run_one {}
  else
    printf "%s\n" "${CONDS[@]}" | xargs -n 1 -P "${N_JOBS}" -I {} bash -lc 'run_one "$@"' _ {}
  fi
fi

echo "[done] all conditions processed."
