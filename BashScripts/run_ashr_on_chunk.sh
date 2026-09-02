#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_ashr_recursive.sh /path/to/dir [N_JOBS] [R_SCRIPT]
#
# Example:
#   ./run_ashr_recursive.sh ./results 8 ./run_ashr_on_chunk.R

DIR="${1:?Usage: $0 /path/to/dir [N_JOBS] [R_SCRIPT]}"
N_JOBS="${2:-6}"
R_SCRIPT="${3:-./run_ashr_on_chunk.R}"

if [[ ! -d "${DIR}" ]]; then
  echo "[error] directory not found: ${DIR}" >&2
  exit 1
fi

if [[ ! -x "${R_SCRIPT}" ]]; then
  echo "[error] R script not found or not executable: ${R_SCRIPT}" >&2
  exit 1
fi

# Find chunk CSVs under the given directory (recursively), run in parallel.
find "${DIR}" -type f -name 'chunk_*.csv' -print0 | sort -z | \
  xargs -0 -n 1 -P "${N_JOBS}" "${R_SCRIPT}"

echo "[done] All chunk files processed under: ${DIR}"
