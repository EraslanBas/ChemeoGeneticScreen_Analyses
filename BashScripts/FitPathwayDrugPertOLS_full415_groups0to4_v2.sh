#!/usr/bin/env bash
# Run FitPathwayDrugPertOLS.py on the full ~415-pathway set (default exclude
# list applied) for groups 00..04 only. Five procs in parallel, each pinned to
# 4 BLAS threads to avoid the oversubscription that throttled the prior 22-proc
# run on this 114-core box. Skip-existing in the script means already-completed
# (pathway, batch) CSVs are reused; only the missing ones get fit.
#
# Per-group missing counts (at launch):
#   group_00:  46 fits   group_01: 119   group_02: 196
#   group_03: 208 fits   group_04: 144
#   Total:    713 fits

set -euo pipefail

PYTHON=/home/beraslan/miniconda/envs/py312/bin/python
SCRIPT=/home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/FitPathwayDrugPertOLS.py
LOG_DIR=/home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/logs
mkdir -p "$LOG_DIR"

# BLAS thread cap per process — 5 procs × 4 threads = 20 threads on 114 cores
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

TS=$(date +%Y%m%d_%H%M%S)
PID_LOG="$LOG_DIR/${TS}_full415_g0to4_pids.log"
echo "Tag: ${TS}_full415_g0to4"
echo "BLAS threads/proc: 4 (5 procs total → 20 threads on 114 cores)"
echo "PID log: $PID_LOG"
echo

for g in 00 01 02 03 04; do
    GROUP="group_${g}"
    LOG_FILE="$LOG_DIR/${TS}_full415_${GROUP}.log"
    nohup "$PYTHON" -u "$SCRIPT" \
        --group "$GROUP" \
        --n-pathways 99999 \
        > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$GROUP  PID=$PID  log=$LOG_FILE" | tee -a "$PID_LOG"
done

echo
echo "All 5 group processes launched."
echo "Watch: tail -f $LOG_DIR/${TS}_full415_group_03.log"
