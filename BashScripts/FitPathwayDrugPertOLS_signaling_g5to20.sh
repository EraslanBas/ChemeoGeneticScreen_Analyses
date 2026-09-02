#!/usr/bin/env bash
# Run FitPathwayDrugPertOLS.py on the 78 *_SIGNALING_PATHWAY pathways that are
# in the union of remaining (missing) pathways across groups 05-20. Skip-existing
# in the (patched) script handles already-completed (pathway, batch) CSVs, so
# only the missing fits actually run.
#
# 4 procs concurrent, processing 16 groups in waves. Each proc gets 6 BLAS
# threads (4 × 6 = 24 threads on 114 cores).

PYTHON=/home/beraslan/miniconda/envs/py312/bin/python
SCRIPT=/home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/FitPathwayDrugPertOLS.py
PATHWAYS_FILE=/home/beraslan/Projects/ChemoGeneticScreens/PathwayORA/KEGG/pathways_signaling_remaining.txt
LOG_DIR=/home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/logs
mkdir -p "$LOG_DIR"

if [ ! -f "$PATHWAYS_FILE" ]; then
    echo "Pathways file missing: $PATHWAYS_FILE" >&2
    exit 1
fi

PATHWAY_LIST=$(tr '\n' ' ' < "$PATHWAYS_FILE")
N_PATHWAYS=$(wc -l < "$PATHWAYS_FILE")
echo "Loaded $N_PATHWAYS signaling pathways from $PATHWAYS_FILE"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

TS=$(date +%Y%m%d_%H%M%S)
PID_LOG="$LOG_DIR/${TS}_signaling_g5to20_pids.log"
echo "Tag: ${TS}_signaling_g5to20  (4 concurrent procs, 6 threads each)"
echo "PID log: $PID_LOG"
echo

GROUPS_LIST="05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20"
MAX_PARALLEL=4

for g in $GROUPS_LIST; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAX_PARALLEL" ]; do
        wait -n 2>/dev/null || sleep 5
    done

    GROUP="group_${g}"
    LOG_FILE="$LOG_DIR/${TS}_signaling_${GROUP}.log"
    nohup "$PYTHON" -u "$SCRIPT" \
        --group "$GROUP" \
        --pathways $PATHWAY_LIST \
        > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$GROUP  PID=$PID  log=$LOG_FILE"
    echo "$GROUP  PID=$PID  log=$LOG_FILE" >> "$PID_LOG"
done

echo
echo "All groups queued. Waiting for the last $MAX_PARALLEL to finish..."
wait
echo "All 16 groups complete."
