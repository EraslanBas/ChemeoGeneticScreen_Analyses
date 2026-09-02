#!/usr/bin/env bash
# After the 173-filtered run finishes for groups 00..04, run the full ~415-pathway
# OLS fit (default exclude list applied) for those same five groups.
#
# Per-group flow:
#   1. Wait for the corresponding 173-filtered PID (from $PID_LOG) to exit.
#   2. Launch python FitPathwayDrugPertOLS.py with no --pathways flag and
#      --n-pathways 99999 so every non-excluded pathway is fit.
#   3. Skip-existing in the script means the ~85-filtered set already done by the
#      prior run is reused; only the ~330 remaining pathways per group get fit.
#
# Logs to SRC/PythonScripts/logs/{TS}_full415_group_NN.log

set -euo pipefail

PYTHON=/home/beraslan/miniconda/envs/py312/bin/python
SCRIPT=/home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/FitPathwayDrugPertOLS.py
LOG_DIR=/home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/logs
PID_LOG=/home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/logs/20260506_210212_173filtered_pids.log
mkdir -p "$LOG_DIR"

if [[ ! -f "$PID_LOG" ]]; then
    echo "PID log not found: $PID_LOG" >&2
    exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
NEW_PID_LOG="$LOG_DIR/${TS}_full415_pids.log"
echo "Tag: ${TS}_full415"
echo "Wait-on PID log: $PID_LOG"
echo "New PID log: $NEW_PID_LOG"
echo

GROUPS=(00 01 02 03 04)

# Per-group launcher (runs in background; waits for prior PID then execs)
launch_after_wait() {
    local group="$1"
    local wait_pid="$2"
    local log="$3"

    {
        # Poll until the prior PID is gone
        while kill -0 "$wait_pid" 2>/dev/null; do
            sleep 30
        done
        echo "[$(date +%H:%M:%S)] prior PID $wait_pid for group_${group} exited; launching full-415 fit"
        exec "$PYTHON" -u "$SCRIPT" \
            --group "group_${group}" \
            --n-pathways 99999
    } > "$log" 2>&1 &
    echo $!
}

for g in "${GROUPS[@]}"; do
    PRIOR_PID=$(awk -v g="group_${g}" '$1 == g {sub("PID=", "", $2); print $2}' "$PID_LOG")
    if [[ -z "$PRIOR_PID" ]]; then
        echo "Could not find prior PID for group_${g} in $PID_LOG" >&2
        continue
    fi
    LOG_FILE="$LOG_DIR/${TS}_full415_group_${g}.log"
    NEW_PID=$(launch_after_wait "$g" "$PRIOR_PID" "$LOG_FILE")
    echo "group_${g}  waiting_for_PID=$PRIOR_PID  watcher_PID=$NEW_PID  log=$LOG_FILE" \
        | tee -a "$NEW_PID_LOG"
done

echo
echo "All 5 watcher processes launched. Each will poll its prior PID every 30 s,"
echo "then exec the full-415 OLS fit when the 173-filtered job for that group exits."
echo
echo "Watch with:  tail -f $LOG_DIR/${TS}_full415_group_03.log"
