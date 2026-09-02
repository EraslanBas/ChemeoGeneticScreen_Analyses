#!/usr/bin/env bash
# Resume FitPathwayDrugPertOLS.py for the 85-pathway subset that intersects the
# notebook-17 173-pathway selection with the current pathways_exclude.txt.
#
# Launches one background process per group (00..21) and writes a per-group
# timestamped log under SRC/PythonScripts/logs/. Each Python process applies
# the script's own skip-if-exists logic, so already-completed (pathway, batch)
# CSVs are not re-fit.
#
# Usage:
#   ./FitPathwayDrugPertOLS_173filtered.sh
#
# All processes run in parallel via nohup. PIDs are appended to the run-tag
# log file so they can be inspected/killed later.

set -euo pipefail

PYTHON=/home/beraslan/miniconda/envs/py312/bin/python
SCRIPT=/home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/FitPathwayDrugPertOLS.py
PATHWAYS_FILE=/home/beraslan/Projects/ChemoGeneticScreens/PathwayORA/KEGG/pathways_173_filtered.txt
LOG_DIR=/home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/logs
mkdir -p "$LOG_DIR"

if [[ ! -f "$PATHWAYS_FILE" ]]; then
    echo "Pathways file missing: $PATHWAYS_FILE" >&2
    exit 1
fi

# Read the 85 pathway names into a bash array
mapfile -t PATHWAYS < "$PATHWAYS_FILE"
echo "Loaded ${#PATHWAYS[@]} pathways from $PATHWAYS_FILE"

TS=$(date +%Y%m%d_%H%M%S)
PID_LOG="$LOG_DIR/${TS}_173filtered_pids.log"
echo "Tag: ${TS}_173filtered"
echo "PID log: $PID_LOG"
echo

for g in $(seq -w 0 21); do
    GROUP="group_${g}"
    LOG_FILE="$LOG_DIR/${TS}_173filtered_${GROUP}.log"
    nohup "$PYTHON" -u "$SCRIPT" \
        --group "$GROUP" \
        --pathways "${PATHWAYS[@]}" \
        > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$GROUP  PID=$PID  log=$LOG_FILE" | tee -a "$PID_LOG"
done

echo
echo "All 22 group processes launched. Watch progress with:"
echo "  tail -f $LOG_DIR/${TS}_173filtered_group_05.log"
