#!/usr/bin/env bash
set -euo pipefail

max_jobs=1
i=0

for f in /processed_datasets/VCI/ChemoGenetic_H1_Basak/LDN-193189/SPLIT/ad_*.h5ad; do
  b=$(basename "$f")
  echo "Starting: $b"
  python /home/beraslan/Projects/ChemoGeneticScreens/SRC/PythonScripts/Compute_DEwilcox.py "$f" /home/beraslan/Projects/ChemoGeneticScreens/DATA/LDN-193189 /processed_datasets/VCI/ChemoGenetic_H1_Basak/LDN-193189/SPLIT/LDN-193189_controls.h5ad &
  i=$((i+1))
  if (( i % max_jobs == 0 )); then
    wait   # throttle to max_jobs
  fi
done

wait
echo "All done."
