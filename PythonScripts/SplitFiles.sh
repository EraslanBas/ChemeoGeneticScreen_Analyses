#!/usr/bin/env bash
set -euo pipefail

# --------------------------------------
# Configuration
# --------------------------------------

# Root paths
INPUT_ROOT="/processed_datasets/VCI/ChemoGenetic_H1_Basak"
OUTPUT_ROOT="/processed_datasets/VCI/ChemoGenetic_H1_Basak"

# List of perturbation conditions
conditions=("Stattic")
#conditions=("CHIR")

# Path to your Python splitting script
SPLIT_SCRIPT="SplitBigAnndata.py"

# --------------------------------------
# Run splitting for each condition
# --------------------------------------

for cond in "${conditions[@]}"; do
    echo "--------------------------------------------"
    echo " Processing condition: ${cond}"
    echo "--------------------------------------------"

    input_path="${INPUT_ROOT}/${cond}/${cond}.h5ad"
    output_path="${OUTPUT_ROOT}/${cond}/SPLIT"

    # Create output directory if missing
    mkdir -p "${output_path}"

    # Call the Python script
    python "${SPLIT_SCRIPT}" "${input_path}" "${output_path}" 

    echo "✅ Done: ${cond}"
    echo
done

echo "All conditions processed successfully."
