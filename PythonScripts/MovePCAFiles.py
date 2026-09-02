#!/usr/bin/env python3

from pathlib import Path
import shutil
import re

# Source and destination roots
src_root = Path("/home/beraslan/Projects/ChemoGeneticScreens/DATA")
dst_root = Path("/processed_datasets/VCI/ChemoGenetic_H1_Basak/PCA_res")

# Pattern for files
pattern = re.compile(r"(.*)_ad_\d+_pca_cosineRes\.csv")

for file in src_root.rglob("*_ad_*_pca_cosineRes.csv"):
    
    fname = file.name
    match = pattern.match(fname)
    
    if match:
        cond = match.group(1)

        dst_dir = dst_root / cond
        dst_dir.mkdir(parents=True, exist_ok=True)

        dst_file = dst_dir / fname

        print(f"Moving: {file} -> {dst_file}")
        shutil.move(str(file), str(dst_file))