import os
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
import re

# --- paths ---
base_dir = Path("/processed_datasets/VCI/ChemoGenetic_H1_Basak/DMSO_round2")
in_path = base_dir / "DMSO_round2.h5ad"
chunk_dir = base_dir / "chunks"
chunk_dir.mkdir(exist_ok=True, parents=True)

# # --- read only in backed mode ---
# dmso_round2 = sc.read_h5ad(in_path, backed="r")

# # --- load mean expressions and compute delta ---
# mean_expr_before = pd.read_csv("./TextFiles/mean_expr_before.csv", index_col=0)
# mean_expr_after  = pd.read_csv("./TextFiles/mean_expr_after.csv", index_col=0)

# dmso_round2_dif = (
#     mean_expr_before.loc["DMSO_round2", :] - mean_expr_after.loc["DMSO_round2", :]
# )

# # align to var_names
# delta_aligned = dmso_round2_dif.reindex(dmso_round2.var_names).fillna(0.0)
# delta_row = delta_aligned.to_numpy().astype(np.float32)[None, :]  # shape (1, n_genes)

# n_cells = dmso_round2.n_obs
# chunk_size = 20000  # tune based on your RAM

# print(f"Total cells: {n_cells}, using chunk_size={chunk_size}")

# chunk_idx = 0
# for start in range(0, n_cells, chunk_size):
#     end = min(n_cells, start + chunk_size)
#     print(f"Processing chunk {chunk_idx}: cells [{start}:{end})")

#     # 1) extract dense X for the chunk
#     X_chunk = dmso_round2.X[start:end, :].toarray().astype(np.float32)  # (chunk_cells, n_genes)

#     # 2) apply per-gene delta (exact dense shift)
#     X_chunk -= delta_row   # broadcasts over rows

#     # 3) slice obs (rows) and reuse full var (genes)
#     obs_chunk = dmso_round2.obs.iloc[start:end].copy()
#     var_chunk = dmso_round2.var.copy()

#     # 4) build a normal in-memory AnnData chunk
#     adata_chunk = sc.AnnData(
#         X=X_chunk,
#         obs=obs_chunk,
#         var=var_chunk,
#     )

#     # 5) write chunk to disk
#     chunk_path = chunk_dir / f"DMSO_round2_chunk_{chunk_idx:03d}.h5ad"
#     adata_chunk.write(chunk_path)
#     print(f"  wrote {chunk_path}")

#     chunk_idx += 1

# # close backed file
# dmso_round2.file.close()
# print("Finished writing all chunks.")




# Pattern to match chunk files
pattern = "DMSO_round2_chunk_*.h5ad"

# Collect file paths
chunk_files = list(chunk_dir.glob(pattern))

# Sort numerically by the chunk index (e.g., 072, 073, ...)
def extract_index(f):
    m = re.search(r"chunk_(\d+)\.h5ad", f.name)
    return int(m.group(1)) if m else -1

chunk_files = sorted(chunk_files, key=extract_index)

print(f"Found {len(chunk_files)} chunks:")
for f in chunk_files:
    print("  ", f.name)

# Load all AnnData objects
adatas = []
for f in chunk_files:
    print(f"Reading {f.name} ...")
    ad = sc.read_h5ad(f)
    adatas.append(ad)

# Concatenate along the cell axis
print("Concatenating all chunks...")
adata_full = sc.concat(adatas, axis=0, join="inner")

# Output path
out_path = Path("/processed_datasets/VCI/ChemoGenetic_H1_Basak/DMSO_round2/DMSO_round2_batchcorrected_concat.h5ad")

print(f"Writing final concatenated AnnData to: {out_path}")
adata_full.write(out_path)

print("Done!")
