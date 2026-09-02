#!/usr/bin/env python3
"""
Replogle variant of ComputeSE.py.

The Replogle_Basak per-cell-line .h5ad files store RAW integer counts in .X and
label the knock-down target in obs['gene'] (control == 'non-targeting'). The
ashr pipeline expects log1p-normalized expression (see Notes/logfc_standard_error.md),
so this script log-normalizes X in memory (normalize_total(target_sum) + log1p)
and then writes the same chunk_XXXXXX.csv files ComputeSE.py produces:

    perturbation, gene, mean_pert, mean_ctrl, mean_diff, se, n_pert, n_ctrl

The Welch mean-difference / SE math is imported verbatim from ComputeSE.py so
there is a single source of truth.

Example:
  python ComputeSE_Replogle.py \
    --adata /processed_datasets/VCI/Replogle_Basak/hepg2.h5ad \
    --out-dir /processed_datasets/VCI/Replogle_Basak/Ashr/hepg2 \
    --group-key gene --control-label non-targeting \
    --target-sum 200000 --min-cells 20 --chunk-perts 100
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# Reuse the exact same Welch math as the ChemoGenetic pipeline.
from ComputeSE import _sum_and_sumsq, _mean_and_var_from_sum


def compute_and_write_chunks_from_adata(
    adata,
    out_dir: Path,
    *,
    group_key: str,
    control_label: str,
    min_cells: int,
    chunk_perts: int,
) -> List[Path]:
    if group_key not in adata.obs:
        raise KeyError(f"--group-key '{group_key}' not found in adata.obs")

    X = adata.X
    if sp.issparse(X):
        X = X.tocsr()

    groups = adata.obs[group_key].astype("string").to_numpy()
    genes = adata.var_names.to_numpy()

    ctrl_mask = groups == control_label
    n_ctrl = int(ctrl_mask.sum())
    if n_ctrl < 2:
        raise ValueError(f"Need >=2 control cells for '{control_label}', found {n_ctrl}.")

    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- control stats once ----
    s_c, ss_c = _sum_and_sumsq(X[ctrl_mask])
    mean_c, var_c = _mean_and_var_from_sum(s_c, ss_c, n_ctrl)

    # ---- perturbations to process ----
    pert_counts = pd.Series(groups).value_counts()
    perts = [p for p in np.unique(groups) if p != control_label]
    perts = [p for p in perts if int(pert_counts.get(p, 0)) >= min_cells]
    if not perts:
        raise ValueError(
            f"No perturbations passed filters (min_cells={min_cells}) excluding '{control_label}'."
        )

    chunk_files: List[Path] = []
    for start in range(0, len(perts), chunk_perts):
        batch = perts[start : start + chunk_perts]
        rows = []
        for pert in batch:
            pert_mask = groups == pert
            n_p = int(pert_mask.sum())
            if n_p < 2:
                continue

            s_p, ss_p = _sum_and_sumsq(X[pert_mask])
            mean_p, var_p = _mean_and_var_from_sum(s_p, ss_p, n_p)

            mean_diff = mean_p - mean_c
            se = np.sqrt(var_p / n_p + var_c / n_ctrl)

            rows.append(
                pd.DataFrame(
                    {
                        "perturbation": pert,
                        "gene": genes,
                        "mean_pert": mean_p,
                        "mean_ctrl": mean_c,
                        "mean_diff": mean_diff,
                        "se": se,
                        "n_pert": n_p,
                        "n_ctrl": n_ctrl,
                    }
                )
            )

        chunk_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
        chunk_path = out_dir / f"chunk_{start:06d}.csv"
        chunk_df.to_csv(chunk_path, index=False)
        chunk_files.append(chunk_path)
        print(f"[ok] {chunk_path.name}: {len(batch)} perts -> {len(chunk_df):,} rows")

    return chunk_files


def main() -> None:
    ap = argparse.ArgumentParser(description="Log-normalize a Replogle h5ad and write ashr chunk CSVs.")
    ap.add_argument("--adata", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--group-key", default="gene")
    ap.add_argument("--control-label", default="non-targeting")
    ap.add_argument("--target-sum", type=float, default=200000.0)
    ap.add_argument("--min-cells", type=int, default=20)
    ap.add_argument("--chunk-perts", type=int, default=100)
    args = ap.parse_args()

    print(f"[load] {args.adata}")
    adata = sc.read_h5ad(str(args.adata))
    print(f"[load] {adata.shape[0]:,} cells x {adata.shape[1]:,} genes")

    print(f"[norm] normalize_total(target_sum={args.target_sum:g}) + log1p")
    sc.pp.normalize_total(adata, target_sum=args.target_sum)
    sc.pp.log1p(adata)

    compute_and_write_chunks_from_adata(
        adata,
        args.out_dir,
        group_key=args.group_key,
        control_label=args.control_label,
        min_cells=args.min_cells,
        chunk_perts=args.chunk_perts,
    )
    print(f"[done] chunks written to {args.out_dir}")


if __name__ == "__main__":
    main()
