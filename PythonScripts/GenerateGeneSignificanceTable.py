#!/usr/bin/env python3
"""
For each drug condition, read: <drug>/<drug>_FDRs.csv
Each file is a matrix-like table:
  - rows = perturbations
  - columns = response genes (FDR values)
  - a column named "target" contains the perturbation label (not used except to exclude from gene columns)

Compute, for each gene and each drug, how many perturbations have FDR < threshold.
Output a single table:
  - rows = genes
  - columns = drug conditions
  - values = count of significant perturbations (FDR < threshold)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


DRUGS = [
    "AR-A014418", "AZD4573", "CHIR-98014", "DMSO_round2",
    "Lexibulin", "PP121", "Romidepsin", "Stattic",
    "Bisindolylmaleimide-I", "DG-172", "DMSO_round2_batch2", "JTE-607",
    "LDN-193189", "LY2090314", "NSC95397", "VX-11e",
]


def read_fdr_matrix(path: Path, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise KeyError(f"Missing required column '{target_col}' in {path}")

    # Keep only gene columns (everything except target_col)
    gene_cols = [c for c in df.columns if c != target_col]

    if not gene_cols:
        raise ValueError(f"No gene columns found in {path} after excluding '{target_col}'")

    # Ensure numeric (coerce errors to NaN)
    g = df[gene_cols].apply(pd.to_numeric, errors="coerce")

    return g  # rows=perts, cols=genes


def counts_sig_per_gene(gene_fdr_df: pd.DataFrame, thresh: float) -> pd.Series:
    # Count number of perturbations with FDR < thresh per gene
    # NaNs are treated as not significant.
    return (gene_fdr_df < thresh).sum(axis=0).astype(int)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Count, per gene and drug context, how many perturbations are significant (FDR<threshold)."
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path("."),
        help="Base directory containing drug subfolders (default: current directory)",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.1,
        help="FDR significance threshold (default: 0.1)",
    )
    ap.add_argument(
        "--target-col",
        type=str,
        default="target",
        help="Column name holding perturbation label (default: target)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("gene_by_drug_nsig_fdr_lt_0p1.csv"),
        help="Output CSV path for the gene x drug table",
    )
    ap.add_argument(
        "--skip-missing",
        action="store_true",
        help="If set, skip drugs whose files are missing instead of erroring.",
    )
    args = ap.parse_args()

    base_dir: Path = args.base_dir
    thresh: float = args.threshold

    counts_by_drug: dict[str, pd.Series] = {}

    for drug in DRUGS:
        f = base_dir / drug / f"{drug}_FDRs.csv"
        if not f.exists():
            msg = f"[missing] {f}"
            if args.skip_missing:
                print(msg + " (skipping)")
                continue
            raise FileNotFoundError(msg)

        g = read_fdr_matrix(f, target_col=args.target_col)
        s = counts_sig_per_gene(g, thresh)
        s.name = drug
        counts_by_drug[drug] = s
        print(f"[ok] {drug}: {g.shape[0]} perturbations x {g.shape[1]} genes")

    if not counts_by_drug:
        raise RuntimeError("No drug files were processed.")

    # Outer join across genes: rows=genes, cols=drugs
    out_df = pd.concat(counts_by_drug.values(), axis=1).fillna(0).astype(int)

    # Optional: sort genes by total number of significant perturbations across drugs (descending)
    out_df["MEDIAN"] = out_df.median(axis=1)
    out_df = out_df.sort_values("MEDIAN", ascending=False)
    
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=True)
    print(f"[done] wrote {args.out} (genes={out_df.shape[0]}, drugs={out_df.shape[1]})")


if __name__ == "__main__":
    main()
