#!/usr/bin/env python3
"""
For each drug condition, read: <drug>/<drug>_FDRs.csv
Each file is a matrix-like table:
  - rows = perturbations
  - columns = response genes (FDR values)
  - a column named "target" contains the perturbation label

Compute, for each perturbation and each drug, how many genes have FDR < threshold.
Output a single table:
  - rows = perturbations
  - columns = drug conditions
  - values = count of significant genes (FDR < threshold)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


DRUGS = [
    "AR-A014418", "AZD4573", "CHIR-98014", "DMSO_round2",
    "Lexibulin", "PP121", "Romidepsin", "Stattic",
    "Bisindolylmaleimide-I", "DG-172", "DMSO_round2_batch2", "JTE-607",
    "LDN-193189", "LY2090314", "NSC95397", "VX-11e",
]


def read_fdr_table(path: Path, target_col: str) -> tuple[pd.Series, pd.DataFrame]:
    """
    Returns:
      targets: Series of perturbation labels (len = n_perts)
      g: DataFrame of FDRs (rows=perts, cols=genes) numeric
    """
    df = pd.read_csv(path)

    if target_col not in df.columns:
        raise KeyError(f"Missing required column '{target_col}' in {path}")

    targets = df[target_col].astype(str)

    gene_cols = [c for c in df.columns if c != target_col]
    if not gene_cols:
        raise ValueError(f"No gene columns found in {path} after excluding '{target_col}'")

    g = df[gene_cols].apply(pd.to_numeric, errors="coerce")
    return targets, g


def counts_sig_genes_per_pert(targets: pd.Series, gene_fdr_df: pd.DataFrame, thresh: float) -> pd.Series:
    """
    Count number of genes with FDR < thresh per perturbation (row-wise).
    Returns a Series indexed by perturbation label.
    If a perturbation label appears multiple times, we SUM across those rows.
    """
    counts = (gene_fdr_df < thresh).sum(axis=1).astype(int)  # per-row gene counts
    s = pd.Series(counts.values, index=targets.values)

    # If duplicated perturbation labels exist, aggregate (sum) across duplicates
    s = s.groupby(level=0).sum()
    return s


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Count, per perturbation and drug context, how many genes are significant (FDR<threshold)."
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path("./../../DATA/"),
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
        default=Path("perturbation_by_drug_nsig_genes_fdr_lt_0p1.csv"),
        help="Output CSV path for the perturbation x drug table",
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

        targets, g = read_fdr_table(f, target_col=args.target_col)
        s = counts_sig_genes_per_pert(targets, g, thresh)
        s.name = drug
        counts_by_drug[drug] = s

        print(f"[ok] {drug}: {g.shape[0]} rows x {g.shape[1]} genes -> {s.shape[0]} perturbations")

    if not counts_by_drug:
        raise RuntimeError("No drug files were processed.")

    # Outer join across perturbations: rows=perturbations, cols=drugs
    out_df = pd.concat(counts_by_drug.values(), axis=1).fillna(0).astype(int)

    # Optional: summary columns + sorting
    num_cols = out_df.columns  # drugs only
    out_df["TOTAL"] = out_df[num_cols].sum(axis=1)
    out_df["MEDIAN"] = out_df[num_cols].median(axis=1)
    out_df = out_df.sort_values("TOTAL", ascending=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=True)
    print(f"[done] wrote {args.out} (perturbations={out_df.shape[0]}, cols={out_df.shape[1]})")


if __name__ == "__main__":
    main()
