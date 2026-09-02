#!/usr/bin/env python3
"""
1. Picks 500 perturbations from the DMSO batch1 raw AnnData with cell counts
   spanning the full distribution (stratified across 10 log-spaced bins).
2. Writes a subset h5ad containing only those perturbations' cells plus a
   downsampled non-targeting control set.
3. Runs the same Welch-style per-(perturbation, gene) (mean_diff, SE)
   calculation that ComputeSE.py performs and writes the result as a single
   long parquet, ready to be joined against glmGamPoi output by R.

Outputs (under --out-dir, default Notes/welch_vs_glmgampoi/):
    subset_DMSO_b1.h5ad
    selected_perts.csv         (perturbation, n_cells)
    welch_se_long.parquet      (perturbation, gene, mean_diff, se, n_pert, n_ctrl)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp


# ----- (mean, var) streaming helpers — identical math to ComputeSE.py -----

def _sum_and_sumsq(X) -> Tuple[np.ndarray, np.ndarray]:
    if sp.issparse(X):
        X = X.tocsr()
        s  = np.asarray(X.sum(axis=0)).ravel().astype(np.float64, copy=False)
        ss = np.asarray(X.multiply(X).sum(axis=0)).ravel().astype(np.float64, copy=False)
    else:
        Xd = np.asarray(X, dtype=np.float64)
        s  = Xd.sum(axis=0)
        ss = (Xd * Xd).sum(axis=0)
    return s, ss


def _mean_and_var(s: np.ndarray, ss: np.ndarray, n: int):
    mean = s / n
    if n < 2:
        return mean, np.zeros_like(mean)
    ex2 = ss / n
    var_pop = np.maximum(ex2 - mean * mean, 0.0)
    return mean, var_pop * (n / (n - 1))


# ----- stratified perturbation sampling -----

def stratified_sample(counts: pd.Series, n_target: int = 500,
                       n_bins: int = 10, *, seed: int = 0,
                       min_cells: int = 5) -> list[str]:
    """Return ~n_target perturbation names with cell counts spanning the full
    log range of `counts` (one Series of n_cells indexed by perturbation)."""
    counts = counts[counts >= min_cells]
    if len(counts) == 0:
        raise RuntimeError("no perturbations with enough cells")
    log_n = np.log10(counts.values.astype(float))
    edges = np.linspace(log_n.min(), log_n.max(), n_bins + 1)
    bins  = np.clip(np.digitize(log_n, edges[1:-1]), 0, n_bins - 1)
    rng = np.random.default_rng(seed)

    per_bin = n_target // n_bins
    picked: list[str] = []
    for b in range(n_bins):
        in_bin = counts.index[bins == b].tolist()
        rng.shuffle(in_bin)
        picked.extend(in_bin[:per_bin])

    # Top up to n_target by sampling any remaining perts not yet picked
    if len(picked) < n_target:
        remaining = [p for p in counts.index if p not in set(picked)]
        rng.shuffle(remaining)
        picked.extend(remaining[: n_target - len(picked)])
    return picked[:n_target]


# ----- main -----

DEFAULT_ADATA = Path(
    "/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/"
    "PertQC_CytoGMM/DualGuideOnly_ntgUPM5/DMSO/scanpy/"
    "ad_gene_guide_complete.gene.h5ad"
)
DEFAULT_OUT = Path("/home/beraslan/Projects/ChemoGeneticScreens/Notes/welch_vs_glmgampoi")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adata", type=Path, default=DEFAULT_ADATA)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-perts", type=int, default=500)
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--min-cells", type=int, default=5)
    ap.add_argument("--n-ctrl", type=int, default=20_000,
                     help="Downsampled non-targeting control cells.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--group-key", default="target_gene")
    ap.add_argument("--control-label", default="non-targeting")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"adata    : {args.adata}")
    print(f"out_dir  : {args.out_dir}")

    # ---- pick perturbations ----
    print("Loading obs (backed-mode read)…")
    a_backed = ad.read_h5ad(args.adata, backed="r")
    counts = a_backed.obs[args.group_key].astype(str).value_counts()
    a_backed.file.close()
    counts = counts[counts.index != args.control_label]
    print(f"  total non-control perts: {len(counts):,}")

    picked = stratified_sample(counts, n_target=args.n_perts,
                                n_bins=args.n_bins, seed=args.seed,
                                min_cells=args.min_cells)
    picked_counts = counts.loc[picked].sort_values()
    picked_counts.rename("n_cells").rename_axis("perturbation").to_frame()\
        .to_csv(args.out_dir / "selected_perts.csv")
    print(f"  picked {len(picked)} perts; n_cells range {int(picked_counts.min())}–{int(picked_counts.max())}")
    print(f"  10-bin counts: {np.histogram(np.log10(picked_counts), bins=args.n_bins)[0].tolist()}")

    # ---- subset adata (perts + downsampled NT) ----
    print("Loading full adata (in-memory) — this can be slow…")
    t0 = time.time()
    a = ad.read_h5ad(args.adata)
    print(f"  full adata: {a.shape}  ({time.time()-t0:.1f}s)")
    grp = a.obs[args.group_key].astype(str).to_numpy()

    pick_set  = set(picked)
    pert_mask = np.isin(grp, list(pick_set))
    ctrl_mask = grp == args.control_label

    rng = np.random.default_rng(args.seed)
    n_ctrl_avail = int(ctrl_mask.sum())
    n_ctrl = min(args.n_ctrl, n_ctrl_avail)
    ctrl_idx = np.flatnonzero(ctrl_mask)
    keep_ctrl_idx = rng.choice(ctrl_idx, size=n_ctrl, replace=False)
    ctrl_keep = np.zeros_like(ctrl_mask, dtype=bool)
    ctrl_keep[keep_ctrl_idx] = True

    keep = pert_mask | ctrl_keep
    a_sub = a[keep, :].copy()
    print(f"  subset: {a_sub.shape}  (pert {int(pert_mask.sum()):,} + ctrl {n_ctrl:,})")

    subset_path = args.out_dir / "subset_DMSO_b1.h5ad"
    a_sub.write_h5ad(subset_path, compression="gzip")
    print(f"  wrote {subset_path}  ({subset_path.stat().st_size / 1e9:.2f} GB)")

    # ---- compute Welch (mean_diff, SE) on the SAME subset, on log1p(adata.X) ----
    print("Computing Welch (mean_diff, SE) per (pert, gene)…")
    if a_sub.X.dtype != np.float32 and a_sub.X.dtype != np.float64:
        a_sub.X = a_sub.X.astype(np.float32)

    sub_grp = a_sub.obs[args.group_key].astype(str).to_numpy()
    genes   = a_sub.var_names.to_numpy()
    ctrl_mask_sub = sub_grp == args.control_label
    n_ctrl_sub = int(ctrl_mask_sub.sum())

    s_c, ss_c = _sum_and_sumsq(a_sub.X[ctrl_mask_sub])
    mean_c, var_c = _mean_and_var(s_c, ss_c, n_ctrl_sub)

    rows = []
    for i, p in enumerate(picked, 1):
        if i % 50 == 0:
            print(f"    [{i:>4}/{len(picked)}] {p}", flush=True)
        m = sub_grp == p
        n_p = int(m.sum())
        if n_p < 2:
            continue
        s_p, ss_p = _sum_and_sumsq(a_sub.X[m])
        mean_p, var_p = _mean_and_var(s_p, ss_p, n_p)
        rows.append(pd.DataFrame({
            "perturbation": p,
            "gene":         genes,
            "mean_diff":    mean_p - mean_c,
            "se":           np.sqrt(var_p / n_p + var_c / n_ctrl_sub),
            "n_pert":       n_p,
            "n_ctrl":       n_ctrl_sub,
        }))

    welch = pd.concat(rows, ignore_index=True)
    welch_path = args.out_dir / "welch_se_long.parquet"
    welch.to_parquet(welch_path, index=False)
    print(f"Wrote {welch_path}  ({len(welch):,} rows)")

    print()
    print("Next step:")
    print(f"  Rscript {Path(__file__).parent.parent / 'RScripts' / 'RunGlmGamPoi_subset.R'} \\")
    print(f"      --adata {subset_path} \\")
    print(f"      --perts-csv {args.out_dir / 'selected_perts.csv'} \\")
    print(f"      --out {args.out_dir / 'glmgampoi_long.csv'}")


if __name__ == "__main__":
    main()
