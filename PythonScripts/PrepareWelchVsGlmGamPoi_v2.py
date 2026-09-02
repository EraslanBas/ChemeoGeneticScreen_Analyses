#!/usr/bin/env python3
"""
Prepare inputs for the Welch-vs-glmGamPoi comparison (Option A).

1. Pick 500 perturbations from the DMSO batch1 dataset, stratified across
   10 log-spaced bins of cell count.
2. Subsample a fixed shared set of `n_ctrl` non-targeting control cells
   (default 20,000) and write their IDs to `shared_nt_cell_ids.csv`.
3. For each picked perturbation, write a per-pert h5ad of (pert cells +
   shared NT cells) sliced from the **raw-counts** adata — glmGamPoi
   consumes these one at a time, in parallel via `xargs -P`.
4. Slice the **log1p-normalised** adata for the SAME cell IDs and compute
   the Welch (mean_diff, SE) per (perturbation, gene) — writes
   `welch_se_long.parquet`. Cell IDs are aligned between the two files,
   so both methods see the exact same perturbation × control cells.

Outputs (under --out-dir, default Notes/welch_vs_glmgampoi/):
    selected_perts.csv             perturbation, n_cells
    shared_nt_cell_ids.csv         cell_id (20k rows)
    per_pert_counts/<pert>.h5ad    one per selected perturbation
    welch_se_long.parquet          pert × gene × {mean_diff, se, n_pert, n_ctrl}
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


# ---------- Welch helpers — identical math to ComputeSE.py ----------

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


# ---------- stratified perturbation sampling ----------

def stratified_sample(counts: pd.Series, n_target: int = 500,
                       n_bins: int = 10, *, seed: int = 0,
                       min_cells: int = 5) -> list[str]:
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
    if len(picked) < n_target:
        remaining = [p for p in counts.index if p not in set(picked)]
        rng.shuffle(remaining)
        picked.extend(remaining[: n_target - len(picked)])
    return picked[:n_target]


# ---------- main ----------

DEFAULT_COUNTS = Path(
    "/processed_datasets/VCI/ChemoGenetic_H1/8cmpd_set1/Full/"
    "PertQC_CytoGMM/DualGuideOnly_ntgUPM5/DMSO/scanpy/"
    "ad_gene_guide_complete.gene.h5ad"
)
DEFAULT_LOG1P = Path(
    "/processed_datasets/VCI/ChemoGenetic_H1_Basak/DMSO_round2/DMSO_round2.h5ad"
)
DEFAULT_OUT = Path("/home/beraslan/Projects/ChemoGeneticScreens/Notes/welch_vs_glmgampoi")
DEFAULT_PER_PERT_DIR = Path(
    "/processed_datasets/VCI/ChemoGenetic_H1_Basak/DMSO_round2/per_pert_counts_500"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--counts-adata", type=Path, default=DEFAULT_COUNTS,
                     help="Raw-counts adata (for glmGamPoi).")
    ap.add_argument("--log1p-adata",  type=Path, default=DEFAULT_LOG1P,
                     help="log1p-normalised adata (for Welch SE).")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT,
                     help="Small artifacts: selected_perts.csv, "
                          "shared_nt_cell_ids.csv, welch_se_long.parquet.")
    ap.add_argument("--per-pert-dir", type=Path, default=DEFAULT_PER_PERT_DIR,
                     help="Per-pert h5ad files (bulk data). Default points "
                          "at /processed_datasets/... to avoid filling the "
                          "project disk.")
    ap.add_argument("--n-perts", type=int, default=500)
    ap.add_argument("--n-bins",  type=int, default=10)
    ap.add_argument("--min-cells", type=int, default=5)
    ap.add_argument("--n-ctrl",  type=int, default=5_000,
                     help="Downsampled NT control cells shared across all "
                          "per-pert h5ad files (default 5,000 — plenty for "
                          "glmGamPoi).")
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--group-key",     default="target_gene")
    ap.add_argument("--control-label", default="non-targeting")
    args = ap.parse_args()

    out_dir   = args.out_dir
    per_dir   = args.per_pert_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_dir.mkdir(parents=True, exist_ok=True)
    print(f"counts adata : {args.counts_adata}")
    print(f"log1p adata  : {args.log1p_adata}")
    print(f"out_dir      : {out_dir}")
    print(f"per_pert_dir : {per_dir}")

    # ---- load both adatas (full) ----
    print("\nLoading counts adata…", flush=True)
    t0 = time.time()
    a_cnt = ad.read_h5ad(args.counts_adata)
    print(f"  counts: {a_cnt.shape}  ({time.time()-t0:.1f}s)")

    t0 = time.time()
    print("Loading log1p adata…", flush=True)
    a_log = ad.read_h5ad(args.log1p_adata)
    print(f"  log1p: {a_log.shape}   ({time.time()-t0:.1f}s)")

    # ---- align: cell-ID and gene-name match ----
    if not (a_cnt.obs_names == a_log.obs_names).all():
        print("[warn] obs_names not pre-aligned — re-aligning log1p to counts order")
        a_log = a_log[a_cnt.obs_names].copy()
    if not (a_cnt.var_names == a_log.var_names).all():
        common_g = a_cnt.var_names.intersection(a_log.var_names)
        print(f"[warn] gene sets differ — intersecting to {len(common_g):,}")
        a_cnt = a_cnt[:, common_g].copy()
        a_log = a_log[:, common_g].copy()
    assert (a_cnt.obs_names == a_log.obs_names).all()
    print(f"  aligned: same {len(a_cnt):,} cells × {a_cnt.shape[1]:,} genes in both")

    tg = a_cnt.obs[args.group_key].astype(str).to_numpy()

    # ---- pick perturbations (stratified by n_cells) ----
    counts = pd.Series(tg).value_counts()
    counts_nondrug = counts[counts.index != args.control_label]
    picked = stratified_sample(counts_nondrug, n_target=args.n_perts,
                                n_bins=args.n_bins, seed=args.seed,
                                min_cells=args.min_cells)
    picked_counts = counts_nondrug.loc[picked].sort_values()
    picked_counts.rename("n_cells").rename_axis("perturbation").to_frame()\
        .to_csv(out_dir / "selected_perts.csv")
    print(f"\npicked {len(picked)} perts; n_cells range "
          f"{int(picked_counts.min())}–{int(picked_counts.max())}")
    print(f"  10-bin counts: {np.histogram(np.log10(picked_counts), bins=args.n_bins)[0].tolist()}")

    # ---- shared NT control subsample ----
    rng = np.random.default_rng(args.seed)
    nt_idx_all = np.flatnonzero(tg == args.control_label)
    n_ctrl = min(args.n_ctrl, len(nt_idx_all))
    nt_keep_idx = rng.choice(nt_idx_all, size=n_ctrl, replace=False)
    nt_keep_idx.sort()
    nt_ids = a_cnt.obs_names[nt_keep_idx].tolist()
    pd.Series(nt_ids, name="cell_id").to_csv(out_dir / "shared_nt_cell_ids.csv", index=False)
    print(f"shared NT cells   : {n_ctrl:,}")
    print(f"  saved IDs to     : {out_dir / 'shared_nt_cell_ids.csv'}")

    # ---- precompute NT control Welch stats once ----
    print("\nWelch: computing NT control stats once…", flush=True)
    nt_mat_log = a_log.X[nt_keep_idx]
    s_c, ss_c = _sum_and_sumsq(nt_mat_log)
    mean_c, var_c = _mean_and_var(s_c, ss_c, n_ctrl)
    genes = a_log.var_names.to_numpy()

    # ---- per-pert: write counts h5ad + accumulate Welch row ----
    print("\nFor each pert: write per-pert counts h5ad and compute Welch SE…", flush=True)
    welch_parts = []
    t_loop = time.time()
    for i, p in enumerate(picked, 1):
        pert_idx = np.flatnonzero(tg == p)
        n_p = int(pert_idx.size)

        # Counts subset for glmGamPoi
        sub_ids = np.concatenate([pert_idx, nt_keep_idx])
        sub_cnt = a_cnt[sub_ids, :].copy()
        # Ensure target_gene labels are the bare "non-targeting" / pert (not factor-coded)
        sub_cnt.obs[args.group_key] = sub_cnt.obs[args.group_key].astype(str)
        out_h5 = per_dir / f"{p}_counts.h5ad"
        # No compression — gzip dominates wall time and disk is ample.
        sub_cnt.write_h5ad(out_h5)

        # Welch on log1p (same indices)
        sub_log = a_log.X[pert_idx]
        s_p, ss_p = _sum_and_sumsq(sub_log)
        mean_p, var_p = _mean_and_var(s_p, ss_p, n_p)
        welch_parts.append(pd.DataFrame({
            "perturbation": p,
            "gene":         genes,
            "mean_diff":    mean_p - mean_c,
            "se":           np.sqrt(var_p / n_p + var_c / n_ctrl),
            "n_pert":       n_p,
            "n_ctrl":       n_ctrl,
        }))

        if i % 25 == 0 or i == len(picked):
            print(f"  [{i:>4}/{len(picked)}] {p}   n_pert={n_p:>6,}   "
                  f"({time.time()-t_loop:.0f}s elapsed)", flush=True)

    welch = pd.concat(welch_parts, ignore_index=True)
    welch_path = out_dir / "welch_se_long.parquet"
    welch.to_parquet(welch_path, index=False)
    print(f"\nWrote Welch table : {welch_path}  ({len(welch):,} rows)")
    print(f"Wrote per-pert h5ad files to: {per_dir}/   ({len(picked)} files)")

    print("\nNext step (parallel):")
    print(f"  cd {Path(__file__).parent.parent / 'BashScripts'}")
    print(f"  ./RunGlmGamPoi_perPertFiles.sh")


if __name__ == "__main__":
    main()
