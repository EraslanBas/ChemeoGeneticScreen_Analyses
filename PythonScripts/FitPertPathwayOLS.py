#!/usr/bin/env python3
"""
Fit OLS of pathway_score ~ perturbation per pathway, with non-targeting cells
as the reference level. Output (per drug): per-(pert, pathway) beta, p-value,
and BH-FDR matrices.

Uses statsmodels.api.OLS directly. The design matrix is built once per drug
(N x P float32 indicator matrix + intercept; non-targeting is the omitted
reference level) and reused across all pathways. Within each fit:

    score = X . beta + epsilon
    beta_{p}   = coefficient for pert p (vs NT)
    SE(beta_p) = sqrt( sigma^2 . [(X'X)^{-1}]_{p,p} )
    sigma^2    = SSR / (N - rank)

beta, SE, and t/p-values are read off `OLSResults` directly.

Inputs (per drug):
  /processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways/
    <drug>_perts941_obs.csv        cell_barcode, target_gene  (~5 MB, fast)
    pathway_scores_<drug>/score_<pathway>.csv    one CSV per pathway

Outputs (per drug, in --out-dir):
  beta_<drug>.parquet              (n_perts x n_pathways)
  se_<drug>.parquet                (n_perts x n_pathways)
  pvalue_<drug>.parquet            (n_perts x n_pathways)
  fdr_<drug>.parquet               (n_perts x n_pathways) BH within pathway
  n_pert_cells_<drug>.parquet      (n_perts x n_pathways) sample sizes

Caveat: cell-level p-values are anti-conservative because cells from the same
biological sample are correlated. For ranking (pert, pathway) hits this is
fine; for defensible inference switch to pseudobulk or include a random effect.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List

import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


DEFAULT_CACHE_DIR = Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways')
DEFAULT_OUT_DIR   = Path('/home/beraslan/Projects/ChemoGeneticScreens/PertPathwayOLS')


def build_design_matrix(pert_arr: np.ndarray,
                         nt_label: str = 'non-targeting',
                         dtype=np.float32,
                         ) -> tuple:
    """Build the OLS design matrix: intercept + dummy columns for every
    perturbation level (NT omitted = reference).

    Returns (X, pert_levels, n_pert_cells) where:
      X is float32 (N, P+1) with columns [intercept, pert_1, pert_2, ...]
      pert_levels is the list of perturbations corresponding to columns 1..P
      n_pert_cells is a Series indexed by pert_level with cell counts
    """
    pert_arr = np.asarray(pert_arr)
    pert_levels = sorted(set(pert_arr) - {nt_label})
    pert_to_col = {p: i + 1 for i, p in enumerate(pert_levels)}
    N = len(pert_arr)
    P = len(pert_levels)
    X = np.zeros((N, P + 1), dtype=dtype)
    X[:, 0] = 1.0
    counts = np.zeros(P, dtype=np.int64)
    for i, p in enumerate(pert_arr):
        if p == nt_label:
            continue
        col = pert_to_col[p]
        X[i, col] = 1.0
        counts[col - 1] += 1
    return X, pert_levels, pd.Series(counts, index=pert_levels, name='n_pert_cells')


def fit_one_pathway(X: np.ndarray,
                     score: np.ndarray,
                     pert_levels: list,
                     n_pert_cells: pd.Series,
                     min_pert_cells: int = 2,
                     ) -> pd.DataFrame:
    """Fit statsmodels OLS once for the supplied design matrix and score
    vector. Returns DataFrame indexed by pert with beta/se/pvalue/n.
    """
    model = sm.OLS(score.astype(np.float64), X.astype(np.float64)).fit()
    # Coefficients 1..P+1 correspond to pert_levels (column 0 is the intercept)
    betas  = model.params[1:]
    ses    = model.bse[1:]
    pvalues = model.pvalues[1:]
    out = pd.DataFrame({
        'beta':         betas,
        'se':           ses,
        'pvalue':       pvalues,
        'n_pert_cells': n_pert_cells.values.astype(np.int32),
    }, index=pert_levels)
    out.index.name = 'perturbation'

    too_few = out['n_pert_cells'] < min_pert_cells
    if too_few.any():
        out.loc[too_few, ['beta', 'se', 'pvalue']] = np.nan
    return out


def process_drug(drug: str,
                  cache_dir: Path,
                  out_dir: Path,
                  min_pert_cells: int) -> None:
    obs_csv     = cache_dir / f'{drug}_perts941_obs.csv'
    scores_dir  = cache_dir / f'pathway_scores_{drug}'
    if not obs_csv.exists():
        raise FileNotFoundError(f'Missing obs csv: {obs_csv}  '
                                  f'(generate with: pd.DataFrame from a.obs_names + a.obs["target_gene"])')
    if not scores_dir.exists():
        raise FileNotFoundError(f'Missing pathway scores dir: {scores_dir}')

    print(f'\n=== {drug} ===')
    print(f'Loading {obs_csv}')
    t0 = time.time()
    obs = pd.read_csv(obs_csv, dtype={'cell_barcode': str, 'target_gene': str})
    bc_to_pert = pd.Series(
        obs['target_gene'].values,
        index=obs['cell_barcode'].values,
        name='target_gene',
    )
    print(f'  {len(bc_to_pert):,} cells, {bc_to_pert.nunique():,} distinct target_gene levels  '
          f'({time.time()-t0:.1f}s)')

    csv_files = sorted(scores_dir.glob('score_*.csv'))
    print(f'\n{len(csv_files)} pathway CSVs in {scores_dir}')

    # Per-pathway result files land here; safe across crashes (skip-existing).
    per_path_dir = out_dir / f'per_pathway_{drug}'
    per_path_dir.mkdir(parents=True, exist_ok=True)
    print(f'Per-pathway results: {per_path_dir}')

    # Build the design matrix ONCE per drug — same X reused across all pathways.
    print('Building design matrix (intercept + dummies for every pert level, NT omitted)...')
    t_X = time.time()
    bc_arr     = bc_to_pert.index.values
    pert_arr   = bc_to_pert.values
    X_full, pert_levels, n_cells_per_pert = build_design_matrix(pert_arr)
    print(f'  X.shape = {X_full.shape}  ({X_full.nbytes / 1e9:.2f} GB float32)  '
          f'pert_levels = {len(pert_levels)}  ({time.time()-t_X:.1f}s)')

    bc_idx = pd.Series(np.arange(len(bc_arr)), index=bc_arr)

    t_total = time.time()
    n_done, n_existing, n_empty = 0, 0, 0
    for i, fp in enumerate(csv_files, 1):
        pname = fp.stem.replace('score_', '')
        out_pq = per_path_dir / f'{pname}.parquet'
        if out_pq.exists():
            n_existing += 1
            continue
        ts = time.time()
        scores = pd.read_csv(fp, index_col=0, dtype={'cell_barcode': str})['score']
        scores.index = scores.index.astype(str)
        rows = bc_idx.reindex(scores.index).dropna().astype(int)
        if rows.empty:
            print(f'  [{i:>3}/{len(csv_files)}] {pname}: no overlap with subset, skipping')
            n_empty += 1
            continue
        X_sub = X_full[rows.values]
        y_sub = scores.loc[rows.index].values
        col_counts = X_sub[:, 1:].sum(axis=0).astype(np.int64)
        n_pert_cells_sub = pd.Series(col_counts, index=pert_levels, name='n_pert_cells')

        res = fit_one_pathway(
            X_sub, y_sub, pert_levels, n_pert_cells_sub,
            min_pert_cells=min_pert_cells,
        )
        # Save immediately — atomic write via .tmp rename
        tmp = out_pq.with_suffix('.parquet.tmp')
        res.to_parquet(tmp)
        os.replace(tmp, out_pq)
        n_done += 1
        if (n_done % 5 == 0) or (i == len(csv_files)):
            elapsed = time.time() - t_total
            rate = elapsed / max(n_done, 1)
            rem  = (len(csv_files) - i) * rate
            print(f'  [{i:>3}/{len(csv_files)}] {pname[:48]:<48s}  '
                  f'fit={time.time()-ts:>5.1f}s  done={n_done} existing={n_existing}  '
                  f'elapsed={elapsed/60:>5.1f}m  ETA={rem/60:>5.1f}m')

    print(f'\nFitted {n_done} new pathways  (existing: {n_existing}, no-overlap: {n_empty})')

    # ----------------------------------------------------------------------
    # Build the matrices from per-pathway parquets
    # ----------------------------------------------------------------------
    print(f'\nBuilding (perts x pathways) matrices from {per_path_dir} ...')
    parquets = sorted(per_path_dir.glob('*.parquet'))
    if not parquets:
        raise RuntimeError(f'No per-pathway parquets in {per_path_dir}')
    betas_d, se_d, pvals_d, n_d = {}, {}, {}, {}
    for pq in parquets:
        pname = pq.stem
        df_p = pd.read_parquet(pq)
        betas_d[pname] = df_p['beta']
        se_d[pname]    = df_p['se']
        pvals_d[pname] = df_p['pvalue']
        n_d[pname]     = df_p['n_pert_cells']
    print(f'  loaded {len(parquets)} pathway results')

    all_perts = sorted(set().union(*(s.index for s in betas_d.values())))
    pathways  = list(betas_d.keys())

    def to_matrix(d: Dict[str, pd.Series], dtype=np.float64) -> pd.DataFrame:
        df = pd.DataFrame(d).reindex(index=all_perts, columns=pathways)
        return df.astype(dtype)

    beta_df  = to_matrix(betas_d, np.float32)
    se_df    = to_matrix(se_d,    np.float32)
    pval_df  = to_matrix(pvals_d, np.float64)
    n_df     = to_matrix(n_d,     'Int32')

    # BH within pathway across perturbations
    print(f'\nBH-correcting within each of {pval_df.shape[1]} pathways across {pval_df.shape[0]} perts...')
    fdr_df = pd.DataFrame(np.nan, index=pval_df.index, columns=pval_df.columns)
    for pname in pval_df.columns:
        p = pval_df[pname].values
        ok = ~np.isnan(p)
        if ok.any():
            _, q, _, _ = multipletests(p[ok], method='fdr_bh')
            qfull = np.full_like(p, np.nan, dtype=np.float64)
            qfull[ok] = q
            fdr_df[pname] = qfull
    fdr_df = fdr_df.astype(np.float32)

    # Save
    out_dir.mkdir(parents=True, exist_ok=True)
    beta_df.to_parquet(out_dir / f'beta_{drug}.parquet')
    se_df.to_parquet(  out_dir / f'se_{drug}.parquet')
    pval_df.to_parquet(out_dir / f'pvalue_{drug}.parquet')
    fdr_df.to_parquet( out_dir / f'fdr_{drug}.parquet')
    n_df.to_parquet(   out_dir / f'n_pert_cells_{drug}.parquet')

    print(f'\nWrote (rows = {beta_df.shape[0]} perts, cols = {beta_df.shape[1]} pathways):')
    for kind in ('beta', 'se', 'pvalue', 'fdr', 'n_pert_cells'):
        fp = out_dir / f'{kind}_{drug}.parquet'
        print(f'  {fp}  ({fp.stat().st_size/1e6:.2f} MB)')

    # Quick summary
    sig_q = (fdr_df < 0.10).sum().sum()
    n_tested = (~fdr_df.isna()).sum().sum()
    print(f'\nSignificant (pert x pathway) at BH q<0.10: {sig_q:,} / {n_tested:,}  '
          f'({100*sig_q/max(n_tested,1):.2f}%)')
    print(f'Total runtime: {(time.time()-t_total)/60:.1f} min')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--drugs', nargs='+',
                    default=['DMSO_round2', 'DMSO_round2_batch2'],
                    help='Drug names to process. Defaults to both DMSOs.')
    ap.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE_DIR,
                    help='Dir holding <drug>_perts941.h5ad and pathway_scores_<drug>/.')
    ap.add_argument('--out-dir', type=Path, default=DEFAULT_OUT_DIR,
                    help='Output directory for parquet matrices.')
    ap.add_argument('--min-pert-cells', type=int, default=2,
                    help='Perts with fewer than this many cells get NaN beta/p-value.')
    args = ap.parse_args()

    for d in args.drugs:
        process_drug(d, args.cache_dir, args.out_dir, args.min_pert_cells)


if __name__ == '__main__':
    main()
