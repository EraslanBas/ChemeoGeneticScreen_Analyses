
#!/usr/bin/env python3
"""
Run MOFA+ on the per-drug PosteriorMean logFC matrices, with drugs as views.

Setup:
- 16 views (one per drug context)
- samples = perturbations (intersection across drugs)
- features = genes (intersection across drugs, with optional |median-logFC| filter)
- Gaussian likelihood (data is Bayesian-shrunk logFC, near-zero-centered)
- ARD on both factors and weights → automatic sparsity:
  * weight ARD switches gene-loading mass off in irrelevant factors
  * factor ARD switches a factor off entirely in views where it's inactive,
    which is exactly the shared/private decomposition we want.

Outputs (in <out_dir>):
  mofa_model.hdf5            — full MOFA model (loadable via mofapy2)
  factors.parquet            — (n_perts, n_factors) factor matrix
  loadings_<drug>.parquet    — (n_genes, n_factors) per-view loadings
  factor_view_alpha.parquet  — (n_factors, n_views) ARD weights:
                               low alpha = factor active in that view
  variance_explained.parquet — per-(view, factor) R^2
  axes.json                  — kept perts, kept genes, drug order
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def load_per_drug(pm_dir: Path) -> Tuple[Dict[str, pd.DataFrame], List[str]]:
    files = sorted(pm_dir.glob('PosteriorMean_matrix_*.csv'))
    if not files:
        raise RuntimeError(f'No PosteriorMean_matrix_*.csv under {pm_dir}')
    drugs: List[str] = []
    mats: Dict[str, pd.DataFrame] = {}
    for fp in files:
        drug = fp.stem.replace('PosteriorMean_matrix_', '')
        drugs.append(drug)
        df = pd.read_csv(fp, index_col=0)
        df.index.name = 'perturbation'
        mats[drug] = df
        print(f'  {drug:30s}  {df.shape[0]:>5d} perts × {df.shape[1]:>6d} genes')
    return mats, drugs


def align(mats: Dict[str, pd.DataFrame],
          gene_median_thr: float | None) -> Tuple[Dict[str, pd.DataFrame], List[str], List[str]]:
    """Intersect perturbation rows + gene columns across all drugs.

    Optionally drop genes whose absolute median logFC across perturbations
    (taken per drug, then min across drugs) is below `gene_median_thr` —
    follows the convention used by the rowbound CSVs.
    """
    common_perts = sorted(set.intersection(*(set(df.index) for df in mats.values())))
    common_genes = sorted(set.intersection(*(set(df.columns) for df in mats.values())))
    print(f'\nIntersection: {len(common_perts)} perts × {len(common_genes)} genes')

    if gene_median_thr is not None and gene_median_thr > 0:
        per_drug_max_abs_median = np.zeros(len(common_genes))
        for d, df in mats.items():
            sub = df.loc[common_perts, common_genes].abs().median(axis=0).values
            per_drug_max_abs_median = np.maximum(per_drug_max_abs_median, sub)
        keep_g = per_drug_max_abs_median >= gene_median_thr
        common_genes = [g for g, k in zip(common_genes, keep_g) if k]
        print(f'After |median logFC| ≥ {gene_median_thr}: {len(common_genes)} genes kept')

    aligned = {d: df.loc[common_perts, common_genes] for d, df in mats.items()}
    return aligned, common_perts, common_genes


def to_mofa_inputs(aligned: Dict[str, pd.DataFrame],
                   drugs: List[str]) -> Tuple[List[List[np.ndarray]], List[str], List[str]]:
    """Build the data matrix mofapy2 expects.

    Layout: data[v][g] is a numpy (n_samples × n_features) matrix. We have a
    single sample group, so g=0; one view per drug.
    """
    data = [[aligned[d].values.astype(np.float32)] for d in drugs]
    return data, list(aligned[drugs[0]].index), list(aligned[drugs[0]].columns)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--pm-dir',     type=Path,
                    default=Path('/home/beraslan/Projects/ChemoGeneticScreens/PosteriorMeanMatrices'))
    ap.add_argument('--out-dir',    type=Path,
                    default=Path('/home/beraslan/Projects/ChemoGeneticScreens/MOFA'))
    ap.add_argument('--n-factors',  type=int, default=30,
                    help='Initial number of factors (ARD will prune unused ones).')
    ap.add_argument('--gene-median-thr', type=float, default=0.0,
                    help='Drop genes whose max-across-drugs |median logFC| is below this. '
                         'Data is Bayesian-shrunk so |logFC| is typically <0.01 — keep at 0 '
                         'to disable, or pass a tiny value like 0.002 to drop dead genes.')
    ap.add_argument('--max-iter',   type=int, default=1000)
    ap.add_argument('--seed',       type=int, default=0)
    ap.add_argument('--scale-views', action='store_true',
                    help='Scale variance-per-view to 1 before fitting (MOFA option).')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Loading per-drug PosteriorMean matrices from {args.pm_dir}')
    mats, drugs = load_per_drug(args.pm_dir)
    print(f'\n{len(drugs)} drug contexts')

    aligned, perts, genes = align(mats, args.gene_median_thr)
    print(f'\nFinal aligned tensor: {len(perts)} perts × {len(genes)} genes × {len(drugs)} views')

    data, sample_names, feature_names = to_mofa_inputs(aligned, drugs)

    # -------------------------------------------------------------------
    # Build & train MOFA+
    # -------------------------------------------------------------------
    from mofapy2.run.entry_point import entry_point
    ent = entry_point()

    ent.set_data_options(scale_views=args.scale_views, scale_groups=False)
    ent.set_data_matrix(
        data,
        views_names=drugs,
        groups_names=['all'],
        samples_names=[sample_names],
        features_names=[feature_names for _ in drugs],
        likelihoods=['gaussian'] * len(drugs),    # force Gaussian — data is shrunk logFC
    )
    ent.set_model_options(
        factors=args.n_factors,
        likelihoods=['gaussian'] * len(drugs),
        spikeslab_weights=True,    # sparsity prior on gene loadings
        ard_factors=True,          # per (view, factor) ARD — drives shared/private split
        ard_weights=True,          # per (view, factor) loading sparsity
    )
    ent.set_train_options(
        iter=args.max_iter,
        convergence_mode='medium',
        startELBO=1,
        freqELBO=10,
        gpu_mode=False,
        seed=args.seed,
        verbose=True,
    )

    print('\nBuilding & training MOFA+ ...')
    t0 = time.time()
    ent.build()
    ent.run()
    print(f'  trained in {(time.time()-t0)/60:.1f} min')

    # -------------------------------------------------------------------
    # Save the canonical HDF5 model and pull tabular outputs
    # -------------------------------------------------------------------
    model_path = args.out_dir / 'mofa_model.hdf5'
    ent.save(str(model_path), save_data=False)
    print(f'\nSaved model → {model_path}  ({model_path.stat().st_size/1e6:.1f} MB)')

    with h5py.File(model_path, 'r') as h:
        # Z is per-group; we have one group ('all')
        z_keys = list(h['expectations/Z'].keys())
        Z = np.asarray(h[f'expectations/Z/{z_keys[0]}'][()]).T   # (n_samples, n_factors)
        # W is per-view
        view_keys = list(h['expectations/W'].keys())
        W = {v: np.asarray(h[f'expectations/W/{v}'][()]).T       # (n_features, n_factors)
             for v in view_keys}
        # ARD on factors: alpha matrix (n_factors, n_views) — large alpha = factor inactive in view
        alpha_keys = list(h['expectations/AlphaW'].keys())
        alpha_per_view = {v: np.asarray(h[f'expectations/AlphaW/{v}'][()]).flatten()
                           for v in alpha_keys}

    pert_idx = sample_names
    factor_cols = [f'factor_{i}' for i in range(Z.shape[1])]
    factor_df = pd.DataFrame(Z, index=pert_idx, columns=factor_cols)
    factor_df.index.name = 'perturbation'
    factor_df.to_parquet(args.out_dir / 'factors.parquet')
    print(f'Saved factors:    {args.out_dir / "factors.parquet"}  '
          f'({factor_df.shape[0]} perts × {factor_df.shape[1]} factors)')

    for v, mat in W.items():
        Wdf = pd.DataFrame(mat, index=feature_names, columns=factor_cols)
        Wdf.index.name = 'gene'
        Wdf.to_parquet(args.out_dir / f'loadings_{v}.parquet')

    alpha_df = pd.DataFrame(
        np.column_stack([alpha_per_view[v] for v in drugs]),
        index=factor_cols,
        columns=drugs,
    )
    alpha_df.index.name = 'factor'
    alpha_df.to_parquet(args.out_dir / 'factor_view_alpha.parquet')
    print(f'Saved per-(factor,view) ARD α: {args.out_dir / "factor_view_alpha.parquet"}')

    # Variance explained per (view, factor) — read from the model file
    with h5py.File(model_path, 'r') as h:
        if 'variance_explained' in h:
            r2_per_factor = {}
            for v in drugs:
                if v in h['variance_explained/r2_per_factor']:
                    g_keys = list(h[f'variance_explained/r2_per_factor/{v}'].keys())
                    r2_per_factor[v] = np.asarray(
                        h[f'variance_explained/r2_per_factor/{v}/{g_keys[0]}'][()]
                    ).flatten()
            r2_df = pd.DataFrame(
                np.column_stack([r2_per_factor[v] for v in drugs]),
                index=factor_cols,
                columns=drugs,
            )
            r2_df.index.name = 'factor'
            r2_df.to_parquet(args.out_dir / 'variance_explained.parquet')
            print(f'Saved variance explained: {args.out_dir / "variance_explained.parquet"}')

    with open(args.out_dir / 'axes.json', 'w') as f:
        json.dump({
            'drugs':         drugs,
            'perturbations': pert_idx,
            'genes':         feature_names,
            'n_factors':     Z.shape[1],
            'gene_median_thr': args.gene_median_thr,
            'seed':          args.seed,
        }, f, indent=2)
    print(f'Saved axes.json:  {args.out_dir / "axes.json"}')

    # -------------------------------------------------------------------
    # Quick console summary of shared vs private factors
    # -------------------------------------------------------------------
    # A factor is "active" in a view if its ARD α is small (loadings non-shrunk).
    # Use the median α across factors as a per-view scale, then call factor k
    # active in view v if α_{k,v} < 2× the global median.
    alpha_arr = alpha_df.values                                    # (n_factors, n_views)
    thr = 2.0 * np.median(alpha_arr)
    active_views_per_factor = (alpha_arr < thr).sum(axis=1)
    n_factors = alpha_arr.shape[0]
    n_shared  = int((active_views_per_factor >= len(drugs) - 1).sum())
    n_private = int((active_views_per_factor == 1).sum())
    n_partial = int(n_factors - n_shared - n_private)
    print(f'\nFactor activity summary (α < 2× median):')
    print(f'  Shared (active in ≥ {len(drugs)-1}/{len(drugs)} views): {n_shared}')
    print(f'  Private (active in exactly 1 view):                       {n_private}')
    print(f'  Partial (active in 2–{len(drugs)-2} views):               {n_partial}')


if __name__ == '__main__':
    main()
