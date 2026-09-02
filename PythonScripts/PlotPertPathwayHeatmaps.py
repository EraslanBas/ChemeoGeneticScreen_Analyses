#!/usr/bin/env python3
"""
Build pert × pathway × drug significance/effect tensors from per-(group,
pathway, batch) OLS CSVs and plot a 2×8 grid of heatmaps:

  Row 1:  7 batch1 drugs (drug:target_gene interactions vs DMSO_round2)
          + DMSO_round2 panel (target_gene main effects, batch1 model)
  Row 2:  7 batch2 drugs (interactions vs DMSO_round2_batch2)
          + DMSO_round2_batch2 panel (main effects, batch2 model)

Each panel cell ∈ {-1, 0, +1} = sign(β) × (pval < threshold).

Filters (applied sequentially):
  - perts: keep those with ≥ --pert-min-sig significant interactions
           (counted across all 14 non-DMSO drugs and all complete pathways)
  - pathways: keep those with ≥ --path-min-sig significant tests
           (counted across kept perts and 14 non-DMSO drugs)

Inputs: /home/beraslan/Projects/ChemoGeneticScreens/PertPathwayDrugInteraction/
        <group>/<pathway>__{batch1,batch2}.csv
"""

from __future__ import annotations
import argparse
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import linkage, leaves_list

warnings.filterwarnings('ignore')

BATCH_DRUGS = {
    'batch1': ['AR-A014418', 'AZD4573', 'CHIR-98014', 'Lexibulin', 'PP121',
                'Romidepsin', 'Stattic', 'DMSO_round2'],
    'batch2': ['Bisindolylmaleimide-I', 'DG-172', 'JTE-607', 'LDN-193189',
                'LY2090314', 'NSC95397', 'VX-11e', 'DMSO_round2_batch2'],
}
DMSO_OF_BATCH = {'batch1': 'DMSO_round2', 'batch2': 'DMSO_round2_batch2'}

DIFF_CMAP = ListedColormap(['cornflowerblue', 'white', 'crimson'])


def discover_complete_pathways(input_dir: Path):
    """Return sorted list of pathways with both-batch CSVs in every group."""
    groups = sorted([d.name for d in input_dir.iterdir()
                      if d.is_dir() and d.name.startswith('group_')])
    if not groups:
        raise SystemExit(f'no group_* dirs in {input_dir}')

    pathways_by_gb = {}
    for g in groups:
        for batch in ('batch1', 'batch2'):
            files = list((input_dir / g).glob(f'*__{batch}.csv'))
            pathways_by_gb[(g, batch)] = {
                f.name.replace(f'__{batch}.csv', '') for f in files
            }

    sets = list(pathways_by_gb.values())
    common = set.intersection(*sets) if sets else set()
    n_total = len(set.union(*sets)) if sets else 0
    print(f'Groups detected: {len(groups)}')
    print(f'Pathways with full coverage ({len(groups)} groups × 2 batches = '
          f'{len(groups)*2} CSVs): {len(common)} / {n_total}')
    return groups, sorted(common)


def load_ols_long(input_dir: Path, groups, pathways):
    """Load all relevant rows (target_gene main effects + drug:target_gene
    interactions) into a single long DataFrame."""
    if not pathways:
        raise SystemExit('no complete pathways yet — wait for more groups to finish')
    rows = []
    t0 = time.time()
    expected = len(groups) * len(pathways) * 2
    n_done = 0
    for g in groups:
        for batch in ('batch1', 'batch2'):
            for p in pathways:
                fp = input_dir / g / f'{p}__{batch}.csv'
                if not fp.exists():
                    continue
                df = pd.read_csv(fp, dtype={'drug': str, 'target_gene': str})
                df = df[df['term_type'].isin(('target_gene', 'drug:target_gene'))]
                if df.empty:
                    continue
                df['group']   = g
                df['batch']   = batch
                df['pathway'] = p
                rows.append(df)
                n_done += 1
                if n_done % 2000 == 0:
                    print(f'  loaded {n_done:,}/{expected:,} CSVs '
                          f'({time.time()-t0:.0f}s)', flush=True)
    long_df = pd.concat(rows, ignore_index=True)
    print(f'  loaded {n_done:,}/{expected:,} CSVs in {time.time()-t0:.0f}s '
          f'({len(long_df):,} rows total)')
    return long_df


def build_tensors(long_df):
    """Pivot long_df into wide pert × (drug, pathway) tables for both
    interactions and DMSO main effects."""
    # Drug:target_gene interactions: drug = the non-DMSO drug
    ints = long_df[long_df['term_type'] == 'drug:target_gene'].copy()
    ints = ints.dropna(subset=['drug', 'target_gene'])

    # target_gene main effects: assign drug = DMSO of that batch
    mes = long_df[long_df['term_type'] == 'target_gene'].copy()
    mes = mes.dropna(subset=['target_gene'])
    mes['drug'] = mes['batch'].map(DMSO_OF_BATCH)

    combo = pd.concat([ints, mes], ignore_index=True)
    print(f'Long combined rows: {len(combo):,}')
    print(f'  drugs: {combo["drug"].nunique()}'
          f'  perts: {combo["target_gene"].nunique()}'
          f'  pathways: {combo["pathway"].nunique()}')

    beta = combo.pivot_table(values='beta',
                              index='target_gene',
                              columns=['drug', 'pathway'])
    pval = combo.pivot_table(values='pvalue',
                              index='target_gene',
                              columns=['drug', 'pathway'])
    return beta, pval


def to_sign_sig(beta, pval, p_threshold):
    """Cell ∈ {-1, 0, +1} = sign(β) when pvalue < threshold else 0."""
    sig = (pval < p_threshold).fillna(False).values
    sn  = np.sign(beta.fillna(0).values).astype(np.int8)
    out = (sn * sig.astype(np.int8))
    return pd.DataFrame(out, index=beta.index, columns=beta.columns)


def filter_perts_pathways(sig, pert_min_sig, path_min_sig):
    """Two-stage filter: perts → pathways."""
    non_dmso = [d for d in sig.columns.get_level_values('drug').unique()
                 if d not in DMSO_OF_BATCH.values()]
    sig_int = sig.loc[:, sig.columns.get_level_values('drug').isin(non_dmso)]

    # Per pert: number of (drug, pathway) cells with |sig|=1
    pert_counts = sig_int.abs().sum(axis=1)
    keep_perts = pert_counts[pert_counts >= pert_min_sig].index.tolist()
    print(f'Perts kept: {len(keep_perts)} / {sig_int.shape[0]} '
          f'(min sig interactions per pert ≥ {pert_min_sig})')
    if not keep_perts:
        raise SystemExit('no perts pass the filter — lower --pert-min-sig')

    # Per pathway: number of (drug, kept_pert) cells with |sig|=1
    sig_int_kp = sig_int.loc[keep_perts]
    by_path = sig_int_kp.abs().groupby(level='pathway', axis=1).sum().sum(axis=0)
    keep_paths = by_path[by_path >= path_min_sig].index.tolist()
    print(f'Pathways kept: {len(keep_paths)} / {len(by_path)} '
          f'(min sig tests per pathway ≥ {path_min_sig})')
    if not keep_paths:
        raise SystemExit('no pathways pass the filter — lower --path-min-sig')

    return keep_perts, keep_paths


def hier_order(mat, axis):
    """Return hierarchical-clustering leaf order along the given axis."""
    if mat.shape[axis] <= 1:
        return mat.axes[axis].tolist()
    arr = mat.values if axis == 0 else mat.values.T
    arr = np.nan_to_num(arr, nan=0.0)
    Z = linkage(arr, method='average', metric='euclidean')
    return [mat.axes[axis][i] for i in leaves_list(Z)]


def plot_grid(sig, beta, keep_perts, keep_paths, out_fp, p_threshold):
    # Subset
    sig_kp = sig.loc[keep_perts]
    sig_kp = sig_kp.loc[:, sig_kp.columns.get_level_values('pathway').isin(keep_paths)]
    beta_kp = beta.loc[keep_perts]
    beta_kp = beta_kp.loc[:, beta_kp.columns.get_level_values('pathway').isin(keep_paths)]

    # Order perts by hierarchical clustering on combined sign-sig matrix
    pert_order = hier_order(sig_kp, axis=0)
    # Order pathways by clustering on |sig| aggregated across drugs
    sigabs_by_path = sig_kp.abs().groupby(level='pathway', axis=1).sum()
    path_order = hier_order(sigabs_by_path[keep_paths], axis=1)

    fig, axes = plt.subplots(2, 8, figsize=(34, 14))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.92, bottom=0.05,
                        wspace=0.06, hspace=0.18)

    for row_i, batch in enumerate(['batch1', 'batch2']):
        non_dmso_drugs = [d for d in BATCH_DRUGS[batch] if d != DMSO_OF_BATCH[batch]]
        dmso = DMSO_OF_BATCH[batch]
        for col_i, drug in enumerate(non_dmso_drugs):
            ax = axes[row_i, col_i]
            try:
                sub = sig_kp.xs(drug, level='drug', axis=1)
            except KeyError:
                ax.set_axis_off(); continue
            sub = sub.reindex(index=pert_order, columns=path_order)
            sns.heatmap(sub, ax=ax, cmap=DIFF_CMAP, vmin=-1, vmax=1, center=0,
                         cbar=False, xticklabels=False, yticklabels=False)
            ax.set_title(f'{drug}\n(vs {dmso})', fontsize=12)

        # Last column: DMSO main-effects panel
        ax = axes[row_i, 7]
        try:
            sub = sig_kp.xs(dmso, level='drug', axis=1)
            sub = sub.reindex(index=pert_order, columns=path_order)
            sns.heatmap(sub, ax=ax, cmap=DIFF_CMAP, vmin=-1, vmax=1, center=0,
                         cbar=False, xticklabels=False, yticklabels=False)
            ax.set_title(f'{dmso}\n(pert main effect)', fontsize=12,
                          fontweight='bold')
        except KeyError:
            ax.set_axis_off()

    fig.supxlabel(f'{len(path_order):,} Pathways', fontsize=18)
    fig.supylabel(f'{len(pert_order):,} Perturbations', fontsize=18)
    fig.legend(
        handles=[
            Patch(facecolor='crimson',          label=f'+1 sig positive (p<{p_threshold})'),
            Patch(facecolor='white', edgecolor='gray', label='0 not significant'),
            Patch(facecolor='cornflowerblue',   label=f'−1 sig negative (p<{p_threshold})'),
        ],
        loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=3, fontsize=12,
        frameon=False,
    )
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fp, dpi=120, bbox_inches='tight')
    print(f'Wrote {out_fp}')

    # Also save the underlying tensors for downstream reuse
    parquet_dir = out_fp.parent / 'tensors'
    parquet_dir.mkdir(parents=True, exist_ok=True)
    sig_kp.to_parquet(parquet_dir / 'sign_sig.parquet')
    beta_kp.to_parquet(parquet_dir / 'beta.parquet')
    pd.Series(pert_order).to_csv(parquet_dir / 'pert_order.csv', index=False, header=['pert'])
    pd.Series(path_order).to_csv(parquet_dir / 'pathway_order.csv', index=False, header=['pathway'])
    print(f'Wrote tensors + orderings to {parquet_dir}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input-dir', type=Path,
                    default=Path('/home/beraslan/Projects/ChemoGeneticScreens/PertPathwayDrugInteraction'))
    ap.add_argument('--output-fig', type=Path,
                    default=Path('/home/beraslan/Projects/ChemoGeneticScreens/PertPathwayDrugInteraction/figures/pert_pathway_drug_grid.png'))
    ap.add_argument('--p-threshold', type=float, default=0.05)
    ap.add_argument('--pert-min-sig', type=int, default=10,
                    help='Min # of significant (drug,pathway) interactions per pert to retain (default 10)')
    ap.add_argument('--path-min-sig', type=int, default=20,
                    help='Min # of significant (drug,kept_pert) tests per pathway to retain (default 20)')
    args = ap.parse_args()

    groups, complete = discover_complete_pathways(args.input_dir)
    long_df = load_ols_long(args.input_dir, groups, complete)
    beta, pval = build_tensors(long_df)
    print(f'beta tensor shape: {beta.shape}  (perts × (drug,pathway))')

    sig = to_sign_sig(beta, pval, args.p_threshold)
    keep_perts, keep_paths = filter_perts_pathways(sig,
                                                     args.pert_min_sig,
                                                     args.path_min_sig)
    plot_grid(sig, beta, keep_perts, keep_paths, args.output_fig, args.p_threshold)


if __name__ == '__main__':
    main()
