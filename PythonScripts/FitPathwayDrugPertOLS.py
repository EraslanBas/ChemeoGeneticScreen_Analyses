#!/usr/bin/env python3
"""
Fit OLS per (group, pathway, batch) with the formula:

    pathway_score ~ C(drug, Treatment(reference=DMSO_for_batch))
                    + C(target_gene, Treatment(reference="non-targeting"))
                    + C(drug):C(target_gene)
                    + <continuous technical covariates from .obs>

Fits are split by experimental batch. Each batch uses its own DMSO control as the
reference level for the drug factor:

  batch1  → drugs = {AR-A014418, AZD4573, CHIR-98014, DMSO_round2,
                      Lexibulin, PP121, Romidepsin, Stattic}
            reference drug = DMSO_round2
  batch2  → drugs = {Bisindolylmaleimide-I, DG-172, DMSO_round2_batch2,
                      JTE-607, LDN-193189, LY2090314, NSC95397, VX-11e}
            reference drug = DMSO_round2_batch2

target_gene reference is always 'non-targeting'.

Default covariates: pct_counts_mt, log1p_total_counts.

Output: one CSV per (pathway, batch) under
  <output_dir>/<group>/<pathway>__<batch>.csv
columns:
  term_type  ∈ {'intercept','drug','target_gene','drug:target_gene','covariate'}
  drug, target_gene, covariate, beta, se, tvalue, pvalue, n_cells

Score-CSV rows are in the same order as the concat AnnData's .obs (the scoring
script writes zip(obs_names, drug, target_gene, score) row-wise), so .obs
covariates are attached by row index — no merge needed.

Trial defaults: 10 pathways from group_00, both batches.
"""

from __future__ import annotations

import argparse
import gc
import os
import re
import resource
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

DEFAULT_EXCLUDE_PATTERN = r'^KEGG_MEDICUS_PATHOGEN_|^KEGG_PATHOGENIC_'
DEFAULT_EXCLUDE_LIST = Path(
    '/home/beraslan/Projects/ChemoGeneticScreens/PathwayORA/KEGG/pathways_exclude.txt'
)


def _load_exclude_list(fp):
    if fp is None:
        return set()
    fp = Path(fp)
    if not fp.exists():
        print(f'  warning: exclude list not found: {fp}')
        return set()
    out = set()
    for line in fp.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        out.add(line)
    return out

warnings.filterwarnings('ignore')


BATCH_SPEC = {
    'batch1': {
        'drugs': ['AR-A014418', 'AZD4573', 'CHIR-98014', 'DMSO_round2',
                   'Lexibulin', 'PP121', 'Romidepsin', 'Stattic'],
        'ref':   'DMSO_round2',
    },
    'batch2': {
        'drugs': ['Bisindolylmaleimide-I', 'DG-172', 'DMSO_round2_batch2',
                   'JTE-607', 'LDN-193189', 'LY2090314', 'NSC95397', 'VX-11e'],
        'ref':   'DMSO_round2_batch2',
    },
}


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB → GB on Linux


def _parse_term(name: str, covariate_cols):
    """Parse a patsy coef name into (term_type, drug, target_gene, covariate)."""
    if name == 'Intercept':
        return ('intercept', None, None, None)
    if name in covariate_cols:
        return ('covariate', None, None, name)

    def _grab(piece: str, key: str):
        if key not in piece:
            return None
        return piece.rsplit('[T.', 1)[1].rstrip(']')

    if ':' in name:
        a, b = name.split(':', 1)
        d = _grab(a, 'C(drug') or _grab(b, 'C(drug')
        t = _grab(a, 'C(target_gene') or _grab(b, 'C(target_gene')
        return ('drug:target_gene', d, t, None)
    if 'C(drug' in name:
        return ('drug', _grab(name, 'C(drug'), None, None)
    if 'C(target_gene' in name:
        return ('target_gene', None, _grab(name, 'C(target_gene'), None)
    return ('unknown', None, None, None)


def fit_one_batch(df_b, batch_name, batch_ref, covariate_cols):
    n_cells = len(df_b)
    n_drugs = df_b['drug'].nunique()
    n_perts = df_b['target_gene'].nunique()

    cc_pair = df_b.groupby(['drug', 'target_gene'], observed=True).size().to_dict()
    cc_drug = df_b.groupby('drug', observed=True).size().to_dict()
    cc_pert = df_b.groupby('target_gene', observed=True).size().to_dict()

    cov_terms = ' + '.join(covariate_cols) if covariate_cols else ''
    formula = (
        f'score ~ C(drug, Treatment(reference="{batch_ref}"))'
        ' + C(target_gene, Treatment(reference="non-targeting"))'
        f' + C(drug, Treatment(reference="{batch_ref}"))'
        ':C(target_gene, Treatment(reference="non-targeting"))'
    )
    if cov_terms:
        formula += ' + ' + cov_terms

    print(f'    [{batch_name}] {n_cells:>8,} cells × {n_drugs} drugs × {n_perts} perts', flush=True)
    t_design = time.time()
    model = smf.ols(formula, data=df_b)
    t_design_done = time.time() - t_design
    print(f'    [{batch_name}] design built  ({t_design_done:.1f}s, n_params={model.df_model + 1:.0f})', flush=True)
    t_fit = time.time()
    res = model.fit()
    t_fit_done = time.time() - t_fit
    print(f'    [{batch_name}] OLS fit       ({t_fit_done:.1f}s, R²={res.rsquared:.4f})', flush=True)

    rows = []
    for name, beta in res.params.items():
        term_type, d, t, cv = _parse_term(name, covariate_cols)
        if term_type == 'drug:target_gene':
            n = int(cc_pair.get((d, t), 0))
        elif term_type == 'drug':
            n = int(cc_drug.get(d, 0))
        elif term_type == 'target_gene':
            n = int(cc_pert.get(t, 0))
        else:
            n = n_cells
        rows.append({
            'term_type':   term_type,
            'drug':        d,
            'target_gene': t,
            'covariate':   cv,
            'beta':        float(beta),
            'se':          float(res.bse[name]),
            'tvalue':      float(res.tvalues[name]),
            'pvalue':      float(res.pvalues[name]),
            'n_cells':     n,
        })
    return pd.DataFrame(rows), res.rsquared, t_design_done, t_fit_done


def fit_one_pathway(score_csv, out_dir, covariate_df, covariate_cols, batches, overwrite):
    pname = score_csv.stem.replace('score_', '')
    print(f'\n[pathway] {pname}', flush=True)
    t0 = time.time()

    if not overwrite and all((out_dir / f'{pname}__{b}.csv').exists() for b in batches):
        for b in batches:
            print(f'    [{b}] exists, skipping')
        print(f'  ↳ pathway done in {time.time()-t0:.2f}s  (skip-fast, no CSV read)', flush=True)
        return []

    df = pd.read_csv(score_csv,
                     dtype={'cell_barcode': str, 'drug': str,
                             'target_gene': str, 'score': np.float32})
    if len(df) != len(covariate_df):
        raise RuntimeError(f'row-count mismatch: score CSV has {len(df):,} '
                            f'but covariate frame has {len(covariate_df):,}')
    for c in covariate_cols:
        df[c] = covariate_df[c].values

    fit_results = []
    for batch_name in batches:
        spec = BATCH_SPEC[batch_name]
        out_fp = out_dir / f'{pname}__{batch_name}.csv'
        if out_fp.exists() and not overwrite:
            print(f'    [{batch_name}] exists, skipping')
            continue
        df_b = df[df['drug'].isin(spec['drugs'])].copy()
        df_b['drug'] = df_b['drug'].astype('category')
        df_b['target_gene'] = df_b['target_gene'].astype('category')
        out_df, rsq, td, tf = fit_one_batch(df_b, batch_name, spec['ref'], covariate_cols)
        tmp = out_fp.with_suffix('.csv.tmp')
        out_df.to_csv(tmp, index=False)
        os.replace(tmp, out_fp)
        print(f'    [{batch_name}] → {out_fp.name}  ({len(out_df)} terms)', flush=True)
        fit_results.append({'pathway': pname, 'batch': batch_name,
                              'n_terms': len(out_df), 'rsq': rsq,
                              't_design': td, 't_fit': tf})
        del df_b, out_df
        gc.collect()

    print(f'  ↳ pathway done in {time.time()-t0:.1f}s  peak RSS={_peak_rss_gb():.1f} GB', flush=True)
    return fit_results


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--group',       default='group_00')
    ap.add_argument('--n-pathways',  type=int, default=10)
    ap.add_argument('--pathways',    nargs='+', default=None)
    ap.add_argument('--batches',     nargs='+', default=['batch1', 'batch2'],
                     choices=list(BATCH_SPEC.keys()))
    ap.add_argument('--score-dir',   type=Path,
                     default=Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways'))
    ap.add_argument('--concat-dir',  type=Path,
                     default=Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways/group_concats'))
    ap.add_argument('--covariates',  nargs='*',
                     default=['pct_counts_mt', 'log1p_total_counts'],
                     help='Continuous .obs columns to add as additive covariates. '
                          'Pass empty list (--covariates) to disable.')
    ap.add_argument('--output-dir',  type=Path,
                     default=Path('/home/beraslan/Projects/ChemoGeneticScreens/PertPathwayDrugInteraction'))
    ap.add_argument('--exclude-pattern', type=str, default=DEFAULT_EXCLUDE_PATTERN,
                     help='Regex on pathway name to exclude (default: skip KEGG MEDICUS '
                          'PATHOGEN entries). Pass --exclude-pattern "" to disable.')
    ap.add_argument('--exclude-list', type=Path, default=DEFAULT_EXCLUDE_LIST,
                     help='Plain-text file of pathway names (one per line, # comments) '
                          'to exclude. Combined with --exclude-pattern. Pass NONE to '
                          'disable.')
    ap.add_argument('--overwrite',   action='store_true')
    args = ap.parse_args()
    if str(args.exclude_list).upper() == 'NONE':
        args.exclude_list = None

    group_score_dir = args.score_dir / f'pathway_scores_concat_{args.group}'
    if not group_score_dir.exists():
        raise SystemExit(f'score dir does not exist: {group_score_dir}')

    if args.pathways:
        score_csvs = [group_score_dir / f'score_{p}.csv' for p in args.pathways]
        missing = [c for c in score_csvs if not c.exists()]
        if missing:
            raise SystemExit(f'missing pathway CSVs: {missing}')
    else:
        score_csvs = sorted(group_score_dir.glob('score_*.csv'))

    excl_set = _load_exclude_list(args.exclude_list)
    if excl_set:
        n_before = len(score_csvs)
        score_csvs = [c for c in score_csvs
                       if c.stem.replace('score_', '') not in excl_set]
        print(f'Excluded {n_before - len(score_csvs)} pathways from list {args.exclude_list}'
              f' ({len(excl_set)} entries)')

    if args.exclude_pattern:
        pat = re.compile(args.exclude_pattern)
        n_before = len(score_csvs)
        score_csvs = [c for c in score_csvs
                       if not pat.search(c.stem.replace('score_', ''))]
        print(f'Excluded {n_before - len(score_csvs)} pathways matching {args.exclude_pattern!r}')

    if not args.pathways:
        score_csvs = score_csvs[:args.n_pathways]

    out_dir = args.output_dir / args.group
    out_dir.mkdir(parents=True, exist_ok=True)

    concat_fp = args.concat_dir / f'concat_{args.group}.h5ad'
    if not concat_fp.exists():
        raise SystemExit(f'concat h5ad missing: {concat_fp}')

    print(f'Group:      {args.group}')
    print(f'Score dir:  {group_score_dir}')
    print(f'Output dir: {out_dir}')
    print(f'Pathways:   {len(score_csvs)} to fit')
    print(f'Batches:    {args.batches}')
    print(f'Covariates: {args.covariates if args.covariates else "(none)"}')
    print(f'Loading covariates from {concat_fp}', flush=True)
    t0_obs = time.time()
    a_backed = ad.read_h5ad(concat_fp, backed='r')
    missing_cov = [c for c in args.covariates if c not in a_backed.obs.columns]
    if missing_cov:
        raise SystemExit(f'covariates missing from .obs: {missing_cov}\n'
                          f'available: {list(a_backed.obs.columns)}')
    covariate_df = a_backed.obs[args.covariates].copy() if args.covariates else pd.DataFrame()
    a_backed.file.close()
    print(f'  loaded {len(covariate_df):,} rows of covariates ({time.time()-t0_obs:.1f}s)')
    print()

    summary = []
    for csv in score_csvs:
        try:
            summary += fit_one_pathway(csv, out_dir, covariate_df,
                                        args.covariates, args.batches, args.overwrite)
        except Exception as e:
            print(f'  ERROR on {csv.stem}: {type(e).__name__}: {e}')
        gc.collect()

    if summary:
        sdf = pd.DataFrame(summary)
        print(f'\n=== Summary ({len(sdf)} pathway×batch fits) ===')
        for col in ['t_design', 't_fit']:
            print(f'  {col:<10s}  median={sdf[col].median():.1f}s  '
                  f'mean={sdf[col].mean():.1f}s  max={sdf[col].max():.1f}s')
        for b in sdf['batch'].unique():
            sub = sdf[sdf['batch'] == b]
            print(f'  {b}:  n={len(sub)}  R² range=[{sub["rsq"].min():.4f}..{sub["rsq"].max():.4f}]  '
                  f'rows/csv={int(sub["n_terms"].iloc[0])}')
        print(f'  total elapsed:  {(sdf["t_design"].sum() + sdf["t_fit"].sum())/60:.1f} min')
        print(f'  peak RSS:       {_peak_rss_gb():.1f} GB')


if __name__ == '__main__':
    main()
