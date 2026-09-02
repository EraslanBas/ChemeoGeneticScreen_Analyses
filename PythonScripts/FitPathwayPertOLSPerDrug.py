#!/usr/bin/env python3
"""
Fit pathway_score ~ C(target_gene) + technical covariates within ONE drug
context. Used to estimate per-perturbation effects separately in each DMSO
batch (DMSO_round2 vs DMSO_round2_batch2) for concordance assessment.

For each (group, drug, pathway):
  - Read the cross-drug concat score CSV
  - Filter rows to cells from `drug`
  - Fit OLS:  score ~ C(target_gene, ref='non-targeting') + <covariates>
  - Save β, SE, t, p per pert

Output: <output_dir>/<group>/<pathway>__<drug>.csv
columns: term_type, target_gene, covariate, beta, se, tvalue, pvalue, n_cells
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

warnings.filterwarnings('ignore')

DEFAULT_DRUGS = ['DMSO_round2', 'DMSO_round2_batch2']
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


def _peak_rss_gb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def _parse_term(name, covariate_cols):
    if name == 'Intercept':
        return ('intercept', None, None)
    if name in covariate_cols:
        return ('covariate', None, name)
    if 'C(target_gene' in name:
        return ('target_gene', name.rsplit('[T.', 1)[1].rstrip(']'), None)
    return ('unknown', None, None)


def fit_one(score_csv, drug, out_fp, covariate_df, covariate_cols):
    df = pd.read_csv(score_csv,
                     dtype={'cell_barcode': str, 'drug': str,
                             'target_gene': str, 'score': np.float32})
    if len(df) != len(covariate_df):
        raise RuntimeError(f'row-count mismatch: csv={len(df)} cov={len(covariate_df)}')
    for c in covariate_cols:
        df[c] = covariate_df[c].values

    mask = (df['drug'] == drug).values
    if not mask.any():
        return None
    df_d = df.loc[mask].copy()
    df_d['target_gene'] = df_d['target_gene'].astype('category')

    cc_pert = df_d.groupby('target_gene', observed=True).size().to_dict()

    formula = 'score ~ C(target_gene, Treatment(reference="non-targeting"))'
    if covariate_cols:
        formula += ' + ' + ' + '.join(covariate_cols)

    model = smf.ols(formula, data=df_d)
    res = model.fit()

    rows = []
    for name, beta in res.params.items():
        term_type, t, cv = _parse_term(name, covariate_cols)
        n = int(cc_pert.get(t, 0)) if term_type == 'target_gene' else len(df_d)
        rows.append({
            'term_type':   term_type,
            'target_gene': t,
            'covariate':   cv,
            'beta':        float(beta),
            'se':          float(res.bse[name]),
            'tvalue':      float(res.tvalues[name]),
            'pvalue':      float(res.pvalues[name]),
            'n_cells':     n,
        })
    out_df = pd.DataFrame(rows)
    tmp = out_fp.with_suffix('.csv.tmp')
    out_df.to_csv(tmp, index=False)
    os.replace(tmp, out_fp)
    return res.rsquared, len(df_d)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--groups', nargs='+', default=None,
                     help='Subset of group names. Default: all available.')
    ap.add_argument('--drugs',  nargs='+', default=DEFAULT_DRUGS)
    ap.add_argument('--score-dir',  type=Path,
                     default=Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways'))
    ap.add_argument('--concat-dir', type=Path,
                     default=Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways/group_concats'))
    ap.add_argument('--covariates', nargs='*',
                     default=['pct_counts_mt', 'log1p_total_counts'])
    ap.add_argument('--output-dir', type=Path,
                     default=Path('/home/beraslan/Projects/ChemoGeneticScreens/PertPathwayDMSOOnly'))
    ap.add_argument('--exclude-pattern', type=str, default=DEFAULT_EXCLUDE_PATTERN,
                     help='Regex on pathway name to exclude (default: skip KEGG MEDICUS '
                          'PATHOGEN entries). Pass --exclude-pattern "" to disable.')
    ap.add_argument('--exclude-list', type=Path, default=DEFAULT_EXCLUDE_LIST,
                     help='Plain-text file of pathway names (one per line, # comments) '
                          'to exclude. Combined with --exclude-pattern. Pass NONE to '
                          'disable.')
    ap.add_argument('--overwrite',  action='store_true')
    args = ap.parse_args()
    if str(args.exclude_list).upper() == 'NONE':
        args.exclude_list = None

    pat = re.compile(args.exclude_pattern) if args.exclude_pattern else None
    excl_set = _load_exclude_list(args.exclude_list)
    if excl_set:
        print(f'Loaded {len(excl_set)} exclude entries from {args.exclude_list}')

    if args.groups is None:
        args.groups = sorted([d.name.replace('pathway_scores_concat_', '')
                               for d in args.score_dir.glob('pathway_scores_concat_*')
                               if d.is_dir()])

    print(f'Groups:     {args.groups}')
    print(f'Drugs:      {args.drugs}')
    print(f'Covariates: {args.covariates}')
    print()

    grand_t = time.time()
    for g in args.groups:
        print(f'\n=== {g} ===', flush=True)
        group_score_dir = args.score_dir / f'pathway_scores_concat_{g}'
        if not group_score_dir.exists():
            print(f'  MISSING score dir: {group_score_dir}')
            continue
        out_dir = args.output_dir / g
        out_dir.mkdir(parents=True, exist_ok=True)

        concat_fp = args.concat_dir / f'concat_{g}.h5ad'
        if not concat_fp.exists():
            print(f'  MISSING concat: {concat_fp}')
            continue

        t_obs = time.time()
        a_backed = ad.read_h5ad(concat_fp, backed='r')
        missing_cov = [c for c in args.covariates if c not in a_backed.obs.columns]
        if missing_cov:
            print(f'  ERROR — covariates missing from .obs: {missing_cov}')
            a_backed.file.close()
            continue
        covariate_df = a_backed.obs[args.covariates].copy() if args.covariates else pd.DataFrame()
        a_backed.file.close()
        print(f'  loaded {len(covariate_df):,} covariate rows ({time.time()-t_obs:.1f}s)', flush=True)

        score_csvs = sorted(group_score_dir.glob('score_*.csv'))
        if excl_set:
            n_before = len(score_csvs)
            score_csvs = [c for c in score_csvs
                           if c.stem.replace('score_', '') not in excl_set]
            if n_before != len(score_csvs):
                print(f'  excluded {n_before - len(score_csvs)} pathways via list')
        if pat is not None:
            n_before = len(score_csvs)
            score_csvs = [c for c in score_csvs
                           if not pat.search(c.stem.replace('score_', ''))]
            if n_before != len(score_csvs):
                print(f'  excluded {n_before - len(score_csvs)} pathways via regex '
                      f'{args.exclude_pattern!r}')
        print(f'  {len(score_csvs)} pathways × {len(args.drugs)} drugs = {len(score_csvs)*len(args.drugs)} fits', flush=True)

        t0 = time.time()
        n_done = n_skipped = n_err = 0
        for pi, csv in enumerate(score_csvs, 1):
            pname = csv.stem.replace('score_', '')
            for drug in args.drugs:
                out_fp = out_dir / f'{pname}__{drug}.csv'
                if out_fp.exists() and not args.overwrite:
                    n_skipped += 1
                    continue
                try:
                    fit_one(csv, drug, out_fp, covariate_df, args.covariates)
                    n_done += 1
                except Exception as e:
                    n_err += 1
                    print(f'  ERROR {pname} / {drug}: {type(e).__name__}: {e}')
            if pi % 50 == 0 or pi == len(score_csvs):
                el = (time.time() - t0) / 60
                pace = el / max(pi, 1)
                eta = pace * (len(score_csvs) - pi)
                print(f'    [{pi:>4}/{len(score_csvs)}]  done={n_done} skipped={n_skipped} err={n_err}'
                      f'  elapsed={el:>5.1f}m  pace={pace:.2f}m/pw  ETA={eta:.0f}m'
                      f'  RSS={_peak_rss_gb():.1f}GB', flush=True)
        print(f'  → {g} finished in {(time.time()-t0)/60:.1f}m', flush=True)
        gc.collect()

    print(f'\n=== ALL GROUPS DONE in {(time.time()-grand_t)/60:.1f}m  '
          f'peak RSS={_peak_rss_gb():.1f}GB ===')


if __name__ == '__main__':
    main()
