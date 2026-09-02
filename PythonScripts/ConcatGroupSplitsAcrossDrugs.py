#!/usr/bin/env python3
"""
Concatenate per-drug group splits into one cross-drug AnnData per group.

For each group g (e.g. group_00):
  Load all 16 <drug>_group_g.h5ad files
  ad.concat them along axis 0 (cells)
  Save concat_<group>.h5ad

The pre-existing `obs['drug']` in each split tags cells by drug context, so
the concat preserves drug identity automatically.

Outputs:
  /processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways/group_concats/
      concat_<group>.h5ad   (one per group)

Skip-existing: if the output already exists, the concat is skipped.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import time
import warnings
from pathlib import Path

import anndata as ad
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


CACHE_DIR = Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways')


def concat_one_group(args):
    gname, drugs, splits_dir, out_dir, skip_existing = args
    out_fp = out_dir / f'concat_{gname}.h5ad'
    if skip_existing and out_fp.exists():
        return (gname, f'  exists, skipped — {out_fp}  ({out_fp.stat().st_size/1e9:.2f} GB)')

    t0 = time.time()
    parts = []
    missing = []
    for d in drugs:
        fp = splits_dir / f'{d}_{gname}.h5ad'
        if not fp.exists():
            missing.append(fp)
            continue
        a = ad.read_h5ad(fp)
        a.obs['drug'] = a.obs.get('drug', d)
        parts.append(a)

    if missing:
        return (gname, f'  MISSING splits ({len(missing)}): e.g. {missing[:2]}')

    big = ad.concat(parts, join='outer', merge='same', axis=0)
    big.obs['drug']        = big.obs['drug'].astype('category')
    big.obs['target_gene'] = big.obs['target_gene'].astype(str).astype('category')
    big.uns['concat_group']  = gname
    big.uns['concat_drugs']  = list(drugs)

    n_pert = int((big.obs['target_gene'].astype(str) != 'non-targeting').sum())
    n_nt   = int((big.obs['target_gene'].astype(str) == 'non-targeting').sum())

    tmp = out_fp.with_suffix('.h5ad.tmp')
    big.write_h5ad(tmp, compression='gzip')
    tmp.rename(out_fp)

    msg = (f'  {gname}: concat {len(drugs)} drugs → {big.n_obs:,} cells × {big.n_vars} vars '
           f'(perts={n_pert:,}, NT={n_nt:,})  {out_fp.stat().st_size/1e9:.2f} GB  '
           f'({time.time()-t0:.1f}s)')

    del parts, big
    gc.collect()
    return (gname, msg)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir',  type=Path, default=CACHE_DIR)
    ap.add_argument('--groups-json',type=Path, default=CACHE_DIR / 'pert_groups.json')
    ap.add_argument('--splits-dir', type=Path, default=CACHE_DIR / 'group_splits')
    ap.add_argument('--out-dir',    type=Path, default=CACHE_DIR / 'group_concats')
    ap.add_argument('--groups',     nargs='+', default=None,
                    help='Subset of group names to concat (default: all).')
    ap.add_argument('--n-parallel', type=int, default=4)
    ap.add_argument('--overwrite',  action='store_true')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.groups_json.read_text())
    all_groups = list(spec['groups'].keys())
    drugs = spec['drugs']
    groups = args.groups if args.groups else all_groups
    print(f'Concatenating {len(groups)} group(s) across {len(drugs)} drugs')
    print(f'Splits dir: {args.splits_dir}')
    print(f'Output dir: {args.out_dir}')
    print(f'Parallelism: {args.n_parallel}')
    print()

    job_args = [(g, drugs, args.splits_dir, args.out_dir, not args.overwrite)
                for g in groups]

    t0 = time.time()
    with mp.get_context('fork').Pool(args.n_parallel) as pool:
        for gname, msg in pool.imap_unordered(concat_one_group, job_args):
            print(msg, flush=True)

    print(f'\nAll groups concatenated in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
