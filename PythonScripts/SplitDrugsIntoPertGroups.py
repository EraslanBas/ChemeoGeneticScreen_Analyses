#!/usr/bin/env python3
"""
Split each per-drug cache into per-(drug, group) sub-h5ads.

Reads the group definitions from `pert_groups.json` (produced earlier) and,
for each drug, subsets the cache to (group_g_perts ∪ {non-targeting}) cells
and writes one sub-h5ad per group.

Usage:
  python SplitDrugsIntoPertGroups.py            # process all drugs
  python SplitDrugsIntoPertGroups.py --drugs DMSO_round2 AZD4573
  python SplitDrugsIntoPertGroups.py --n-parallel 4    # default sequential

Outputs:
  /processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways/group_splits/
      <drug>_group<gg>.h5ad   (one per (drug, group) pair)

Skip-existing: if the output already exists, the split is skipped, so this
is safe to interrupt and re-run.
"""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import time
import warnings
from pathlib import Path
from typing import Dict, List

import anndata as ad
import numpy as np

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


CACHE_DIR = Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways')


def split_one_drug(args):
    drug, groups, out_dir, skip_existing = args
    cache_fp = CACHE_DIR / f'{drug}.h5ad'
    if not cache_fp.exists():
        return (drug, f'MISSING cache: {cache_fp}')

    out_paths = {gname: out_dir / f'{drug}_{gname}.h5ad' for gname in groups}

    # Skip-existing pre-check — if every group's split already exists, skip
    # the load entirely.
    if skip_existing and all(p.exists() for p in out_paths.values()):
        return (drug, f'all {len(groups)} group splits already exist — skipping load')

    t0 = time.time()
    a = ad.read_h5ad(cache_fp)
    t_load = time.time() - t0
    tg = a.obs['target_gene'].astype(str).values
    is_nt = tg == 'non-targeting'
    n_cells_total = a.n_obs

    msg_lines = [f'{drug}  {a.n_obs:,} cells × {a.n_vars} vars  loaded in {t_load:.1f}s']

    n_written = n_skipped = 0
    for gname, perts in groups.items():
        out_fp = out_paths[gname]
        if skip_existing and out_fp.exists():
            n_skipped += 1
            continue
        ts = time.time()
        keep_mask = np.isin(tg, list(perts)) | is_nt
        sub = a[keep_mask].copy()
        sub.uns['split_drug']        = drug
        sub.uns['split_group']       = gname
        sub.uns['split_n_perts_set'] = len(perts)
        # Atomic write
        tmp = out_fp.with_suffix('.h5ad.tmp')
        sub.write_h5ad(tmp, compression='gzip')
        tmp.rename(out_fp)
        n_written += 1
        n_pert = int((tg[keep_mask] != 'non-targeting').sum())
        n_nt   = int((tg[keep_mask] == 'non-targeting').sum())
        msg_lines.append(
            f'  {gname}: {sub.n_obs:>7,d} cells (perts={n_pert:>6,d}, NT={n_nt:>5,d})  '
            f'{out_fp.stat().st_size/1e6:>5.0f} MB  ({time.time()-ts:.1f}s)'
        )
        del sub
        gc.collect()

    del a
    gc.collect()
    msg_lines.append(f'  → wrote {n_written} new split(s), skipped {n_skipped} existing  '
                      f'(drug total {time.time()-t0:.1f}s)')
    return (drug, '\n'.join(msg_lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir', type=Path, default=CACHE_DIR)
    ap.add_argument('--groups-json', type=Path, default=CACHE_DIR / 'pert_groups.json')
    ap.add_argument('--out-dir', type=Path, default=CACHE_DIR / 'group_splits')
    ap.add_argument('--drugs', nargs='+', default=None,
                    help='Subset of drugs to process (default: all in groups_json).')
    ap.add_argument('--n-parallel', type=int, default=1,
                    help='Number of drugs to process in parallel (default 1 = sequential).')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = json.loads(args.groups_json.read_text())
    drugs = args.drugs if args.drugs else spec['drugs']
    groups = spec['groups']  # {group_name: [pert_list]}
    print(f'Loaded {len(drugs)} drugs × {len(groups)} groups from {args.groups_json}')
    print(f'Output dir: {args.out_dir}')
    print(f'Parallelism: {args.n_parallel} drug(s) at a time')
    print()

    job_args = [(d, groups, args.out_dir, not args.overwrite) for d in drugs]

    t0 = time.time()
    if args.n_parallel == 1:
        for ja in job_args:
            d, msg = split_one_drug(ja)
            print(msg, flush=True)
    else:
        with mp.get_context('fork').Pool(args.n_parallel) as pool:
            for d, msg in pool.imap_unordered(split_one_drug, job_args):
                print(msg, flush=True)

    print(f'\nAll drugs processed in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
