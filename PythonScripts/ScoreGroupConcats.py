#!/usr/bin/env python3
"""
Score 552 KEGG pathways with sc.tl.score_genes on each cross-drug group concat.

Concatenated h5ads (one per group) live at <group_concats_dir>/concat_<group>.h5ad
and contain cells from all 16 drug contexts pooled, restricted to the perts
in that group + all NT cells.

Groups are processed sequentially. Within each group, N pathways are scored
in parallel via mp.Pool. Each pathway's per-cell score is written
incrementally as a CSV with `cell_barcode,drug,target_gene,score` columns —
the extra `drug` and `target_gene` columns are useful for downstream OLS.

Outputs:
  /processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways/pathway_scores_concat_<group>/
      score_<pathway>.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import multiprocessing as mp
import os
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


CACHE_DIR = Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways')


# Module globals shared with forked workers via Linux fork-COW
_ADATA = None
_BARCODES = None
_DRUG_COL  = None
_PERT_COL  = None
_BG_POOL_SET = None
_OUT_DIR = None
_N_BINS = 25
_CTRL_SIZE = 50
_SEED = 0
_MIN_MEMBERS = 3


def parse_gmt(path: Path) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for line in open(path):
        parts = line.rstrip('\n').split('\t')
        if len(parts) < 3:
            continue
        name, _url, *genes = parts
        genes = {g for g in genes if g}
        if genes:
            out[name] = genes
    return out


def _score_one_pathway(args: Tuple[str, set]) -> Tuple[str, str]:
    pname, pgenes = args
    out_csv = _OUT_DIR / f'score_{pname}.csv'
    if out_csv.exists():
        return ('existed', pname)
    members = [g for g in _ADATA.var_names if g in pgenes]
    if len(members) < _MIN_MEMBERS:
        return ('too_small', pname)
    pool = list(_BG_POOL_SET | set(members))
    score_name = '_score'
    try:
        sc.tl.score_genes(
            _ADATA,
            gene_list=members,
            gene_pool=pool,
            n_bins=_N_BINS,
            ctrl_size=_CTRL_SIZE,
            score_name=score_name,
            random_state=_SEED,
            use_raw=False,
        )
        score = _ADATA.obs[score_name].values.astype(np.float32)
        del _ADATA.obs[score_name]
    except Exception as e:
        return ('error', pname, repr(e))

    tmp = out_csv.with_suffix('.csv.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cell_barcode', 'drug', 'target_gene', 'score'])
        w.writerows(zip(_BARCODES, _DRUG_COL, _PERT_COL, score))
    os.replace(str(tmp), str(out_csv))
    return ('ok', pname)


def score_one_group(group: str,
                     concat_dir: Path,
                     pathway_genes: Dict[str, set],
                     n_parallel: int,
                     n_bins: int,
                     ctrl_size: int,
                     seed: int,
                     min_members: int) -> None:
    concat_fp = concat_dir / f'concat_{group}.h5ad'
    if not concat_fp.exists():
        print(f'[{group}] MISSING concat file: {concat_fp}')
        return

    out_dir = CACHE_DIR / f'pathway_scores_concat_{group}'
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = [(p, g) for p, g in pathway_genes.items()
                if not (out_dir / f'score_{p}.csv').exists()]
    n_done = len(pathway_genes) - len(pending)
    print(f'\n=== {group} ===  ({len(pathway_genes)} pathways, '
          f'{n_done} already done, {len(pending)} to score)')
    if not pending:
        print(f'  all pathways already scored — skipping')
        return

    print(f'Loading {concat_fp}', flush=True)
    t0 = time.time()
    a = ad.read_h5ad(concat_fp)
    print(f'  {a.n_obs:,} cells × {a.n_vars} vars  ({time.time()-t0:.1f}s)', flush=True)

    if 'background_gene' not in a.var.columns:
        raise RuntimeError(f'{concat_fp}: var.background_gene flag missing')
    bg_pool = set(a.var_names[a.var['background_gene'].values].tolist())

    barcodes = a.obs_names.astype(str).values
    drug_col = a.obs['drug'].astype(str).values
    pert_col = a.obs['target_gene'].astype(str).values
    print(f'  bg pool = {len(bg_pool)} genes; pending pathways = {len(pending)}', flush=True)

    global _ADATA, _BARCODES, _DRUG_COL, _PERT_COL, _BG_POOL_SET, _OUT_DIR, _N_BINS, _CTRL_SIZE, _SEED, _MIN_MEMBERS
    _ADATA = a
    _BARCODES = barcodes
    _DRUG_COL = drug_col
    _PERT_COL = pert_col
    _BG_POOL_SET = bg_pool
    _OUT_DIR = out_dir
    _N_BINS = n_bins
    _CTRL_SIZE = ctrl_size
    _SEED = seed
    _MIN_MEMBERS = min_members

    n_ok = n_existed = n_small = n_err = 0
    err_log: List[Tuple[str, str]] = []
    t_total = time.time()

    ctx = mp.get_context('fork')
    with ctx.Pool(n_parallel, maxtasksperchild=64) as pool:
        for i, res in enumerate(pool.imap_unordered(_score_one_pathway, pending, chunksize=1), 1):
            status = res[0]
            if status == 'ok':       n_ok += 1
            elif status == 'existed':n_existed += 1
            elif status == 'too_small': n_small += 1
            elif status == 'error':
                n_err += 1; err_log.append((res[1], res[2]))
            if i % 25 == 0 or i == len(pending):
                el = time.time() - t_total
                print(f'  [{i:>4}/{len(pending)}] '
                      f'ok={n_ok} existed={n_existed} small={n_small} err={n_err}  '
                      f'elapsed={el/60:>5.1f}m  ({el/i:.1f}s/pw)', flush=True)

    del _ADATA; _ADATA = None
    del a
    gc.collect()

    print(f'\n  done {group}: scored={n_ok}  small={n_small}  existed={n_existed}  errors={n_err}  '
          f'time={(time.time()-t_total)/60:.1f}m')
    if err_log:
        for n, e in err_log[:5]:
            print(f'    err {n}: {e}')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir',   type=Path, default=CACHE_DIR)
    ap.add_argument('--concat-dir',  type=Path, default=CACHE_DIR / 'group_concats')
    ap.add_argument('--groups-json', type=Path, default=CACHE_DIR / 'pert_groups.json')
    ap.add_argument('--pathway-list', type=Path,
                    default=Path('/home/beraslan/Projects/ChemoGeneticScreens/PathwayORA/KEGG/'
                                  'pathway_representatives_all_kegg_jaccard60.txt'))
    ap.add_argument('--gmt-paths', type=Path, nargs='+', default=[
        Path('/home/beraslan/Projects/ModuleFinder/MuVI/msigdb/c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt'),
        Path('/home/beraslan/Projects/ModuleFinder/MuVI/msigdb/c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt'),
    ])
    ap.add_argument('--groups',     nargs='+', default=None)
    ap.add_argument('--n-parallel', type=int, default=20)
    ap.add_argument('--n-bins',     type=int, default=25)
    ap.add_argument('--ctrl-size',  type=int, default=50)
    ap.add_argument('--seed',       type=int, default=0)
    ap.add_argument('--min-members',type=int, default=3)
    args = ap.parse_args()

    import json
    spec = json.loads(args.groups_json.read_text())
    groups = args.groups if args.groups else list(spec['groups'].keys())

    wanted_pathways = [ln.strip() for ln in args.pathway_list.read_text().splitlines()
                        if ln.strip() and not ln.startswith('#')]
    kegg = {}
    for gmt in args.gmt_paths:
        kegg.update(parse_gmt(gmt))
    pathway_genes = {p: kegg[p] for p in wanted_pathways if p in kegg}
    print(f'Pathways to score: {len(pathway_genes)}')
    print(f'Groups (sequential): {groups}')
    print(f'Per-group parallelism: {args.n_parallel} pathways')

    t0 = time.time()
    for g in groups:
        score_one_group(g, args.concat_dir, pathway_genes,
                         args.n_parallel, args.n_bins, args.ctrl_size,
                         args.seed, args.min_members)

    print(f'\nAll groups done in {(time.time()-t0)/60:.1f} min')


if __name__ == '__main__':
    main()
