#!/usr/bin/env python3
"""
Score KEGG pathways with sc.tl.score_genes across all 16 drug caches.

- Drugs are processed SEQUENTIALLY (one cache loaded at a time → bounded memory).
- Within each drug, N pathways are scored IN PARALLEL via a multiprocessing.Pool.
- Each pathway's per-cell score is written as its own CSV atomically the moment
  it's computed → robust to crashes (skip-existing makes restart cheap).

Inputs
------
  /processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways/<drug>.h5ad
  /home/beraslan/Projects/ChemoGeneticScreens/PathwayORA/KEGG/
      pathway_representatives_all_kegg_jaccard60.txt

Outputs
-------
  /processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways/pathway_scores_<drug>/
      score_<pathway>.csv   (one per pathway, columns: cell_barcode, score)

Methodology
-----------
For each pathway P with member genes G_P (intersected with the cache var_names),
we set scanpy's `gene_pool = (background_genes from var.background_gene) ∪ G_P`,
then call `sc.tl.score_genes(adata, gene_list=G_P, gene_pool=pool, n_bins=25,
ctrl_size=50, score_name='_score', random_state=0, use_raw=False)`. The
gene_pool inclusion of G_P is required so scanpy can index member-gene bins;
scanpy auto-excludes G_P from the control sample, so controls always come
from the background pool only.

Pathways with fewer than `--min-members` (default 3) members in the cache are
skipped (no CSV written).
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
import h5py
import numpy as np
import pandas as pd
import scanpy as sc

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# Module globals — populated in each drug's load step, inherited by mp.Pool
# workers via Linux fork copy-on-write. Workers do NOT mutate them.
_ADATA: ad.AnnData = None
_BARCODES: np.ndarray = None
_BG_POOL_SET: set = None
_OUT_DIR: Path = None
_N_BINS: int = 25
_CTRL_SIZE: int = 50
_SEED: int = 0
_MIN_MEMBERS: int = 3


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def parse_gmt(path: Path) -> Dict[str, set]:
    out: Dict[str, set] = {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip('\n').split('\t')
            if len(parts) < 3:
                continue
            name, _url, *genes = parts
            genes = {g for g in genes if g}
            if genes:
                out[name] = genes
    return out


def load_reduced_drugs(reduced_h5: Path) -> List[str]:
    with h5py.File(reduced_h5, 'r') as h:
        return [s.decode() for s in h['drugs'][:]]


# --------------------------------------------------------------------------
# Worker — runs in forked child process
# --------------------------------------------------------------------------
def _score_one_pathway(args: Tuple[str, set]) -> Tuple[str, str]:
    """Score one pathway, write its CSV atomically, return status string."""
    pname, pgenes = args
    out_csv = _OUT_DIR / f'score_{pname}.csv'
    if out_csv.exists():
        return ('existed', pname)

    # Members in this cache
    members = [g for g in _ADATA.var_names if g in pgenes]
    if len(members) < _MIN_MEMBERS:
        return ('too_small', pname)

    pool = list(_BG_POOL_SET | set(members))

    # gene_pool restriction puts the bin-matched controls in the bg-only
    # subset; scanpy excludes pathway members from being chosen as controls.
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
        # Don't accumulate score columns across tasks in the same worker
        del _ADATA.obs[score_name]
    except Exception as e:
        return ('error', pname, repr(e))

    # Atomic write
    tmp = out_csv.with_suffix('.csv.tmp')
    with open(tmp, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cell_barcode', 'score'])
        w.writerows(zip(_BARCODES, score))
    os.replace(str(tmp), str(out_csv))

    return ('ok', pname)


# --------------------------------------------------------------------------
# Per-drug driver
# --------------------------------------------------------------------------
def score_one_drug(drug: str,
                    cache_dir: Path,
                    pathway_genes: Dict[str, set],
                    n_parallel: int,
                    n_bins: int,
                    ctrl_size: int,
                    seed: int,
                    min_members: int) -> None:
    cache_path = cache_dir / f'{drug}.h5ad'
    if not cache_path.exists():
        print(f'[{drug}] MISSING cache: {cache_path}')
        return
    out_dir = cache_dir / f'pathway_scores_{drug}'
    out_dir.mkdir(parents=True, exist_ok=True)

    # Skip-existing pre-check
    pending = [(p, g) for p, g in pathway_genes.items()
                if not (out_dir / f'score_{p}.csv').exists()]
    n_done = len(pathway_genes) - len(pending)
    print(f'\n=== {drug} ===  ({len(pathway_genes)} pathways, '
          f'{n_done} already done, {len(pending)} to score)')
    if not pending:
        print(f'  all pathways already scored — skipping')
        return

    print(f'Loading {cache_path}', flush=True)
    t0 = time.time()
    a = ad.read_h5ad(cache_path)
    print(f'  {a.n_obs:,} cells × {a.n_vars} vars  ({time.time()-t0:.1f}s)', flush=True)

    if 'background_gene' not in a.var.columns:
        raise RuntimeError(f'{cache_path}: var.background_gene flag missing — was this cache built with the new BuildDrugCaches?')
    bg_pool = set(a.var_names[a.var['background_gene'].values].tolist())
    barcodes = a.obs_names.astype(str).values
    print(f'  bg pool = {len(bg_pool)} genes; pending pathways = {len(pending)}', flush=True)

    # Stash for forked workers
    global _ADATA, _BARCODES, _BG_POOL_SET, _OUT_DIR, _N_BINS, _CTRL_SIZE, _SEED, _MIN_MEMBERS
    _ADATA = a
    _BARCODES = barcodes
    _BG_POOL_SET = bg_pool
    _OUT_DIR = out_dir
    _N_BINS = n_bins
    _CTRL_SIZE = ctrl_size
    _SEED = seed
    _MIN_MEMBERS = min_members

    n_ok = n_existed = n_small = n_err = 0
    t_total = time.time()
    err_log: List[Tuple[str, str]] = []

    # Use 'fork' start method so workers inherit _ADATA + bg_pool via COW
    ctx = mp.get_context('fork')
    with ctx.Pool(n_parallel, maxtasksperchild=64) as pool:
        for i, res in enumerate(pool.imap_unordered(_score_one_pathway, pending, chunksize=1), 1):
            status = res[0]
            if status == 'ok':
                n_ok += 1
            elif status == 'existed':
                n_existed += 1
            elif status == 'too_small':
                n_small += 1
            elif status == 'error':
                n_err += 1
                err_log.append((res[1], res[2]))
            if i % 25 == 0 or i == len(pending):
                elapsed = time.time() - t_total
                print(f'  [{i:>4}/{len(pending)}] '
                      f'ok={n_ok} existed={n_existed} small={n_small} err={n_err}  '
                      f'elapsed={elapsed/60:>5.1f}m  ({elapsed/i:.1f}s/pw)', flush=True)

    # Free the cache before moving on
    del _ADATA
    _ADATA = None
    del a
    gc.collect()

    print(f'\n  done {drug}: scored={n_ok}  small=<{min_members} members={n_small}  '
          f'existed={n_existed}  errors={n_err}  '
          f'time={(time.time()-t_total)/60:.1f}m')
    if err_log:
        print(f'  Errors:')
        for n, e in err_log[:10]:
            print(f'    {n}: {e}')
        if len(err_log) > 10:
            print(f'    ... and {len(err_log)-10} more')


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cache-dir', type=Path,
                    default=Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways'))
    ap.add_argument('--pathway-list', type=Path,
                    default=Path('/home/beraslan/Projects/ChemoGeneticScreens/PathwayORA/KEGG/pathway_representatives_all_kegg_jaccard60.txt'))
    ap.add_argument('--gmt-paths', type=Path, nargs='+', default=[
        Path('/home/beraslan/Projects/ModuleFinder/MuVI/msigdb/c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt'),
        Path('/home/beraslan/Projects/ModuleFinder/MuVI/msigdb/c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt'),
    ])
    ap.add_argument('--reduced-h5', type=Path,
                    default=Path('/home/beraslan/Projects/ChemoGeneticScreens/PathwayORA/KEGG/KEGG_ORA_tensor_reduced.h5'))
    ap.add_argument('--drugs', nargs='+', default=None,
                    help='Drug names to process (default: all 16 from reduced tensor).')
    ap.add_argument('--n-parallel', type=int, default=20,
                    help='Pathways scored in parallel within each drug (default 20).')
    ap.add_argument('--n-bins', type=int, default=25)
    ap.add_argument('--ctrl-size', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--min-members', type=int, default=3)
    args = ap.parse_args()

    # Load pathway names & gene sets
    print(f'Reading pathway list: {args.pathway_list}')
    wanted = [ln.strip() for ln in args.pathway_list.read_text().splitlines()
               if ln.strip() and not ln.startswith('#')]
    print(f'  {len(wanted)} pathways requested')

    kegg = {}
    for gmt in args.gmt_paths:
        kegg.update(parse_gmt(gmt))
    print(f'KEGG GMTs loaded: {len(kegg)} pathways total')

    pathway_genes = {p: kegg[p] for p in wanted if p in kegg}
    missing = set(wanted) - set(pathway_genes)
    if missing:
        print(f'  warning: {len(missing)} requested pathways not in GMTs (e.g. {sorted(missing)[:3]})')
    print(f'pathways to score: {len(pathway_genes)}')

    # Drug list
    drugs = args.drugs if args.drugs else load_reduced_drugs(args.reduced_h5)
    print(f'Drugs (sequential): {drugs}')
    print(f'Per-drug parallelism: {args.n_parallel} pathways simultaneously')

    t_grand = time.time()
    for drug in drugs:
        score_one_drug(drug, args.cache_dir, pathway_genes,
                        args.n_parallel, args.n_bins, args.ctrl_size,
                        args.seed, args.min_members)

    print(f'\nAll drugs done in {(time.time()-t_grand)/60:.1f} min')


if __name__ == '__main__':
    main()
