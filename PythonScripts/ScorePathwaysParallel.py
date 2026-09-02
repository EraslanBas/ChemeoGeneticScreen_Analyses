#!/usr/bin/env python3
"""
Parallel pathway scoring for one batch.

For the top-N most context-dependent pathways (ranked by mean-over-perts of
cross-drug variance in the reduced KEGG ORA tensor), compute per-cell
module scores and write one CSV per pathway.

Score = mean(member-gene expr) − mean(bin-matched-control expr), the same
quantity as scanpy's `sc.tl.score_genes` but vectorised: each pathway is a
pair of sparse@dense matvecs (X @ p_col, X @ c_col).

Parallelism: fork-based mp.Pool. Each worker scores ONE pathway. The big
sparse X, var_names, gene-bin assignment, etc. are populated as module
globals in the parent process and inherited by workers via copy-on-write,
so only one copy of X lives in RAM regardless of worker count.
"""

from __future__ import annotations

# Limit BLAS threads BEFORE numpy import so N workers don't fight
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import argparse
import csv as _csv
import gc
import multiprocessing as mp
import time
import warnings
from pathlib import Path
from typing import Dict, List

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    ad.settings.allow_write_nullable_strings = True
except AttributeError:
    pass


DEFAULT_PROJECT      = Path('/home/beraslan/Projects/ChemoGeneticScreens')
DEFAULT_CACHE_DIR    = Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways')
DEFAULT_SAMPLE_SHEET = DEFAULT_PROJECT / 'TextFiles' / 'SampleSheet.csv'
DEFAULT_GMT_PATHS = [
    Path('/home/beraslan/Projects/ModuleFinder/MuVI/msigdb/c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt'),
    Path('/home/beraslan/Projects/ModuleFinder/MuVI/msigdb/c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt'),
]


# --- Shared via fork-COW (populated in main, read by workers) ---
_X            = None     # scipy.sparse.csr_matrix, shape (n_cells, n_genes)
_GENE_BINS    = None     # np.ndarray (n_genes,) int — quantile bin per gene
_GENE_IDX     = None     # dict gene_name -> col index
_OBS_NAMES    = None     # np.ndarray (n_cells,) str — cell barcodes
_PATHWAY_GENES = None    # dict pathway_name -> set of member gene names
_OUT_DIR      = None     # Path
_CTRL_SIZE    = None     # int
_SEED         = None     # int
_N_VARS       = None     # int


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


def compute_top_context_pathways(reduced_h5: Path, n_top: int) -> List[str]:
    """Rank pathways by mean-over-perts of cross-drug variance of -log10(q_bh)."""
    with h5py.File(reduced_h5, 'r') as h:
        T = h['neg_log10_q'][:]                     # (n_drugs, n_perts, n_pathways)
        pathways = [s.decode() for s in h['pathways'][:]]
    pert_path_var = np.nanvar(T, axis=0)            # (n_perts, n_pathways)
    context_score = np.nanmean(pert_path_var, axis=0)  # (n_pathways,)
    order = np.argsort(context_score)[::-1]
    return [pathways[i] for i in order[:n_top]]


def _score_one(pname: str):
    """Worker: score one pathway and write its CSV."""
    pgenes = _PATHWAY_GENES[pname]
    members = [_GENE_IDX[g] for g in pgenes if g in _GENE_IDX]
    if len(members) < 3:
        return pname, 0.0, 0, 'skip:<3 members'

    rng = np.random.default_rng(_SEED + (hash(pname) & 0x7FFFFFFF))

    p_col = np.zeros(_N_VARS, dtype=np.float32)
    p_col[members] = 1.0 / len(members)

    member_bins = _GENE_BINS[members]
    members_set = set(members)
    ctrl_indices: List[int] = []
    for b in np.unique(member_bins):
        n_in_bin = int((member_bins == b).sum())
        bin_pool = np.array(
            [g for g in np.where(_GENE_BINS == b)[0] if g not in members_set]
        )
        n_sample = min(_CTRL_SIZE * n_in_bin, len(bin_pool))
        if n_sample > 0:
            ctrl_indices.extend(rng.choice(bin_pool, n_sample, replace=False))
    n_ctrl = len(ctrl_indices)
    if n_ctrl == 0:
        return pname, 0.0, len(members), 'skip:no controls'

    c_col = np.zeros(_N_VARS, dtype=np.float32)
    c_col[ctrl_indices] = 1.0 / n_ctrl

    t0 = time.time()
    score = ((_X @ p_col) - (_X @ c_col)).astype(np.float32)
    t_compute = time.time() - t0

    out_path = _OUT_DIR / f'score_{pname}.csv'
    pd.DataFrame({
        'cell_barcode': _OBS_NAMES,
        'score':        score,
    }).to_csv(out_path, index=False)
    return pname, t_compute, len(members), 'OK'


def main():
    global _X, _GENE_BINS, _GENE_IDX, _OBS_NAMES
    global _PATHWAY_GENES, _OUT_DIR, _CTRL_SIZE, _SEED, _N_VARS

    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch',        required=True,
                    choices=['Round2_batch1', 'Round2_batch2'],
                    help='Which experimental batch to score')
    ap.add_argument('--n-pathways',   type=int, default=100,
                    help='Top-N most context-dependent pathways to score')
    ap.add_argument('--n-workers',    type=int, default=16)
    ap.add_argument('--n-bins',       type=int, default=25)
    ap.add_argument('--ctrl-size',    type=int, default=50)
    ap.add_argument('--seed',         type=int, default=0)
    ap.add_argument('--project',      type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--cache-dir',    type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument('--gmt-paths',    type=Path, nargs='+', default=DEFAULT_GMT_PATHS)
    ap.add_argument('--reduced-h5',   type=Path, default=None)
    ap.add_argument('--concat-cache', type=Path, default=None)
    ap.add_argument('--out-dir',      type=Path, default=None)
    args = ap.parse_args()

    args.reduced_h5   = args.reduced_h5   or (args.project / 'PathwayORA' / 'KEGG' / 'KEGG_ORA_tensor_reduced.h5')
    args.concat_cache = args.concat_cache or (args.cache_dir / f'all_drugs_concat_{args.batch}.h5ad')
    args.out_dir      = args.out_dir      or (args.cache_dir / f'pathway_scores_{args.batch}')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    print(f'Batch:        {args.batch}')
    print(f'Concat cache: {args.concat_cache}')
    print(f'Output dir:   {args.out_dir}')
    print(f'Workers:      {args.n_workers}')
    print(f'Pathways:     top-{args.n_pathways} by cross-drug variance')

    # ---------------------------------------------------------------
    # 0. Pick top-N context-dependent pathways
    # ---------------------------------------------------------------
    print(f'\n[0] Selecting top-{args.n_pathways} pathways from {args.reduced_h5}...')
    t0 = time.time()
    top_pathways = compute_top_context_pathways(args.reduced_h5, args.n_pathways)
    print(f'  picked {len(top_pathways)} ({time.time()-t0:.1f}s)')

    # Load GMTs and keep only the gene sets we need
    kegg: Dict[str, set] = {}
    for p in args.gmt_paths:
        if p.exists():
            kegg.update(parse_gmt(p))
    pathway_genes = {p: kegg[p] for p in top_pathways if p in kegg}
    missing = set(top_pathways) - set(pathway_genes)
    if missing:
        print(f'  WARN: {len(missing)} pathways not in any GMT, skipping: '
              f'{list(missing)[:3]}...')

    # ---------------------------------------------------------------
    # 1. Read concat cache
    # ---------------------------------------------------------------
    print(f'\n[1] Reading concat cache...')
    t0 = time.time()
    big = ad.read_h5ad(args.concat_cache)
    print(f'  loaded: {big.n_obs:,} cells × {big.n_vars} genes ({time.time()-t0:.1f}s)')

    # ---------------------------------------------------------------
    # 2. Per-gene means + quantile bins (one-time, parent process)
    # ---------------------------------------------------------------
    print(f'\n[2] Per-gene means + {args.n_bins}-bin assignment...')
    t0 = time.time()
    X = big.X
    if not sp.issparse(X):
        raise RuntimeError('Expected sparse X')
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()
    gene_means = np.asarray(X.mean(axis=0)).flatten()
    bin_edges  = np.quantile(gene_means, np.linspace(0, 1, args.n_bins + 1))
    gene_bins  = np.digitize(gene_means, bin_edges[1:-1])
    print(f'  ({time.time()-t0:.1f}s)')

    # ---------------------------------------------------------------
    # 3. Stage globals for fork-COW workers, free non-essentials
    # ---------------------------------------------------------------
    _X             = X
    _GENE_BINS     = gene_bins
    _GENE_IDX      = {g: i for i, g in enumerate(big.var_names)}
    _OBS_NAMES     = np.asarray(big.obs_names, dtype=str)
    _PATHWAY_GENES = pathway_genes
    _OUT_DIR       = args.out_dir
    _CTRL_SIZE     = args.ctrl_size
    _SEED          = args.seed
    _N_VARS        = big.n_vars

    del big
    gc.collect()
    print(f'  ready: {len(pathway_genes)} pathways, {len(_OBS_NAMES):,} cells')

    # ---------------------------------------------------------------
    # 4. Parallel scoring
    # ---------------------------------------------------------------
    print(f'\n[3] Scoring {len(pathway_genes)} pathways with {args.n_workers} workers...')
    t0 = time.time()
    pnames = list(pathway_genes.keys())
    results = []
    if args.n_workers > 1:
        with mp.Pool(args.n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_score_one, pnames), 1):
                results.append(r)
                if i % 10 == 0 or i == len(pnames):
                    elapsed = time.time() - t0
                    print(f'    [{i:>3}/{len(pnames)}] '
                          f'last: {r[0][:60]:<60s} '
                          f't_compute={r[1]:>5.1f}s n_members={r[2]:>3d} {r[3]}  '
                          f'elapsed={elapsed:.0f}s')
    else:
        for p in pnames:
            r = _score_one(p)
            results.append(r)
            print(f'  {r[0]}  t={r[1]:.1f}s  {r[3]}')

    elapsed = time.time() - t0
    n_ok = sum(1 for r in results if r[3] == 'OK')
    print(f'\n[done] {n_ok}/{len(results)} pathways scored in '
          f'{elapsed:.0f}s ({elapsed/60:.1f} min)')

    # Save per-pathway timings
    timings_csv = args.out_dir / '_run_timings.csv'
    pd.DataFrame(
        results, columns=['pathway', 't_compute_s', 'n_members', 'status']
    ).to_csv(timings_csv, index=False)
    print(f'  timings → {timings_csv}')


if __name__ == '__main__':
    main()
