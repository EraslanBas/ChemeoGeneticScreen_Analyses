#!/usr/bin/env python3
"""
Batched pathway scoring for one batch — one big sparse@dense matmul.

Equivalent math to scanpy's sc.tl.score_genes (bin-matched module score),
implemented as TWO sparse@dense matrix multiplies:

    P : (n_genes, n_pathways) member-weight matrix       (cols sum to 1 over members)
    C : (n_genes, n_pathways) bin-matched control matrix (cols sum to 1 over controls)

    score_matrix = X @ P  −  X @ C            shape (n_cells, n_pathways)

Each X@P / X@C is a single sparse_csr @ dense_matrix call, much more cache-
efficient than 100 sequential matvecs (which the per-pathway parallel scorer
hit memory-bandwidth contention on). Output: one CSV per pathway.
"""

from __future__ import annotations

import argparse
import gc
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


DEFAULT_PROJECT   = Path('/home/beraslan/Projects/ChemoGeneticScreens')
DEFAULT_CACHE_DIR = Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways')
DEFAULT_GMT_PATHS = [
    Path('/home/beraslan/Projects/ModuleFinder/MuVI/msigdb/c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt'),
    Path('/home/beraslan/Projects/ModuleFinder/MuVI/msigdb/c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt'),
]


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
        T = h['neg_log10_q'][:]
        pathways = [s.decode() for s in h['pathways'][:]]
    pert_path_var = np.nanvar(T, axis=0)
    context_score = np.nanmean(pert_path_var, axis=0)
    order = np.argsort(context_score)[::-1]
    return [pathways[i] for i in order[:n_top]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--batch',        required=True,
                    choices=['Round2_batch1', 'Round2_batch2'])
    ap.add_argument('--n-pathways',   type=int, default=100)
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
    print(f'Pathways:     top-{args.n_pathways} by cross-drug variance')

    # ---------------------------------------------------------------
    # 0. Pick top-N pathways
    # ---------------------------------------------------------------
    print(f'\n[0] Selecting top-{args.n_pathways} pathways...')
    t0 = time.time()
    top_pathways = compute_top_context_pathways(args.reduced_h5, args.n_pathways)
    kegg: Dict[str, set] = {}
    for p in args.gmt_paths:
        if p.exists():
            kegg.update(parse_gmt(p))
    pathway_genes = {p: kegg[p] for p in top_pathways if p in kegg}
    missing = set(top_pathways) - set(pathway_genes)
    if missing:
        print(f'  WARN: {len(missing)} pathways not in any GMT, skipping')
    print(f'  picked {len(pathway_genes)} ({time.time()-t0:.1f}s)')

    # ---------------------------------------------------------------
    # 1. Load concat
    # ---------------------------------------------------------------
    print(f'\n[1] Reading concat cache...')
    t0 = time.time()
    big = ad.read_h5ad(args.concat_cache)
    print(f'  loaded: {big.n_obs:,} cells × {big.n_vars} genes ({time.time()-t0:.1f}s)')

    # ---------------------------------------------------------------
    # 2. Per-gene means + bins
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

    var_names = list(big.var_names)
    gene_idx  = {g: i for i, g in enumerate(var_names)}
    obs_names = np.asarray(big.obs_names, dtype=str)
    n_vars    = big.n_vars
    n_cells   = big.n_obs

    # ---------------------------------------------------------------
    # 3. Build P and C as (n_genes, n_pathways) dense matrices
    # ---------------------------------------------------------------
    print(f'\n[3] Building P and C weight matrices...')
    t0 = time.time()
    pnames = [p for p in pathway_genes if len(set(pathway_genes[p]) & set(gene_idx)) >= 3]
    print(f'  {len(pnames)}/{len(pathway_genes)} pathways have ≥3 members in cache')

    rng = np.random.default_rng(args.seed)
    P = np.zeros((n_vars, len(pnames)), dtype=np.float32)
    C = np.zeros((n_vars, len(pnames)), dtype=np.float32)
    valid = np.zeros(len(pnames), dtype=bool)

    for k, pname in enumerate(pnames):
        members = [gene_idx[g] for g in pathway_genes[pname] if g in gene_idx]
        P[members, k] = 1.0 / len(members)

        member_bins = gene_bins[members]
        members_set = set(members)
        ctrl_indices: List[int] = []
        for b in np.unique(member_bins):
            n_in_bin = int((member_bins == b).sum())
            bin_pool = np.array(
                [g for g in np.where(gene_bins == b)[0] if g not in members_set]
            )
            n_sample = min(args.ctrl_size * n_in_bin, len(bin_pool))
            if n_sample > 0:
                ctrl_indices.extend(rng.choice(bin_pool, n_sample, replace=False))
        if ctrl_indices:
            C[ctrl_indices, k] = 1.0 / len(ctrl_indices)
            valid[k] = True
    print(f'  P/C: ({n_vars}, {len(pnames)}) each   '
          f'({time.time()-t0:.1f}s)   {valid.sum()} valid pathways')

    del big
    gc.collect()

    # ---------------------------------------------------------------
    # 4. Two big sparse@dense matmuls
    # ---------------------------------------------------------------
    print(f'\n[4] X @ P  ({n_cells:,} × {n_vars} sparse @ {n_vars} × {len(pnames)} dense)...')
    t0 = time.time()
    pathway_means = (X @ P).astype(np.float32)        # (n_cells, n_pathways)
    print(f'  pathway_means: {pathway_means.shape}  ({time.time()-t0:.1f}s)')

    print(f'\n[5] X @ C ...')
    t0 = time.time()
    ctrl_means = (X @ C).astype(np.float32)
    print(f'  ctrl_means:    {ctrl_means.shape}     ({time.time()-t0:.1f}s)')

    print(f'\n[6] score = pathway_means − ctrl_means')
    score_matrix = pathway_means - ctrl_means
    score_matrix[:, ~valid] = np.nan
    del pathway_means, ctrl_means, P, C
    gc.collect()

    # ---------------------------------------------------------------
    # 7. Write per-pathway CSVs
    # ---------------------------------------------------------------
    print(f'\n[7] Writing {valid.sum()} per-pathway CSVs...')
    t0 = time.time()
    timings = []
    for k, pname in enumerate(pnames):
        if not valid[k]:
            timings.append((pname, 0.0, 'skip:invalid'))
            continue
        ts = time.time()
        out_path = args.out_dir / f'score_{pname}.csv'
        pd.DataFrame({
            'cell_barcode': obs_names,
            'score':        score_matrix[:, k],
        }).to_csv(out_path, index=False)
        timings.append((pname, time.time() - ts, 'OK'))
        if (k + 1) % 10 == 0:
            print(f'    [{k+1:>3}/{len(pnames)}] '
                  f'last write {timings[-1][1]:.1f}s  cum={time.time()-t0:.0f}s')
    print(f'  total CSV write: {time.time()-t0:.1f}s')

    pd.DataFrame(timings, columns=['pathway', 't_write_s', 'status']).to_csv(
        args.out_dir / '_run_timings.csv', index=False
    )
    print(f'\n[done] {args.batch}: {valid.sum()} pathways scored + saved.')


if __name__ == '__main__':
    main()
