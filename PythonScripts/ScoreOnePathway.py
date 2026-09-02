#!/usr/bin/env python3
"""
Single-pathway timing test for the parallelizable per-pathway scoring
strategy.

Loads + concatenates the 16 per-drug caches, computes per-gene means and
25-quantile bin assignments (one-time work), then scores ONE pathway and
writes its score to disk as `cell_barcode,score` CSV. Prints per-stage
timings so we can decide how to parallelize and project total runtime.
"""

from __future__ import annotations

import argparse
import gc
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# anndata >= 0.11 errors on writing pd.StringDtype-backed categoricals unless opted-in
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


def load_batch_drugs(sample_sheet: Path, batch: str) -> List[str]:
    """Read SampleSheet.csv and return drug names whose ExperimentRound == batch.
    Uses csv module (not pandas) to tolerate the BOM and trailing-comma row."""
    import csv as _csv
    out: List[str] = []
    with open(sample_sheet, encoding='utf-8-sig', newline='') as f:
        reader = _csv.reader(f)
        next(reader, None)  # skip header
        for row in reader:
            if len(row) < 2:
                continue
            sample, expt = row[0].strip(), row[1].strip()
            if expt == batch and sample:
                out.append(sample)
    return out


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


def load_reduced_axes(h5_path: Path) -> Tuple[List[str], List[str], List[str]]:
    with h5py.File(h5_path, 'r') as h:
        drugs    = [s.decode() for s in h['drugs'][:]]
        perts    = [s.decode() for s in h['perturbations'][:]]
        pathways = [s.decode() for s in h['pathways'][:]]
    return drugs, perts, pathways


def load_drug_cache(drug: str, cache_dir: Path) -> ad.AnnData:
    fp = cache_dir / f'{drug}.h5ad'
    if not fp.exists():
        raise FileNotFoundError(f'Cache missing: {fp}')
    a = ad.read_h5ad(fp)
    a.obs['drug'] = drug
    return a


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project',       type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--cache-dir',     type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument('--out-dir',       type=Path, default=None,
                    help='Per-pathway CSV output dir. Default: <cache-dir>/pathway_scores[_<batch>]')
    ap.add_argument('--reduced-h5',    type=Path, default=None)
    ap.add_argument('--gmt-paths',     type=Path, nargs='+', default=DEFAULT_GMT_PATHS)
    ap.add_argument('--pathway-index', type=int,  default=0,
                    help='Index into reduced-h5 pathways[] (default 0 = first)')
    ap.add_argument('--n-bins',        type=int,  default=25)
    ap.add_argument('--ctrl-size',     type=int,  default=50)
    ap.add_argument('--seed',          type=int,  default=0)
    ap.add_argument('--concat-cache',  type=Path, default=None,
                    help='Path to concatenated AnnData. Read if exists, else built+saved.')
    ap.add_argument('--rebuild-concat', action='store_true',
                    help='Force rebuild of concat cache even if it exists')
    ap.add_argument('--batch',         type=str, default=None,
                    help='If set, restrict to drugs in this ExperimentRound '
                         '(e.g. "Round2_batch1"). Per-batch concat + scoring.')
    ap.add_argument('--sample-sheet',  type=Path, default=DEFAULT_SAMPLE_SHEET)
    ap.add_argument('--stop-after-concat', action='store_true',
                    help='Build + save concat cache and exit (skip scoring + CSV).')
    ap.add_argument('--method', choices=['fast', 'scanpy'], default='fast',
                    help="Scoring method. 'fast' = vectorised X@P, X@C matvecs "
                         "(default). 'scanpy' = sc.tl.score_genes (for timing comparison).")
    args = ap.parse_args()

    args.reduced_h5 = args.reduced_h5 or (
        args.project / 'PathwayORA' / 'KEGG' / 'KEGG_ORA_tensor_reduced.h5'
    )

    suffix = f'_{args.batch}' if args.batch else ''
    args.concat_cache = args.concat_cache or (
        args.cache_dir / f'all_drugs_concat{suffix}.h5ad'
    )
    args.out_dir = args.out_dir or (
        args.cache_dir / f'pathway_scores{suffix}'
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    drugs, _perts, pathways = load_reduced_axes(args.reduced_h5)
    if args.batch:
        batch_drugs = load_batch_drugs(args.sample_sheet, args.batch)
        keep = [d for d in drugs if d in batch_drugs]
        if not keep:
            raise RuntimeError(
                f'No reduced-tensor drugs match batch {args.batch!r}. '
                f'reduced drugs: {drugs}; batch drugs: {batch_drugs}'
            )
        print(f'Batch filter {args.batch!r}: {len(keep)} drugs of {len(drugs)} reduced-tensor drugs match')
        print(f'  drugs: {keep}')
        drugs = keep

    if not 0 <= args.pathway_index < len(pathways):
        raise ValueError(
            f'pathway-index {args.pathway_index} out of range [0, {len(pathways)})'
        )
    pname = pathways[args.pathway_index]
    print(f'Reduced tensor: {len(drugs)} drugs, {len(pathways)} pathways')
    print(f'Scoring pathway[{args.pathway_index}]: {pname}')

    kegg_all: Dict[str, set] = {}
    for gmt in args.gmt_paths:
        kegg_all.update(parse_gmt(gmt))
    if pname not in kegg_all:
        raise KeyError(f'Pathway {pname} not present in any GMT')
    pgenes = kegg_all[pname]
    print(f'Pathway has {len(pgenes)} member genes in GMT')

    # -----------------------------------------------------------------
    # Stage 0a — load + concat 16 caches (or read existing concat cache)
    # -----------------------------------------------------------------
    t_save = 0.0
    if args.concat_cache.exists() and not args.rebuild_concat:
        print(f'\n[0a] Reading concat cache: {args.concat_cache}')
        t_stage = time.time()
        big = ad.read_h5ad(args.concat_cache)
        print(f'  loaded: {big.n_obs:,} cells × {big.n_vars} genes '
              f'({time.time()-t_stage:.1f}s)')
        t_0a = time.time() - t_stage
    else:
        print(f'\n[0a] Loading + concatenating {len(drugs)} caches...')
        t_stage = time.time()
        parts = []
        for d in drugs:
            td = time.time()
            a = load_drug_cache(d, args.cache_dir)
            parts.append(a)
            print(f'  {d:28s} {a.n_obs:>9,d} cells ({time.time()-td:.1f}s)')
        big = ad.concat(parts, join='outer', merge='same')
        big.obs['drug']        = big.obs['drug'].astype(str).astype('category')
        big.obs['target_gene'] = big.obs['target_gene'].astype(str).astype('category')
        del parts
        gc.collect()
        print(f'  concat: {big.n_obs:,} cells × {big.n_vars} genes '
              f'({time.time()-t_stage:.1f}s)')
        t_0a = time.time() - t_stage

        # Concat auto-promotes X.indices to int64 because total nnz > 2.1B
        # exceeds int32 range. But indices only refer to columns (≤ 3,480 here),
        # which fits in int32 trivially. Indptr legitimately needs int64
        # (cumulative nnz), but downcasting indices saves ~57 GB per batch.
        if sp.issparse(big.X) and big.X.indices.dtype == np.int64:
            print(f'  downcasting X.indices  int64 → int32  '
                  f'(col dim={big.X.shape[1]} fits in int32)')
            big.X.indices = big.X.indices.astype(np.int32)

        # One-time write of unscored concat artifact for all future runs.
        # Uncompressed: I/O-bound (~1-3 min), much faster than gzip (~30 min).
        print(f'\n[0a-save] Writing concat artifact (uncompressed) → {args.concat_cache}')
        ts = time.time()
        big.write_h5ad(args.concat_cache)
        sz = args.concat_cache.stat().st_size / 1e9
        t_save = time.time() - ts
        print(f'  wrote {sz:.1f} GB ({t_save:.1f}s)')

        if args.stop_after_concat:
            print('\n--stop-after-concat: skipping scoring/CSV. Done.')
            return

    # -----------------------------------------------------------------
    # Scoring branch — fast (vectorised) vs scanpy (sc.tl.score_genes)
    # -----------------------------------------------------------------
    if args.method == 'scanpy':
        print(f'\n[1] sc.tl.score_genes  ({len(pgenes)} member genes)')
        members_present = [g for g in pgenes if g in big.var_names]
        if len(members_present) < 3:
            raise RuntimeError(
                f'Pathway {pname} has < 3 members in cache; cannot score'
            )
        print(f'  members present: {len(members_present)} / {len(pgenes)}')
        t_stage = time.time()
        sc.tl.score_genes(
            big,
            gene_list=members_present,
            score_name='_score',
            n_bins=args.n_bins,
            ctrl_size=args.ctrl_size,
            random_state=args.seed,
            use_raw=False,
        )
        score = big.obs['_score'].values.astype(np.float32)
        t_scanpy = time.time() - t_stage
        print(f'  scored {big.n_obs:,} cells in {t_scanpy:.1f}s')
        t_0b = t_1a = 0.0
        t_1b = t_scanpy

        # Skip the fast-method stages 0b/1a/1b, jump to CSV write
        out_path = args.out_dir / f'score_{pname}_scanpy.csv'
        print(f'\n[1c] Writing CSV → {out_path}')
        t_stage = time.time()
        df = pd.DataFrame({
            'cell_barcode': big.obs_names.astype(str),
            'score':        score,
        })
        df.to_csv(out_path, index=False)
        sz = out_path.stat().st_size / 1e9
        print(f'  wrote {sz:.2f} GB ({time.time()-t_stage:.1f}s)')
        t_1c = time.time() - t_stage

        one_time = t_0a + t_save
        per_path = t_1b + t_1c
        total    = one_time + per_path
        print(f'\n=== Timing summary (scanpy) ===')
        print(f'  0a load+concat (or read cache): {t_0a:>8.1f}s')
        if t_save > 0:
            print(f'  0a-save write concat cache    : {t_save:>8.1f}s  (one-time)')
        print(f'  1  sc.tl.score_genes           : {t_1b:>8.1f}s')
        print(f'  1c write CSV                  : {t_1c:>8.1f}s')
        print(f'  ----------------------------------')
        print(f'  Total this run                : {total:>8.1f}s '
              f'({total/60:.1f} min)')
        return

    # -----------------------------------------------------------------
    # Stage 0b — gene means + 25-bin assignments
    # -----------------------------------------------------------------
    print(f'\n[0b] Per-gene means + {args.n_bins}-quantile bin assignment...')
    t_stage = time.time()
    X = big.X
    if not sp.issparse(X):
        raise RuntimeError('Expected sparse X')
    if not sp.isspmatrix_csr(X):
        X = X.tocsr()
    gene_means = np.asarray(X.mean(axis=0)).flatten()
    bin_edges  = np.quantile(gene_means, np.linspace(0, 1, args.n_bins + 1))
    gene_bins  = np.digitize(gene_means, bin_edges[1:-1])
    print(f'  ({time.time()-t_stage:.1f}s)')
    t_0b = time.time() - t_stage

    # -----------------------------------------------------------------
    # Stage 1a — build P[:,0] and C[:,0]
    # -----------------------------------------------------------------
    print(f'\n[1a] Building member/control weight columns...')
    t_stage = time.time()
    var_names = list(big.var_names)
    gene_idx  = {g: i for i, g in enumerate(var_names)}
    members   = [gene_idx[g] for g in pgenes if g in gene_idx]
    print(f'  members present in cache: {len(members)} / {len(pgenes)}')
    if len(members) < 3:
        raise RuntimeError(f'Pathway {pname} has < 3 members in cache; cannot score')

    rng = np.random.default_rng(args.seed)
    p_col = np.zeros(big.n_vars, dtype=np.float32)
    p_col[members] = 1.0 / len(members)

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
    n_ctrl = len(ctrl_indices)
    if n_ctrl == 0:
        raise RuntimeError(f'Pathway {pname} has no valid controls')
    c_col = np.zeros(big.n_vars, dtype=np.float32)
    c_col[ctrl_indices] = 1.0 / n_ctrl
    print(f'  {n_ctrl} control genes ({time.time()-t_stage:.1f}s)')
    t_1a = time.time() - t_stage

    # -----------------------------------------------------------------
    # Stage 1b — score = X @ p_col − X @ c_col
    # -----------------------------------------------------------------
    print(f'\n[1b] Sparse@dense matvec (X @ p_col, X @ c_col)...')
    t_stage = time.time()
    pathway_mean = X @ p_col
    ctrl_mean    = X @ c_col
    score        = (pathway_mean - ctrl_mean).astype(np.float32)
    print(f'  score shape: {score.shape}, dtype: {score.dtype} '
          f'({time.time()-t_stage:.1f}s)')
    t_1b = time.time() - t_stage

    # -----------------------------------------------------------------
    # Stage 1c — write CSV
    # -----------------------------------------------------------------
    out_path = args.out_dir / f'score_{pname}.csv'
    print(f'\n[1c] Writing CSV → {out_path}')
    t_stage = time.time()
    df = pd.DataFrame({'cell_barcode': big.obs_names.astype(str), 'score': score})
    df.to_csv(out_path, index=False)
    sz = out_path.stat().st_size / 1e9
    print(f'  wrote {sz:.2f} GB ({time.time()-t_stage:.1f}s)')
    t_1c = time.time() - t_stage

    # -----------------------------------------------------------------
    # Timing summary + projection for parallel run across 211 pathways
    # -----------------------------------------------------------------
    one_time = t_0a + t_0b
    per_path = t_1a + t_1b + t_1c
    total    = one_time + per_path
    print(f'\n=== Timing summary ===')
    print(f'  0a load+concat (or read cache): {t_0a:>8.1f}s')
    if t_save > 0:
        print(f'  0a-save write concat cache    : {t_save:>8.1f}s  (one-time)')
    print(f'  0b gene means + bins          : {t_0b:>8.1f}s')
    print(f'  1a build P/C columns          : {t_1a:>8.1f}s')
    print(f'  1b X@P, X@C, diff             : {t_1b:>8.1f}s')
    print(f'  1c write CSV                  : {t_1c:>8.1f}s')
    print(f'  ----------------------------------')
    print(f'  One-time setup            : {one_time:>8.1f}s '
          f'({one_time/60:.1f} min)')
    print(f'  Per-pathway compute       : {per_path:>8.1f}s')
    print(f'  Total (this run, 1 pw)    : {total:>8.1f}s')
    print(f'\n  Serial projection (all 211): '
          f'{(one_time + 211*per_path)/60:.1f} min')
    print(f'  Parallel projections (per-pathway across N workers):')
    for n in (4, 8, 16, 32, 64):
        proj = one_time + 211 * per_path / n
        print(f'    N={n:>3}:  {proj:>7.0f}s = {proj/60:.1f} min')


if __name__ == '__main__':
    main()
