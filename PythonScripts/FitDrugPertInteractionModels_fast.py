#!/usr/bin/env python3
"""
Fast fallback for FitDrugPertInteractionModels.py.

Drop-in replacement that swaps the slow `sc.tl.score_genes` loop (which adds
211 columns one-by-one to a 16.5M-row pandas obs, ~8 hr) for a single pair
of sparse@dense matrix multiplies (~5 min) computing the same quantity.

Algorithm — bin-matched module score, identical recipe to scanpy's:
  1. mean expression per gene across cells
  2. quantile-bin genes by mean expression (default 25 bins)
  3. for each pathway, sample ctrl_size × n_members "control genes" from
     the same bins as members
  4. score(cell, pathway) = mean(member_expr) − mean(ctrl_expr)

Implementation — represent as two (n_genes, n_pathways) weight matrices P
and C (each column normalised to mean):
  pathway_means = X @ P     (sparse matmul, ~minutes)
  ctrl_means    = X @ C
  scores        = pathway_means − ctrl_means

Output: drop-in compatible with the slow script. Writes
  /processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways/all_drugs_scored.h5ad
  /home/beraslan/Projects/ChemoGeneticScreens/PathwayORA/KEGG/interaction_models.parquet
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
DEFAULT_PROJECT   = Path('/home/beraslan/Projects/ChemoGeneticScreens')
DEFAULT_CACHE_DIR = Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways')
DEFAULT_GMT_PATHS = [
    Path('/home/beraslan/Projects/ModuleFinder/MuVI/msigdb/c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt'),
    Path('/home/beraslan/Projects/ModuleFinder/MuVI/msigdb/c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt'),
]


# --------------------------------------------------------------------------
# Module globals — populated in main(), inherited by workers via fork-COW
# --------------------------------------------------------------------------
_DRUG_CODES: np.ndarray   = None
_SCORE_MAT:  np.ndarray   = None
_PERT_INDICES: List[np.ndarray] = None
_NT_IDX: np.ndarray       = None
_DRUG_LEVELS: List[str]   = None


# --------------------------------------------------------------------------
# Helpers
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


def load_reduced_axes(h5_path: Path) -> Tuple[List[str], List[str], List[str]]:
    with h5py.File(h5_path, 'r') as h:
        drugs    = [s.decode() for s in h['drugs'][:]]
        perts    = [s.decode() for s in h['perturbations'][:]]
        pathways = [s.decode() for s in h['pathways'][:]]
    return drugs, perts, pathways


def load_drug_cache(drug: str, cache_dir: Path) -> ad.AnnData:
    fp = cache_dir / f'{drug}.h5ad'
    if not fp.exists():
        raise FileNotFoundError(f'Cache missing: {fp}. Run BuildDrugCaches.py first.')
    a = ad.read_h5ad(fp)
    a.obs['drug'] = drug
    return a


# --------------------------------------------------------------------------
# Fast scoring — vectorised sc.tl.score_genes equivalent
# --------------------------------------------------------------------------
def fast_score_pathways(
    X,                           # sparse (n_cells, n_genes) log1p
    var_names: List[str],
    pathway_genes: Dict[str, set],
    pathway_names: List[str],
    n_bins: int = 25,
    ctrl_size: int = 50,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns
    -------
    score_mat : float32 (n_cells, n_pathways), NaN columns where pathway has < 3 members
    valid_mask : bool (n_pathways,)
    """
    n_cells, n_genes = X.shape
    n_pw = len(pathway_names)
    rng = np.random.default_rng(seed)

    gene_idx = {g: i for i, g in enumerate(var_names)}

    # 1. Per-gene mean expression
    print(f'  computing per-gene means over {n_cells:,} cells × {n_genes} genes...')
    t0 = time.time()
    if sp.issparse(X):
        gene_means = np.asarray(X.mean(axis=0)).flatten()
    else:
        gene_means = X.mean(axis=0).astype(np.float64)
    print(f'    {time.time() - t0:.1f}s')

    # 2. Quantile-bin by mean expression
    bin_edges = np.quantile(gene_means, np.linspace(0, 1, n_bins + 1))
    gene_bins = np.digitize(gene_means, bin_edges[1:-1])  # (n_genes,) int

    # 3. Build pathway weight matrix P and control weight matrix C
    print(f'  building pathway / control weight matrices ({n_pw} pathways)...')
    t0 = time.time()
    P = np.zeros((n_genes, n_pw), dtype=np.float32)
    C = np.zeros((n_genes, n_pw), dtype=np.float32)
    valid = np.zeros(n_pw, dtype=bool)
    n_dropped_small = 0

    for k, pname in enumerate(pathway_names):
        members = [gene_idx[g] for g in pathway_genes[pname] if g in gene_idx]
        if len(members) < 3:
            n_dropped_small += 1
            continue
        valid[k] = True
        P[members, k] = 1.0 / len(members)

        # bin-matched controls, excluding the members themselves
        member_bins = gene_bins[members]
        ctrl_indices: List[int] = []
        members_set = set(members)
        for b in np.unique(member_bins):
            n_in_bin = int((member_bins == b).sum())
            bin_pool = np.array(
                [g for g in np.where(gene_bins == b)[0] if g not in members_set]
            )
            n_sample = min(ctrl_size * n_in_bin, len(bin_pool))
            if n_sample > 0:
                ctrl_indices.extend(rng.choice(bin_pool, n_sample, replace=False))
        n_ctrl = len(ctrl_indices)
        if n_ctrl > 0:
            C[ctrl_indices, k] = 1.0 / n_ctrl
        else:
            valid[k] = False                     # no controls => can't score
    if n_dropped_small:
        print(f'    {n_dropped_small} pathways had < 3 members → NaN')
    print(f'    {time.time() - t0:.1f}s')

    # 4. Two big sparse@dense matmuls
    print('  pathway_means = X @ P  ...')
    t0 = time.time()
    pathway_means = (X @ P).astype(np.float32)   # (n_cells, n_pw)
    print(f'    {time.time() - t0:.1f}s')

    print('  ctrl_means    = X @ C  ...')
    t0 = time.time()
    ctrl_means = (X @ C).astype(np.float32)
    print(f'    {time.time() - t0:.1f}s')

    score_mat = pathway_means - ctrl_means
    # NaN-out invalid pathways
    score_mat[:, ~valid] = np.nan

    return score_mat, valid


# --------------------------------------------------------------------------
# OLS worker — identical to FitDrugPertInteractionModels.py
# --------------------------------------------------------------------------
def _fit_one(args):
    pert_i, pathway_i = args
    pert_idx = _PERT_INDICES[pert_i]
    n_pert_total = len(pert_idx)
    n_ctrl_total = len(_NT_IDX)

    if n_pert_total < 5:
        return (pert_i, pathway_i, np.nan, np.nan, np.nan, np.nan,
                n_pert_total, n_ctrl_total)

    indices = np.concatenate([pert_idx, _NT_IDX])
    score_arr = _SCORE_MAT[indices, pathway_i]
    drug_arr  = _DRUG_CODES[indices]

    valid = ~np.isnan(score_arr)
    if not valid.any():
        return (pert_i, pathway_i, np.nan, np.nan, np.nan, np.nan,
                n_pert_total, n_ctrl_total)

    score_arr = score_arr[valid]
    drug_arr  = drug_arr[valid]
    perturbed_full = np.concatenate([
        np.ones(n_pert_total,  dtype=np.int8),
        np.zeros(n_ctrl_total, dtype=np.int8),
    ])
    perturbed = perturbed_full[valid]

    n_pert = int((perturbed == 1).sum())
    n_ctrl = int((perturbed == 0).sum())
    drugs_with_pert = int(np.unique(drug_arr[perturbed == 1]).size)
    drugs_with_ctrl = int(np.unique(drug_arr[perturbed == 0]).size)
    if drugs_with_pert < 2 or drugs_with_ctrl < 2 or n_pert < 5:
        return (pert_i, pathway_i, np.nan, np.nan, np.nan, np.nan, n_pert, n_ctrl)

    df = pd.DataFrame({
        'drug':      drug_arr,
        'score':     score_arr,
        'perturbed': perturbed,
    })

    try:
        m_full  = ols('score ~ C(drug) * perturbed', data=df).fit()
        m_noint = ols('score ~ C(drug) + perturbed', data=df).fit()
        f_stat, p_int, _ = m_full.compare_f_test(m_noint)
        anova = anova_lm(m_noint, typ=2)
        p_drug = float(anova.loc['C(drug)',   'PR(>F)'])
        p_pert = float(anova.loc['perturbed', 'PR(>F)'])
        return (pert_i, pathway_i, float(f_stat), float(p_int),
                p_drug, p_pert, n_pert, n_ctrl)
    except Exception:
        return (pert_i, pathway_i, np.nan, np.nan, np.nan, np.nan, n_pert, n_ctrl)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    global _DRUG_CODES, _SCORE_MAT, _PERT_INDICES, _NT_IDX, _DRUG_LEVELS

    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project',     type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--cache-dir',   type=Path, default=DEFAULT_CACHE_DIR)
    ap.add_argument('--reduced-h5',  type=Path, default=None)
    ap.add_argument('--gmt-paths',   type=Path, nargs='+', default=DEFAULT_GMT_PATHS)
    ap.add_argument('--results-out', type=Path, default=None)
    ap.add_argument('--master-out',  type=Path, default=None)
    ap.add_argument('--no-save-master', action='store_true')
    ap.add_argument('--n-bins',      type=int, default=25)
    ap.add_argument('--ctrl-size',   type=int, default=50)
    ap.add_argument('--seed',        type=int, default=0)
    ap.add_argument('--n-workers',   type=int, default=max(mp.cpu_count() - 2, 1))
    args = ap.parse_args()

    args.reduced_h5  = args.reduced_h5  or (args.project / 'PathwayORA' / 'KEGG' / 'KEGG_ORA_tensor_reduced.h5')
    args.results_out = args.results_out or (args.project / 'PathwayORA' / 'KEGG' / 'interaction_models.parquet')
    args.master_out  = args.master_out  or (args.cache_dir / 'all_drugs_scored.h5ad')
    args.results_out.parent.mkdir(parents=True, exist_ok=True)

    drugs, perts, pathways = load_reduced_axes(args.reduced_h5)
    print(f'Reduced tensor:  {len(drugs)} drugs, {len(perts)} perts, {len(pathways)} pathways')

    kegg_all: Dict[str, set] = {}
    for gmt in args.gmt_paths:
        kegg_all.update(parse_gmt(gmt))
    pathway_genes = {p: kegg_all[p] for p in pathways}

    # ----------------------------------------------------------------------
    # 1 + 2. Load + concat caches
    # ----------------------------------------------------------------------
    print(f'\n[1+2] Loading and concatenating {len(drugs)} caches...')
    parts = []
    t0 = time.time()
    for d in drugs:
        td = time.time()
        a = load_drug_cache(d, args.cache_dir)
        parts.append(a)
        print(f'  {d:30s}  {a.n_obs:>7,d} cells  ({time.time()-td:.1f}s)')
    big = ad.concat(parts, join='outer', merge='same')
    big.obs['drug']        = big.obs['drug'].astype('category')
    big.obs['target_gene'] = big.obs['target_gene'].astype('string').astype('category')
    del parts; gc.collect()
    print(f'  concatenated: {big.n_obs:,} cells × {big.n_vars} genes  '
          f'({time.time()-t0:.1f}s)')

    # ----------------------------------------------------------------------
    # 3. FAST score (the whole reason this script exists)
    # ----------------------------------------------------------------------
    print(f'\n[3] Fast score: {len(pathways)} pathways via vectorised matmul')
    t0 = time.time()
    score_mat, valid = fast_score_pathways(
        big.X, list(big.var_names), pathway_genes, pathways,
        n_bins=args.n_bins, ctrl_size=args.ctrl_size, seed=args.seed,
    )
    print(f'  total scoring time: {time.time()-t0:.1f}s '
          f'({(~valid).sum()} pathways NaN, {valid.sum()} valid)')

    # ----------------------------------------------------------------------
    # 4. (Optional) Save master h5ad with score columns appended to obs
    # ----------------------------------------------------------------------
    score_cols = [f'score_{p}' for p in pathways]
    if not args.no_save_master:
        print(f'\n[4] Saving master scored AnnData → {args.master_out}')
        t0 = time.time()
        score_df = pd.DataFrame(score_mat, columns=score_cols, index=big.obs.index)
        big.obs = pd.concat([big.obs, score_df], axis=1, copy=False)
        big.write_h5ad(args.master_out, compression='gzip')
        sz = args.master_out.stat().st_size / 1e9
        print(f'  wrote {sz:.1f} GB in {time.time()-t0:.1f}s')
    else:
        print('\n[4] Skipping master save (--no-save-master)')

    # ----------------------------------------------------------------------
    # 5. Build OLS arrays as module globals (fork-COW)
    # ----------------------------------------------------------------------
    print('\n[5] Building OLS arrays')
    t0 = time.time()
    _DRUG_LEVELS = list(big.obs['drug'].cat.categories)
    _DRUG_CODES  = big.obs['drug'].cat.codes.values.astype(np.int8)
    _SCORE_MAT   = score_mat                              # already float32 (n_cells, n_pw)

    target_gene_arr = big.obs['target_gene'].astype(str).values
    pert_idx_dict = {p: np.where(target_gene_arr == p)[0].astype(np.int32) for p in perts}
    _PERT_INDICES = [pert_idx_dict[p] for p in perts]
    _NT_IDX = np.where(target_gene_arr == 'non-targeting')[0].astype(np.int32)
    print(f'  drug codes: {_DRUG_CODES.shape}, levels={len(_DRUG_LEVELS)}')
    print(f'  score matrix: {_SCORE_MAT.shape} ({_SCORE_MAT.nbytes/1e9:.1f} GB)')
    print(f'  {len(_PERT_INDICES)} perts, {len(_NT_IDX):,} non-targeting cells')
    del big; gc.collect()
    print(f'  built in {time.time()-t0:.1f}s')

    # ----------------------------------------------------------------------
    # 6. Fit OLS (parallel, fork-COW shared globals)
    # ----------------------------------------------------------------------
    n_jobs = len(_PERT_INDICES) * len(pathways)
    print(f'\n[6] Fitting {n_jobs:,} OLS interaction models with {args.n_workers} workers')
    t0 = time.time()
    jobs = [(pi, ki) for pi in range(len(_PERT_INDICES))
                     for ki in range(len(pathways))]
    rows = []
    if args.n_workers > 1:
        with mp.Pool(args.n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_fit_one, jobs, chunksize=64), 1):
                rows.append(r)
                if i % 10000 == 0 or i == n_jobs:
                    print(f'    ...{i:>8,d}/{n_jobs:,}')
    else:
        for j in jobs:
            rows.append(_fit_one(j))
    print(f'  fit in {time.time()-t0:.1f}s')

    # ----------------------------------------------------------------------
    # Build results + BH within pathway
    # ----------------------------------------------------------------------
    res = pd.DataFrame(rows, columns=[
        'pert_i', 'pathway_i',
        'F_interaction', 'p_interaction', 'p_drug_main', 'p_pert_main',
        'n_pert_cells', 'n_control_cells',
    ])
    res.insert(0, 'perturbation', [perts[i]    for i in res['pert_i']])
    res.insert(1, 'pathway',      [pathways[i] for i in res['pathway_i']])
    res = res.drop(columns=['pert_i', 'pathway_i'])

    res['q_interaction_bh'] = np.nan
    for pathway, idx in res.groupby('pathway').groups.items():
        p = res.loc[idx, 'p_interaction'].values
        ok = ~np.isnan(p)
        if ok.any():
            _, q, _, _ = multipletests(p[ok], method='fdr_bh')
            qfull = np.full_like(p, np.nan)
            qfull[ok] = q
            res.loc[idx, 'q_interaction_bh'] = qfull

    res.to_parquet(args.results_out, index=False)
    sig = (res['q_interaction_bh'] < 0.10).sum()
    print(f'\nWrote {args.results_out}  ({len(res):,} rows)')
    print(f'Significant context-dependent (pert × pathway) at BH-q < 0.10: '
          f'{sig:,} / {res.shape[0]:,}')


if __name__ == '__main__':
    main()
