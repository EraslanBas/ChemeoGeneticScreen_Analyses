#!/usr/bin/env python3
"""
Fit drug × genetic-perturbation interaction models on KEGG pathway scores.

This is the IN-MEMORY pipeline (no chunking) — assumes per-drug caches built
by BuildDrugCaches.py at:
    /processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways/<drug>.h5ad

Pipeline
--------
  1. Read the reduced tensor (gives the 1,378 perts × 211 pathways universe).
  2. Load all 16 per-drug AnnData caches and concatenate into one big AnnData.
  3. Score every reduced KEGG pathway once on the concatenated AnnData via
     `sc.tl.score_genes` — all drugs share the same expression-bin reference.
  4. Save the master scored AnnData (re-usable for any downstream analysis).
  5. Build numpy arrays for the OLS step:  drug codes, score matrix,
     per-perturbation cell-index arrays, non-targeting index array.
     Stash them in module-level globals so worker processes can share them
     via Linux fork copy-on-write — no pickling of multi-GB structures.
  6. For each (perturbation × pathway): fit OLS interaction model
        score ~ C(drug) + perturbed + C(drug):perturbed
     and F-test the interaction term against the no-interaction model.
  7. BH-correct interaction p-values within each pathway across perturbations.
  8. Save (perturbation × pathway) interaction-model table as parquet.

Memory note: peak RAM is ~150 GB (sparse X 120 GB + score matrix 22 GB +
small overhead). Designed for hosts with ≥256 GB RAM; trivially fits in
a 900 GB host.
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
import scanpy as sc
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
# Module globals — populated in main(), inherited by worker processes via
# Linux fork copy-on-write. DO NOT mutate inside workers.
# --------------------------------------------------------------------------
_DRUG_CODES: np.ndarray   = None    # (n_cells,) int8
_SCORE_MAT:  np.ndarray   = None    # (n_cells, n_pathways) float32
_PERT_INDICES: List[np.ndarray] = None   # list of int32 arrays, one per pert
_NT_IDX: np.ndarray       = None    # int32 array of non-targeting cell indices
_DRUG_LEVELS: List[str]   = None    # category labels for _DRUG_CODES


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
        raise FileNotFoundError(
            f'Cache missing: {fp}. Run BuildDrugCaches.py first.')
    a = ad.read_h5ad(fp)
    a.obs['drug'] = drug
    return a


# --------------------------------------------------------------------------
# Worker — top-level for multiprocessing.Pool
# --------------------------------------------------------------------------
def _fit_one(args):
    pert_i, pathway_i = args
    pert_idx = _PERT_INDICES[pert_i]
    n_pert_total = len(pert_idx)
    n_ctrl_total = len(_NT_IDX)

    if n_pert_total < 5:
        return (pert_i, pathway_i, np.nan, np.nan, np.nan, np.nan,
                n_pert_total, n_ctrl_total)

    # Slice score column + drug codes for the cells in {pert} ∪ {non-targeting}
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

    # Linux: ensure fork start method so workers inherit module globals via COW
    try:
        mp.set_start_method('fork', force=True)
    except RuntimeError:
        pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project',     type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--cache-dir',   type=Path, default=DEFAULT_CACHE_DIR,
                    help='Per-drug AnnData cache dir built by BuildDrugCaches.py')
    ap.add_argument('--reduced-h5',  type=Path, default=None)
    ap.add_argument('--gmt-paths',   type=Path, nargs='+', default=DEFAULT_GMT_PATHS)
    ap.add_argument('--results-out', type=Path, default=None)
    ap.add_argument('--master-out',  type=Path, default=None,
                    help='Master scored AnnData output. Default: '
                         '<cache-dir>/all_drugs_scored.h5ad. Pass /dev/null to skip.')
    ap.add_argument('--no-save-master', action='store_true',
                    help='Skip writing the master scored AnnData.')
    ap.add_argument('--stop-after-master', action='store_true',
                    help='Run only steps 1–4 (load, concat, score, save master). '
                         'Skip OLS fitting — useful to validate the scored AnnData '
                         'before kicking off the long fit.')
    ap.add_argument('--rescore', action='store_true',
                    help='Force re-doing steps 1–4 even if the master scored '
                         'AnnData already exists. Default: load existing master '
                         'and skip straight to OLS fitting.')
    ap.add_argument('--n-workers',   type=int, default=max(mp.cpu_count() - 2, 1))
    args = ap.parse_args()

    args.reduced_h5  = args.reduced_h5  or (args.project / 'PathwayORA' / 'KEGG' / 'KEGG_ORA_tensor_reduced.h5')
    args.results_out = args.results_out or (args.project / 'PathwayORA' / 'KEGG' / 'interaction_models.parquet')
    args.master_out  = args.master_out  or (args.cache_dir / 'all_drugs_scored.h5ad')
    args.results_out.parent.mkdir(parents=True, exist_ok=True)

    drugs, perts, pathways = load_reduced_axes(args.reduced_h5)
    print(f'Reduced tensor:  {len(drugs)} drugs, {len(perts)} perturbations, '
          f'{len(pathways)} pathways  ({args.reduced_h5})')

    # Pathway gene sets
    kegg_all: Dict[str, set] = {}
    for gmt in args.gmt_paths:
        kegg_all.update(parse_gmt(gmt))
    missing = [p for p in pathways if p not in kegg_all]
    if missing:
        raise RuntimeError(f'{len(missing)} pathways missing from GMT: {missing[:5]}...')
    pathway_genes = {p: kegg_all[p] for p in pathways}

    score_cols = [f'score_{p}' for p in pathways]

    if (not args.rescore) and args.master_out.exists():
        # ------------------------------------------------------------------
        # Fast path: a previously-scored master AnnData exists, so skip
        # steps 1–4 entirely and just load it.
        # ------------------------------------------------------------------
        print(f'\n[1-4/6] Loading existing master scored AnnData '
              f'(--rescore not set)\n  from {args.master_out}')
        t0 = time.time()
        big = ad.read_h5ad(args.master_out)
        big.obs['drug']        = big.obs['drug'].astype('category')
        big.obs['target_gene'] = big.obs['target_gene'].astype('category')
        present_score_cols = [c for c in big.obs.columns if c.startswith('score_')]
        if len(present_score_cols) != len(pathways):
            raise RuntimeError(
                f'Master AnnData has {len(present_score_cols)} score columns but '
                f'reduced tensor expects {len(pathways)}. Re-run with --rescore '
                f'to rebuild from per-drug caches.')
        print(f'  loaded: {big.n_obs:,} cells × {big.n_vars} genes, '
              f'{len(present_score_cols)} score columns  ({time.time()-t0:.1f}s)')
    else:
        # ------------------------------------------------------------------
        # 1 & 2. Load + concat all 16 per-drug caches
        # ------------------------------------------------------------------
        print(f'\n[1/6] Loading {len(drugs)} per-drug caches from {args.cache_dir}')
        parts = []
        t0 = time.time()
        for d in drugs:
            td = time.time()
            a = load_drug_cache(d, args.cache_dir)
            parts.append(a)
            print(f'  {d:30s}  {a.n_obs:>7,d} cells × {a.n_vars} vars  ({time.time()-td:.1f}s)')
        print(f'  loaded in {time.time()-t0:.1f}s')

        print('\n[2/6] Concatenating')
        t0 = time.time()
        big = ad.concat(parts, join='outer', merge='same')
        big.obs['drug']        = big.obs['drug'].astype('category')
        big.obs['target_gene'] = big.obs['target_gene'].astype('string').astype('category')
        del parts; gc.collect()
        print(f'  concatenated: {big.n_obs:,} cells × {big.n_vars} genes '
              f'({time.time()-t0:.1f}s)')

        # ------------------------------------------------------------------
        # 3. Score 211 pathways once on the concatenated AnnData
        # ------------------------------------------------------------------
        print(f'\n[3/6] Scoring {len(pathways)} pathways with sc.tl.score_genes')
        t0 = time.time()
        for pname, pgenes in pathway_genes.items():
            present = list(big.var_names.intersection(pgenes))
            col = f'score_{pname}'
            if len(present) < 3:
                big.obs[col] = np.nan
            else:
                sc.tl.score_genes(big, gene_list=present, score_name=col,
                                  random_state=0, use_raw=False)
        print(f'  scored {len(score_cols)} pathways in {time.time()-t0:.1f}s')

        # ------------------------------------------------------------------
        # 4. Save the master scored AnnData (reusable artifact)
        # ------------------------------------------------------------------
        if not args.no_save_master:
            print(f'\n[4/6] Saving master scored AnnData to {args.master_out}')
            t0 = time.time()
            big.write_h5ad(args.master_out, compression='gzip')
            sz = args.master_out.stat().st_size / 1e9
            print(f'  wrote {sz:.1f} GB in {time.time()-t0:.1f}s')
        else:
            print('\n[4/6] Skipping master save (--no-save-master)')

        if args.stop_after_master:
            print('\n--stop-after-master: skipping OLS fit. Inspect the master '
                  'scored AnnData and re-run without this flag to fit models.')
            return

    # ----------------------------------------------------------------------
    # 5. Build numpy arrays for OLS (set as module globals so fork-COW
    #    workers inherit them with zero copy)
    # ----------------------------------------------------------------------
    print('\n[5/6] Building OLS arrays')
    t0 = time.time()
    _DRUG_LEVELS = list(big.obs['drug'].cat.categories)
    _DRUG_CODES  = big.obs['drug'].cat.codes.values.astype(np.int8)
    _SCORE_MAT   = big.obs[score_cols].to_numpy(dtype=np.float32, na_value=np.nan)

    target_gene_arr = big.obs['target_gene'].astype(str).values
    pert_idx_dict = {p: np.where(target_gene_arr == p)[0].astype(np.int32)
                     for p in perts}
    _PERT_INDICES = [pert_idx_dict[p] for p in perts]
    _NT_IDX = np.where(target_gene_arr == 'non-targeting')[0].astype(np.int32)
    n_cells, n_paths = _SCORE_MAT.shape
    score_mat_gb = _SCORE_MAT.nbytes / 1e9
    print(f'  drug codes: {_DRUG_CODES.shape}, levels={_DRUG_LEVELS}')
    print(f'  score matrix: {_SCORE_MAT.shape} (float32, {score_mat_gb:.1f} GB)')
    print(f'  {len(_PERT_INDICES)} perturbations, {len(_NT_IDX):,} non-targeting cells')
    del big; gc.collect()    # free the big AnnData; we kept the arrays
    print(f'  built in {time.time()-t0:.1f}s')

    # ----------------------------------------------------------------------
    # 6. Fit OLS for every (perturbation × pathway) — parallel via fork-COW
    # ----------------------------------------------------------------------
    n_jobs = len(_PERT_INDICES) * len(pathways)
    print(f'\n[6/6] Fitting {n_jobs:,} OLS interaction models with '
          f'{args.n_workers} workers')
    t0 = time.time()
    jobs = [(pi, ki) for pi in range(len(_PERT_INDICES))
                     for ki in range(len(pathways))]
    rows = []
    if args.n_workers > 1:
        with mp.Pool(args.n_workers) as pool:
            for i, r in enumerate(pool.imap_unordered(_fit_one, jobs, chunksize=64), 1):
                rows.append(r)
                if i % 10000 == 0 or i == n_jobs:
                    print(f'    ...{i:>8,d}/{n_jobs:,} '
                          f'({(time.time()-t0)/60:.1f} min)')
    else:
        for i, j in enumerate(jobs, 1):
            rows.append(_fit_one(j))
            if i % 10000 == 0:
                print(f'    ...{i:>8,d}/{n_jobs:,}')
    print(f'  fit in {time.time()-t0:.1f}s ({(time.time()-t0)/60:.1f} min)')

    # ----------------------------------------------------------------------
    # Build results DataFrame + BH correction within pathway
    # ----------------------------------------------------------------------
    print('\nBuilding results DataFrame + BH correction')
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
        valid = ~np.isnan(p)
        if valid.any():
            _, q, _, _ = multipletests(p[valid], method='fdr_bh')
            qfull = np.full_like(p, np.nan)
            qfull[valid] = q
            res.loc[idx, 'q_interaction_bh'] = qfull

    res.to_parquet(args.results_out, index=False)
    print(f'\nWrote {args.results_out}  ({len(res):,} rows)')
    sig = (res['q_interaction_bh'] < 0.10).sum()
    print(f'Significant context-dependent (pert × pathway) at BH-q < 0.10: '
          f'{sig:,} / {res.shape[0]:,}')


if __name__ == '__main__':
    main()
