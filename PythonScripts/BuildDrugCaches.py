#!/usr/bin/env python3
"""
Build per-drug subsetted AnnData caches for downstream pathway-interaction work.

For each of the 16 drug contexts in the reduced tensor, this script
  1. reads /processed_datasets/VCI/ChemoGenetic_H1_Basak/<drug>/<drug>.h5ad ONCE
     (backed mode so only kept cells are materialised),
  2. keeps cells whose `target_gene` is in the reduced-tensor perturbations
     (default 1,378) — and additionally keeps a random subsample of size
     `--n-ctrl` (default 10,000) of `non-targeting` cells,
  3. restricts the gene axis to the union of (a) gene members across ALL
     KEGG pathways from the supplied GMTs (default = legacy + medicus = 844
     pathways → ≈ 5,500 genes in the tested universe) and (b) `--n-bg`
     stratified background genes (default 2,000) drawn evenly across
     `--n-bg-bins` quantile bins of mean expression (default 25 bins → 80
     per bin) from non-pathway genes, so `sc.tl.score_genes` has a richer
     bin-matched control pool for any pathway,
  4. writes the subsetted AnnData to
     /processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways/<drug>.h5ad.

Background-gene selection is computed ONCE globally from
`var/mean_counts` of the `--bg-ref-drug` source h5ad (default
DMSO_round2 — non-perturbed control with most cells), so all 16 caches
share the same gene universe and concat works without column padding.

The resulting per-drug files load in seconds, so the subsequent
chunk-by-chunk scoring + interaction-model fit can iterate without
repeatedly re-reading the multi-GB source h5ads.

Run once before FitDrugPertInteractionModels.py — the model script will
detect these caches and use them.
"""

from __future__ import annotations

import argparse
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

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------
DEFAULT_PROJECT   = Path('/home/beraslan/Projects/ChemoGeneticScreens')
DEFAULT_DATA_ROOT = Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak')
DEFAULT_OUT_DIR   = Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak_Pathways')
DEFAULT_GMT_PATHS = [
    Path('/home/beraslan/Projects/ModuleFinder/MuVI/msigdb/c2.cp.kegg_legacy.v2024.1.Hs.symbols.gmt'),
    Path('/home/beraslan/Projects/ModuleFinder/MuVI/msigdb/c2.cp.kegg_medicus.v2024.1.Hs.symbols.gmt'),
]


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


def select_background_genes(src_h5ad: Path,
                            pathway_genes: set,
                            n_bg: int,
                            n_bins: int,
                            seed: int) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """Pick `n_bg` non-pathway genes evenly distributed across `n_bins`
    quantile bins of `var/mean_counts` from the reference source h5ad.

    Returns (background_gene_symbols, bin_edges, bin_assignments_for_bg).
    """
    with h5py.File(src_h5ad, 'r') as h:
        var_index_key = '_index' if '_index' in h['var'] else h['var'].attrs.get('_index', '_index')
        all_genes  = np.array([s.decode() if isinstance(s, bytes) else s
                               for s in h['var'][var_index_key][:]])
        mean_cnts  = h['var/mean_counts'][:].astype(np.float64)

    is_pathway = np.isin(all_genes, list(pathway_genes))
    nonpw_idx  = np.where(~is_pathway)[0]
    nonpw_genes = all_genes[nonpw_idx]
    nonpw_means = mean_cnts[nonpw_idx]

    # Quantile bin edges over non-pathway genes only (so each bin holds
    # ~equal counts of candidate genes rather than ~equal expression range).
    bin_edges = np.quantile(nonpw_means, np.linspace(0, 1, n_bins + 1))
    bin_edges[0]  = -np.inf
    bin_edges[-1] = np.inf
    bin_assign    = np.digitize(nonpw_means, bin_edges[1:-1])

    rng = np.random.default_rng(seed)
    per_bin = n_bg // n_bins
    leftover = n_bg - per_bin * n_bins

    selected_local: List[int] = []
    for b in range(n_bins):
        bin_mask = (bin_assign == b)
        bin_pool = np.where(bin_mask)[0]
        if len(bin_pool) == 0:
            continue
        n_take = min(per_bin, len(bin_pool))
        selected_local.extend(rng.choice(bin_pool, size=n_take, replace=False))

    # Distribute any leftover slots across bins that still have unused candidates
    if leftover > 0:
        remaining = np.setdiff1d(np.arange(len(nonpw_idx)), np.array(selected_local))
        if len(remaining) >= leftover:
            selected_local.extend(rng.choice(remaining, size=leftover, replace=False))

    selected_local = np.array(selected_local, dtype=np.int64)
    bg_genes  = nonpw_genes[selected_local].tolist()
    bg_bins   = bin_assign[selected_local]
    return bg_genes, bin_edges, bg_bins


# --------------------------------------------------------------------------
# Per-drug subset
# --------------------------------------------------------------------------
def build_one_drug(drug: str,
                   src_root: Path,
                   out_dir: Path,
                   keep_perts: set,
                   score_genes_union: set,
                   pathway_gene_set: set,
                   background_gene_set: set,
                   n_ctrl: int,
                   seed: int,
                   skip_existing: bool,
                   all_perts: bool = False) -> Tuple[str, str]:
    src = src_root / drug / f'{drug}.h5ad'
    out = out_dir / f'{drug}.h5ad'

    if skip_existing and out.exists():
        return (drug, f'  exists, skipped — {out}  ({out.stat().st_size / 1e9:.2f} GB)')

    if not src.exists():
        return (drug, f'  MISSING {src}')

    # Each worker creates its own deterministic RNG (per-drug seed)
    rng = np.random.default_rng(seed + hash(drug) % 1_000_000)

    t0 = time.time()
    a = ad.read_h5ad(src, backed='r')
    n_total = a.n_obs

    tg = a.obs['target_gene'].astype(str).values
    if all_perts:
        # Keep every perturbed cell (anything not non-targeting), regardless of
        # whether the perturbation made it into the reduced tensor.
        pert_mask = (tg != 'non-targeting')
    else:
        pert_mask = np.isin(tg, list(keep_perts))
    nt_idx = np.where(tg == 'non-targeting')[0]
    n_nt_total = len(nt_idx)
    if n_ctrl > 0 and n_nt_total > n_ctrl:
        nt_keep = rng.choice(nt_idx, size=n_ctrl, replace=False)
    else:
        nt_keep = nt_idx
    keep_mask = pert_mask.copy()
    keep_mask[nt_keep] = True

    sub = a[keep_mask].to_memory()
    a.file.close()

    # Subset genes to KEGG pathway-member union ∪ stratified background genes
    # (intersected with the source's var_names — same fixed set for every drug
    # so concat aligns without column padding).
    gene_keep = [g for g in sub.var_names if g in score_genes_union]
    sub = sub[:, gene_keep].copy()

    # Mark which kept genes are pathway members vs background, so downstream
    # scoring can restrict the bin-matched control pool to background genes
    # and exclude pathway-member overlap.
    is_pw = np.isin(sub.var_names.values, list(pathway_gene_set))
    is_bg = np.isin(sub.var_names.values, list(background_gene_set))
    sub.var['kegg_pathway_member'] = is_pw
    sub.var['background_gene']     = is_bg

    # Compact obs: keep target_gene + a few QC columns + add drug
    sub.obs['drug'] = drug
    keep_obs_cols = [c for c in ['target_gene', 'drug', 'batch',
                                  'n_genes_by_counts', 'log1p_n_genes_by_counts',
                                  'total_counts', 'log1p_total_counts',
                                  'pct_counts_mt']
                     if c in sub.obs.columns or c == 'drug']
    sub.obs = sub.obs[keep_obs_cols].copy()
    sub.obs['target_gene'] = sub.obs['target_gene'].astype('category')
    sub.obs['drug']        = sub.obs['drug'].astype('category')

    # Annotate provenance
    sub.uns['source_h5ad']     = str(src)
    sub.uns['n_cells_total']   = int(n_total)
    sub.uns['n_nt_total']      = int(n_nt_total)
    sub.uns['n_nt_kept']       = int(len(nt_keep))
    sub.uns['n_pert_kept']     = int(pert_mask.sum())
    sub.uns['n_keep_perts']    = int(len(keep_perts))
    sub.uns['n_keep_genes']    = int(sub.n_vars)
    sub.uns['n_pathway_genes'] = int(is_pw.sum())
    sub.uns['n_background_genes'] = int(is_bg.sum())

    out_dir.mkdir(parents=True, exist_ok=True)
    sub.write_h5ad(out, compression='gzip')
    sz = out.stat().st_size / 1e9
    elapsed = time.time() - t0

    msg = (f'  {drug:30s}  cells={sub.n_obs:>7,d} (perts={int(pert_mask.sum()):,}, '
           f'nt={len(nt_keep):,})  vars={sub.n_vars:>5d}  '
           f'{sz:5.2f} GB  ({elapsed:.1f}s)')

    del sub
    gc.collect()
    return (drug, msg)


def _worker(args_tuple):
    """Top-level worker for multiprocessing.Pool — must be picklable."""
    return build_one_drug(*args_tuple)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--project',    type=Path, default=DEFAULT_PROJECT)
    ap.add_argument('--data-root',  type=Path, default=DEFAULT_DATA_ROOT)
    ap.add_argument('--out-dir',    type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument('--reduced-h5', type=Path, default=None)
    ap.add_argument('--gmt-paths',  type=Path, nargs='+', default=DEFAULT_GMT_PATHS)
    ap.add_argument('--n-ctrl',     type=int,  default=10_000,
                    help='Subsample non-targeting cells to this many per drug.')
    ap.add_argument('--n-bg',       type=int,  default=2_000,
                    help='Number of non-pathway background genes to add. '
                         'Distributed across --n-bg-bins quantile bins.')
    ap.add_argument('--n-bg-bins',  type=int,  default=25,
                    help='Number of quantile bins for stratified background '
                         'gene sampling (default 25 → 80 genes/bin at n-bg=2000).')
    ap.add_argument('--bg-ref-drug', type=str, default='DMSO_round2',
                    help='Source drug whose var/mean_counts is used to bin '
                         'genes for background sampling. Default: DMSO_round2.')
    ap.add_argument('--seed',       type=int,  default=0)
    ap.add_argument('--drugs', nargs='+', default=None,
                    help='Process only these drugs (default: all 16 from the reduced tensor).')
    ap.add_argument('--overwrite',  action='store_true',
                    help='Re-build caches even if the output file already exists.')
    ap.add_argument('--n-parallel', type=int, default=6,
                    help='Number of drugs to process in parallel (default 6). '
                         'Set to 1 for sequential.')
    ap.add_argument('--all-perts', action='store_true',
                    help='Keep every perturbed cell (target_gene != non-targeting), '
                         'not just cells whose KO appears in the reduced tensor. '
                         'Caches are ~5–10× larger but include every distinct KO.')
    ap.add_argument('--pathway-list', type=Path, default=None,
                    help='Optional file with one pathway name per line. If given, '
                         'restrict the pathway-gene universe to these pathways only '
                         '(intersected with the GMTs). Use to subset to a curated '
                         'rep-set such as pathway_representatives_all_kegg_jaccard60.txt.')
    args = ap.parse_args()

    args.reduced_h5 = args.reduced_h5 or (args.project / 'PathwayORA' / 'KEGG' / 'KEGG_ORA_tensor_reduced.h5')
    args.out_dir.mkdir(parents=True, exist_ok=True)

    drugs, perts, pathways = load_reduced_axes(args.reduced_h5)
    print(f'Reduced tensor:  {len(drugs)} drugs, {len(perts)} perturbations, '
          f'{len(pathways)} pathways  ({args.reduced_h5})')

    kegg_all: Dict[str, set] = {}
    for gmt in args.gmt_paths:
        kegg_all.update(parse_gmt(gmt))
    print(f'KEGG pathways loaded: {len(kegg_all)} (union of {len(args.gmt_paths)} GMTs)')

    # Optional restriction to a curated pathway list (e.g. Jaccard-collapsed reps)
    if args.pathway_list is not None:
        wanted = [ln.strip() for ln in args.pathway_list.read_text().splitlines()
                   if ln.strip() and not ln.startswith('#')]
        wanted_set = set(wanted)
        missing = wanted_set - set(kegg_all)
        kegg_all = {k: v for k, v in kegg_all.items() if k in wanted_set}
        print(f'Restricted to {args.pathway_list.name}: {len(kegg_all)} of {len(wanted)} requested pathways found in GMTs')
        if missing:
            print(f'  warning: {len(missing)} pathways requested but not in GMTs (e.g. {sorted(missing)[:3]})')

    pathway_gene_set: set = set().union(*kegg_all.values())
    print(f'Pathway-gene universe: {len(pathway_gene_set)} unique gene symbols')

    # Pick 2,000 (default) stratified background genes ONCE, from the
    # bg-ref-drug source. Same set for all 16 drugs → concat aligns cleanly.
    bg_src = args.data_root / args.bg_ref_drug / f'{args.bg_ref_drug}.h5ad'
    if not bg_src.exists():
        raise RuntimeError(f'Background-reference source missing: {bg_src}')
    print(f'\nSelecting {args.n_bg} background genes from {bg_src}')
    print(f'  binning by var/mean_counts into {args.n_bg_bins} quantile bins, seed={args.seed}')
    bg_genes, bin_edges, bg_bins = select_background_genes(
        src_h5ad=bg_src,
        pathway_genes=pathway_gene_set,
        n_bg=args.n_bg,
        n_bins=args.n_bg_bins,
        seed=args.seed,
    )
    bg_per_bin = np.bincount(bg_bins, minlength=args.n_bg_bins)
    print(f'  selected {len(bg_genes)} background genes  '
          f'(per-bin counts min/median/max = {bg_per_bin.min()}/'
          f'{int(np.median(bg_per_bin))}/{bg_per_bin.max()})')
    background_gene_set = set(bg_genes)
    overlap = pathway_gene_set & background_gene_set
    if overlap:
        raise RuntimeError(f'BUG: {len(overlap)} background genes overlap pathway set: {list(overlap)[:5]}')

    score_genes_union: set = pathway_gene_set | background_gene_set
    print(f'Total gene universe (pathway ∪ background): {len(score_genes_union)}')

    if args.drugs:
        drugs_to_process = [d for d in args.drugs if d in drugs]
        unknown = set(args.drugs) - set(drugs)
        if unknown:
            print(f'  warning: ignoring unknown drugs {sorted(unknown)}')
    else:
        drugs_to_process = drugs
    print(f'Drugs to process: {len(drugs_to_process)}')

    keep_perts = set(perts)

    n_par = max(1, min(args.n_parallel, len(drugs_to_process)))
    print(f'\nWriting per-drug caches to {args.out_dir}')
    print(f'(n-ctrl={args.n_ctrl} non-targeting cells subsampled per drug, seed={args.seed}, '
          f'n-parallel={n_par}, all_perts={args.all_perts})\n')

    job_args = [
        (drug, args.data_root, args.out_dir, keep_perts, score_genes_union,
         pathway_gene_set, background_gene_set,
         args.n_ctrl, args.seed, not args.overwrite, args.all_perts)
        for drug in drugs_to_process
    ]

    t_total = time.time()
    if n_par == 1:
        for d_idx, ja in enumerate(job_args, 1):
            print(f'[{d_idx}/{len(job_args)}] {ja[0]}')
            _, msg = _worker(ja)
            print(msg, flush=True)
    else:
        completed = 0
        with mp.Pool(n_par) as pool:
            for drug, msg in pool.imap_unordered(_worker, job_args):
                completed += 1
                print(f'[{completed}/{len(job_args)}] {drug}\n{msg}', flush=True)

    print(f'\nAll caches built in {time.time() - t_total:.1f}s.')
    print(f'Output dir: {args.out_dir}')
    total_gb = sum(p.stat().st_size for p in args.out_dir.glob('*.h5ad')) / 1e9
    print(f'Total cache size: {total_gb:.1f} GB across {len(list(args.out_dir.glob("*.h5ad")))} files')


if __name__ == '__main__':
    main()
