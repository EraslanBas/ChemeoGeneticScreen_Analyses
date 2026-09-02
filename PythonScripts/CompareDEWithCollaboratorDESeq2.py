#!/usr/bin/env python3
"""
Parallel runner for the per-drug DE comparison between Basak's pipeline (ashr-shrunk
posterior means + FDR matrices) and Tyler's pipeline (DESeq2-on-pseudobulk after KD-
efficiency filter, output in /large_storage/gilbertlab/tfair/Set{1,2}_all_FINAL/DESeq2).

Mirrors `SRC/Notebooks/20_CompareDEWithCollaboratorDESeq2.ipynb` but parallelises across
drugs and writes a single combined PDF at the end.

Usage:
  python CompareDEWithCollaboratorDESeq2.py                   # all 16 drugs, 8 workers
  python CompareDEWithCollaboratorDESeq2.py --drugs DMSO_round2 CHIR-98014
  python CompareDEWithCollaboratorDESeq2.py --n-workers 6 --nt-subsample 20000
  python CompareDEWithCollaboratorDESeq2.py --out-dir /tmp/jaccard_run

The script reads Tyler's CSVs once per drug (one process per drug), produces one PDF page
per drug in a tmp dir, then merges them into <out-dir>/per_drug_figures_combined.pdf.

Stalling note: the largest single cost per drug is materialising the NT-cell subset of the
source h5ad in backed mode. The `--nt-subsample N` flag samples N NT cells (default 20000)
for the mean-expression calculation instead of all of them — the per-gene mean over 20k
cells is statistically indistinguishable from the full set for plotting purposes.

Outputs (under --out-dir, default = FDR_matrices/jaccard_outputs/):
  per_drug_figures_combined.pdf           — one page per drug, in input order
  per_drug_pdfs/<drug>.pdf                — per-drug single-page PDFs
  jaccard_per_target.csv                  — long-format Jaccard rows
  per_drug_summary.csv                    — pooled log2FC corr + tested-gene-universe stats
"""

from __future__ import annotations

# Force a headless backend BEFORE pyplot import so worker processes don't try to open
# a GUI when they fork.
import matplotlib
matplotlib.use('Agg', force=True)

import argparse
import gc
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import anndata as ad
import scipy.sparse as sp
from scipy import stats
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --------------------------------------------------------------------------
# Defaults / constants
# --------------------------------------------------------------------------
DEFAULT_PROJ        = Path('/home/beraslan/Projects/ChemoGeneticScreens')
DEFAULT_OUR_FDR_DIR = DEFAULT_PROJ / 'FDR_matrices'
DEFAULT_OUR_PM_DIR  = DEFAULT_PROJ / 'PosteriorMeanMatrices'
DEFAULT_OUR_RAW_DIR = Path('/processed_datasets/VCI/ChemoGenetic_H1_Basak')
DEFAULT_SET1_DIR    = Path('/large_storage/gilbertlab/tfair/Set1_all_FINAL/DESeq2')
DEFAULT_SET2_DIR    = Path('/large_storage/gilbertlab/tfair/Set2_all_FINAL/DESeq2')

DEFAULT_DRUGS_ALL: list[str] = [
    'AR-A014418', 'AZD4573', 'CHIR-98014', 'DMSO_round2', 'Lexibulin',
    'PP121', 'Romidepsin', 'Stattic',
    'Bisindolylmaleimide-I', 'DG-172', 'DMSO_round2_batch2', 'JTE-607',
    'LDN-193189', 'LY2090314', 'NSC95397', 'VX-11e',
]

N_THRESHOLDS    = 30
THRESHOLDS_LOG2 = np.concatenate([[0.0],
                                   np.logspace(np.log10(0.01), np.log10(5.0), N_THRESHOLDS)])
LN2 = float(np.log(2))
THRESHOLDS_NAT  = THRESHOLDS_LOG2 * LN2

N_CELL_BINS   = [2, 50, 100, 200, 400, 800, 1600, 3000, 1_000_000]
N_CELL_LABELS = ['2-50', '50-100', '100-200', '200-400',
                 '400-800', '800-1600', '1600-3000', '3000+']

_LOG_TICKS = np.array([
    0.005, 0.007,
    0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.07,
    0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70,
    1.00, 1.50, 2.00, 3.00, 4.00, 5.00, 7.00,
    10.0, 15.0, 20.0,
])

DESEQ2_COLNAMES = ['gene', 'baseMean', 'log2FoldChange', 'lfcSE', 'stat', 'pvalue', 'padj']
_SUFFIX_RE      = re.compile(r'_(P\d+(?:P\d+)?|ENST[\d.]+)_[AB]$')


def _fmt_tick(x: float) -> str:
    if x >= 10: return f'{x:.0f}'
    if x >= 1:  return f'{x:.1f}'
    return f'{x:g}'


def pert_to_gene(pert_id: str) -> str:
    return _SUFFIX_RE.sub('', pert_id.split('|')[0])


def build_drug_specs(set1_dir: Path, set2_dir: Path) -> list[dict]:
    return [
        {'set': 'batch1', 'their_dir': set1_dir, 'their_drug': 'AR.A014418',           'our_drug': 'AR-A014418'},
        {'set': 'batch1', 'their_dir': set1_dir, 'their_drug': 'AZD4573',              'our_drug': 'AZD4573'},
        {'set': 'batch1', 'their_dir': set1_dir, 'their_drug': 'CHIR.98014',           'our_drug': 'CHIR-98014'},
        {'set': 'batch1', 'their_dir': set1_dir, 'their_drug': 'DMSO',                 'our_drug': 'DMSO_round2'},
        {'set': 'batch1', 'their_dir': set1_dir, 'their_drug': 'Lexibulin',            'our_drug': 'Lexibulin'},
        {'set': 'batch1', 'their_dir': set1_dir, 'their_drug': 'PP121',                'our_drug': 'PP121'},
        {'set': 'batch1', 'their_dir': set1_dir, 'their_drug': 'Romidepsin',           'our_drug': 'Romidepsin'},
        {'set': 'batch1', 'their_dir': set1_dir, 'their_drug': 'Stattic',              'our_drug': 'Stattic'},
        {'set': 'batch2', 'their_dir': set2_dir, 'their_drug': 'Bisindolylmaleimide.I','our_drug': 'Bisindolylmaleimide-I'},
        {'set': 'batch2', 'their_dir': set2_dir, 'their_drug': 'DG.172',               'our_drug': 'DG-172'},
        {'set': 'batch2', 'their_dir': set2_dir, 'their_drug': 'DMSO',                 'our_drug': 'DMSO_round2_batch2'},
        {'set': 'batch2', 'their_dir': set2_dir, 'their_drug': 'JTE.607',              'our_drug': 'JTE-607'},
        {'set': 'batch2', 'their_dir': set2_dir, 'their_drug': 'LDN.193189',           'our_drug': 'LDN-193189'},
        {'set': 'batch2', 'their_dir': set2_dir, 'their_drug': 'LY2090314',            'our_drug': 'LY2090314'},
        {'set': 'batch2', 'their_dir': set2_dir, 'their_drug': 'NSC95397',             'our_drug': 'NSC95397'},
        {'set': 'batch2', 'their_dir': set2_dir, 'their_drug': 'VX.11e',               'our_drug': 'VX-11e'},
    ]


# --------------------------------------------------------------------------
# Loading + computation
# --------------------------------------------------------------------------
def load_one_drug(spec: dict, cfg: dict) -> dict:
    """Load Tyler's + Basak's DE outputs for one drug, plus n_cells and NT-mean expr."""
    their_dir, their_drug, our_drug = spec['their_dir'], spec['their_drug'], spec['our_drug']
    suffix = f'_{cfg["contrast"]}_in_{their_drug}_{cfg["filter"]}.csv'
    files  = sorted(f for f in os.listdir(their_dir) if f.endswith(suffix))

    # ---- Tyler's side ----
    their_sig_acc:    dict[str, dict[float, set[str]]] = defaultdict(lambda: {float(t): set() for t in THRESHOLDS_LOG2})
    their_tested_acc: dict[str, set[str]]              = defaultdict(set)
    their_l2fc_acc:   dict[str, list[pd.Series]]       = defaultdict(list)

    padj_thr = cfg['padj_thr']
    for fname in files:
        gene = pert_to_gene(fname[:-len(suffix)])
        df = pd.read_csv(their_dir / fname,
                         header=0, names=DESEQ2_COLNAMES,
                         usecols=[0, 2, 6], index_col=0)
        idx     = df.index.astype(str).values
        padj    = df['padj'].values
        l2fc    = df['log2FoldChange'].values
        padj_ok = ~np.isnan(padj) & (padj <= padj_thr)
        fc_ok   = ~np.isnan(l2fc)

        their_tested_acc[gene].update(idx)
        their_l2fc_acc[gene].append(df['log2FoldChange'].astype(np.float32))
        per_tau = their_sig_acc[gene]
        for tau in THRESHOLDS_LOG2:
            mask = padj_ok & fc_ok & (np.abs(l2fc) > tau)
            per_tau[float(tau)].update(idx[mask])

    their_sig    = {g: {t: frozenset(s) for t, s in d.items()} for g, d in their_sig_acc.items()}
    their_tested = {g: frozenset(s) for g, s in their_tested_acc.items()}
    their_npairs = {g: len(sl) for g, sl in their_l2fc_acc.items()}

    their_l2fc_rows = {}
    for g, sl in their_l2fc_acc.items():
        their_l2fc_rows[g] = sl[0] if len(sl) == 1 else pd.concat(sl, axis=1).mean(axis=1)
    their_l2fc = pd.DataFrame(their_l2fc_rows).T.astype(np.float32)

    # ---- Basak's side: FDR + PM matrices ----
    fdr_thr = cfg['fdr_thr']
    fdr_fp = cfg['our_fdr_dir'] / f'{our_drug}_FDRs.csv'
    pm_fp  = cfg['our_pm_dir']  / f'PosteriorMean_matrix_{our_drug}.csv'
    if not fdr_fp.exists() or not pm_fp.exists():
        raise FileNotFoundError(f'missing matrices for {our_drug}')
    fdr_df = pd.read_csv(fdr_fp, index_col=0)
    pm_df  = pd.read_csv(pm_fp,  index_col=0)
    common_perts = fdr_df.index.intersection(pm_df.index)
    common_genes = fdr_df.columns.intersection(pm_df.columns)
    fdr_df = fdr_df.loc[common_perts, common_genes]
    pm_df  = pm_df.loc[common_perts,  common_genes]

    fdr_arr = fdr_df.values
    pm_arr  = pm_df.values
    abs_pm  = np.abs(pm_arr)
    fdr_ok  = ~np.isnan(fdr_arr) & (fdr_arr < fdr_thr)
    pm_ok   = ~np.isnan(pm_arr)
    cols    = common_genes.astype(str).values

    our_sig:    dict[str, dict[float, frozenset[str]]] = {}
    our_tested: dict[str, frozenset[str]]              = {}
    for i, tg in enumerate(common_perts.astype(str)):
        per_tau = {}
        for tau_l2, tau_nat in zip(THRESHOLDS_LOG2, THRESHOLDS_NAT):
            mask = fdr_ok[i] & pm_ok[i] & (abs_pm[i] > tau_nat)
            per_tau[float(tau_l2)] = frozenset(cols[mask].tolist())
        our_sig[tg]    = per_tau
        our_tested[tg] = frozenset(cols[~np.isnan(fdr_arr[i])].tolist())

    our_l2fc = (pm_df / LN2).astype(np.float32)

    # ---- per-pert cell counts + per-gene mean log1p expr in NT cells ----
    src_h5 = cfg['our_raw_dir'] / our_drug / f'{our_drug}.h5ad'
    n_cells_per_pert: dict[str, int] = {}
    mean_expr_ctrl_per_gene: pd.Series = pd.Series(dtype=np.float32)
    if src_h5.exists():
        a = ad.read_h5ad(src_h5, backed='r')
        try:
            tg_series = a.obs['target_gene'].astype(str)
            n_cells_per_pert = tg_series.value_counts().to_dict()
            is_nt = tg_series.isin(['non-targeting', 'non_targeting']).values
            nt_idx = np.flatnonzero(is_nt)
            if nt_idx.size > 0:
                # Subsample NT cells for the mean computation — much faster than
                # materialising hundreds of thousands of cells out of backed sparse.
                sub = cfg.get('nt_subsample')
                if sub is not None and 0 < sub < nt_idx.size:
                    rng = np.random.default_rng(0)
                    nt_idx = np.sort(rng.choice(nt_idx, size=sub, replace=False))
                a_nt = a[nt_idx].to_memory()
                X_nt = a_nt.X
                if sp.issparse(X_nt):
                    mean_ctrl = np.asarray(X_nt.mean(axis=0)).ravel()
                else:
                    mean_ctrl = np.asarray(X_nt).mean(axis=0)
                mean_expr_ctrl_per_gene = pd.Series(
                    mean_ctrl.astype(np.float32),
                    index=a_nt.var_names.astype(str),
                    name='mean_expr_ctrl',
                )
                del a_nt
        finally:
            try: a.file.close()
            except Exception: pass
        del a

    return {
        'their_l2fc':              their_l2fc,
        'their_sig':               their_sig,
        'their_tested':            their_tested,
        'their_npairs':            their_npairs,
        'our_l2fc':                our_l2fc,
        'our_sig':                 our_sig,
        'our_tested':              our_tested,
        'n_cells_per_pert':        n_cells_per_pert,
        'mean_expr_ctrl_per_gene': mean_expr_ctrl_per_gene,
    }


def compute_jaccard_for_drug(data: dict, spec: dict) -> pd.DataFrame:
    rows = []
    shared = sorted(set(data['their_sig']) & set(data['our_sig']))
    for tg in shared:
        common_pp = data['their_tested'].get(tg, frozenset()) & data['our_tested'].get(tg, frozenset())
        if not common_pp:
            for tau in THRESHOLDS_LOG2:
                rows.append({'batch': spec['set'], 'drug_ours': spec['our_drug'],
                             'drug_theirs': spec['their_drug'], 'target_gene': tg,
                             'tau_log2': tau,
                             'n_their_sig': 0, 'n_our_sig': 0, 'n_intersect': 0,
                             'n_union': 0, 'jaccard': np.nan,
                             'n_tested_intersect': 0,
                             'n_guide_pairs_theirs': data['their_npairs'].get(tg, 0)})
            continue
        for tau in THRESHOLDS_LOG2:
            a = data['their_sig'][tg][tau] & common_pp
            b = data['our_sig'][tg][tau]   & common_pp
            inter = a & b
            union = a | b
            rows.append({
                'batch':              spec['set'],
                'drug_ours':          spec['our_drug'],
                'drug_theirs':        spec['their_drug'],
                'target_gene':        tg,
                'tau_log2':           tau,
                'n_their_sig':        len(a),
                'n_our_sig':          len(b),
                'n_intersect':        len(inter),
                'n_union':            len(union),
                'jaccard':            (len(inter) / len(union)) if union else np.nan,
                'n_tested_intersect': len(common_pp),
                'n_guide_pairs_theirs': data['their_npairs'].get(tg, 0),
            })
    return pd.DataFrame(rows)


def compute_pooled_corr(data: dict) -> dict:
    A = data['their_l2fc']
    B = data['our_l2fc']
    common_perts = A.index.intersection(B.index)
    common_genes = A.columns.intersection(B.columns)
    a = A.loc[common_perts, common_genes].values.ravel()
    b = B.loc[common_perts, common_genes].values.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 10 or np.std(a[ok]) == 0 or np.std(b[ok]) == 0:
        return {'n': n, 'pearson': float('nan'), 'spearman': float('nan')}
    return {
        'n':        n,
        'pearson':  float(stats.pearsonr(a[ok],  b[ok])[0]),
        'spearman': float(stats.spearmanr(a[ok], b[ok])[0]),
    }


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def _apply_log_tau_ticks(ax, lo: float, hi: float):
    ax.set_xscale('log')
    ax.set_xlim(lo * 0.95, hi * 1.05)
    xt = _LOG_TICKS[(_LOG_TICKS >= lo) & (_LOG_TICKS <= hi)]
    ax.set_xticks(xt)
    ax.set_xticklabels([_fmt_tick(t) for t in xt], rotation=0)
    ax.minorticks_off()


def plot_drug(spec: dict, data: dict, jacc_drug: pd.DataFrame, corr: dict,
              cfg: dict, *, scatter_lim: float = 5.0, ma_ylim: float = 5.0):
    """4×3 figure mirroring the notebook's `plot_drug`."""
    FDR_THR  = cfg['fdr_thr']
    PADJ_THR = cfg['padj_thr']

    fig, axes = plt.subplots(4, 3, figsize=(20, 22))

    # ---------- (0,0): log2FC scatter ----------
    A = data['their_l2fc']
    B = data['our_l2fc']
    common_perts = A.index.intersection(B.index)
    common_genes = A.columns.intersection(B.columns)
    a = A.loc[common_perts, common_genes].values.ravel()
    b = B.loc[common_perts, common_genes].values.ravel()
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    L = scatter_lim

    ax = axes[0, 0]
    ax.hexbin(b, a, gridsize=70, mincnt=1, cmap='viridis', bins='log',
              extent=(-L, L, -L, L))
    ax.plot([-L, L], [-L, L], ls='--', color='crimson', lw=0.8, label='y = x')
    ax.axhline(0, color='gray', lw=0.4); ax.axvline(0, color='gray', lw=0.4)
    ax.set_xlim(-L, L); ax.set_ylim(-L, L)
    ax.set_xlabel(r"Basak's  $\log_2\,\mathrm{FC}$")
    ax.set_ylabel(r"Tyler's  $\log_2\,\mathrm{FC}$")
    ax.set_title(f'log2FC scatter — ρ={corr["pearson"]:.3f}  ρ_s={corr["spearman"]:.3f}'
                 f'  (n={corr["n"]:,})', fontsize=10)
    ax.legend(loc='upper left', fontsize=8)

    # ---------- stratify perts by cell count ----------
    n_series = pd.Series(data['n_cells_per_pert'], name='n_cells')
    pert_bin = pd.cut(n_series, bins=N_CELL_BINS, labels=N_CELL_LABELS,
                       include_lowest=True)
    pert_to_bin = pert_bin.dropna().astype(str).to_dict()
    jdf = jacc_drug.copy()
    jdf['n_cells_bin'] = jdf['target_gene'].map(pert_to_bin)
    jdf = jdf[jdf['n_cells_bin'].notna()]
    plot_df = jdf[jdf['tau_log2'] > 0]

    palette = sns.color_palette('viridis', n_colors=len(N_CELL_LABELS))
    x_lo = float(plot_df['tau_log2'].min()) if len(plot_df) else 0.01
    x_hi = float(plot_df['tau_log2'].max()) if len(plot_df) else 5.0

    def _strat_curves(ax, *, value_col, ylabel, title,
                       logy=False, dropna_col=None,
                       legend_loc='upper right', ylim=None):
        src = plot_df.dropna(subset=[dropna_col]) if dropna_col else plot_df
        for color, lab in zip(palette, N_CELL_LABELS):
            sub = src[src['n_cells_bin'] == lab]
            if sub.empty:
                continue
            g = sub.groupby('tau_log2')[value_col]
            med = g.median()
            lo  = g.quantile(0.25)
            hi  = g.quantile(0.75)
            n   = sub['target_gene'].nunique()
            ax.plot(med.index.values, med.values, color=color, lw=1.8,
                    label=f'{lab}  (n={n})')
            ax.fill_between(med.index.values, lo.values, hi.values,
                             color=color, alpha=0.15, linewidth=0)
        _apply_log_tau_ticks(ax, x_lo, x_hi)
        if logy:
            ax.set_yscale('symlog', linthresh=1)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_xlabel(r'$|\log_2\,\mathrm{FC}|$ threshold $\tau$  (log scale)')
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=10)
        ax.legend(loc=legend_loc, fontsize=7, frameon=True, title='# cells (Basak)')
        ax.grid(alpha=0.3, which='major')

    _strat_curves(axes[0, 1], value_col='jaccard',
                  ylabel='Jaccard  (median ± IQR)',
                  title=f"Jaccard: Basak's vs Tyler's sig sets, vs τ   (FDR<{FDR_THR}, padj≤{PADJ_THR})",
                  logy=False, dropna_col='jaccard',
                  legend_loc='upper left', ylim=(0, 1.0))
    _strat_curves(axes[0, 2], value_col='n_intersect',
                  ylabel='# genes called sig by both, per target  (median ± IQR)',
                  title="Intersection size: Basak's ∩ Tyler's, vs τ",
                  logy=True)

    axes[1, 0].set_axis_off()
    _strat_curves(axes[1, 1], value_col='n_our_sig',
                  ylabel="# Basak's sig genes per target  (median ± IQR)",
                  title=f"Basak's sig-gene count vs τ   (FDR<{FDR_THR} ∧ |log2FC|>τ)",
                  logy=True)
    _strat_curves(axes[1, 2], value_col='n_their_sig',
                  ylabel="# Tyler's sig genes per target  (median ± IQR)",
                  title=f"Tyler's sig-gene count vs τ   (padj≤{PADJ_THR} ∧ |log2FC|>τ)",
                  logy=True)

    # ---------- MA plot helpers ----------
    mean_ctrl = data.get('mean_expr_ctrl_per_gene', pd.Series(dtype=np.float32))

    def _ma_setup(ax, l2fc_df):
        if mean_ctrl is None or mean_ctrl.empty:
            ax.text(0.5, 0.5, '(no NT mean expression available)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_axis_off(); return None
        common = l2fc_df.columns.intersection(mean_ctrl.index)
        if len(common) == 0:
            ax.text(0.5, 0.5, '(no shared genes between l2fc & ctrl-mean)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_axis_off(); return None
        mean_x = mean_ctrl.loc[common].values.astype(np.float32)
        Y      = l2fc_df[common].values.astype(np.float32)
        return mean_x, Y, common

    def _ma_render(ax, mean_x, x_flat, y_flat, keep, label, title_extra):
        if not keep.any():
            ax.text(0.5, 0.5, '(no points to plot)',
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
            ax.set_axis_off(); return
        x_lo = float(np.nanmin(mean_x)); x_hi = float(np.nanmax(mean_x))
        ax.hexbin(x_flat[keep], y_flat[keep], gridsize=80, mincnt=1,
                  cmap='viridis', bins='log',
                  extent=(x_lo, x_hi, -ma_ylim, ma_ylim))
        ax.axhline(0, color='crimson', lw=0.6, ls='--')
        ax.set_xlim(x_lo, x_hi)
        ax.set_ylim(-ma_ylim, ma_ylim)
        ax.set_xlabel('mean log1p expression of gene in NT cells of this drug')
        ax.set_ylabel(f"{label}'s  $\\log_2\\,\\mathrm{{FC}}$")
        ax.set_title(f"{label}'s MA plot — {title_extra}", fontsize=10)
        ax.grid(alpha=0.3)

    def _ma_panel_all(ax, l2fc_df, label):
        setup = _ma_setup(ax, l2fc_df)
        if setup is None: return
        mean_x, Y, _ = setup
        n_perts, n_genes = Y.shape
        x_flat = np.tile(mean_x, n_perts)
        y_flat = Y.ravel()
        keep = np.isfinite(x_flat) & np.isfinite(y_flat)
        _ma_render(ax, mean_x, x_flat, y_flat, keep, label,
                   f'all (target, gene) cells  ({n_perts:,} × {n_genes:,})')

    def _ma_panel_sig(ax, l2fc_df, sig_dict, label, threshold_str):
        setup = _ma_setup(ax, l2fc_df)
        if setup is None: return
        mean_x, Y, common = setup
        n_perts, n_genes = Y.shape
        gene_to_col = {g: i for i, g in enumerate(common.astype(str).tolist())}
        sig_mask = np.zeros_like(Y, dtype=bool)
        for row_i, tg in enumerate(l2fc_df.index.astype(str).tolist()):
            per_tau = sig_dict.get(tg)
            if per_tau is None: continue
            for g in per_tau.get(0.0, ()):
                ci = gene_to_col.get(g)
                if ci is not None: sig_mask[row_i, ci] = True
        x_flat = np.tile(mean_x, n_perts)
        y_flat = Y.ravel()
        keep   = sig_mask.ravel() & np.isfinite(x_flat) & np.isfinite(y_flat)
        n_sig  = int(keep.sum())
        _ma_render(ax, mean_x, x_flat, y_flat, keep, label,
                   f'significant only ({threshold_str}; n={n_sig:,})')

    axes[2, 0].set_axis_off()
    _ma_panel_all(axes[2, 1], data['our_l2fc'],   'Basak')
    _ma_panel_all(axes[2, 2], data['their_l2fc'], 'Tyler')

    axes[3, 0].set_axis_off()
    _ma_panel_sig(axes[3, 1], data['our_l2fc'],   data['our_sig'],
                  'Basak', f'FDR<{FDR_THR}')
    _ma_panel_sig(axes[3, 2], data['their_l2fc'], data['their_sig'],
                  'Tyler', f'padj≤{PADJ_THR}')

    fig.suptitle(f"{spec['our_drug']}  [{spec['set']}]   "
                 f"(Tyler's drug name: {spec['their_drug']})",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    return fig


def plot_sig_counts_per_pert(spec: dict, jacc_drug: pd.DataFrame, data: dict, cfg: dict,
                              *, n_cols: int = 6):
    """Second per-drug figure: grid of scatter panels, one per τ in THRESHOLDS_LOG2.
    Each panel:  x = Basak's # sig genes per target, y = Tyler's # sig genes per target.
    Colored by cell-count bin.  Crimson dashed = y = x.  Symlog axes (linthresh=1) so
    counts of 0 sit cleanly on the axis.

    'Significant' uses the joint criterion at that τ:
        Basak  : FDR  < cfg['fdr_thr']  ∧ |log2FC| > τ
        Tyler  : padj ≤ cfg['padj_thr'] ∧ |log2FC| > τ
    """
    FDR_THR  = cfg['fdr_thr']
    PADJ_THR = cfg['padj_thr']

    n_series = pd.Series(data['n_cells_per_pert'], name='n_cells')
    pert_bin = pd.cut(n_series, bins=N_CELL_BINS, labels=N_CELL_LABELS, include_lowest=True)
    pert_to_bin = pert_bin.dropna().astype(str).to_dict()
    jdf = jacc_drug.copy()
    jdf['n_cells_bin'] = jdf['target_gene'].map(pert_to_bin)
    jdf = jdf[jdf['n_cells_bin'].notna()]

    taus = sorted(jdf['tau_log2'].unique())
    n_taus = len(taus)
    n_rows = (n_taus + n_cols - 1) // n_cols

    palette = sns.color_palette('viridis', n_colors=len(N_CELL_LABELS))
    color_map = dict(zip(N_CELL_LABELS, palette))

    overall_max = max(int(jdf['n_our_sig'].max()),
                      int(jdf['n_their_sig'].max()), 1)
    lim = overall_max * 1.10

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(3.0 * n_cols, 3.0 * n_rows),
                              sharex=True, sharey=True)
    axes_flat = np.atleast_1d(axes).flatten()

    for i, tau in enumerate(taus):
        ax = axes_flat[i]
        sub = jdf[jdf['tau_log2'] == tau]
        for lab in N_CELL_LABELS:
            bin_sub = sub[sub['n_cells_bin'] == lab]
            if bin_sub.empty:
                continue
            ax.scatter(bin_sub['n_our_sig'], bin_sub['n_their_sig'],
                       s=5, alpha=0.5, color=color_map[lab], edgecolors='none')
        ax.plot([0, lim], [0, lim], ls='--', color='crimson', lw=0.7)
        ax.set_xscale('symlog', linthresh=1)
        ax.set_yscale('symlog', linthresh=1)
        ax.set_xlim(-0.5, lim)
        ax.set_ylim(-0.5, lim)
        ax.set_title(f'τ = {tau:.3g}   (n={len(sub):,} perts)', fontsize=9)
        ax.grid(alpha=0.25, which='major')

    for i in range(n_taus, len(axes_flat)):
        axes_flat[i].set_axis_off()

    fig.supxlabel("# Basak's sig genes per target   (symlog)", fontsize=11)
    fig.supylabel("# Tyler's sig genes per target   (symlog)", fontsize=11)

    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=color_map[lab],
                           markeredgecolor='none', markersize=8, label=lab)
               for lab in N_CELL_LABELS]
    fig.legend(handles=handles, loc='lower right',
               bbox_to_anchor=(0.995, 0.005),
               title='# cells (Basak)', fontsize=9, ncol=1, frameon=True)

    fig.suptitle(
        f"{spec['our_drug']}  [{spec['set']}]   "
        f"Per-pert sig-gene count: Basak vs Tyler at each |log2FC| threshold τ\n"
        f"(FDR<{FDR_THR} for Basak, padj≤{PADJ_THR} for Tyler; one panel per τ; "
        f"crimson = y=x)",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Worker entry point (must be picklable / importable at top level)
# --------------------------------------------------------------------------
def process_drug(job_args):
    spec, cfg, per_drug_pdf_dir = job_args
    drug = spec['our_drug']
    t0 = time.time()
    pid = os.getpid()

    try:
        print(f'[{drug}] pid={pid} start', flush=True)
        data = load_one_drug(spec, cfg)
        print(f'[{drug}] loaded in {time.time()-t0:.1f}s', flush=True)

        t1 = time.time()
        jacc_drug = compute_jaccard_for_drug(data, spec)
        corr = compute_pooled_corr(data)
        print(f'[{drug}] jaccard+corr in {time.time()-t1:.1f}s  ρ={corr["pearson"]:.3f}', flush=True)

        t2 = time.time()
        # Two pages per drug: main 4×3 figure, then the per-τ Basak-vs-Tyler-count grid.
        fig1 = plot_drug(spec, data, jacc_drug, corr, cfg)
        fig2 = plot_sig_counts_per_pert(spec, jacc_drug, data, cfg)
        pdf_path = per_drug_pdf_dir / f'{drug}.pdf'
        with PdfPages(pdf_path) as pdf:
            pdf.savefig(fig1, bbox_inches='tight')
            pdf.savefig(fig2, bbox_inches='tight')
        plt.close(fig1)
        plt.close(fig2)
        print(f'[{drug}] plot+save in {time.time()-t2:.1f}s', flush=True)

        # Universe stats
        n_shared_targets = len(set(data['their_sig']) & set(data['our_sig']))
        pp_intersects = [
            len(data['their_tested'][tg] & data['our_tested'][tg])
            for tg in (set(data['their_tested']) & set(data['our_tested']))
        ]
        pp_arr = np.asarray(pp_intersects) if pp_intersects else np.array([0])
        universe = {
            'batch':                spec['set'],
            'drug_ours':            spec['our_drug'],
            'drug_theirs':          spec['their_drug'],
            'n_ours_genes':         data['our_l2fc'].shape[1],
            'n_theirs_genes':       data['their_l2fc'].shape[1],
            'n_shared_targets':     n_shared_targets,
            'pp_intersect_min':     int(pp_arr.min()),
            'pp_intersect_median':  int(np.median(pp_arr)),
            'pp_intersect_max':     int(pp_arr.max()),
        }
        corr_row = {'batch': spec['set'], 'drug_ours': spec['our_drug'],
                    'drug_theirs': spec['their_drug'], **corr}

        del data; gc.collect()
        print(f'[{drug}] DONE in {time.time()-t0:.1f}s', flush=True)
        return {
            'drug_ours': drug,
            'pdf_path':  pdf_path,
            'jacc':      jacc_drug,
            'corr':      corr_row,
            'universe':  universe,
            'elapsed':   time.time() - t0,
            'error':     None,
        }
    except Exception as e:
        tb = traceback.format_exc()
        print(f'[{drug}] FAILED: {e}\n{tb}', flush=True)
        return {
            'drug_ours': drug,
            'pdf_path':  None,
            'jacc':      None,
            'corr':      None,
            'universe':  None,
            'elapsed':   time.time() - t0,
            'error':     str(e),
        }


# --------------------------------------------------------------------------
# PDF merge
# --------------------------------------------------------------------------
def merge_pdfs(pdf_paths: list[Path], out_path: Path) -> bool:
    """Concatenate per-drug PDFs into one. Returns True on success."""
    pdf_paths = [p for p in pdf_paths if p is not None and Path(p).exists()]
    if not pdf_paths:
        print('No per-drug PDFs to merge.')
        return False
    # Try pypdf first (modern), then PyPDF2 (legacy).
    try:
        from pypdf import PdfWriter
        writer = PdfWriter()
        for p in pdf_paths:
            writer.append(str(p))
        with open(out_path, 'wb') as f:
            writer.write(f)
        writer.close()
        return True
    except ImportError:
        pass
    try:
        from PyPDF2 import PdfMerger
        merger = PdfMerger()
        for p in pdf_paths:
            merger.append(str(p))
        with open(out_path, 'wb') as f:
            merger.write(f)
        merger.close()
        return True
    except ImportError:
        pass
    print('Neither `pypdf` nor `PyPDF2` is installed; cannot merge.')
    print(f'Per-drug PDFs left at:  {pdf_paths[0].parent}')
    return False


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def build_argparser():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--drugs', nargs='+', default=None,
                    help='Subset of drugs (our_drug names). Default: all 16.')
    ap.add_argument('--n-workers', type=int, default=8,
                    help='Number of parallel worker processes (one drug per worker).')
    ap.add_argument('--filter', default='25pct',
                    help="Tyler's KD-quantile filter (default 25pct).")
    ap.add_argument('--fdr-thr', type=float, default=0.01)
    ap.add_argument('--padj-thr', type=float, default=0.01)
    ap.add_argument('--nt-subsample', type=int, default=20_000,
                    help='Subsample N NT cells per drug for the mean-expr panel. '
                         'Default 20000. Pass 0 (or a negative number) to use all NT cells.')
    ap.add_argument('--out-dir', type=Path,
                    default=DEFAULT_OUR_FDR_DIR / 'jaccard_outputs',
                    help='Output directory (default: FDR_matrices/jaccard_outputs/).')
    ap.add_argument('--our-fdr-dir', type=Path, default=DEFAULT_OUR_FDR_DIR)
    ap.add_argument('--our-pm-dir',  type=Path, default=DEFAULT_OUR_PM_DIR)
    ap.add_argument('--our-raw-dir', type=Path, default=DEFAULT_OUR_RAW_DIR)
    ap.add_argument('--set1-dir',    type=Path, default=DEFAULT_SET1_DIR)
    ap.add_argument('--set2-dir',    type=Path, default=DEFAULT_SET2_DIR)
    ap.add_argument('--start-method', choices=['fork', 'spawn'], default='fork',
                    help='Multiprocessing start method. fork is faster on Linux; '
                         'spawn is safer if you see hangs.')
    return ap


def main():
    args = build_argparser().parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_drug_pdf_dir = out_dir / 'per_drug_pdfs'
    per_drug_pdf_dir.mkdir(exist_ok=True)

    all_specs = build_drug_specs(args.set1_dir, args.set2_dir)
    valid_names = {s['our_drug'] for s in all_specs}
    if args.drugs:
        unknown = set(args.drugs) - valid_names
        if unknown:
            raise SystemExit(f'Unknown drug name(s): {sorted(unknown)}\n'
                             f'Known: {sorted(valid_names)}')
        specs_to_run = [s for s in all_specs if s['our_drug'] in set(args.drugs)]
    else:
        specs_to_run = all_specs

    nt_subsample = args.nt_subsample if args.nt_subsample > 0 else None
    cfg = {
        'contrast':     'target',
        'filter':       args.filter,
        'fdr_thr':      args.fdr_thr,
        'padj_thr':     args.padj_thr,
        'our_fdr_dir':  args.our_fdr_dir,
        'our_pm_dir':   args.our_pm_dir,
        'our_raw_dir':  args.our_raw_dir,
        'nt_subsample': nt_subsample,
    }

    n_workers = max(1, min(args.n_workers, len(specs_to_run)))
    print(f'Running {len(specs_to_run)} drug(s) with {n_workers} worker(s); '
          f'nt_subsample={nt_subsample}; out={out_dir}')
    print(f'Drugs: {[s["our_drug"] for s in specs_to_run]}', flush=True)

    job_args = [(spec, cfg, per_drug_pdf_dir) for spec in specs_to_run]

    t_total = time.time()
    if n_workers == 1:
        results = [process_drug(j) for j in job_args]
    else:
        ctx = mp.get_context(args.start_method)
        with ctx.Pool(n_workers, maxtasksperchild=1) as pool:
            # imap_unordered prints results as soon as each drug finishes
            results = []
            for r in pool.imap_unordered(process_drug, job_args):
                results.append(r)

    # Sort results to match the input drug order
    by_drug = {r['drug_ours']: r for r in results}
    ordered = [by_drug[s['our_drug']] for s in specs_to_run if s['our_drug'] in by_drug]

    failed = [r for r in ordered if r['error'] is not None]
    succeeded = [r for r in ordered if r['error'] is None]
    print(f'\n=== {len(succeeded)} succeeded, {len(failed)} failed   '
          f'(wall: {time.time()-t_total:.1f}s) ===')
    for r in failed:
        print(f'  FAILED: {r["drug_ours"]}: {r["error"]}')

    if not succeeded:
        raise SystemExit('No drugs completed successfully.')

    # Merge per-drug PDFs into the combined one (in input order)
    combined_pdf = out_dir / 'per_drug_figures_combined.pdf'
    pdf_paths = [r['pdf_path'] for r in succeeded]
    if merge_pdfs(pdf_paths, combined_pdf):
        sz = combined_pdf.stat().st_size / 1e6
        print(f'\nwrote combined PDF: {combined_pdf}  ({sz:.1f} MB, {len(pdf_paths)} pages)')
    else:
        print(f'\nper-drug PDFs left at: {per_drug_pdf_dir}')

    # Aggregate CSVs
    jacc_df = pd.concat([r['jacc'] for r in succeeded], ignore_index=True)
    corr_df = pd.DataFrame([r['corr'] for r in succeeded])
    universe_df = pd.DataFrame([r['universe'] for r in succeeded])
    summary_df = corr_df.merge(universe_df,
                                on=['batch', 'drug_ours', 'drug_theirs'],
                                how='left').sort_values(['batch', 'pearson'],
                                                          ascending=[True, False])

    jacc_path    = out_dir / 'jaccard_per_target.csv'
    summary_path = out_dir / 'per_drug_summary.csv'
    jacc_df.to_csv(jacc_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    print(f'wrote {jacc_path}  ({len(jacc_df):,} rows)')
    print(f'wrote {summary_path}  ({len(summary_df)} drugs)')


if __name__ == '__main__':
    main()
