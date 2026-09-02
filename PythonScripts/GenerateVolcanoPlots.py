#!/usr/bin/env python3
"""
Volcano plots per drug context.

For each drug:
  x  =  shrunken logFC  (from PosteriorMeanMatrices/PosteriorMean_matrix_<drug>.csv,
                         row = perturbation, col = gene)
  y  =  -log10(p_value) (from DATA/<drug>/<drug>_res.csv, long-format
                         (target, feature, p_value) — multiple rows per
                         (target, feature) collapsed to the minimum p_value)

The _res.csv files are huge (10s of millions of rows) so we stream them in
chunks and write a per-drug parquet cache (`<drug>_minp.parquet`) so re-runs
are fast. Use --refresh to recompute.

Usage:
  python GenerateVolcanoPlots.py                              # all drugs, default paths
  python GenerateVolcanoPlots.py --drugs AZD4573 Stattic      # subset
  python GenerateVolcanoPlots.py --grid                       # also write a 4x4 grid of all drugs
  python GenerateVolcanoPlots.py --refresh                    # ignore cached minp parquet

Defaults:
  PosteriorMeanMatrices/PosteriorMean_matrix_<drug>.csv
  DATA/<drug>/<drug>_res.csv
  Figures/Volcano/  (output)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


# ----------------- p_value aggregation -----------------

def aggregate_minp(res_csv: Path, chunksize: int = 2_000_000, verbose: bool = True) -> pd.DataFrame:
    """Stream the long _res.csv and collapse to one row per (target, feature)
    holding the minimum p_value across source files (or any duplicate
    entries). Returns columns: target, feature, p_value, n_obs."""
    t0 = time.time()
    parts = []
    rows_total = 0
    for i, chunk in enumerate(pd.read_csv(
        res_csv,
        usecols=["target", "feature", "p_value"],
        dtype={"target": str, "feature": str, "p_value": np.float64},
        chunksize=chunksize,
    )):
        rows_total += len(chunk)
        g = chunk.groupby(["target", "feature"], sort=False, observed=True)
        gdf = pd.DataFrame({
            "p_value": g["p_value"].min(),
            "n_obs":   g.size(),
        }).reset_index()
        parts.append(gdf)
        if verbose:
            print(f"    [{i:02d}] read {len(chunk):,} rows  "
                  f"(cum {rows_total:,})  elapsed {time.time()-t0:.1f}s", flush=True)
    if not parts:
        raise RuntimeError(f"No rows read from {res_csv}")
    combined = pd.concat(parts, ignore_index=True)
    g = combined.groupby(["target", "feature"], sort=False, observed=True)
    out = pd.DataFrame({
        "p_value": g["p_value"].min(),
        "n_obs":   g["n_obs"].sum(),
    }).reset_index()
    if verbose:
        print(f"    aggregated to {len(out):,} (target × feature) rows "
              f"({time.time()-t0:.1f}s)", flush=True)
    return out


def get_or_build_minp(res_csv: Path, cache_dir: Path, *,
                      refresh: bool = False, chunksize: int = 2_000_000) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{res_csv.stem.replace('_res','')}_minp.parquet"
    if cache.exists() and not refresh:
        print(f"  [cache] reading {cache.name}", flush=True)
        return pd.read_parquet(cache)
    minp = aggregate_minp(res_csv, chunksize=chunksize)
    minp.to_parquet(cache)
    print(f"  [cache] wrote {cache.name}  ({len(minp):,} rows)", flush=True)
    return minp


# ----------------- volcano plot -----------------

def _merge_logfc_and_p(pm_long: pd.DataFrame, minp: pd.DataFrame) -> pd.DataFrame:
    df = pd.merge(pm_long, minp, on=["target", "feature"], how="inner")
    df["nlogp"] = -np.log10(np.clip(df["p_value"].values, 1e-300, 1.0))
    return df


def _pm_to_long(pm: pd.DataFrame) -> pd.DataFrame:
    """Stack a (perturbation × gene) matrix to long [target, feature, logFC]."""
    long = pm.stack(dropna=True).rename("logFC").reset_index()
    long.columns = ["target", "feature", "logFC"]
    return long


def plot_volcano_for_drug(
    drug: str,
    pm_dir: Path,
    data_dir: Path,
    cache_dir: Path,
    out_dir: Path,
    *,
    p_threshold: float = 0.05,
    logfc_threshold: float = 0.5,
    refresh: bool = False,
    chunksize: int = 2_000_000,
    sample_n: int | None = None,
    ax: plt.Axes | None = None,
) -> pd.DataFrame:
    """Generate the volcano plot for one drug. Returns the merged dataframe."""
    pm_path  = pm_dir / f"PosteriorMean_matrix_{drug}.csv"
    res_path = data_dir / drug / f"{drug}_res.csv"
    if not pm_path.exists():
        raise FileNotFoundError(pm_path)
    if not res_path.exists():
        raise FileNotFoundError(res_path)

    print(f"\n=== {drug} ===")
    print(f"  loading posterior matrix: {pm_path.name}", flush=True)
    pm = pd.read_csv(pm_path, index_col=0)
    pm_long = _pm_to_long(pm)
    print(f"  posterior matrix: {pm.shape[0]:,} perts × {pm.shape[1]:,} genes  "
          f"({len(pm_long):,} cells)", flush=True)

    print(f"  aggregating min p_value across {res_path.name}", flush=True)
    minp = get_or_build_minp(res_path, cache_dir, refresh=refresh, chunksize=chunksize)

    df = _merge_logfc_and_p(pm_long, minp)
    print(f"  merged: {len(df):,} (pert × gene) cells with both logFC and p_value", flush=True)

    # Categorize cells
    pos_sig = (df["p_value"] < p_threshold) & (df["logFC"] >  logfc_threshold)
    neg_sig = (df["p_value"] < p_threshold) & (df["logFC"] < -logfc_threshold)
    other   = ~(pos_sig | neg_sig)

    n_pos = int(pos_sig.sum())
    n_neg = int(neg_sig.sum())
    n_oth = int(other.sum())
    print(f"  {n_pos:,} positive-sig | {n_neg:,} negative-sig | {n_oth:,} other", flush=True)

    # Optional sub-sample of the grey background for plotting speed
    if sample_n is not None and other.sum() > sample_n:
        other_idx = df.index[other]
        keep_other = np.random.default_rng(0).choice(other_idx, size=sample_n, replace=False)
        plot_mask_other = pd.Series(False, index=df.index)
        plot_mask_other.loc[keep_other] = True
    else:
        plot_mask_other = other

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        owns_fig = True
    else:
        owns_fig = False

    ax.scatter(df.loc[plot_mask_other, "logFC"], df.loc[plot_mask_other, "nlogp"],
               s=2, alpha=0.04, color="lightgrey", rasterized=True,
               label=f"n.s.  (n={n_oth:,})")
    ax.scatter(df.loc[neg_sig, "logFC"], df.loc[neg_sig, "nlogp"],
               s=3, alpha=0.25, color="cornflowerblue", rasterized=True,
               label=f"down  (n={n_neg:,})")
    ax.scatter(df.loc[pos_sig, "logFC"], df.loc[pos_sig, "nlogp"],
               s=3, alpha=0.25, color="crimson", rasterized=True,
               label=f"up  (n={n_pos:,})")

    ax.axhline(-math.log10(p_threshold), ls="--", color="black", lw=0.6)
    ax.axvline( logfc_threshold, ls="--", color="black", lw=0.6)
    ax.axvline(-logfc_threshold, ls="--", color="black", lw=0.6)
    ax.set_xlabel("shrunken logFC  (ashr posterior mean)")
    ax.set_ylabel(r"$-\log_{10}(p\mathrm{-value})$")
    ax.set_title(f"{drug}   ({len(df):,} (pert × gene) cells)")
    ax.legend(loc="upper right", fontsize=8, markerscale=2)

    if owns_fig:
        plt.tight_layout()
        out_fp = out_dir / f"volcano_{drug}.png"
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_fp, dpi=150, bbox_inches="tight")
        print(f"  wrote {out_fp}")
        plt.close()
    return df


def plot_volcano_grid(
    drugs: list[str],
    pm_dir: Path, data_dir: Path, cache_dir: Path, out_dir: Path,
    *, p_threshold: float, logfc_threshold: float,
    refresh: bool, chunksize: int, sample_n: int | None,
):
    n = len(drugs)
    n_cols = 4
    n_rows = int(math.ceil(n / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.5 * n_rows),
                              sharex=False, sharey=False)
    axes = np.array(axes).flatten()
    for ax, drug in zip(axes, drugs):
        try:
            plot_volcano_for_drug(
                drug, pm_dir, data_dir, cache_dir, out_dir,
                p_threshold=p_threshold,
                logfc_threshold=logfc_threshold,
                refresh=refresh,
                chunksize=chunksize,
                sample_n=sample_n,
                ax=ax,
            )
        except FileNotFoundError as e:
            ax.set_title(f"{drug}  (missing)")
            ax.text(0.5, 0.5, str(e), ha="center", va="center",
                     transform=ax.transAxes, fontsize=8, color="firebrick")
            ax.set_axis_off()
    for ax in axes[len(drugs):]:
        ax.set_axis_off()
    out_dir.mkdir(parents=True, exist_ok=True)
    grid_fp = out_dir / "volcano_grid_all_drugs.png"
    plt.tight_layout()
    plt.savefig(grid_fp, dpi=130, bbox_inches="tight")
    print(f"\nGrid figure → {grid_fp}")
    plt.close()


# ----------------- CLI -----------------

DEFAULT_DRUGS = [
    "AR-A014418", "AZD4573", "Bisindolylmaleimide-I", "CHIR-98014",
    "DG-172", "DMSO_round2", "DMSO_round2_batch2", "JTE-607",
    "LDN-193189", "LY2090314", "Lexibulin", "NSC95397",
    "PP121", "Romidepsin", "Stattic", "VX-11e",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pm-dir",     type=Path,
                     default=Path("/home/beraslan/Projects/ChemoGeneticScreens/PosteriorMeanMatrices"))
    ap.add_argument("--data-dir",   type=Path,
                     default=Path("/home/beraslan/Projects/ChemoGeneticScreens/DATA"))
    ap.add_argument("--out-dir",    type=Path,
                     default=Path("/home/beraslan/Projects/ChemoGeneticScreens/Figures/Volcano"))
    ap.add_argument("--cache-dir",  type=Path,
                     default=Path("/home/beraslan/Projects/ChemoGeneticScreens/Figures/Volcano/minp_cache"))
    ap.add_argument("--drugs",      nargs="+", default=None,
                     help=f"Subset of drugs to plot (default: all 16)")
    ap.add_argument("--p-threshold",    type=float, default=0.05)
    ap.add_argument("--logfc-threshold", type=float, default=0.5)
    ap.add_argument("--refresh",    action="store_true",
                     help="Ignore cached *_minp.parquet and recompute.")
    ap.add_argument("--chunksize",  type=int, default=2_000_000)
    ap.add_argument("--sample-n",   type=int, default=400_000,
                     help="Random sub-sample of non-significant cells to scatter "
                          "(for plotting speed; default 400k). Set to 0 to plot all.")
    ap.add_argument("--grid",       action="store_true",
                     help="Also generate a 4×4 grid figure with all drugs.")
    args = ap.parse_args()

    drugs = args.drugs or DEFAULT_DRUGS
    sample_n = None if args.sample_n in (0, None) else int(args.sample_n)

    print(f"PosteriorMean dir : {args.pm_dir}")
    print(f"DATA dir          : {args.data_dir}")
    print(f"output dir        : {args.out_dir}")
    print(f"cache dir         : {args.cache_dir}")
    print(f"drugs             : {drugs}")
    print(f"thresholds        : p < {args.p_threshold}  |  |logFC| > {args.logfc_threshold}")

    if args.grid:
        plot_volcano_grid(
            drugs, args.pm_dir, args.data_dir, args.cache_dir, args.out_dir,
            p_threshold=args.p_threshold,
            logfc_threshold=args.logfc_threshold,
            refresh=args.refresh,
            chunksize=args.chunksize,
            sample_n=sample_n,
        )
    else:
        for d in drugs:
            try:
                plot_volcano_for_drug(
                    d, args.pm_dir, args.data_dir, args.cache_dir, args.out_dir,
                    p_threshold=args.p_threshold,
                    logfc_threshold=args.logfc_threshold,
                    refresh=args.refresh,
                    chunksize=args.chunksize,
                    sample_n=sample_n,
                )
            except FileNotFoundError as e:
                print(f"  [skip] {d}: missing file {e}", flush=True)


if __name__ == "__main__":
    # Only force the headless Agg backend when running as a script — leave
    # the notebook's inline backend alone when this module is imported.
    matplotlib.use("Agg", force=True)
    main()
