#!/usr/bin/env python3
"""
Compare DMSO-replicate concordance against DMSO-vs-drug concordance, as a
function of a |logFC| threshold tau.

Background
----------
06_AssessConcordanceBetweenTechnicalReplicates / 20_CompareDEWithCollaboratorDESeq2
established a per-perturbation Jaccard concordance between the two DMSO replicates
(DMSO_round2 = batch1, DMSO_round2_batch2 = batch2): for each perturbation, sweep a
|posterior-mean logFC| threshold tau and Jaccard the hit-gene sets of the two
matrices (signed and unsigned). Outputs live in Notes/dmso_replicate_jaccard/.

This script reuses exactly that hit-set definition and threshold grid, but adds the
14 drug contexts. For every perturbation it computes, at each tau:

  * J_replicate(p, tau)  = Jaccard(DMSO_round2, DMSO_round2_batch2)        [cross-batch baseline]
  * J_drug(p, d, tau)    = Jaccard(DMSO_of_d's_batch, drug d)              [WITHIN-batch]

Within-batch pairing removes batch confounding, so a drop from J_replicate to
J_drug reflects the drug's modulation of the perturbation, not batch variation.

Significance
------------
At each tau, paired one-sided Wilcoxon signed-rank test (J_replicate > J_drug)
across perturbations matched between the replicate baseline and that drug. Run
per drug and for all drugs pooled. We then report, per drug and pooled:
  * the smallest tau at which the replicate is significantly more concordant
    (p < alpha) and stays significant for all larger tau,
  * the tau of maximum separation (largest median(J_rep - J_drug) among
    significant tau),
  * effect sizes (median difference, fraction of perts with J_rep > J_drug).

Hit-set definition (matches the existing replicate sweep -- |logFC| only, no FDR gate):
  unsigned: hit_A = |A| > tau ; inter = hit_A & hit_B ; union = hit_A | hit_B
  signed  : inter = (A>tau & B>tau) | (A<-tau & B<-tau) ; union = (|A|>tau)|(|B|>tau)

Outputs (default Notes/dmso_vs_drug_concordance/):
  jaccard_long_unsigned.csv / jaccard_long_signed.csv   per (comparison, pert, tau)
  significance_unsigned.csv / significance_signed.csv    per (drug|POOLED, tau): p, effect sizes
  crossover_summary.csv                                  per (mode, drug|POOLED): crossover + peak tau
  fig_jaccard_vs_tau_{mode}.png                          median Jaccard curves, replicate vs drugs
  fig_separation_vs_tau_{mode}.png                       median(J_rep-J_drug) and -log10 p vs tau
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
PROJ = Path("/home/beraslan/Projects/ChemoGeneticScreens")
PM_DIR = PROJ / "PosteriorMeanMatrices"
PERT_COUNTS = PROJ / "Notes" / "dmso_replicate_jaccard" / "pert_cell_counts.csv"

DMSO_B1 = "DMSO_round2"
DMSO_B2 = "DMSO_round2_batch2"

BATCH1_DRUGS = ["AR-A014418", "AZD4573", "CHIR-98014", "Lexibulin",
                "PP121", "Romidepsin", "Stattic"]
BATCH2_DRUGS = ["Bisindolylmaleimide-I", "DG-172", "JTE-607", "LDN-193189",
                "LY2090314", "NSC95397", "VX-11e"]
DMSO_OF_BATCH = {"batch1": DMSO_B1, "batch2": DMSO_B2}

# |logFC| (natural-log units) threshold grid -- matches the existing replicate sweep:
# [0.0] + logspace(log10(0.01), log10(1.0), 40)
N_THRESHOLDS = 40
THRESHOLDS = np.concatenate([[0.0],
                             np.logspace(np.log10(0.01), np.log10(1.0), N_THRESHOLDS)])

MODES = ("unsigned", "signed")
ALPHA = 0.05


# --------------------------------------------------------------------------- #
# Core Jaccard computation
# --------------------------------------------------------------------------- #
def pairwise_jaccard_wide(A: np.ndarray, B: np.ndarray, thresholds: np.ndarray,
                          mode: str):
    """Row-wise (per-perturbation) Jaccard of hit-gene sets between aligned
    matrices A and B, for every threshold.

    A, B : (n_pert, n_gene) float arrays, already aligned on rows and columns.
    Returns J, NA, NB, NI, NU each shaped (n_pert, n_thresholds).
    """
    absA, absB = np.abs(A), np.abs(B)
    n, T = A.shape[0], len(thresholds)
    J = np.full((n, T), np.nan, dtype=np.float64)
    NA = np.zeros((n, T), dtype=np.int32)
    NB = np.zeros((n, T), dtype=np.int32)
    NI = np.zeros((n, T), dtype=np.int32)
    NU = np.zeros((n, T), dtype=np.int32)
    for j, t in enumerate(thresholds):
        hitA = absA > t
        hitB = absB > t
        if mode == "unsigned":
            inter = hitA & hitB
        else:  # signed: same-direction agreement
            inter = ((A > t) & (B > t)) | ((A < -t) & (B < -t))
        union = hitA | hitB
        ni = inter.sum(axis=1)
        nu = union.sum(axis=1)
        NA[:, j] = hitA.sum(axis=1)
        NB[:, j] = hitB.sum(axis=1)
        NI[:, j] = ni
        NU[:, j] = nu
        with np.errstate(invalid="ignore", divide="ignore"):
            J[:, j] = np.where(nu > 0, ni / nu, np.nan)
    return J, NA, NB, NI, NU


def align(df_a: pd.DataFrame, df_b: pd.DataFrame):
    """Align two matrices on common perturbations (rows) and genes (cols)."""
    perts = df_a.index.intersection(df_b.index)
    genes = df_a.columns.intersection(df_b.columns)
    a = df_a.loc[perts, genes]
    b = df_b.loc[perts, genes]
    return a, b, perts


def jaccard_frame(ref_df, other_df, thresholds, mode, label, drug, batch):
    """Compute wide Jaccard (index=pert, cols=tau) plus a tidy long DataFrame."""
    a, b, perts = align(ref_df, other_df)
    J, NA, NB, NI, NU = pairwise_jaccard_wide(a.values, b.values, thresholds, mode)
    Jwide = pd.DataFrame(J, index=perts, columns=thresholds)

    # tidy long form
    nrow = len(perts)
    long = pd.DataFrame({
        "comparison": label,
        "drug": drug,
        "batch": batch,
        "perturbation": np.repeat(perts.values, len(thresholds)),
        "threshold": np.tile(thresholds, nrow),
        "n_sig_ref": NA.ravel(),
        "n_sig_other": NB.ravel(),
        "n_intersection": NI.ravel(),
        "n_union": NU.ravel(),
        "jaccard": J.ravel(),
    })
    return Jwide, long


# --------------------------------------------------------------------------- #
# Significance testing
# --------------------------------------------------------------------------- #
def paired_wilcoxon_greater(x: np.ndarray, y: np.ndarray):
    """One-sided paired Wilcoxon signed-rank: H1 = x stochastically > y.
    Returns (p_value, n_pairs, median_diff, frac_x_greater)."""
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = int(ok.sum())
    if n == 0:
        return np.nan, 0, np.nan, np.nan
    diff = x - y
    med_diff = float(np.median(diff))
    frac_greater = float(np.mean(diff > 0))
    nz = np.count_nonzero(diff)
    if nz == 0:
        return np.nan, n, med_diff, frac_greater
    try:
        _, p = stats.wilcoxon(x, y, alternative="greater", zero_method="wilcox")
    except ValueError:
        p = np.nan
    return float(p), n, med_diff, frac_greater


def run_significance(J_rep: pd.DataFrame, drug_J: dict[str, pd.DataFrame],
                     thresholds: np.ndarray):
    """For each drug and POOLED, paired Wilcoxon (J_rep > J_drug) per tau."""
    rows = []
    pooled_pairs = {t: ([], []) for t in thresholds}  # tau -> (rep_vals, drug_vals)

    for drug, Jd in drug_J.items():
        common = J_rep.index.intersection(Jd.index)
        for t in thresholds:
            xr = J_rep.loc[common, t].values
            yd = Jd.loc[common, t].values
            p, n, md, fg = paired_wilcoxon_greater(xr, yd)
            rows.append({
                "drug": drug, "threshold": t, "n_pairs": n,
                "median_J_rep": float(np.nanmedian(xr)) if n else np.nan,
                "median_J_drug": float(np.nanmedian(yd)) if n else np.nan,
                "median_diff": md, "frac_rep_greater": fg, "p_value": p,
            })
            ok = np.isfinite(xr) & np.isfinite(yd)
            pooled_pairs[t][0].append(xr[ok])
            pooled_pairs[t][1].append(yd[ok])

    # pooled across all drugs (each replicate value reused per drug -- matched pairs)
    for t in thresholds:
        xr = np.concatenate(pooled_pairs[t][0]) if pooled_pairs[t][0] else np.array([])
        yd = np.concatenate(pooled_pairs[t][1]) if pooled_pairs[t][1] else np.array([])
        p, n, md, fg = paired_wilcoxon_greater(xr, yd)
        rows.append({
            "drug": "POOLED", "threshold": t, "n_pairs": n,
            "median_J_rep": float(np.nanmedian(xr)) if n else np.nan,
            "median_J_drug": float(np.nanmedian(yd)) if n else np.nan,
            "median_diff": md, "frac_rep_greater": fg, "p_value": p,
        })

    sig = pd.DataFrame(rows)
    # BH-correct across all (drug x tau) tests within this mode
    finite = sig["p_value"].notna()
    sig["p_adj_bh"] = np.nan
    if finite.any():
        sig.loc[finite, "p_adj_bh"] = _bh(sig.loc[finite, "p_value"].values)
    return sig


def _bh(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def crossover_summary(sig: pd.DataFrame, mode: str, alpha: float = ALPHA):
    """Per drug/POOLED: smallest tau that is significant and stays significant
    for all larger tau (stable crossover), plus tau of maximum separation."""
    rows = []
    for drug, sub in sig.groupby("drug"):
        sub = sub[sub["threshold"] > 0].sort_values("threshold")
        taus = sub["threshold"].values
        padj = sub["p_adj_bh"].values
        mdiff = sub["median_diff"].values
        sigmask = (padj < alpha)

        # stable crossover: smallest tau such that it and all larger taus are sig
        stable_tau = np.nan
        sig_suffix = np.array([sigmask[i:].all() for i in range(len(sigmask))])
        if sig_suffix.any():
            stable_tau = float(taus[np.argmax(sig_suffix)])

        first_sig = float(taus[np.argmax(sigmask)]) if sigmask.any() else np.nan

        # tau of maximum separation among significant taus (fallback: overall)
        if sigmask.any():
            idx_pool = np.where(sigmask)[0]
        else:
            idx_pool = np.arange(len(taus))
        peak_i = idx_pool[np.nanargmax(mdiff[idx_pool])]

        rows.append({
            "mode": mode, "drug": drug,
            "first_significant_tau": first_sig,
            "stable_crossover_tau": stable_tau,
            "peak_separation_tau": float(taus[peak_i]),
            "peak_median_diff": float(mdiff[peak_i]),
            "peak_median_J_rep": float(sub["median_J_rep"].values[peak_i]),
            "peak_median_J_drug": float(sub["median_J_drug"].values[peak_i]),
            "peak_frac_rep_greater": float(sub["frac_rep_greater"].values[peak_i]),
            "n_significant_taus": int(sigmask.sum()),
            "n_taus": int(len(taus)),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def plot_jaccard_vs_tau(J_rep, drug_J, batch_of, mode, out_png):
    fig, ax = plt.subplots(figsize=(8, 6))
    taus = THRESHOLDS[THRESHOLDS > 0]
    # replicate baseline -- bold black
    med_rep = np.nanmedian(J_rep[taus].values, axis=0)
    ax.plot(taus, med_rep, color="black", lw=3.0, zorder=10,
            label=f"DMSO replicate (b1 vs b2)  n={J_rep.shape[0]}")
    cmap = plt.get_cmap("tab20")
    for i, (drug, Jd) in enumerate(sorted(drug_J.items())):
        med = np.nanmedian(Jd[taus].values, axis=0)
        ls = "-" if batch_of[drug] == "batch1" else "--"
        ax.plot(taus, med, color=cmap(i % 20), lw=1.4, ls=ls, alpha=0.85,
                label=f"{drug} ({batch_of[drug]})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$|\log\,\mathrm{FC}|$ threshold $\tau$  (natural-log units, log scale)")
    ax.set_ylabel("median per-perturbation Jaccard")
    ax.set_title(f"DMSO-replicate vs DMSO-vs-drug concordance  ({mode})\n"
                 "solid = batch1 (vs DMSO_round2), dashed = batch2 (vs DMSO_round2_batch2)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def plot_separation_vs_tau(sig, mode, out_png, alpha=ALPHA):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    taus_all = np.sort(sig["threshold"].unique())
    taus = taus_all[taus_all > 0]
    cmap = plt.get_cmap("tab20")
    drugs = [d for d in sorted(sig["drug"].unique()) if d != "POOLED"]

    for i, drug in enumerate(drugs):
        sub = sig[(sig["drug"] == drug) & (sig["threshold"] > 0)].sort_values("threshold")
        axes[0].plot(sub["threshold"], sub["median_diff"], color=cmap(i % 20),
                     lw=1.3, alpha=0.85, label=drug)
        axes[1].plot(sub["threshold"], -np.log10(sub["p_adj_bh"].clip(lower=1e-300)),
                     color=cmap(i % 20), lw=1.3, alpha=0.85, label=drug)
    pooled = sig[(sig["drug"] == "POOLED") & (sig["threshold"] > 0)].sort_values("threshold")
    axes[0].plot(pooled["threshold"], pooled["median_diff"], color="black", lw=3, label="POOLED")
    axes[1].plot(pooled["threshold"], -np.log10(pooled["p_adj_bh"].clip(lower=1e-300)),
                 color="black", lw=3, label="POOLED")

    axes[0].axhline(0, color="gray", lw=0.8, ls=":")
    axes[0].set_ylabel(r"median$(J_{\mathrm{rep}} - J_{\mathrm{drug}})$  (separation)")
    axes[0].set_title(f"Concordance separation vs $\\tau$  ({mode})")
    axes[1].axhline(-np.log10(alpha), color="red", lw=1.0, ls="--",
                    label=f"BH p={alpha}")
    axes[1].set_ylabel(r"$-\log_{10}$ BH-adjusted p (paired Wilcoxon, $J_{rep}>J_{drug}$)")
    axes[1].set_title(f"Significance vs $\\tau$  ({mode})")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel(r"$|\log\,\mathrm{FC}|$ threshold $\tau$ (log scale)")
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=6, ncol=2, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pm-dir", type=Path, default=PM_DIR)
    ap.add_argument("--out-dir", type=Path,
                    default=PROJ / "Notes" / "dmso_vs_drug_concordance")
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--min-cells", type=int, default=0,
                    help="optional: restrict to perturbations with >= this many "
                         "cells (min over the two replicates), via pert_cell_counts.csv")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    batch_of = {d: "batch1" for d in BATCH1_DRUGS}
    batch_of.update({d: "batch2" for d in BATCH2_DRUGS})
    all_ctx = [DMSO_B1, DMSO_B2] + BATCH1_DRUGS + BATCH2_DRUGS

    print(f"Loading {len(all_ctx)} posterior-mean matrices from {args.pm_dir} ...")
    mats = {}
    for ctx in all_ctx:
        fp = args.pm_dir / f"PosteriorMean_matrix_{ctx}.csv"
        mats[ctx] = pd.read_csv(fp, index_col=0)
        print(f"  {ctx:24s} {mats[ctx].shape}")

    keep_perts = None
    if args.min_cells > 0 and PERT_COUNTS.exists():
        pc = pd.read_csv(PERT_COUNTS, index_col=0)
        col = "n_cells_min" if "n_cells_min" in pc.columns else pc.columns[-1]
        keep_perts = set(pc.index[pc[col] >= args.min_cells].astype(str))
        print(f"min-cells={args.min_cells}: restricting to {len(keep_perts)} perts")
        mats = {k: v.loc[v.index.astype(str).isin(keep_perts)] for k, v in mats.items()}

    crossover_rows = []
    for mode in MODES:
        print(f"\n=== mode = {mode} ===")
        # replicate baseline
        Jrep_wide, rep_long = jaccard_frame(
            mats[DMSO_B1], mats[DMSO_B2], THRESHOLDS, mode,
            label="replicate", drug="DMSO_replicate", batch="cross")
        print(f"  replicate baseline: {Jrep_wide.shape[0]} perts")

        drug_J = {}
        long_parts = [rep_long]
        for drug in BATCH1_DRUGS + BATCH2_DRUGS:
            b = batch_of[drug]
            dmso = DMSO_OF_BATCH[b]
            Jd_wide, d_long = jaccard_frame(
                mats[dmso], mats[drug], THRESHOLDS, mode,
                label="drug", drug=drug, batch=b)
            drug_J[drug] = Jd_wide
            long_parts.append(d_long)
            print(f"  {drug:24s} ({b}, vs {dmso}): {Jd_wide.shape[0]} perts")

        # save long jaccard
        long_df = pd.concat(long_parts, ignore_index=True)
        long_fp = args.out_dir / f"jaccard_long_{mode}.csv"
        long_df.to_csv(long_fp, index=False)
        print(f"  wrote {long_fp}  ({len(long_df):,} rows)")

        # significance
        sig = run_significance(Jrep_wide, drug_J, THRESHOLDS)
        sig_fp = args.out_dir / f"significance_{mode}.csv"
        sig.to_csv(sig_fp, index=False)
        print(f"  wrote {sig_fp}")

        # crossover summary
        cs = crossover_summary(sig, mode, alpha=args.alpha)
        crossover_rows.append(cs)

        # figures
        plot_jaccard_vs_tau(Jrep_wide, drug_J, batch_of, mode,
                            args.out_dir / f"fig_jaccard_vs_tau_{mode}.png")
        plot_separation_vs_tau(sig, mode,
                               args.out_dir / f"fig_separation_vs_tau_{mode}.png",
                               alpha=args.alpha)

        # concise console readout for the pooled test
        pooled = cs[cs["drug"] == "POOLED"].iloc[0]
        print(f"  POOLED  stable-crossover tau = {pooled['stable_crossover_tau']}, "
              f"peak-separation tau = {pooled['peak_separation_tau']:.4f} "
              f"(median diff = {pooled['peak_median_diff']:.4f})")

    summary = pd.concat(crossover_rows, ignore_index=True)
    summary_fp = args.out_dir / "crossover_summary.csv"
    summary.to_csv(summary_fp, index=False)
    print(f"\nwrote {summary_fp}")
    print("\n==== crossover summary ====")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
