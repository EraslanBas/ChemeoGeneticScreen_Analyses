# Scripts &amp; notebooks reference

Every real code file under `SRC/`, grouped by pipeline stage, with what it
actually does (read from each file's docstring/header, not guessed from the
filename). Skipped: `.ipynb_checkpoints/`, `__pycache__/`, `*.log`,
`nohup.out`, `Notebooks/Up` (stray 40KB JSON, not a notebook),
`Notebooks/06_InspectDMSOReplicateDiscrepancy.ipynb.bak`.

## Shared utilities / config

These exist in two places — `Notebooks/` and `RScripts/`/`PythonScripts/` —
so a notebook and a standalone script can each load them via a relative/cwd
path without reaching into the other's directory.

- `libraries.py` — common third-party imports (scanpy, matplotlib, numpy,
  pandas, anndata); identical in both locations.
- `util.py` — small shared helpers, e.g. `sample_adata()` (random cell
  subsampling) and `assess_on_target_knockdown()`; identical in both
  locations.
- `parameters.py` — project paths and the drug-condition name lists
  (`drugConditions`, `drugConditions_round2`, `drugConditions_round2_batch2`,
  etc.), including the "Orig" vs. corrected drug-name mappings. The two
  copies are equivalent — they differ only in how `drugConditions_round2`
  happens to be line-wrapped, not in its values.
- `Main.R` / `Conf.R` — R library imports and the `conditions` list used by
  legacy plotting notebooks; identical in both locations.
- `Utilities.R` — misc R helpers: `%ni%`, `extract_parts()` (parse
  `<pert>_<cells>_<umi>_de_results.csv` filenames), `computeR2()`,
  `save_pheatmap_pdf()`, `lappend()`, and a mouse→human gene-name converter
  (`convertMouseGeneList`, via biomaRt) plus DAVID/GO wrappers used by older
  analyses — some referenced paths look stale (e.g. `/home/jovyan/...`).

## 1. Preprocessing / normalization

- `PythonScripts/LogNormalizeData.py` — despite the name, this both
  normalizes (total-count target_sum + log1p, via `run_differential_expression`
  helpers) and actually runs `pdex.parallel_differential_expression`; runs
  side-effecting code at import time.
- `PythonScripts/CorrectBatch.py` — ComBat-style batch-effect correction on
  DMSO_round2; contains a large commented-out earlier version.
- `PythonScripts/GenerateControlCounts.py` — builds a common-gene-intersected
  count table across multiple h5ad files in backed mode (memory-safe var-name
  intersection).
- `PythonScripts/SplitBigAnndata.py` — splits one large h5ad into per-perturbation
  chunks (`max_per_file` cap), backed-mode reads.
- `PythonScripts/SubsetPerts.py` — one-off: filters RGFP condition to cells
  passing a knockdown-efficiency threshold and a preselected perturbation list.
- `PythonScripts/BuildDrugCaches.py` — builds the 16 per-drug AnnData caches used
  by the whole pathway-analysis branch: subsets to reduced-tensor perturbations
  + NT controls, restricts genes to the KEGG-pathway gene union plus
  quantile-bin-matched background genes, so all 16 caches share one gene
  universe.
- `PythonScripts/MakeChemogeneticLeidenSplit.py` — builds the Leiden-stratified
  train/val/test perturbation split (25/5/70) and the `state3` TOML config
  used for downstream perturbation-prediction model training.
- `PythonScripts/MovePCAFiles.py` — one-off file-mover for PCA result CSVs into
  a per-condition directory layout.
- `Notebooks/01_Data_outlook.ipynb` — initial QC / data exploration.
- `Notebooks/0_PlotGeneratedData.ipynb`, `0_PlotGeneratedDataR.ipynb` — early
  plotting of generated/simulated data (Python and R versions).
- `Notebooks/03_BatchEffectCorrection_combat.ipynb` — notebook version of
  ComBat batch correction.
- `Notebooks/05_AssessBatchEffects.ipynb` (R) — assesses batch effects via
  `Main.R`/`Utilities.R`/`Conf.R` helpers.

## 2. Differential expression

### R (glmGamPoi — Gamma-Poisson GLM on raw counts)

- `RScripts/RunGlmGamPoi_onePert.R` — fits glmGamPoi DE for one per-pert h5ad
  (perturbation cells + shared NT controls) at a time; atomic CSV write,
  skip-if-exists, safe under `xargs`-level parallelism.
- `RScripts/RunGlmGamPoi_subset.R` — same idea but reads one pre-subset h5ad
  covering many perturbations plus a `--perts-csv` allowlist; derives the SE
  from the 1-d.f. F-statistic (`SE = |lfc| / sqrt(f_statistic)`).
- `RScripts/ComputeDE_glmGAMPoi.r` — original per-file driver: loads one h5ad,
  refits a glmGamPoi model **per perturbation** against pooled controls.
- `RScripts/ComputeDE_glmGAMPoi_2.r` — refactor of the above: fits **one**
  joint design matrix across all perturbations (`min_cells` filter, `n_cores`
  parallel contrasts) instead of refitting per perturbation — much faster.
  Has a harmless typo (`n_ores`) in an unreached message-formatting branch.
- `RScripts/run_ashr_on_chunk.R` — 13-line `ashr::ash()` wrapper: reads a
  `(mean_diff, se)` chunk CSV, writes back the empirical-Bayes shrunk
  posterior result.

### Python (Welch t-test + ashr shrinkage path)

- `PythonScripts/ComputeSE.py` — streaming per-(perturbation, gene) Welch mean
  difference and SE computation on sparse expression data, chunked by
  perturbation count for parallel downstream `ashr` fitting. Docstring
  advertises `--no-manifest`/`--consolidate` flags and a `manifest.tsv`
  output that the function never actually writes (dead/unimplemented).
- `PythonScripts/ComputeSE_Replogle.py` — same Welch math (imported verbatim
  from `ComputeSE.py`) adapted for the Replogle_Basak cell-line datasets,
  which store raw counts and need in-memory log-normalization first.
- `PythonScripts/AshrGenerateMatrix.py` / `AshrGenerateMatrix.sh` — assembles
  per-chunk `AshrResult_chunk_*.csv` files into a single
  perturbation × gene `PosteriorMean_matrix.csv`.
- `PythonScripts/MergeAshrRes.py` — merges each `AshrResult_chunk_*.csv` back
  with its corresponding `chunk_*.csv` metadata (n_pert, n_ctrl, etc.).
- `PythonScripts/RunAshrPipeline_Replogle.py` — end-to-end orchestrator for
  the Replogle cell lines: `ComputeSE_Replogle.py` → `run_ashr_on_chunk.R`
  (parallel) → `MergeAshrRes.py` → `AshrGenerateMatrix.py`, each step
  skip-if-exists for resumability.
- `PythonScripts/Compute_DEwilcox.py` — CLI wrapper around
  `pdex.parallel_differential_expression` (Wilcoxon-based) for one input file
  vs. one control file.
- `PythonScripts/Compute_DEControl.py` — builds a **null DE background** by
  repeatedly sampling fake "perturbation" groups from the control population
  and running DE against the rest of the controls — the empirical null
  distribution for p-values/effect sizes.
- `PythonScripts/Compute_DEDrugVersusDMSO.py` — one-off: runs
  `parallel_differential_expression` for a fixed list of drug conditions vs.
  the DMSO_round2 control.
- `PythonScripts/ConcatDERes.py` — concatenates per-file DE result CSVs
  matching a glob pattern into one combined DataFrame, tagging origin file.
- `PythonScripts/PrepareDMSOb1SubsetAndWelchSE.py` — for the Welch-vs-glmGamPoi
  validation: samples 500 perturbations (stratified by cell count) from DMSO
  batch1, writes a subset h5ad, and computes the matching Welch (mean_diff,
  SE) as a parquet.
- `PythonScripts/PrepareWelchVsGlmGamPoi_v2.py` — v2 of the above: additionally
  writes per-perturbation h5ads (pert cells + a *fixed shared* NT-cell subset)
  so glmGamPoi and the Welch calculation see exactly the same cells.
- `PythonScripts/CompareDEWithCollaboratorDESeq2.py` — parallel per-drug
  comparison between this pipeline's ashr/FDR output and a collaborator's
  DESeq2-on-pseudobulk pipeline; renders a combined multi-page PDF.
- `PythonScripts/CompareDMSOReplicateVsDrugConcordance.py` — compares
  DMSO-replicate (cross-batch) concordance against DMSO-vs-drug concordance
  across a sweep of |logFC| thresholds, with a paired Wilcoxon signed-rank
  test — tests whether a drug's effect on a perturbation exceeds batch noise.
- `PythonScripts/GenerateVolcanoPlots.py` — per-drug volcano plots (shrunken
  logFC vs. −log10 p-value), with a parquet cache of the min-p-value per
  (target, feature) since the raw p-value files are tens of millions of rows.
- `PythonScripts/GenerateBetaMatrices.py` — pivots long-format per-drug result
  tables into wide perturbation × gene FDR matrices.

### Notebooks

- `Notebooks/CompareWelchVsGlmGamPoi.ipynb` — the actual comparison/plotting
  for the Welch-vs-glmGamPoi validation (Option A, DMSO batch1, 500 perts).
- `Notebooks/ComputeDEDrugs.ipynb` — interactive driver for
  `pdex.parallel_differential_expression` across drug conditions.
- `Notebooks/ComputeLogFCStandardErrors.ipynb` — interactive/exploratory
  version of the Welch SE computation.
- `Notebooks/06_AssessConcordanceBetweenTechnicalReplicates.ipynb` — pairwise
  Jaccard concordance between the three available DMSO FDR matrices
  (Round1, batch1, batch2), independent of posterior-mean values.
- `Notebooks/06_InspectDMSOReplicateDiscrepancy.ipynb` — sweeps a |logFC|
  threshold τ and measures DMSO-batch1-vs-batch2 discrepancy in the
  ashr-shrunken logFC vectors.
- `Notebooks/20_CompareDEWithCollaboratorDESeq2.ipynb` — notebook counterpart
  of `CompareDEWithCollaboratorDESeq2.py`, one self-contained analysis block
  per drug.
- `Notebooks/OLD/04_PlotDEGs_Drugs.ipynb`, `OLD/04_PlotDEGs_KnockDowns.ipynb`
  — legacy R plotting notebooks (superseded), kept for reference.

## 3. Significance tables

- `PythonScripts/GenerateGeneSignificanceTable.py` — for each drug's
  `_FDRs.csv` (perturbations × genes), counts how many perturbations call
  each gene significant (FDR &lt; threshold); output is genes × drugs.
- `PythonScripts/GeneratePerturbationSignificanceTable.py` — same idea
  transposed: perturbations × drugs, counting significant genes per
  perturbation.
- `Notebooks/02_PlotSignificantPerturbations.ipynb` — saves the union of
  perturbations significant in ≥1 drug context (`PertSignificance/
  union_significant_perts.csv`), consumed by notebooks 10/11/12.

## 4. Embeddings / geometry / distance

- `PythonScripts/ComputeDistancesBetweenConditions.py` — pools and
  subsamples control cells across conditions (`*_controls.h5ad`) for
  cross-condition distance comparisons.
- `PythonScripts/Compute_CosineDistanceToDrugControl_v2.py` — computes each
  perturbation's cosine distance to its drug's control centroid, with a
  control-null distribution for significance.
- `PythonScripts/CompareDMSOReplicateVsDrugConcordance.py` — see DE section
  above (it's also a distance/concordance script).
- `PythonScripts/AssessTargetKnockDownEfficiency.py` — loops over round-2 drug
  conditions calling `util.assess_on_target_knockdown()` to QC CRISPR
  knockdown efficiency per target.
- `PythonScripts/AddPFI1ToEnergyDistances.py` — extends the existing 21×21
  energy-distance matrix with a new PFI1 row/column, projecting PFI1 controls
  into the *existing* PCA basis (verified to reproduce stored PCA to ~1e-6)
  rather than recomputing PCA from scratch.
- `PythonScripts/AssessEnergyDistBetweenDrugs.py` — pairwise multivariate
  energy-distance computation between drugs on the shared PCA space; rough
  exploratory state (large commented-out blocks, debug print markers, a
  trailing-comma tuple bug on an otherwise-unused variable).
- `PythonScripts/PerturbFlowFunctions.py` — shared normalization/DE/distance
  helper functions reused by several of the scripts above.
- `Notebooks/07_AssessEnergyDistBetweenDrugs.ipynb` — interactive version of
  the energy-distance analysis (source of the method `AddPFI1ToEnergyDistances.py`
  mirrors).
- `Notebooks/ComputeEnergyDistance.ipynb` — energy-distance exploration.
- `Notebooks/10_InspectGeometryConcordance.ipynb` — builds a second
  loosely-filtered rowbound posterior-mean CSV (`_anyDrugSig_geneMed40.csv`)
  used by notebooks 11/12.
- `Notebooks/11_GeneratePerturbationEmbeddings.ipynb`,
  `12_GenerateGeneEmbeddings.ipynb` — PCA-based perturbation and gene
  embeddings from the rowbound posterior-mean matrices.
- `Notebooks/18_PerturbationManifoldMotion.ipynb` — measures how much each
  knockdown's transcriptional signature moves across the 16 drug contexts in
  PCA space ("manifold motion" = context dependence).
- `Notebooks/23_ContextSpecificity_vs_PCs.ipynb` — refines context specificity
  as cosine distance to the batch-matched DMSO baseline, sweeping number of
  PCs used.
- `Notebooks/ClusterPerturbations.ipynb` — clusters perturbations (feeds the
  Leiden split used by `MakeChemogeneticLeidenSplit.py`).
- `Notebooks/05_PlotFoldChanges.ipynb` — small fold-change plotting notebook.

## 5. Other

- `Notebooks/BpNetPerturb.ipynb` — a dilated-convolution, BPNet-style PyTorch
  model applied to this project's perturbation data.

## 6. Pathway analysis

### ORA (over-representation analysis) on DE genes

- `Notebooks/15_KEGG_pathway_ORA.ipynb` — per (perturbation, drug) pair: masks
  logFCs by FDR threshold, builds significant-gene sets, hypergeometric test
  against KEGG pathway gene sets.
- `Notebooks/16_KEGG_pathway_tensor.ipynb` — runs the same ORA logic across
  all 16 drugs and stacks into a `(drug × perturbation × pathway)` HDF5
  tensor.
- `Notebooks/17_InspectKEGGTensor.ipynb` — loads the tensor and renders
  per-drug `-log10(q_bh)` heatmaps.

### Per-cell pathway scoring + drug×gene interaction OLS

- `PythonScripts/ScoreOnePathway.py` — single-pathway timing/prototype test
  for the vectorized scoring strategy (per-stage timing printed for capacity
  planning).
- `PythonScripts/ScorePathwaysParallel.py` — per-drug, top-N
  most-context-dependent pathways scored in parallel via `mp.Pool`
  (fork-based, shared read-only globals for the big sparse matrix).
- `PythonScripts/ScorePathwaysBatched.py` — reformulates
  `sc.tl.score_genes`-equivalent scoring as two sparse@dense matrix
  multiplies (member-weight matrix P, bin-matched-control matrix C) —
  much faster than per-pathway matvecs.
- `PythonScripts/ScorePathwaysSeqDrugs.py` — scores all KEGG pathways across
  all 16 drug caches, drugs processed sequentially (bounded memory), pathways
  parallel within each drug; atomic per-pathway CSV writes for crash safety.
- `PythonScripts/ScoreGroupConcats.py` — same scoring, but on cross-drug group
  concats (`concat_<group>.h5ad`, all 16 drugs pooled per perturbation group)
  instead of per-drug caches.
- `PythonScripts/ConcatGroupSplitsAcrossDrugs.py` — builds those group concats
  from `SplitDrugsIntoPertGroups.py` outputs, skip-existing.
- `PythonScripts/SplitDrugsIntoPertGroups.py` — splits each per-drug cache
  into per-(drug, group) sub-h5ads based on `pert_groups.json` group
  definitions.
- `PythonScripts/FitPathwayDrugPertOLS.py` — the main interaction model: per
  (group, pathway, batch), fits
  `pathway_score ~ C(drug) + C(target_gene) + C(drug):C(target_gene) + covariates`,
  with each batch's own DMSO context as the reference drug level.
- `PythonScripts/FitPertPathwayOLS.py` — per-drug, fits
  `pathway_score ~ perturbation` (NT as reference) directly via
  `statsmodels.api.OLS` with a precomputed design matrix reused across all
  pathways; outputs beta/SE/FDR matrices.
- `PythonScripts/FitPathwayPertOLSPerDrug.py` — per-(group, drug, pathway)
  variant of the above, used to compare per-perturbation effects between the
  two DMSO batches for concordance checks.
- `PythonScripts/FitDrugPertInteractionModels.py` — in-memory (no chunking)
  end-to-end version: scores KEGG pathways once on the full 16-drug
  concatenated cache, then fits the drug×perturbation interaction OLS and
  BH-corrects within pathway.
- `PythonScripts/FitDrugPertInteractionModels_fast.py` — drop-in replacement
  for the `sc.tl.score_genes` step in the above (~8h → ~5min) using the same
  sparse@dense matmul trick as `ScorePathwaysBatched.py`.
- `PythonScripts/PlotPertPathwayHeatmaps.py` / `Notebooks/PlotPertPathwayHeatmaps.ipynb`
  — builds pert × pathway × drug sign/significance tensors from the OLS CSVs
  and renders a 2×8 grid of heatmaps (one row per batch, drug columns +
  DMSO main-effects column), with significance-count filters on both axes.

See `docs/methods_notes/pathway_ols_notes.md` for the scoring/OLS methodology
write-up, and `docs/methods_notes/OLS_resume_runbook.md` for the operational
runbook used to relaunch this pipeline's large batch jobs (the `BashScripts/
FitPathwayDrugPertOLS_*.sh` orchestrators below implement that runbook).

## 7. Train/test split design

- `Notebooks/22_PerturbationFeatures.ipynb` — computes per-gene power (cell
  count), effect size, and context-specificity features used to diversify
  the test set.
- `Notebooks/21_SelectTrainTestSplit.ipynb` — combinatorial search over
  6-drug test-set candidates spanning easy → hard, given those features.
- `PythonScripts/MakeChemogeneticLeidenSplit.py` — see Preprocessing section
  (produces the final train/val/test split and `state3` TOML).

## 8. Orchestration (shell)

- `BashScripts/ComputeSE.sh` — runs `ComputeSE.py` across all drug conditions
  with a fixed parameter set.
- `BashScripts/run_ashr_on_chunk.sh` — recursively finds chunk CSVs under a
  directory and runs `run_ashr_on_chunk.R` on each, `N_JOBS`-way parallel.
- `BashScripts/CopyAshrPosteriorMeanMatrices.sh` — collects each drug's
  `PosteriorMean_matrix.csv` into the shared `PosteriorMeanMatrices/`
  directory, renaming by drug.
- `BashScripts/Compute_DE.sh` — throttled (`max_jobs`) loop over
  per-perturbation split files calling `Compute_DEwilcox.py`.
- `BashScripts/RunGlmGamPoi_perPertFiles.sh` — parallel glmGamPoi across
  per-pert h5ads produced by `PrepareWelchVsGlmGamPoi_v2.py`.
- `BashScripts/RunWelchVsGlmGamPoi.sh` — end-to-end driver for the
  Welch-vs-glmGamPoi validation (calls `PrepareDMSOb1SubsetAndWelchSE.py`
  then `RunGlmGamPoi_subset.R`).
- `BashScripts/GenerateVolcanoPlots.sh` — wrapper for
  `GenerateVolcanoPlots.py` with fixed defaults.
- `BashScripts/FitDrugPertInteractionModels.sh` — manual-invocation wrapper
  for `FitDrugPertInteractionModels.py` (smoke-test / full / custom-args
  modes), logs to a timestamped file.
- `BashScripts/FitPathwayDrugPertOLS_173filtered.sh`,
  `FitPathwayDrugPertOLS_full415_groups00to04.sh`,
  `FitPathwayDrugPertOLS_full415_groups0to4_v2.sh`,
  `FitPathwayDrugPertOLS_signaling_g5to20.sh` — successive batch-job
  orchestrators for `FitPathwayDrugPertOLS.py`, each covering a specific
  pathway subset/group range and building on skip-existing behavior from the
  prior run (see `docs/methods_notes/OLS_resume_runbook.md` for why there are
  several of these).
- `PythonScripts/AshrGenerateMatrix.sh` — in-place wrapper for
  `AshrGenerateMatrix.py` across a base directory of per-drug results.
- `PythonScripts/SplitFiles.sh` — wrapper for `SplitBigAnndata.py` over a
  configurable list of conditions.
