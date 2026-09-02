# ChemoGeneticScreens — analysis pipeline

Code for analyzing a chemogenetic single-cell perturbation screen: CRISPR
knockouts crossed with 16 small-molecule drug contexts (plus DMSO vehicle
controls across two batches), read out by single-cell RNA-seq. The question
throughout is *how does a genetic perturbation's transcriptional effect
change depending on which drug the cell is also exposed to* — i.e. context
specificity / drug×gene interaction.

This directory contains only code and documentation. Raw data, intermediate
results (FDR/posterior-mean matrices, pathway scores, model checkpoints,
figures) live as siblings of `SRC/` in the parent `ChemoGeneticScreens/`
project directory and are intentionally not tracked here (see `.gitignore`).

## Directory guide

| Path | Contents |
|---|---|
| `RScripts/` | glmGamPoi (Gamma-Poisson GLM) differential expression and `ashr` empirical-Bayes shrinkage — both invoked per-perturbation or per-chunk from the shell scripts below. |
| `PythonScripts/` | The production pipeline: normalization, the Welch-t-test DE path, significance tables, embeddings/distances, and the per-cell pathway-scoring + OLS interaction pipeline. (Gene-program fitting — MOFA+/cNMF/ICA/VAE — now lives in the sibling `ModuleFinder_VAE` project.) |
| `Notebooks/` | Numbered, mostly-sequential analysis notebooks (`01_...` → `23_...`) plus assorted exploratory/plotting notebooks. These are where the pipeline was originally developed; several were later hardened into the scripts in `PythonScripts/`. |
| `BashScripts/` | Orchestration: wrappers that fan a Python/R script out across drugs, perturbation groups, or pathway batches, usually with skip-existing resumability and per-run logs. |
| `docs/` | Extended documentation: `docs/scripts_reference.md` (per-file reference), `docs/DEPENDENCIES.md`, and `docs/methods_notes/` — copies of the methods memos from the parent project's `Notes/` folder that justify specific pipeline choices. |

See `docs/scripts_reference.md` for a per-file description of every script and
notebook, organized by pipeline stage.

## Pipeline overview

```
raw counts (per drug context, per-cell h5ad)
        │
        ▼
1. QC / normalization           LogNormalizeData.py, 01_Data_outlook.ipynb
   (total-count norm, target_sum=200,000, log1p)
        │
        ▼
2. Batch correction              CorrectBatch.py, 03_BatchEffectCorrection_combat.ipynb,
   (ComBat)                      05_AssessBatchEffects.ipynb
        │
        ▼
3. Differential expression  ──────────────────┬───────────────────────────
   computed TWO independent ways               │
   and cross-validated against each other       │
        │                                       │
   Welch t-test path                       glmGamPoi path (Gamma-Poisson GLM
   ComputeSE.py (streaming mean/SE)         on raw counts)
     → run_ashr_on_chunk.R (EB shrinkage)   RunGlmGamPoi_onePert.R /
     → AshrGenerateMatrix.py,               RunGlmGamPoi_subset.R,
       MergeAshrRes.py                      ComputeDE_glmGAMPoi[.r/_2.r]
     → PosteriorMeanMatrices/, FDR_matrices/
        │                                       │
        └──────────────┬────────────────────────┘
                CompareWelchVsGlmGamPoi.ipynb, PrepareWelchVsGlmGamPoi_v2.py
                (see docs/methods_notes/CompositionalBias_scRNA_DE.md for why
                 they disagree, and logfc_standard_error.md /
                 ashr_vs_single_normal_shrinkage.md for the statistics)
        │
        ▼
4. Significance tables            GenerateGeneSignificanceTable.py,
   (counts of FDR<τ hits          GeneratePerturbationSignificanceTable.py,
    per gene/pert × drug)         02_PlotSignificantPerturbations.ipynb
        │
        ▼
5. Geometry / embeddings          10_InspectGeometryConcordance.ipynb →
   (PCA on rowbound               11_GeneratePerturbationEmbeddings.ipynb,
    posterior-mean matrices)      12_GenerateGeneEmbeddings.ipynb
        │
        ▼
7. Pathway analysis  ─────────────────────────┬──────────────────────────────
        │                                      │
   ORA on FDR-filtered DE genes            Per-cell pathway scoring + OLS
   15_KEGG_pathway_ORA.ipynb                interaction model
   → 16_KEGG_pathway_tensor.ipynb           BuildDrugCaches.py → ScoreOnePathway.py /
   → 17_InspectKEGGTensor.ipynb             ScorePathwaysParallel.py / ScorePathwaysBatched.py
                                             → FitPathwayDrugPertOLS.py / FitPertPathwayOLS.py
                                             → PlotPertPathwayHeatmaps.py
        │
        ▼
8. Context-specificity / manifold analysis   18_PerturbationManifoldMotion.ipynb,
   (cosine distance of each (drug, gene)     23_ContextSpecificity_vs_PCs.ipynb,
    signature to its batch-matched DMSO      Compute_CosineDistanceToDrugControl_v2.py
    baseline, in PCA space)
        │
        ▼
9. Train / test split design                 22_PerturbationFeatures.ipynb (power,
   (for downstream perturbation-prediction   effect size, context-specificity per gene)
    model evaluation)                        → 21_SelectTrainTestSplit.ipynb,
                                              MakeChemogeneticLeidenSplit.py

Cross-checks along the way:
  20_CompareDEWithCollaboratorDESeq2.ipynb / CompareDEWithCollaboratorDESeq2.py
    — validates our DE against a collaborator's independent DESeq2 pipeline
  CompareDMSOReplicateVsDrugConcordance.py
    — checks that a drug's effect on a perturbation's signature exceeds
      DMSO-replicate (batch) noise
```

## Why two DE methods?

The Welch/ashr path is fast and works directly on log1p-normalized data, but
total-count normalization can introduce *compositional bias*: a few strongly
up- or down-regulated genes distort the normalization factor for every other
gene, producing spurious "significant" passenger genes. glmGamPoi models raw
counts directly (Gamma-Poisson GLM) and is not subject to this bias, but is
much more expensive to run per-perturbation. Running both and comparing
(`CompareWelchVsGlmGamPoi.ipynb`) is the check that the Welch-path calls are
trustworthy where they agree, and flags where they don't. See
`docs/methods_notes/CompositionalBias_scRNA_DE.md`.

## The gene-program modeling work (step 6)

All shared-vs-drug-private factor-model code — not just the VAEs — now lives
together in a sibling project, `~/Projects/ModuleFinder_VAE`, and is out of
scope for this repo:

- **MOFA+** (linear; `RunMOFA_PosteriorMatrices.py`, `19_MOFA_PosteriorMatrices.ipynb`)
  — ARD-driven shared/private decomposition, drugs as views.
- **cNMF** and **ICA** (`RunCNMF.py`/`cNMFForDrugNTCs.ipynb`,
  `08_GenePrograms_ICA.ipynb`) — matrix-factorization baselines.
- **Two VAE architectures** (deep) — a drug-invariant version of the same
  decomposition (product-of-experts shared latent + adversarial drug
  classifier via gradient reversal; a group-lasso single-block alternative).
- `PlotModelArchitecture.ipynb` — renders the comparison figures across all
  four approaches.
- The associated result/output directories (`MOFA/`, `cnmf_analysis/`,
  `vae_beta_results/`, `vae_gene_programs/`) moved along with the code.

They were grouped together because they're one self-contained modeling
effort compared head-to-head. See
`docs/methods_notes/identifiability_in_GFA.md` for the theoretical grounding
shared across all four approaches.

## Known duplication / rough edges

Documented in detail in `docs/scripts_reference.md`, but worth flagging up
front: `Main.R`, `Conf.R`, `libraries.py`, and `util.py` each exist as two
byte-identical copies (once in `Notebooks/`, once in `RScripts/`/
`PythonScripts/`) so notebooks and standalone scripts can each load them via
a relative/cwd-based path without cross-referencing the other's directory —
this is intentional, not accidental drift. `parameters.py` also exists in
both locations; the two copies differ only in how one list is line-wrapped,
not in content — but since they're independent files, keep them in sync by
hand if you edit the values.
