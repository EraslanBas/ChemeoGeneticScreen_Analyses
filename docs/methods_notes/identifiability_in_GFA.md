---
title: "Identifiability in observational data and in group factor analysis"
subtitle: "When labels are needed, when multi-view structure substitutes for them"
date: ""
geometry: "margin=2.2cm"
fontsize: 10pt
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
  - \usepackage{longtable}
  - \usepackage{array}
---

# Question

> Are there identifiability issues in observational data that get resolved only when we also have a label per data point? Is this also true for the group factor analysis (GFA) setting?

Mostly yes — but the multi-view setting changes the picture meaningfully.

# 1. The single-view story

For nonlinear representation learning on a single view of i.i.d. observational data, **the nonlinear ICA model is fundamentally non-identifiable** (Hyvärinen & Pajunen 1999). For any nonlinear mixing $f$ producing the data, infinitely many other nonlinear mixings give the same observed distribution. The standard fix is conditioning on an auxiliary variable $u$:
$$
p(\mathbf{z} \mid u) \;=\; \mathcal{N}\!\big(\boldsymbol{\mu}(u),\; \boldsymbol{\Sigma}(u)\big),
$$
which Hyvärinen & Morioka (2016, "TCL"; using a time index $u$) and Khemakhem et al. (2020, "iVAE"; any auxiliary) showed restores identifiability up to a *component-wise* transform. The label, time index, environment, or class breaks the rotational/mixing ambiguity.

So in a single-view nonlinear ICA setting your statement is correct: per-sample auxiliary information is essentially required.

# 2. What changes in GFA / multi-view

Multi-view structure is itself an identifiability-restoring signal — you don't necessarily need per-sample labels.

## 2.1 Linear GFA (Klami 2014; MOFA, MuVI)

Identifiable up to factor permutation/sign **without labels**, via two combined mechanisms:

1. **Sparsity priors on per-view loadings** (ARD / spike-and-slab) — break the rotation ambiguity that plagues vanilla factor analysis. With column-sparse $W_v$, almost every rotation destroys sparsity, so the maximum-a-posteriori solution is essentially unique up to permutation/sign.
2. **The multi-view structure itself** — different sharing patterns across views constrain which factors can be active where. A rotation that mixes a shared and a view-private factor is penalized by the per-view ARD.

Linear GFA never required per-sample labels. Identifiability comes from the priors plus the view structure.

## 2.2 Nonlinear / deep multi-view (Gresele 2020; Daunhawer 2023)

Two formal results that hold *without* per-sample labels:

- **Gresele, Locatello, Schölkopf 2020 — "Multi-view Nonlinear ICA"**. Two views with shared content $\mathbf{z}^c$ and view-private style $\mathbf{z}^p_v$ are sufficient to identify $\mathbf{z}^c$ up to a component-wise transform, under smoothness + independence assumptions on the data-generating process. **Multiple views act as the auxiliary signal**.

- **Daunhawer, Bauer, Locatello et al. (ICLR 2023) — "Identifiability Results for Multimodal Contrastive Learning"**. Multimodal contrastive learning (CLIP-style) provably recovers $\mathbf{z}^c$ under analogous assumptions. Same lesson: views serve as auxiliaries to each other.

So for the *shared* content in a multi-view setup, identifiability holds without labels. For the *private* content per view, the situation is weaker — you typically need additional structure (sparsity, independence, monotonicity, sign constraints) or you fall back on labels.

# 3. Where labels still help even in multi-view

Three places per-sample labels strengthen identifiability beyond what multi-view alone gives:

1. **Recovering private factors** that aren't pinned down by the multi-view assumption.
2. **Going from "up to component-wise transform" to "up to permutation/sign"** — a tighter form of identifiability that's easier to interpret biologically.
3. **Causal / counterfactual claims** — iVAE-family results give this only with auxiliaries.

# 4. Implications for the chemogenetic-screens setup

The drug × pert × gene-logFC data has a **doubly-identifying structure**:

- **Multi-view**: 16 drug contexts as views.
- **Per-sample auxiliary**: each cell carries a `target_gene` label (the perturbation), giving $\approx$ 2{,}400 distinct classes. In iVAE language, this is exactly the auxiliary that stratifies the latent prior.

Identifiability-wise this is unusually favorable. Concretely:

- **For linear GFA** (e.g. MOFA+ on the per-drug PosteriorMean matrices): rely on per-view ARD + view structure. No need for labels in the model — they're implicit because each row is a different perturbation.
- **For deep multi-view models** (e.g. the `14_2` / `14_3` VAE notebooks): condition the encoder on `target_gene` to put yourself fully in the iVAE setting — that delivers component-wise identifiability for *both* shared and private.
- **Cell-level OLS on `sc.tl.score_genes` outputs**: the per-cell `target_gene` label is exactly the auxiliary, and the joint OLS estimates per-perturbation coefficients $\beta_p$ that are well-defined under this setup.

# 5. Summary table

| Setting | Needs per-sample labels? | Identifiability granularity |
|---|---|---|
| **Single-view nonlinear ICA** (vanilla VAE on observational data) | **Yes** — the only fix | Up to component-wise transform (with auxiliary) |
| **Linear GFA** (Klami, MOFA, MuVI) | No — ARD + multi-view structure suffices | Up to permutation / sign |
| **Multi-view nonlinear ICA**, *shared content* (Gresele 2020) | No — views substitute for labels | Up to component-wise transform |
| **Multi-view nonlinear ICA**, *private content* | Often weak without extra assumptions | Need sparsity / independence priors or labels |
| **Multi-view + per-sample labels** (iVAE-flavour) | Both signals available | Tightest identifiability |

# 6. References

- Hyvärinen, A. & Pajunen, P. (1999). *Nonlinear independent component analysis: existence and uniqueness results.* Neural Networks 12(3).
- Hyvärinen, A. & Morioka, H. (2016). *Unsupervised feature extraction by time-contrastive learning and nonlinear ICA.* NeurIPS.
- Klami, A., Virtanen, S. & Kaski, S. (2014). *Group factor analysis.* JMLR / IEEE T-NNLS.
- Khemakhem, I., Kingma, D., Monti, R. & Hyvärinen, A. (2020). *Variational autoencoders and nonlinear ICA: A unifying framework.* AISTATS (iVAE).
- Gresele, L., von Kügelgen, J., Stimper, V., Schölkopf, B. & Besserve, M. (2020). *The Incomplete Rosetta Stone Problem: Identifiability Results for Multi-View Nonlinear ICA.* UAI / NeurIPS workshops.
- Argelaguet, R. et al. (2018, 2020). *MOFA / MOFA+: Multi-Omics Factor Analysis.* Mol. Syst. Biol., Genome Biology.
- Qoku, A. & Buettner, F. (2023). *MuVI: A Multi-View Latent Variable Model.* AISTATS.
- Daunhawer, I., Bizeul, A., Palumbo, E., Marx, A. & Vogt, J. (2023). *Identifiability Results for Multimodal Contrastive Learning.* ICLR.
