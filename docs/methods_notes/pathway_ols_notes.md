---
title: "From cells to per-(perturbation, pathway) effects"
subtitle: "scanpy `sc.tl.score_genes` + OLS fit on the resulting cell-level scores"
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

# 1. Inputs

For each drug context $d$ we have an AnnData $X^{(d)} \in \mathbb{R}^{N_d \times G}$ of log1p-normalized expression for $N_d$ cells over $G$ genes. Each cell carries a `target_gene` label $t_i \in \mathcal{P} \cup \{\text{non-targeting}\}$ where $\mathcal{P}$ is the set of perturbations; "non-targeting" (NT) is the control identity. We restrict to a subset of cells (the perturbations from the reduced KEGG ORA tensor + a random NT subsample) and a subset of genes (KEGG-pathway-member union $\cup$ stratified background genes). For a pathway $k$ we have a member-gene set $G_k \subseteq \{1, \dots, G\}$ from MSigDB.

# 2. Cell-level pathway score (`sc.tl.score_genes`) and OLS fit

The score implemented in `scanpy.tl.score_genes` is the *bin-matched module score* of Tirosh et al. (2016). Given a gene set $G_k$ and a pool of candidate genes $\mathcal{P}_g \subseteq \{1, \dots, G\}$ (the `gene_pool` argument; default = all genes), it computes:

1. **Per-gene mean expression** over all $N$ cells in the AnnData:
$$
\bar{x}_g \;=\; \frac{1}{N} \sum_{i=1}^{N} X_{i, g}, \qquad g \in \mathcal{P}_g.
$$

2. **Bin genes by mean expression**. Divide $\mathcal{P}_g$ into $B$ (default 25) quantile bins of $\bar{x}_g$; let $b(g) \in \{1, \dots, B\}$ denote gene $g$'s bin.

3. **Bin-matched controls**. For each pathway member $g \in G_k \cap \mathcal{P}_g$ that has a defined bin, sample $S$ (default 50) control genes uniformly without replacement from the bin $\{g' \in \mathcal{P}_g : b(g') = b(g),\; g' \notin G_k\}$. Pool them:
$$
C_k \;=\; \bigcup_{g \in G_k \cap \mathcal{P}_g} \mathrm{Sample}_S\!\big(\{g' \in \mathcal{P}_g : b(g') = b(g),\; g' \notin G_k\}\big).
$$

4. **Score**. For each cell $i$:
$$
\boxed{\;\mathrm{score}_{k, i} \;=\; \underbrace{\frac{1}{\,|G_k|\,} \sum_{g \in G_k} X_{i, g}}_{\text{mean expression of pathway members}} \;\;-\;\; \underbrace{\frac{1}{\,|C_k|\,} \sum_{c \in C_k} X_{i, c}}_{\text{mean expression of bin-matched controls}}\;}
$$

5. **Joint pathway-level interaction OLS.** Fix one pathway $k$ and one experimental batch $\mathcal{D}$ (the set of drugs sharing one DMSO control). Pool all cells across drugs in $\mathcal{D}$, let $y_i = \mathrm{score}_{k, i}$, $d_i \in \mathcal{D}$ be the drug, $t_i \in \mathcal{P} \cup \{\mathrm{NT}\}$ the perturbation, and $z_i \in \mathbb{R}^{Q}$ continuous technical covariates (we use `pct_counts_mt`, `log1p_total_counts`). Fit
$$
y_i \;=\; \alpha
\;+\; \sum_{d \neq d_0}\! \beta_d^{\mathrm{drug}} \mathbb{1}[d_i = d]
\;+\; \sum_{p \neq \mathrm{NT}}\! \beta_p^{\mathrm{pert}} \mathbb{1}[t_i = p]
\;+\; \sum_{d \neq d_0,\, p \neq \mathrm{NT}}\! \boxed{\delta_{d,p}}\, \mathbb{1}[d_i = d]\,\mathbb{1}[t_i = p]
\;+\; \gamma^{\top} z_i \;+\; \varepsilon_i,
$$
with reference levels $d_0 = $ DMSO of the batch, $t = \mathrm{NT}$. The interaction coefficient $\delta_{d,p}$ (rows of `term_type = "drug:target_gene"` in the per-pathway CSV) is the pathway-$k$ change in score for perturbation $p$ in drug $d$ *over and above* the additive drug and pert main effects — i.e. the **drug-context-specific genetic interaction** on this pathway. All coefficients are estimated jointly in one OLS solve via `statsmodels.formula.api.ols` with t-statistics and two-sided p-values; we fit batch1 and batch2 separately so each uses its own DMSO reference.

The bin-matching keeps high-expressing housekeeping genes from dominating the control pool, which would otherwise drag every cell's score downward. Pathways with $|G_k \cap \mathcal{P}_g| < 3$ are skipped (NaN score).

In our pipeline we set `gene_pool = (background_genes) ∪ (this_pathway_members)`. Because scanpy excludes pathway members from being chosen as controls, this forces $C_k \subseteq \text{background\_genes}$ — guaranteeing that no control gene is itself a member of *some other* KEGG pathway in the same cache.

# 3. BH correction and filtering

Per pathway $k$ we apply Benjamini–Hochberg correction across the $|\mathcal{P}|$ p-values:
$$
q_p^{(k)} \;=\; \min_{p' \,:\, p^{(k)}_{p'} \,\geq\, p^{(k)}_p}\!\Big[\, p^{(k)}_{p'} \cdot \frac{|\mathcal{P}|}{\mathrm{rank}(p^{(k)}_{p'})} \,\Big].
$$

Perturbations with $n_p < 2$ get $\hat{\beta}_p = \mathrm{SE} = p_p = q_p = \mathrm{NaN}$ (insufficient cells to estimate within-group mean / variance). Subsequent analyses define a `(pert, pathway)` cell as significant when $q_p^{(k)} < \tau$ (we use $\tau = 0.10$ as the default).

# 4. Output structure

For each drug $d$ the pipeline writes five $|\mathcal{P}| \times K$ matrices (rows = perturbations, columns = pathways):

| File | Quantity |
|---|---|
| `beta_<drug>.parquet` | $\hat{\beta}_{p,k}$ |
| `se_<drug>.parquet` | $\mathrm{SE}(\hat{\beta}_{p,k})$ |
| `pvalue_<drug>.parquet` | $p_{p,k}$ |
| `fdr_<drug>.parquet` | $q_{p,k}$ (BH within pathway) |
| `n_pert_cells_<drug>.parquet` | $n_p$ — sample size per (pert, pathway) cell |

Per-pathway intermediate files at `per_pathway_<drug>/<pathway>.parquet` are also written (one fit at a time) so the run is restart-safe.

# 5. Caveats

- **Anti-conservative p-values.** Cells from the same biological sample / library are correlated, so the OLS independence assumption is violated. The effective sample size is smaller than $N$, which inflates the t-statistic. For *ranking* (pert, pathway) hits this is fine; for defensible per-coefficient inference, switch to pseudobulk (mean per sample) or include a sample-level random effect (mixed-model OLS).

- **Score is bin-pool-dependent.** The means $\bar{x}_g$ used for binning, and therefore the resulting $C_k$, depend on the cell pool fed to `score_genes`. Scoring the same pathway on a different cell subset can shift the score by a constant — but $\hat{\beta}_p$, being a *difference* of group means, is invariant under additive shifts.

- **Member-gene coverage.** A pathway with $|G_k \cap \mathcal{P}_g| < 3$ is unscored (NaN). Even at $\geq 3$, very small pathways have noisier scores.

- **Multiple testing across pathways.** BH is applied within pathway across perts; if you also want a global FDR control across pathways, multiply $q_p^{(k)}$ by an additional cross-pathway correction (Bonferroni on $K$, or a hierarchical FDR like Benjamini–Bogomolov).
