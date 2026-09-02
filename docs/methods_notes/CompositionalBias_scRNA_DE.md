---
title: "Compositional Bias in Single-Cell RNA-seq Differential Expression"
subtitle: "Are lowly or highly expressed passenger genes more affected?"
date: "2026-06-15"
geometry: margin=1in
fontsize: 11pt
---

# The question

In single-cell RNA-seq differential expression (DE) testing, the data are
*compositional*: because of library-size (total-count) normalization, each cell
is effectively a vector of relative abundances. If you artificially change the
expression of certain ("driver") genes, the apparent expression of other
("passenger") genes can shift too — even though their absolute expression did
not change.

**Which passenger genes are more affected: lowly or highly expressed ones?**

# Short answer

The spurious **fold-change is essentially the same size for every passenger
gene**, but the spurious change is far more likely to be **detected** (called
significant) in **highly expressed genes**. So in practice it is the highly
expressed genes that *look* differentially expressed.

# Why the bias is a constant offset in log space

Library-size normalization turns each cell into fractions,
$$ p_g = \frac{c_g}{T}, \qquad T = \sum_g c_g , $$
where $c_g$ is the raw count of gene $g$ and $T$ is the cell's total count.

Suppose some driver genes gain $\Delta T$ extra counts, so the new total is
$T' = T + \Delta T$. A passenger gene's *absolute* counts $c_g$ are unchanged,
but its normalized value becomes
$$ p_g' = \frac{c_g}{T'} = p_g \cdot \frac{T}{T'} . $$

The factor $T/T'$ is **the same for every passenger gene**. In terms of
log-fold-change,
$$ \log \frac{p_g'}{p_g} = \log \frac{T}{T'} = \text{constant, independent of } g . $$

So the *magnitude* of the artifactual logFC does **not** depend on whether a
passenger gene is low- or high-expressed: every unchanged gene is shifted down
by the same amount.

# Why detectability depends on expression level

Whether that constant shift crosses the significance threshold depends on the
gene's **noise**, not just its effect size. The test statistic is roughly
$$ z \approx \frac{\text{shift}}{\text{SE}} . $$
Counts are sampled (Poisson / negative-binomial), so:

- **Highly expressed genes** have low *relative* sampling noise (small
  coefficient of variation, few dropouts) $\Rightarrow$ small SE $\Rightarrow$
  the constant compositional shift produces a large, significant statistic.
  **These light up as false positives.**

- **Lowly expressed genes** have high relative noise and many zeros
  $\Rightarrow$ large SE $\Rightarrow$ the same shift is buried in noise and
  rarely reaches significance.

**Conclusion:** highly expressed passenger genes are the ones more affected in
terms of spuriously appearing DE.

# A second-order effect on the driver side

Changing a *highly* expressed gene moves $T$ a lot (large $\Delta T$, hence a
large $T/T'$ shift on everyone), whereas perturbing a *lowly* expressed gene
barely changes the total and causes almost no compositional artifact. For the
passenger genes the question is about, however, it is still the highly expressed
ones that get falsely called.

# Practical implications

- A few strongly induced, abundant genes can create a "smear" of apparent
  down-regulation across many other genes — concentrated among high-mean genes
  with a consistent sign. That pattern is a signature of a compositional
  artifact rather than real biology.

- Robust size-factor methods resist this better than plain total-count
  normalization, because they estimate the normalization from the bulk / median
  of genes rather than letting a few dominant genes set $T$:
  DESeq2 (median-of-ratios), edgeR (TMM), scran (pooled size factors).

- Sanity check your hits: if they are dominated by high-mean genes that all move
  in the same direction, suspect compositionality.

## Relevance to this project

The pipeline normalizes to a fixed `target_sum` (200000) followed by `log1p` —
i.e. total-count normalization — so this caveat applies directly. The
glmGamPoi path models raw counts with size factors and is more robust; the
Welch-on-`log1p` path is more exposed to the artifact. This is a useful lens for
the `CompareWelchVsGlmGamPoi` comparison.
