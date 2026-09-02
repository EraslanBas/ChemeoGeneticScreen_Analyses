---
title: "Per-(perturbation, gene) standard errors of the logFC estimates"
subtitle: "How `ComputeSE.py` produces the `se` column fed to `ashr`"
date: ""
geometry: "margin=2.2cm"
fontsize: 11pt
header-includes:
  - \usepackage{amsmath}
  - \usepackage{amssymb}
  - \usepackage{booktabs}
  - \usepackage{array}
  - \usepackage{microtype}
---

# 1. Setup, per gene $g$

For a single perturbation $p$ we have two groups of cells in the AnnData fed to
`ComputeSE.py`:

- **Perturbed cells** (`target_gene == p`): $n_p$ cells with log1p-normalized
  expression values $x^{(p)}_1, \dots, x^{(p)}_{n_p}$ taken from `adata.X`.
- **Control cells** (`target_gene == non-targeting`): $n_c$ cells with values
  $x^{(c)}_1, \dots, x^{(c)}_{n_c}$ from the same matrix.

The estimand is the difference of population means on the log1p scale,

$$
\beta_{p,g} \;=\; \mu^{(p)}_g - \mu^{(c)}_g,
$$

estimated by the corresponding sample means,

$$
\widehat{\beta}_{p,g} \;=\; \bar{x}^{(p)} - \bar{x}^{(c)}.
$$

# 2. Variance of the estimator

Treating cells as **independent draws** within each group, the sampling variance
of each group mean is

$$
\mathrm{Var}\!\bigl(\bar{x}^{(p)}\bigr) \;=\; \frac{\sigma^{(p)\,2}_g}{n_p},
\qquad
\mathrm{Var}\!\bigl(\bar{x}^{(c)}\bigr) \;=\; \frac{\sigma^{(c)\,2}_g}{n_c}.
$$

The two group means are computed from **disjoint sets of cells**, so
$\bar{x}^{(p)}$ and $\bar{x}^{(c)}$ are independent. For independent random
variables variances add — regardless of whether we are summing or
subtracting:

$$
\mathrm{Var}(X - Y) \;=\; \mathrm{Var}(X) + \mathrm{Var}(Y)
\qquad \text{when } X \perp\!\!\!\perp Y.
$$

There is no minus sign on the right; the sign of the linear combination
does not matter once the terms are independent. Applying this with
$X = \bar{x}^{(p)}$ and $Y = \bar{x}^{(c)}$ gives the variance of the
estimator:

$$
\mathrm{Var}\!\bigl(\widehat{\beta}_{p,g}\bigr)
   \;=\; \mathrm{Var}\!\bigl(\bar{x}^{(p)}\bigr) +
        \mathrm{Var}\!\bigl(\bar{x}^{(c)}\bigr)
   \;=\; \frac{\sigma^{(p)\,2}_g}{n_p}
       + \frac{\sigma^{(c)\,2}_g}{n_c}.
$$

This is the **Welch / Behrens–Fisher** variance for a two-sample mean
difference — it does **not** assume equal variances between groups.

Two sanity-check corollaries:

- *Why the SE shrinks with $n$.* Each group mean's variance has $n$ in
  the denominator. Doubling either group's cell count roughly halves
  that group's contribution to the variance.
- *What can break this formula.* If cells within a group are **not**
  independent (e.g.\ they share a library, batch, clone, or cell-cycle
  state), then
  $\mathrm{Var}\!\bigl(\bar{x}^{(p)}\bigr) > \sigma^{(p)\,2}_g/n_p$
  because the *effective* sample size is smaller than $n$. The
  resulting SE is **anti-conservative** — see §5.

# 3. Estimating $\sigma^{2}_g$

The unknown population variances are replaced by the **unbiased sample
variances** (Bessel correction):

$$
\widehat{\sigma}^{(p)\,2}_g \;=\; \frac{1}{n_p - 1}
\sum_{i=1}^{n_p}\!\bigl(x^{(p)}_i - \bar{x}^{(p)}\bigr)^{\!2},
$$

and analogously for the controls. `ComputeSE.py` computes this in a streaming,
sparse-aware way without ever materialising the centered matrix. It carries
column-wise sums $S_g = \sum_i x_i$ and sums of squares $T_g = \sum_i x_i^2$:

```python
s, ss = _sum_and_sumsq(X[group_mask])     # one pass over the sparse matrix
```

A closed-form identity then gives the variance:

$$
\widehat{\sigma}^{2}_g \;=\;
   \frac{n}{n-1}\!\left( \frac{T_g}{n} - \bar{x}^{\,2} \right)
   \;=\; \frac{n}{n-1} \cdot
\underbrace{\bigl(\overline{x^{2}} - \bar{x}^{\,2}\bigr)}_{\text{population variance}}.
$$

In code:

```python
ex2     = ss / n
var_pop = np.maximum(ex2 - mean * mean, 0.0)
var     = var_pop * (n / (n - 1))     # Bessel correction → unbiased
```

The `np.maximum(\ldots, 0.0)` is a numerical guard: with `float32` input and
large means, $\overline{x^{2}} - \bar{x}^{\,2}$ can be a tiny negative number
because of catastrophic cancellation. Clamp it to zero rather than let
`np.sqrt` later return a `NaN`.

# 4. Final SE formula

After computing the group means and unbiased variances, the per-(perturbation,
gene) standard error is the square root of the Welch variance:

$$
\boxed{
\mathrm{SE}\!\bigl(\widehat{\beta}_{p,g}\bigr)
\;=\;
\sqrt{\, \frac{\widehat{\sigma}^{(p)\,2}_g}{n_p}
       + \frac{\widehat{\sigma}^{(c)\,2}_g}{n_c} \,}.
}
$$

```python
mean_diff = mean_p - mean_c
se        = np.sqrt(var_p / n_p + var_c / n_ctrl)
```

This $(\widehat{\beta}, \mathrm{SE})$ pair per `(pert, gene)` is exactly what
`ashr::ash(betahat, sebetahat)` consumes downstream to produce the
posterior-mean shrunken estimates and `lfsr`.

# 5. Assumptions and where they fail

| Assumption | Holds in our data? |
|---|---|
| Cells within a group are i.i.d. | **Strongly violated.** Cells share library / batch / cell-cycle / clonal structure, so the effective sample size is $< n$. The SE is therefore **anti-conservative** — true uncertainty is larger than computed. |
| Sample means are approximately Gaussian (CLT) | Roughly OK for moderately-expressed genes when $n$ is in the hundreds. Breaks for very small $n$ or very sparse genes. |
| $\widehat{\sigma}^{2}_g$ is a reliable estimate of $\sigma^{2}_g$ | OK at large $n$. **Breaks at small $n$**: the sample variance can be exactly $0$ just because all cells in a small group happen to have the same value at a sparse gene — the SE then collapses to $\approx 0$ and \texttt{ashr} treats the estimate as perfectly precise. This is the failure mode behind the small-$n$ posterior-mean inflation observed in the DMSO replicate diagnostic. |
| Mean difference on the log1p scale $\approx$ log fold-change | Approximate. For highly-expressed genes $\mathrm{log1p}(x) \approx \log(x)$, so the difference is in **natural-log units**. For low-count genes log1p departs from $\log$ appreciably. |
| Perturbed and control cells are independent | True — no cell is in both groups. |

# 6. In one sentence

We compute, gene by gene,
$\widehat{\beta} = \bar{x}_{\text{pert}} - \bar{x}_{\text{ctrl}}$ and
$\mathrm{SE} = \sqrt{\widehat{\sigma}^{2}_{\text{pert}}/n_{\text{pert}} + \widehat{\sigma}^{2}_{\text{ctrl}}/n_{\text{ctrl}}}$
\— the textbook Welch SE for a two-sample mean difference \— on
log1p-normalized expression, treating each cell as one independent
observation.
