---
title: "Single-Normal empirical-Bayes shrinkage vs.\\ ashr"
geometry: margin=1in
fontsize: 11pt
---

# Is this shrinkage the same as `ashr`?

The candidate formula:

$$\sigma^2_{\text{prior}} \;=\; \max\!\Big(\,\overline{\hat\beta^2} \;-\; \overline{se^2},\ 0\Big)$$

$$\hat\beta_{\text{shrunk}} \;=\; \hat\beta \cdot \frac{\sigma^2_{\text{prior}}}{\sigma^2_{\text{prior}} + se^2}$$

**Short answer:** same family, but `ashr` is strictly more general.

## What the formula above is

This is single-Normal empirical Bayes (a James–Stein / `limma`-style shrinker):

- Prior: $\beta \sim \mathcal{N}(0,\,\sigma^2_{\text{prior}})$
- Likelihood: $\hat\beta \mid \beta \sim \mathcal{N}(\beta,\,se^2)$
- Posterior mean: $\hat\beta \cdot \sigma^2_{\text{prior}} / (\sigma^2_{\text{prior}} + se^2)$
- $\sigma^2_{\text{prior}}$ estimated by method of moments:
  $\operatorname{Var}(\hat\beta) \approx \sigma^2_{\text{prior}} + \overline{se^2}$,
  so $\sigma^2_{\text{prior}} \approx \overline{\hat\beta^2} - \overline{se^2}$, clipped at 0.

One scalar prior variance, one Normal component, no null spike.

## How `ashr` (Stephens 2017) differs

1. **Mixture prior, not a single Normal.**
   `ashr` fits
   $$g(\beta) \;=\; \pi_0\,\delta_0 \;+\; \sum_k \pi_k\,\mathcal{N}(0,\,\sigma_k^2)$$
   (or uniform components) — a flexible **unimodal** prior. The
   $\{\pi_k,\sigma_k\}$ are estimated by EM. The effective shrinkage factor
   is therefore not $\sigma^2/(\sigma^2+se^2)$ everywhere; it is a
   mixture-weighted blend that depends on where $\hat\beta$ falls relative
   to each component.

2. **Explicit point mass at 0** ($\pi_0\,\delta_0$).
   This is what lets `ashr` produce **lfsr** (local false sign rate) and
   **lfdr** (local false discovery rate). The single-Normal formula above
   has no null component — it shrinks everything toward $0$ by a smooth
   factor but never assigns "probably exactly zero" mass.

3. **Different shrinkage shape.**
   With a single Normal prior, the shrinkage factor depends only on
   $se^2$ (the same $\sigma^2_{\text{prior}}$ for every observation).
   `ashr`'s posterior mean is more aggressive near zero (the null
   component pulls hard) and gentler in the tails (heavier-tailed
   components dominate). Two observations with the same $se$ but
   different $|\hat\beta|$ are **differently** shrunk by `ashr`; under
   the single-Normal formula they receive the same proportional shrinkage.

## When the two coincide

The single-Normal formula = `ashr` restricted to one mixture component
($K=1$), $\pi_0 = 0$, and $\sigma_1$ fixed by method-of-moments instead
of MLE. In that degenerate case the two posterior-mean formulas
agree exactly.

## Practical implication for this project

If `PosteriorMeanMatrices/` was produced by R's `ashr` (via
`SRC/RScripts/run_ashr_on_chunk.R`), the values will differ from what
the single-Normal formula would give — especially for:

- genes near zero (`ashr` shrinks **more**, because the $\pi_0$ component pulls hard),
- large-effect genes (`ashr` shrinks **less**, because heavier-tailed mixture components dominate).

The two will agree in spirit (rank order roughly preserved), not in
numbers.
