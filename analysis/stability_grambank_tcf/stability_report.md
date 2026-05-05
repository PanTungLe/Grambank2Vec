# Seed Stability Report — GRAMBANK TCF

**Settings:** database=grambank, architecture=tcf, K=5 seeds, 10 pairwise comparisons

**Checkpoints:**
  - `checkpoints/grambank_tcf_s42`
  - `checkpoints/grambank_tcf_s43`
  - `checkpoints/grambank_tcf_s44`
  - `checkpoints/grambank_tcf_s45`
  - `checkpoints/grambank_tcf_s46`

---

## Probe A — Nearest-Neighbour Jaccard@10

Jaccard similarity of the top-K neighbour sets across all seed pairs.  Values near 1 indicate the neighbour structure is stable across random initialisations.

| Feature-value | Mean Jaccard@10 | Std |
|---------------|---------------------|-----|
| GB203=1 | 0.596 | 0.116 |
| GB193=1 | 0.479 | 0.109 |
| GB203=0 | 0.255 | 0.082 |
| GB193=0 | 0.169 | 0.090 |

**Overall mean Jaccard@10:** 0.375 ± 0.170

---

## Probe B — Silhouette Stability

Global silhouette score (metric=cosine) per seed.  Negative values are expected — see Phase 4 notes.

| Seed | Silhouette |
|------|------------|
| 0 (`s42`) | -0.3829 |
| 1 (`s43`) | -0.3971 |
| 2 (`s44`) | -0.3811 |
| 3 (`s45`) | -0.3721 |
| 4 (`s46`) | -0.3776 |

**Mean:** -0.3822  **Std:** 0.0083

---

## Probe C — Greenberg Residual Stability

Residual = ||(a_pos − a_neg) − (b_pos − b_neg)||₂.  Low residual (low empirical p) → universal supported.

| Universal | Residual mean ± std | p-value mean ± std |
|-----------|--------------------|--------------------|
| Greenberg-U4 (verb-final/Postpos vs verb-initial/Prep) | SKIPPED | — |
| Adj-N order vs Verb-final (head-final cluster) | SKIPPED | — |

---

## Procrustes Stability (language embeddings)

Standardised Procrustes disparity between every pair of seed language-embedding matrices.  Lower → more similar geometry.  Values near 0 indicate the latent language space is stable across random initialisations.

**Mean disparity:** 0.2217  ± 0.0286  (range [0.1687, 0.2634])

**Pairwise disparity matrix:**

```
     s42  s43  s44  s45  s46
s42  0.0000  0.2017  0.1917  0.2348  0.1687
s43  0.2017  0.0000  0.2376  0.2634  0.2130
s44  0.1917  0.2376  0.0000  0.2584  0.2089
s45  0.2348  0.2634  0.2584  0.0000  0.2389
s46  0.1687  0.2130  0.2089  0.2389  0.0000
```

---

## Summary

| Metric | Value |
|--------|-------|
| Mean Jaccard@10 (Probe A) | 0.375 ± 0.170 |
| Silhouette mean ± std (Probe B) | -0.3822 ± 0.0083 |
| Procrustes disparity mean ± std | 0.2217 ± 0.0286 |
