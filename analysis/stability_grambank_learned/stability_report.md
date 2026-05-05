# Seed Stability Report — GRAMBANK LEARNED

**Settings:** database=grambank, architecture=learned, K=5 seeds, 10 pairwise comparisons

**Checkpoints:**
  - `checkpoints/grambank_learned_s42`
  - `checkpoints/grambank_learned_s43`
  - `checkpoints/grambank_learned_s44`
  - `checkpoints/grambank_learned_s45`
  - `checkpoints/grambank_learned_s46`

---

## Probe A — Nearest-Neighbour Jaccard@10

Jaccard similarity of the top-K neighbour sets across all seed pairs.  Values near 1 indicate the neighbour structure is stable across random initialisations.

| Feature-value | Mean Jaccard@10 | Std |
|---------------|---------------------|-----|
| GB203=1 | 0.827 | 0.129 |
| GB133=0 | 0.791 | 0.097 |
| GB193=1 | 0.776 | 0.103 |
| GB074=1 | 0.748 | 0.124 |
| GB074=0 | 0.748 | 0.124 |
| GB133=1 | 0.745 | 0.097 |
| GB131=0 | 0.678 | 0.134 |
| GB131=1 | 0.676 | 0.125 |
| GB075=1 | 0.650 | 0.131 |
| GB075=0 | 0.650 | 0.131 |
| GB203=0 | 0.434 | 0.088 |
| GB193=0 | 0.346 | 0.074 |

**Overall mean Jaccard@10:** 0.672 ± 0.138

---

## Probe B — Silhouette Stability

Global silhouette score (metric=cosine) per seed.  Negative values are expected — see Phase 4 notes.

| Seed | Silhouette |
|------|------------|
| 0 (`s42`) | -0.5397 |
| 1 (`s43`) | -0.5396 |
| 2 (`s44`) | -0.5400 |
| 3 (`s45`) | -0.5377 |
| 4 (`s46`) | -0.5409 |

**Mean:** -0.5396  **Std:** 0.0010

---

## Probe C — Greenberg Residual Stability

Residual = ||(a_pos − a_neg) − (b_pos − b_neg)||₂.  Low residual (low empirical p) → universal supported.

| Universal | Residual mean ± std | p-value mean ± std |
|-----------|--------------------|--------------------|
| Greenberg-U4 (verb-final/Postpos vs verb-initial/Prep) | 0.980 ± 0.030 | 0.296 ± 0.021 |
| Adj-N order vs Verb-final (head-final cluster) | 2.256 ± 0.028 | 0.883 ± 0.006 |

---

## Procrustes Stability (language embeddings)

Standardised Procrustes disparity between every pair of seed language-embedding matrices.  Lower → more similar geometry.  Values near 0 indicate the latent language space is stable across random initialisations.

**Mean disparity:** 0.0724  ± 0.0027  (range [0.0673, 0.0770])

**Pairwise disparity matrix:**

```
     s42  s43  s44  s45  s46
s42  0.0000  0.0730  0.0703  0.0673  0.0711
s43  0.0730  0.0000  0.0745  0.0739  0.0770
s44  0.0703  0.0745  0.0000  0.0705  0.0748
s45  0.0673  0.0739  0.0705  0.0000  0.0717
s46  0.0711  0.0770  0.0748  0.0717  0.0000
```

---

## Summary

| Metric | Value |
|--------|-------|
| Mean Jaccard@10 (Probe A) | 0.672 ± 0.138 |
| Silhouette mean ± std (Probe B) | -0.5396 ± 0.0010 |
| Procrustes disparity mean ± std | 0.0724 ± 0.0027 |
