# Seed Stability Report — WALS LEARNED

**Settings:** database=wals, architecture=learned, K=5 seeds, 10 pairwise comparisons

**Checkpoints:**
  - `checkpoints/wals_learned_s42`
  - `checkpoints/wals_learned_s43`
  - `checkpoints/wals_learned_s44`
  - `checkpoints/wals_learned_s45`
  - `checkpoints/wals_learned_s46`

---

## Probe A — Nearest-Neighbour Jaccard@10

Jaccard similarity of the top-K neighbour sets across all seed pairs.  Values near 1 indicate the neighbour structure is stable across random initialisations.

| Feature-value | Mean Jaccard@10 | Std |
|---------------|---------------------|-----|
| 85A=Prepositions | 0.927 | 0.089 |
| 85A=Postpositions | 0.873 | 0.083 |
| 81A=SOV | 0.858 | 0.103 |
| 81A=SVO | 0.806 | 0.088 |
| 55A=Absent | 0.776 | 0.103 |
| 81A=OVS | 0.714 | 0.092 |
| 13A=Complex tone system | 0.684 | 0.077 |
| 55A=Obligatory | 0.661 | 0.116 |
| 13A=No tones | 0.646 | 0.104 |
| 81A=VOS | 0.631 | 0.087 |
| 81A=VSO | 0.622 | 0.122 |
| 55A=Optional | 0.505 | 0.143 |
| 13A=Simple tone system | 0.472 | 0.134 |

**Overall mean Jaccard@10:** 0.706 ± 0.133

---

## Probe B — Silhouette Stability

Global silhouette score (metric=cosine) per seed.  Negative values are expected — see Phase 4 notes.

| Seed | Silhouette |
|------|------------|
| 0 (`s42`) | -0.3649 |
| 1 (`s43`) | -0.3631 |
| 2 (`s44`) | -0.3616 |
| 3 (`s45`) | -0.3618 |
| 4 (`s46`) | -0.3655 |

**Mean:** -0.3634  **Std:** 0.0016

---

## Probe C — Greenberg Residual Stability

Residual = ||(a_pos − a_neg) − (b_pos − b_neg)||₂.  Low residual (low empirical p) → universal supported.

| Universal | Residual mean ± std | p-value mean ± std |
|-----------|--------------------|--------------------|
| Greenberg-U4 (SOV/Post vs VSO/Prep) | 1.422 ± 0.035 | 0.183 ± 0.022 |
| Word-order/Adposition (SVO/Prep vs SOV/Post) | 1.411 ± 0.046 | 0.268 ± 0.022 |
| Tone vs Classifier (SimpleTone/NoClass vs NoTone/ObligClass) | 2.984 ± 0.080 | 0.975 ± 0.004 |

---

## Procrustes Stability (language embeddings)

Standardised Procrustes disparity between every pair of seed language-embedding matrices.  Lower → more similar geometry.  Values near 0 indicate the latent language space is stable across random initialisations.

**Mean disparity:** 0.0431  ± 0.0014  (range [0.0412, 0.0461])

**Pairwise disparity matrix:**

```
     s42  s43  s44  s45  s46
s42  0.0000  0.0433  0.0431  0.0420  0.0429
s43  0.0433  0.0000  0.0412  0.0423  0.0461
s44  0.0431  0.0412  0.0000  0.0418  0.0451
s45  0.0420  0.0423  0.0418  0.0000  0.0430
s46  0.0429  0.0461  0.0451  0.0430  0.0000
```

---

## Summary

| Metric | Value |
|--------|-------|
| Mean Jaccard@10 (Probe A) | 0.706 ± 0.133 |
| Silhouette mean ± std (Probe B) | -0.3634 ± 0.0016 |
| Procrustes disparity mean ± std | 0.0431 ± 0.0014 |
