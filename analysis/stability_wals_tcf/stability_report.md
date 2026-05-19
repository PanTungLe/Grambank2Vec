# Seed Stability Report — WALS TCF

**Settings:** database=wals, architecture=tcf, K=5 seeds, 10 pairwise comparisons

**Checkpoints:**
  - `checkpoints/wals_tcf_s42`
  - `checkpoints/wals_tcf_s43`
  - `checkpoints/wals_tcf_s44`
  - `checkpoints/wals_tcf_s45`
  - `checkpoints/wals_tcf_s46`

---

## Probe A — Nearest-Neighbour Jaccard@10

Jaccard similarity of the top-K neighbour sets across all seed pairs.  Values near 1 indicate the neighbour structure is stable across random initialisations.

| Feature-value | Mean Jaccard@10 | Std |
|---------------|---------------------|-----|
| 81A=SOV | 0.424 | 0.091 |
| 85A=Prepositions | 0.403 | 0.074 |
| 85A=Postpositions | 0.388 | 0.165 |
| 81A=SVO | 0.311 | 0.069 |
| 55A=Absent | 0.295 | 0.127 |
| 13A=No tones | 0.288 | 0.139 |
| 81A=VSO | 0.134 | 0.064 |
| 55A=Obligatory | 0.112 | 0.091 |
| 13A=Complex tone system | 0.104 | 0.073 |
| 55A=Optional | 0.084 | 0.055 |
| 81A=OVS | 0.049 | 0.039 |
| 13A=Simple tone system | 0.033 | 0.053 |
| 81A=VOS | 0.027 | 0.037 |

**Overall mean Jaccard@10:** 0.204 ± 0.144

---

## Probe B — Silhouette Stability

Global silhouette score (metric=cosine) per seed.  Negative values are expected — see Phase 4 notes.

| Seed | Silhouette |
|------|------------|
| 0 (`s42`) | -0.7530 |
| 1 (`s43`) | -0.7511 |
| 2 (`s44`) | -0.7517 |
| 3 (`s45`) | -0.7545 |
| 4 (`s46`) | -0.7500 |

**Mean:** -0.7521  **Std:** 0.0016

---

## Probe C — Greenberg Residual Stability

Residual = ||(a_pos − a_neg) − (b_pos − b_neg)||₂.  Low residual (low empirical p) → universal supported.

| Universal | Residual mean ± std | p-value mean ± std |
|-----------|--------------------|--------------------|
| Greenberg-U4 (SOV/Post vs VSO/Prep) | 1.425 ± 0.071 | 0.658 ± 0.038 |
| Word-order/Adposition (SVO/Prep vs SOV/Post) | 0.363 ± 0.024 | 0.099 ± 0.008 |
| Tone vs Classifier (SimpleTone/NoClass vs NoTone/ObligClass) | 2.711 ± 0.225 | 0.957 ± 0.016 |

---

## Procrustes Stability (language embeddings)

Standardised Procrustes disparity between every pair of seed language-embedding matrices.  Lower → more similar geometry.  Values near 0 indicate the latent language space is stable across random initialisations.

**Mean disparity:** 0.1256  ± 0.0049  (range [0.1182, 0.1308])

**Pairwise disparity matrix:**

```
     s42  s43  s44  s45  s46
s42  0.0000  0.1265  0.1292  0.1308  0.1182
s43  0.1265  0.0000  0.1306  0.1303  0.1221
s44  0.1292  0.1306  0.0000  0.1295  0.1192
s45  0.1308  0.1303  0.1295  0.0000  0.1198
s46  0.1182  0.1221  0.1192  0.1198  0.0000
```

---

## Summary

| Metric | Value |
|--------|-------|
| Mean Jaccard@10 (Probe A) | 0.204 ± 0.144 |
| Silhouette mean ± std (Probe B) | -0.7521 ± 0.0016 |
| Procrustes disparity mean ± std | 0.1256 ± 0.0049 |
