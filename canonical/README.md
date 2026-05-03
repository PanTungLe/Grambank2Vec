# Canonical Model Training and Analysis

This directory implements the canonical-model sub-pipeline for the
Grambank2Vec thesis.  It trains stable, full-data typological embeddings
and runs geometry probes to answer:

- **RQ2 (second clause):** Do learned embeddings capture typologically
  meaningful geometric structure?
- **RQ1 (second clause):** Is the latent typological space similar across
  WALS and Grambank?

---

## Directory layout

```
canonical/
  train_canonical.py      Full-data canonical training (Phases 2–3)
  analyze_geometry.py     Geometry probes A/B/C (Phase 4)
  compare_databases.py    Cross-database Procrustes + CCA (Phase 5)
  seed_stability.py       Seed-stability analysis (Phase 6)
  utils.py                Shared helpers (seeding, data utils)
  api_audit.md            Notes on upstream API contracts
  tests/                  Unit tests for all modules
```

---

## Canonical training (`train_canonical.py`)

Trains either the **T-CF** (binary Bernoulli, sigmoid) or **Learned**
(multiclass softmax, cross-entropy) model on the **full** dataset with a
5% random held-out validation split used only for early stopping.

### Key design choices

| Choice | Rationale |
|--------|-----------|
| `Adam(weight_decay=0)` | Global weight decay causes sparse-embedding collapse via Adam's adaptive denominator; per-batch L2 inside the model is used instead. |
| Glottocodes as join key | Both WALS and Grambank are indexed by Glottocode in `lang2id.json`, enabling cross-database intersection without ambiguity. |
| Seeded DataLoader generator | `torch.Generator().manual_seed(seed)` ensures bit-identical training runs under the same seed. |
| Early stopping (patience=10) | Caps training at `--n_epochs` (default 200) but stops sooner when val loss plateaus. |

### Usage

```bash
# WALS — Learned architecture
python canonical/train_canonical.py \
    --database wals --architecture learned \
    --data_path /path/to/wals \
    --out_dir checkpoints/wals_learned_s42 \
    --seed 42

# WALS — T-CF architecture
python canonical/train_canonical.py \
    --database wals --architecture tcf \
    --data_path /path/to/wals \
    --out_dir checkpoints/wals_tcf_s42 \
    --seed 42

# Grambank (requires cloned Grambank CLDF repo)
python canonical/train_canonical.py \
    --database grambank --architecture learned \
    --data_path /path/to/grambank \
    --out_dir checkpoints/grambank_learned_s42 \
    --seed 42
```

### Output artifacts

| File | Description |
|------|-------------|
| `lang_embeddings.npy` | `(n_langs, d)` language embedding matrix |
| `featvalue_embeddings.npy` | `(n_fv, d)` feature-value embeddings (Learned) |
| `binarycol_embeddings.npy` | `(n_cols, d)` binary-column embeddings (T-CF) |
| `featvalue2id.json` | `"81A=SOV" → row_index` (Learned) |
| `binarycol2id.json` | `"81A=SOV" → row_index` (T-CF) |
| `feat2values.json` | `"81A" → ["SOV", "SVO", …]` |
| `lang2id.json` | Glottocode → row_index (cross-database join key) |
| `lang2id_full.json` | Native language ID → row_index (all languages) |
| `config.json` | All hyperparameters + final training stats |
| `training_log.csv` | Per-epoch train loss, val loss, embedding norms |
| `model_best.pt` | Best PyTorch checkpoint |

---

## Geometry probes (`analyze_geometry.py`)

Three probes on a canonical checkpoint, addressing RQ2 second clause.

### Probe A — Nearest neighbours

Top-K cosine-nearest feature-value neighbours for curated WALS/Grambank
targets (word-order, adposition, classifier, tone features).

### Probe B — Silhouette by feature membership

Groups all feature-value embeddings by their parent feature ID and
computes the silhouette score using cosine distance.  **Negative global
silhouette is expected** — the softmax objective actively pushes same-feature
values apart (mutual exclusion), making the cluster structure typologically
meaningful (implicational groupings) rather than within-feature.

### Probe C — Greenberg analogy probes

Vector-arithmetic residual test:
`||(a_pos − a_neg) − (b_pos − b_neg)||₂`
compared against a random-quadruple null distribution (1 000 permutations).
Low residual and low empirical p-value support the tested universal.

### Usage

```bash
python canonical/analyze_geometry.py \
    --checkpoint_dir checkpoints/wals_learned_s42 \
    --output_dir analysis/wals_learned_s42 \
    --top_k 10
```

---

## Cross-database comparison (`compare_databases.py`)

Aligns two canonical checkpoints on their shared-language Glottocode
intersection and tests latent-space similarity (RQ1 second clause).

| Step | Method | Low value means… |
|------|--------|-----------------|
| 2 | Orthogonal Procrustes disparity | Spaces are geometrically similar |
| 3 | Permutation test (500 perms) | p < 0.05 → better than random |
| 4 | CCA canonical correlations | High corr → shared variance |
| 5 | Family-preservation probe (optional) | Same-family languages cluster |

### Usage

```bash
python canonical/compare_databases.py \
    --checkpoint_a checkpoints/wals_learned_s42 \
    --checkpoint_b checkpoints/grambank_learned_s42 \
    --output_dir  analysis/wals_vs_grambank_learned \
    --n_perm 500 --seed 42
```

---

## Seed stability (`seed_stability.py`)

Trains K seeds per (database, architecture) setting and measures
representational stability across random initialisations.

```bash
python canonical/seed_stability.py \
    --checkpoints checkpoints/wals_tcf_s42 checkpoints/wals_tcf_s43 \
                  checkpoints/wals_tcf_s44 checkpoints/wals_tcf_s45 \
                  checkpoints/wals_tcf_s46 \
    --output_dir analysis/stability_wals_tcf
```

Outputs `stability_report.md` and `stability_metrics.json`.

---

## Tests

```bash
python -m pytest canonical/tests/ -v
```

71 tests across four modules; all pass without network access or CLDF data.
