# Downstream Transfer Experiment — Lab Server Runbook

## Prerequisites

```bash
source venv/bin/activate
pip install conllu  # optional — we use our own minimal parser
```

Verify GPU:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Full Run (GPU, all 59 languages, all 5 seeds)

Estimated time: ~4–6 hours on a single GPU.

```bash
cd ~/Grambank2Vec

# Phase 1: Extract representations (fast, ~5 min, no GPU needed)
python downstream/run_downstream.py \
    --ckpt_wals checkpoints/wals_learned_s42 \
    --ckpt_grambank checkpoints/grambank_learned_s42 \
    --out_dir downstream_results \
    --device cuda \
    --seeds 42,43,44,45,46 \
    --skip_download \
    --skip_transfer

# Phase 2: Download UD treebanks (~5-10 min, network-dependent)
python downstream/ud_data.py \
    --out_dir downstream_results/ud_data

# Phase 3: Build transfer matrix (~3-5 hrs with GPU)
python downstream/transfer_matrix.py \
    --manifest downstream_results/ud_data/manifest.json \
    --out_dir downstream_results/transfer \
    --device cuda \
    --n_epochs 30 \
    --patience 5 \
    --max_train_sents 5000 \
    --max_test_sents 500

# Phase 4: Evaluate (fast, ~5 min, no GPU needed)
python downstream/run_downstream.py \
    --out_dir downstream_results \
    --only_eval \
    --uriel_parquet analysis/conditioning_uriel_plus/uriel_plus_vectors.parquet
```

## Recommended Single-Command Run (GPU, seed 42 only, ~1.5 hrs)

```bash
python downstream/run_downstream.py \
    --ckpt_wals checkpoints/wals_learned_s42 \
    --ckpt_grambank checkpoints/grambank_learned_s42 \
    --out_dir downstream_results \
    --device cuda \
    --n_epochs 30 \
    --patience 5 \
    --max_train_sents 5000 \
    --max_test_sents 500 \
    --seeds 42
```

## Smoke Test (CPU, 3 languages, ~2-5 min)

```bash
python downstream/run_downstream.py \
    --out_dir downstream_smoke \
    --smoke \
    --device cpu
```

## Key Output Files

| File | Description |
|------|-------------|
| `downstream_results/transfer/transfer_matrix.csv` | N×N accuracy matrix |
| `downstream_results/distances/dist_*.csv` | Pairwise distance matrices |
| `downstream_results/results/evaluation_summary.csv` | Main results table |
| `downstream_results/results/per_target_*.csv` | Per-language breakdown |

## What the Results Mean

The `evaluation_summary.csv` has one row per representation:
- `repr_wals_s42_A` — WALS direct language embedding
- `repr_wals_s42_B` — WALS expected value embedding (main contribution)
- `repr_wals_s42_C` — WALS predicted binary profile
- `repr_wals_s42_B_masked30/50/70` — B under 30/50/70% feature masking
- `repr_grambank_s42_*` — same for Grambank
- `joint_B_wals_grambank` — concatenated WALS+Grambank B
- `uriel_geo`, `uriel_phylo`, `uriel_geo_phylo` — URIEL+ baselines
- `random` — sanity check

Key metrics:
- **Spearman ρ**: higher = better rank correlation with actual transfer
- **NDCG@3**: higher = predicted top-3 sources are actually good sources
- **Regret@1**: lower = less accuracy lost by choosing top-predicted source
- **Pairwise accuracy**: higher = pairwise comparison of source quality correct

## Thesis Claim Being Tested

> Categorical learned feature-value embeddings (Repr B) predict
> cross-lingual transfer better than binary typological vectors (Repr C
> and URIEL baselines), especially when the target language has sparse
> typological coverage (masking experiments).

A positive result would show:
1. Repr B > Repr C in Spearman ρ (learned geometry helps)
2. Repr B_masked50 ≈ Repr B (graceful degradation under sparsity)
3. Repr C_masked50 << Repr C (binary vectors fail when features are missing)

