# RQ4 Full Factorial — Lab Server Runbook

## Overview

This runbook describes how to run the full RQ4 conditioning experiment on a lab server.

**Estimated cost**: ~24 000 individual model trainings (baseline + conditioned × all branch/frac/repeat combinations).

**DO NOT run `--full` locally** — use a lab server with ≥8 CPU cores.

---

## Prerequisites

```bash
# 1. Activate your Python environment
conda activate grambank2vec   # or: source venv/bin/activate

# 2. Install dependencies (if not already installed)
pip install torch scikit-learn pandas scipy pyarrow fancyimpute cvxpy

# 3. Install URIEL+
cp -r /path/to/URIELPlus/urielplus /usr/local/lib/python3.*/dist-packages/

# 4. Verify smoke tests pass
python canonical/uriel_plus_loader.py --imputation familymean --smoke
python canonical/conditioning_pipeline.py \
    --database grambank \
    --architecture learned \
    --data_path /path/to/grambank \
    --imputation familymean \
    --seed 42 \
    --smoke
```

---

## Step 1 — Build URIEL+ Vectors (once)

```bash
for method in familymean knn softimpute; do
    python canonical/uriel_plus_loader.py \
        --imputation $method \
        --full
done
```

Outputs: `analysis/conditioning_uriel_plus/uriel_plus_vectors_{method}.parquet`

---

## Step 2 — Full Factorial (all 60 configurations)

Run all combinations of:
- **database**: wals, grambank
- **architecture**: tcf, learned
- **imputation**: familymean, knn, softimpute
- **seed**: 42, 43, 44, 45, 46

### Option A: Sequential (single machine)

```bash
WALS_PATH=/path/to/wals
GRAMBANK_PATH=/path/to/grambank

for db in wals grambank; do
    DATA_PATH=$( [ "$db" = "wals" ] && echo $WALS_PATH || echo $GRAMBANK_PATH )
    for arch in tcf learned; do
        for method in familymean knn softimpute; do
            for seed in 42 43 44 45 46; do
                echo "=== $db / $arch / $method / seed=$seed ==="
                python canonical/conditioning_pipeline.py \
                    --database $db \
                    --architecture $arch \
                    --data_path $DATA_PATH \
                    --imputation $method \
                    --seed $seed \
                    --n_epochs 50 \
                    --patience 10 \
                    --full \
                    --resume
            done
        done
    done
done
```

### Option B: Parallel (SLURM / GNU parallel)

```bash
# Generate all 60 job commands
python3 -c "
import itertools, os
WALS = '/path/to/wals'
GRAMBANK = '/path/to/grambank'
for db, arch, method, seed in itertools.product(
    ['wals', 'grambank'],
    ['tcf', 'learned'],
    ['familymean', 'knn', 'softimpute'],
    [42, 43, 44, 45, 46],
):
    data = WALS if db == 'wals' else GRAMBANK
    print(f'python canonical/conditioning_pipeline.py '
          f'--database {db} --architecture {arch} '
          f'--data_path {data} --imputation {method} '
          f'--seed {seed} --n_epochs 50 --patience 10 '
          f'--full --resume')
" > jobs.txt

# Run with GNU parallel (N = number of CPU cores)
parallel -j8 < jobs.txt

# Or submit as SLURM array job:
# sbatch --array=1-60 run_conditioning_array.sh
```

### Expected runtime

| Configuration | Approx time per run |
|--------------|---------------------|
| grambank / learned | ~3–5 hours |
| grambank / tcf | ~2–3 hours |
| wals / learned | ~1–2 hours |
| wals / tcf | ~1 hour |

Total wall time (sequential): ~7–10 days. Parallel across 8 cores: ~1–2 days.

---

## Step 3 — Resume Interrupted Runs

The pipeline writes a checkpoint CSV after every (branch, frac, repeat) combo. To resume:

```bash
python canonical/conditioning_pipeline.py \
    --database grambank \
    --architecture learned \
    --data_path /path/to/grambank \
    --imputation familymean \
    --seed 42 \
    --full \
    --resume  # ← skips already-completed rows
```

---

## Step 4 — Aggregate and Analyse

Once all 60 runs complete:

```bash
# Merge all CSVs into a single results file
python3 -c "
import pandas as pd, glob
dfs = [pd.read_csv(f) for f in glob.glob('analysis/conditioning_results/*.csv')
       if not f.endswith('_config.csv')]
merged = pd.concat(dfs, ignore_index=True)
merged.to_csv('analysis/conditioning_results/all_results.csv', index=False)
merged.to_parquet('analysis/conditioning_results/all_results.parquet', index=False)
print(f'Total rows: {len(merged)}')
"

# Run the analysis (Phase 9)
python canonical/conditioning_analyze.py \
    --results analysis/conditioning_results/all_results.parquet \
    --strata analysis/conditioning_strata.csv \
    --out_dir analysis/conditioning_analysis
```

---

## Output Files

After a successful run, each configuration produces:
- `analysis/conditioning_results/{db}_{arch}_{method}_s{seed}.csv` — result rows
- `analysis/conditioning_results/config_{db}_{arch}_{method}_s{seed}.json` — run config

Each row in the CSV has:
```
database, architecture, imputation, seed, branch, macroarea,
in_branch_frac, repeat, model_type (baseline|conditioned), f1
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError: uriel_plus_vectors_*.parquet` | Run Step 1 first |
| `RuntimeError: CUDA out of memory` | Add `--device cpu` |
| `fancyimpute` import error | `pip install fancyimpute cvxpy` |
| Run interrupted mid-combo | Add `--resume`; the CSV checkpoint is up-to-date |
| `prepare_categorical` returns empty kept_feature_names | Data path points to wrong database |

---

## Constraint 3 Verification

Before running `--full`, confirm that the URIEL+ parquet contains ONLY
geo and phylo vectors (no typological data):

```python
import pyarrow.parquet as pq
t = pq.read_table('analysis/conditioning_uriel_plus/uriel_plus_vectors_familymean.parquet')
print(t.schema)  # Must show: geo_vec (list<float32>), phylo_vec (list<float32>)
# geo_dim=299 (geocoord_features.npz), phylo_dim=3718 (family_features.npz)
# NEVER features.npz (typological data[1])
```
