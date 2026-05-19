# RQ4 Full Factorial — Lab Server Runbook

## Overview

This runbook describes how to run the full RQ4 conditioning experiment on a lab server.

**Estimated cost**: ~12 000 conditioned model trainings (baseline F1 is loaded
from pre-computed CSVs; no baseline retraining).

**DO NOT run `--full` locally** — use a lab server with ≥8 CPU cores.

---

## Prerequisites

```bash
# 1. Activate your Python environment
conda activate grambank2vec   # or: source venv/bin/activate

# 2. Install dependencies (if not already installed)
pip install torch scikit-learn pandas scipy pyarrow

# 3. Install URIEL+
cp -r /path/to/URIELPlus/urielplus /usr/local/lib/python3.*/dist-packages/

# 4. Verify smoke tests pass
python canonical/uriel_plus_loader.py --smoke
python canonical/conditioning_pipeline.py \
    --database grambank \
    --architecture learned \
    --data_path /path/to/grambank \
    --seed 42 \
    --smoke
```

---

## Step 1 — Build URIEL+ Vectors (once)

```bash
python canonical/uriel_plus_loader.py --full
```

Output: `analysis/conditioning_uriel_plus/uriel_plus_vectors.parquet`

This produces a single parquet with `glottocode, geo_vec, phylo_vec, has_geo,
has_phylo`.  Approximately 6% of languages have `has_geo=False` and receive
zero vectors by design — no imputation is performed.

---

## Step 2 — Full Factorial (all 20 configurations)

Run all combinations of:
- **database**: wals, grambank
- **architecture**: tcf, learned
- **seed**: 42, 43, 44, 45, 46

Baseline F1 scores are loaded automatically from
`analysis/baselines/comparison_results_{database}.csv` — no baseline models
are trained during this step.

### Option A: Sequential (single machine)

```bash
WALS_PATH=/path/to/wals
GRAMBANK_PATH=/path/to/grambank

for db in wals grambank; do
    DATA_PATH=$( [ "$db" = "wals" ] && echo $WALS_PATH || echo $GRAMBANK_PATH )
    for arch in tcf learned; do
        for seed in 42 43 44 45 46; do
            echo "=== $db / $arch / seed=$seed ==="
            python canonical/conditioning_pipeline.py \
                --database $db \
                --architecture $arch \
                --data_path $DATA_PATH \
                --seed $seed \
                --n_epochs 50 \
                --patience 10 \
                --full \
                --resume
        done
    done
done
```

### Option B: Parallel (SLURM / GNU parallel)

```bash
# Generate all 20 job commands
python3 -c "
import itertools
WALS = '/path/to/wals'
GRAMBANK = '/path/to/grambank'
for db, arch, seed in itertools.product(
    ['wals', 'grambank'],
    ['tcf', 'learned'],
    [42, 43, 44, 45, 46],
):
    data = WALS if db == 'wals' else GRAMBANK
    print(f'python canonical/conditioning_pipeline.py '
          f'--database {db} --architecture {arch} '
          f'--data_path {data} '
          f'--seed {seed} --n_epochs 50 --patience 10 '
          f'--full --resume')
" > jobs.txt

# Run with GNU parallel (N = number of CPU cores)
parallel -j8 < jobs.txt

# Or submit as SLURM array job:
# sbatch --array=1-20 run_conditioning_array.sh
```

### Expected runtime

| Configuration | Approx time per run |
|--------------|---------------------|
| grambank / learned | ~2–3 hours |
| grambank / tcf | ~1–2 hours |
| wals / learned | ~45–90 min |
| wals / tcf | ~30–60 min |

Total wall time (sequential): ~3–5 days. Parallel across 8 cores: ~8–12 hours.

---

## Step 3 — Resume Interrupted Runs

The pipeline writes a checkpoint CSV after every (branch, frac, repeat, conditioning) combo. To resume:

```bash
python canonical/conditioning_pipeline.py \
    --database grambank \
    --architecture learned \
    --data_path /path/to/grambank \
    --seed 42 \
    --full \
    --resume  # ← skips already-completed rows
```

---

## Step 4 — Aggregate and Analyse

Once all 20 runs complete:

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
- `analysis/conditioning_results/{db}_{arch}_s{seed}.csv` — result rows
- `analysis/conditioning_results/config_{db}_{arch}_s{seed}.json` — run config

Each row in the CSV has:
```
database, architecture, seed, branch, macroarea,
in_branch_frac, repeat, conditioning, model_type (baseline|conditioned), f1
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `FileNotFoundError: uriel_plus_vectors.parquet` | Run Step 1 first |
| `FileNotFoundError: comparison_results_*.csv` | Ensure `analysis/baselines/` is present |
| `RuntimeError: CUDA out of memory` | Add `--device cpu` |
| Run interrupted mid-combo | Add `--resume`; the CSV checkpoint is up-to-date |
| `prepare_categorical` returns empty kept_feature_names | Data path points to wrong database |

---

## Constraint 3 Verification

Before running `--full`, confirm that the URIEL+ parquet contains ONLY
geo and phylo vectors (no typological data):

```python
import pyarrow.parquet as pq
t = pq.read_table('analysis/conditioning_uriel_plus/uriel_plus_vectors.parquet')
print(t.schema)
# Must show only: glottocode, geo_vec, phylo_vec, has_geo, has_phylo
# NEVER features.npz (typological data[1])
assert set(t.schema.names) == {"glottocode", "geo_vec", "phylo_vec",
                                "has_geo", "has_phylo"}
```
