# UFAL Compatibility Fixes

Compatibility-fixed versions of UFAL's [ST2020 lang_embedding scripts](https://github.com/ufal/ST2020/tree/master/scripts/lang_embedding) for Python 3.11 / numpy 1.26 / sklearn 1.x / pandas 2.x / TF 2.15.

## What changed vs original

### `dataset.py`
| Fix | Reason |
|-----|--------|
| Absolute data paths via `__file__` | Original uses `../../data/` relative to cwd; fails outside lang_embedding/ |
| `np.array(..., dtype=object)` in `create_dataset` | numpy ≥ 1.24 raises ValueError for `[int, (a,b), (a,b), ...]` inhomogeneous arrays |
| `sklearn.compute_class_weight(..., classes=..., y=...)` | sklearn ≥ 1.0 requires keyword arguments |
| Pre-built `self._lang_pairs` set per language | numpy 1.26 `(a,b) in object_array` broadcasts instead of element-comparing; sets fix both correctness and speed |
| `preprocess()` handles `index` column | `test_x.csv` uses `index` not `Unnamed: 0` as index column |

### `callbacks.py`
| Fix | Reason |
|-----|--------|
| Absolute data paths | Same as above |
| `_load()` helper drops `index` or `Unnamed: 0` | `test_x.csv` / `dev_x.csv` differ in index column name |
| `fill_test()` column alignment guard | test_x has 2 fewer WALS features than dev_x; naive concat fails |

### `run_replication.py` (new)
Replaces `run.py` with:
- Argparse CLI (`--epochs`, `--steps`, `--embed`, `--clusters`, `--no-knn`, `--train-on-test`)
- Calls `keras_model.fit()` directly (bypasses `Model.train()` which hardcodes callbacks)
- `--no-knn` actually disables KNN (saves ~25s/epoch)
- `--train-on-test` switches training to `test_x` (trains test embeddings inline)
- ModelCheckpoint every 20 epochs for crash recovery
- Clean summary at end

## Installation

```bash
pip install tensorflow==2.15 tensorflow-addons numpy pandas scikit-learn
git clone https://github.com/ufal/ST2020.git
# Copy these 3 files into ST2020/scripts/lang_embedding/
cp dataset.py callbacks.py run_replication.py /path/to/ST2020/scripts/lang_embedding/
```

## Usage

```bash
# Replicate published 0.755 (uses dev_x, evaluates on dev each epoch)
python run_replication.py --epochs 200 --steps 1000 --embed 512 --clusters 300 --no-knn

# Timing: ~38s/epoch on 4-core CPU → ~2.2 hours total

# Alternative: train test embeddings inline (for test neural accuracy)
python run_replication.py --epochs 200 --steps 1000 --no-knn --train-on-test
```

## Key findings

1. **UFAL's evaluation metric is micro accuracy** (total correct / total blanks), NOT the macro-genus accuracy used in the SIGTYP competition.
2. **0.755 = combined DEV accuracy** (probabilistic 0.711 + neural on trained dev embeddings 0.698).
3. **Test neural accuracy is ~random** with original `dev_x` setting, because test embeddings are never trained — the Filler callback uses `lang_id = cnt + 1208` which indexes into the embedding table at positions that receive zero gradient updates during the 200-epoch run.
4. **The commented-out line** `# self.train_dataset = self.create_dataset(pd.concat([self.train_y, self.test_x]))` would train test embeddings inline (alternative experiment).
5. **Our PyTorch reimplementation gap** (0.72 vs 0.755) is explained by (a) micro vs macro metric difference and (b) post-training fine-tuning (100 steps) vs inline training (200 epochs × 1000 steps) for dev embeddings.
