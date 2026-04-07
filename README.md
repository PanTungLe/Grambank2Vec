# Replicating "A Probabilistic Generative Model of Linguistic Typology"

Bjerva, Kementchedjhieva, Cotterell & Augenstein (NAACL-HLT 2019)

## Files

| File | Description |
|------|-------------|
| `model.py` | Core PyTorch models: `TypologicalMF` (Section 3) and `TypologicalMF_SemiSup` (Section 4) |
| `model_learned.py` | Learned feature-value embeddings with softmax: `TypologicalMF_Learned` |
| `data_preparation.py` | Loads WALS and Grambank from CLDF format, loads eBible texts, binarises features, matches languages |
| `char_lm.py` | Character-level LSTM LM for pre-training language embeddings (Östling & Tiedemann 2017) |
| `model_learned.py` | Learned feature-value embedding model with softmax prediction |
| `compare_models.py` | Side-by-side comparison of binary baseline and learned model |
| `evaluation_pipeline.py` | Branch-based splitting, experiment runner, F1 evaluation with argmax decoding |
| `compare_models.py` | Runs binary baseline (T-CF) and learned embeddings side-by-side on identical splits |
| `run_comparison.py` | Unified entry point for running comparisons on WALS or Grambank |

## Supported Databases

| Database | Languages | Features | Feature type | Genus source |
|----------|-----------|----------|-------------|--------------|
| [WALS](https://github.com/cldf-datasets/wals) | ~2,600 | ~200 | Multi-valued categorical | `Genus` column |
| [Grambank](https://github.com/glottobank/grambank) | ~2,400 | ~200 | Mostly binary (0/1), some 3+ valued | Glottolog classification or `Family` column |

## Requirements

```bash
pip install torch numpy pandas scikit-learn
```

---

## Step 1: Clone Data Sources

### WALS (World Atlas of Language Structures)

```bash
git clone https://github.com/cldf-datasets/wals.git
```

This gives you the CLDF StructureDataset in the `cldf/` subdirectory:

- `cldf/languages.csv` — language metadata (ID, Name, Genus, Family, Macroarea, ...)
- `cldf/values.csv` — observed feature values (Language_ID, Parameter_ID, Value, Code_ID, ...)
- `cldf/codes.csv` — maps numeric Code_IDs to human-readable value labels
- `cldf/parameters.csv` — feature definitions (e.g. "81A" = "Order of Subject, Object and Verb")

Our `data_preparation.py` reads these files directly — no manual conversion needed.

### Grambank

```bash
git clone https://github.com/glottobank/grambank.git
```

Grambank uses Glottocodes as language IDs and has a similar CLDF layout under `cldf/`.
Features are mostly binary (0/1 = yes/no) but some have 3+ values. Values like `"?"` 
(uncertain/not determinable) are automatically filtered out during loading.

#### Genus groupings for Grambank

Grambank does not have a `Genus` column like WALS. Two options for the branch-based
evaluation protocol:

1. **Glottolog classification** (recommended): provides accurate genus-level groupings.
   Requires cloning the Glottolog CLDF dataset:

   ```bash
   git clone https://github.com/glottolog/glottolog-cldf.git
   ```

2. **Family column** (fallback): uses Grambank's `Family` column directly. Coarser
   than WALS genus — results may not be directly comparable.

### eBible (Multilingual Bible Corpus)

Two Bible corpora are supported. Use `--bible_source` to choose which one(s):

| Source | Repository | Translations |
|--------|-----------|-------------|
| `ebible` (default) | [BibleNLP/ebible](https://github.com/BibleNLP/ebible) | ~1079 |
| `parabible` | [LingConLab/parabible](https://github.com/LingConLab/parabible) | ~1846 |
| `both` | Both of the above (merged, keeping the longer text per language) | Combined |

```bash
# Clone manually (optional — run_all.py auto-clones when needed)
git clone https://github.com/BibleNLP/ebible.git
git clone https://github.com/LingConLab/parabible.git
```

The eBible corpus stores each translation as a plain-text file with one verse
per line in the `corpus/` subdirectory. Filenames follow
`<languageCode>-<variant>.txt` (e.g. `eng-eng_kjv.txt`). Blank lines indicate
missing verses and `<range>` tokens mark grouped verses; both are automatically
skipped by the loader.

The ParaBible corpus (~1846 translations) distributes its text files as a
downloadable zip archive. The `run_all.py` script handles downloading and
extraction automatically. Text files use TAB-separated `<verse_id>\t<text>`
lines.

`run_all.py` will clone the required repositories automatically based on the
`--bible_source` setting.

---

## Step 2: Prepare Data

### WALS

```bash
python data_preparation.py \
    --wals_repo /path/to/wals \
    --output_csv wals_prepared.csv
```

This reads the CLDF CSVs, joins languages with their feature values (using
human-readable labels from `codes.csv`), applies frequency filters (≥10 features
per language, ≥10 languages per feature value), and saves a flat CSV.

### Grambank

Grambank data is loaded directly by the experiment scripts (no separate
preparation step needed):

```python
from data_preparation import load_grambank_cldf

# With Glottolog genus mapping (recommended):
df, feature_cols = load_grambank_cldf(
    "/path/to/grambank",
    genus_source="glottolog",
    glottolog_repo_dir="/path/to/glottolog-cldf",
)

# With Family-based genus (no Glottolog needed):
df, feature_cols = load_grambank_cldf(
    "/path/to/grambank",
    genus_source="family",
)
```

---

## Step 3: (Optional) Train Language Embeddings

The semi-supervised extension requires 64-dimensional language embeddings from
a character-level LSTM language model. Skip this step to run only the core
T-CF model.

### 3a. Prepare Bible texts

```bash
# Using eBible (default)
python data_preparation.py \
    --wals_repo /path/to/wals \
    --ebible_dir /path/to/ebible/corpus \
    --charlm_output charlm_data \
    --output_csv wals_prepared.csv

# Using ParaBible
python data_preparation.py \
    --wals_repo /path/to/wals \
    --bible_source parabible \
    --parabible_dir /path/to/parabible/corpus-txt \
    --charlm_output charlm_data

# Using both (merged)
python data_preparation.py \
    --wals_repo /path/to/wals \
    --bible_source both \
    --ebible_dir /path/to/ebible/corpus \
    --parabible_dir /path/to/parabible/corpus-txt \
    --charlm_output charlm_data
```

This loads Bible texts, filters to Latin/Cyrillic/Greek scripts (matching the
paper), and writes per-language text files. It also attempts to match WALS
language codes to Bible translation IDs.

**Important:** The automatic matching is heuristic. You will likely need to
create a manual mapping CSV (`wals_code,bible_lang_id`) for best coverage.

### 3b. Train the character-level LM

```bash
python char_lm.py \
    --bible_source ebible \
    --ebible_dir /path/to/ebible/corpus \
    --output_dir charlm_output \
    --hidden_dim 1024 \
    --char_emb_dim 128 \
    --lang_emb_dim 64 \
    --max_epochs 30 \
    --patience 5 \
    --device cuda   # use 'cpu' if no GPU
```

Architecture (following Östling & Tiedemann 2017):
- 2-layer stacked character LSTM with 1024-dim hidden states
- 128-dim character embeddings
- 64-dim language embeddings concatenated at every time step
- Adam optimiser with early stopping

Output: `charlm_output/lang_embeddings.npy` — shape `(n_languages, 64)`

### 3c. Align embeddings with WALS

The embedding rows must be ordered to match the WALS languages after filtering.
Use `charlm_output/idx2lang.json` (maps embedding index → language ID) and the
matched language list from Step 2 to create a properly aligned `.npy` file.

---

## Step 4: Run Experiments

### Unified comparison (recommended)

The `run_comparison.py` script runs both the binary baseline (T-CF) and
learned feature-value embedding model side-by-side on identical splits:

```bash
# WALS
python run_comparison.py --database wals --wals_repo /path/to/wals

# Grambank with Glottolog genus mapping
python run_comparison.py --database grambank \
    --grambank_repo /path/to/grambank \
    --glottolog_repo /path/to/glottolog-cldf

# Grambank with Family-based genus (no Glottolog needed)
python run_comparison.py --database grambank \
    --grambank_repo /path/to/grambank \
    --genus_source family
```

### Binary baseline only (T-CF)

```bash
# WALS
python evaluation_pipeline.py --database wals \
    --wals_repo /path/to/wals \
    --output_csv results.csv

# Grambank
python evaluation_pipeline.py --database grambank \
    --grambank_repo /path/to/grambank \
    --glottolog_repo /path/to/glottolog-cldf \
    --output_csv results_grambank.csv
```

### With semi-supervised extension (WALS only)

```bash
python evaluation_pipeline.py --database wals \
    --wals_repo /path/to/wals \
    --pretrained_embs charlm_output/lang_embeddings_aligned.npy \
    --output_csv results.csv
```

### Quick smoke test (no data needed):

```bash
python model.py
```

---

## Step 5: Interpret Results

The output CSV contains per-run results. The script prints an aggregate summary
matching Table 1 from the paper:

| In-branch % | T-CF F1 | SemiSup F1 |
|-------------|---------|------------|
| 0%          | ~0.40   | ~0.39      |
| 1%          | ~0.46   | ~0.53      |
| 5%          | ~0.66   | ~0.76      |
| 10%         | ~0.78   | ~0.90      |
| 20%         | ~0.88   | ~0.98      |

---

## Learned Feature-Value Embedding Model

In addition to the binary baseline from Bjerva et al. (2019), this repository
includes a **learned feature-value embedding model** (`model_learned.py`) that
addresses two limitations of the one-hot binarisation approach.

### How it differs from the binary baseline

| Aspect | Binary (T-CF) | Learned |
|--------|--------------|---------|
| Feature representation | One-hot binary columns | Single categorical column per feature |
| Value geometry | Orthogonal (one-hot) | Learned dense embeddings |
| Prediction | Independent sigmoids + post-hoc argmax | Softmax over values (natively categorical) |
| Loss | Binary cross-entropy | Categorical cross-entropy (NLL) |
| Mutual exclusivity | Enforced post-hoc | Built into architecture |

### Theoretical motivation

The binary baseline treats each feature value as an independent binary toggle --
a "Principles & Parameters" style approach. This means SOV and OVS are as
different as SOV and SVO, even though both SOV and OVS are verb-final. The model
can also predict SOV=0.8 and SVO=0.7 simultaneously, which is incoherent.

The learned model treats values as points in a shared embedding space. Values of
the same feature that behave similarly across languages (e.g. verb-final orders)
can learn to cluster together. The softmax ensures predictions are proper
probability distributions over mutually exclusive outcomes.

### Running the comparison

```bash
# Compare both models on specific branches
python compare_models.py --wals_repo /path/to/wals --branches Germanic Slavic

# Full comparison with custom hyperparameters
python compare_models.py --wals_repo /path/to/wals --embed_dim 64 --n_epochs 10

# Smoke test (no data needed)
python model_learned.py
```

### What to look for in results

- **F1 scores**: Compare T-CF (binary) vs Learned at each in-branch fraction.
  The learned model should benefit especially at low in-branch fractions where
  sharing structure between values matters most.
- **Value embedding geometry**: After training on real WALS data, check whether
  typologically similar values cluster together. For example, for word order
  (feature 81A), SOV and OVS should have smaller cosine distance than SOV and
  SVO, reflecting their shared verb-final property.
- **Coherent predictions**: The learned model cannot produce incoherent
  predictions like P(SOV)=0.8 and P(SVO)=0.7 -- the softmax ensures the
  probabilities sum to 1.

---

## Key Implementation Details

### WALS CLDF loading
The CLDF format stores values in long format (`values.csv`), where each row
is one (language, feature, value) triple. We pivot this to wide format, using
human-readable value labels from `codes.csv` (e.g. "SOV" rather than numeric
codes). The `Genus` column in `languages.csv` corresponds to what the paper
calls "branch" — this is the unit for train/eval splitting.

### Grambank CLDF loading
Grambank uses the same CLDF structure but with Glottocodes instead of WALS
codes. Values are treated as categorical string labels (via `codes.csv`), never
as numeric integers. Observations with `Value="?"` (uncertain/not determinable)
and labels like "Not known", "Uncertain", or "Not applicable" are automatically
filtered out before the data is pivoted to wide format.

### Grambank and model behaviour
Grambank features are mostly binary (yes=1, no=0). For the **binary baseline**
(T-CF), binarisation mostly produces single columns rather than one-hot
expansions, so the binary matrix is closer to the original data than for WALS.
For the **learned embedding model**, binary features get two value embeddings
each. The model's advantage over the baseline comes primarily from the shared
value embedding space allowing cross-feature learning, and from features with
3+ values getting proper categorical treatment via softmax.

### Grambank genus groupings
Grambank does not have a `Genus` column. Two strategies are supported:
- `genus_source="glottolog"`: extracts a configurable level of the Glottolog
  classification tree (default level 2, which approximates WALS genus). Supports
  Glottolog CLDF versions with a Classification column, or with Family_ID +
  Parent_ID columns for hierarchy reconstruction.
- `genus_source="family"`: uses the Family column directly (coarser).

### Binarisation
Multi-valued features (e.g. Feature 81A with 7 word-order values) are
one-hot encoded into binary columns. At test time, predictions are decoded by
taking argmax over predicted probabilities within each original feature.
The function is named `binarise_features()` (database-agnostic); `binarise_wals`
is kept as an alias for backwards compatibility.

### Split integrity
All binary columns from the same original feature for a given language go
to the same split (train or eval). This prevents leaking partial information
about multi-valued features.

### L2 regularisation
Weight decay of 0.1 in Adam corresponds to the Gaussian prior
p(λ) = N(0, σ²I) with σ² = 10.

### Evaluation scope
Only branches with ≥4 languages (after filtering to those with ≥10 features)
are evaluated. For WALS this yields ~36 branches and ~448 languages. For
Grambank the number depends on the genus source and filtering thresholds.
