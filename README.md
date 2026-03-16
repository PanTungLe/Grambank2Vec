# Replicating "A Probabilistic Generative Model of Linguistic Typology"

Bjerva, Kementchedjhieva, Cotterell & Augenstein (NAACL-HLT 2019)

## Files

| File | Description |
|------|-------------|
| `model.py` | Core PyTorch models: `TypologicalMF` (Section 3) and `TypologicalMF_SemiSup` (Section 4) |
| `data_preparation.py` | Loads WALS from CLDF format, loads eBible texts, binarises features, matches languages |
| `char_lm.py` | Character-level LSTM LM for pre-training language embeddings (Östling & Tiedemann 2017) |
| `evaluation_pipeline.py` | Branch-based splitting, experiment runner, F1 evaluation with argmax decoding |

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

## Step 2: Prepare WALS Data

```bash
python data_preparation.py \
    --wals_repo /path/to/wals \
    --output_csv wals_prepared.csv
```

This reads the CLDF CSVs, joins languages with their feature values (using
human-readable labels from `codes.csv`), applies frequency filters (≥10 features
per language, ≥10 languages per feature value), and saves a flat CSV.

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

### Core model only (T-CF):

```bash
python evaluation_pipeline.py \
    --wals_repo /path/to/wals \
    --embed_dim 64 \
    --n_epochs 10 \
    --batch_size 64 \
    --l2_reg 0.1 \
    --n_repeats 5 \
    --output_csv results.csv
```

### With semi-supervised extension:

```bash
python evaluation_pipeline.py \
    --wals_repo /path/to/wals \
    --pretrained_embs charlm_output/lang_embeddings_aligned.npy \
    --embed_dim 64 \
    --n_epochs 10 \
    --batch_size 64 \
    --l2_reg 0.1 \
    --n_repeats 5 \
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

## Key Implementation Details

### WALS CLDF loading
The CLDF format stores values in long format (`values.csv`), where each row
is one (language, feature, value) triple. We pivot this to wide format, using
human-readable value labels from `codes.csv` (e.g. "SOV" rather than numeric
codes). The `Genus` column in `languages.csv` corresponds to what the paper
calls "branch" — this is the unit for train/eval splitting.

### Binarisation
Multi-valued WALS features (e.g. Feature 81A with 7 word-order values) are
one-hot encoded into binary columns. At test time, predictions are decoded by
taking argmax over predicted probabilities within each original feature.

### Split integrity
All binary columns from the same original feature for a given language go
to the same split (train or eval). This prevents leaking partial information
about multi-valued features.

### L2 regularisation
Weight decay of 0.1 in Adam corresponds to the Gaussian prior
p(λ) = N(0, σ²I) with σ² = 10.

### Evaluation scope
Only branches with ≥4 languages (after filtering to those with ≥10 features)
are evaluated. This yields ~36 branches and ~448 languages.

### Script filtering for Bible data
The paper only uses Bible translations in Latin, Cyrillic, and Greek scripts.
`data_preparation.py` implements this filter using Unicode character name
inspection.
