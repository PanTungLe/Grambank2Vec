# API Audit — existing codebase, for canonical-training scaffolding

This document catalogues the actual class/function signatures in the
existing pipeline so that `canonical/train_canonical.py` (Phase 2) can call
into them without reinventing data structures. Read this before reading the
prompt's "I think the API is X" assumptions — those names need translating.

Smoke test confirmed: `python model.py` runs end-to-end on the synthetic
toy data with the freshly installed PyTorch 2.x. The repo is on
`claude/canonical-model-training-ijbwt`, working tree clean.

---

## 1. Model classes

### 1a. T-CF (binary one-hot Bernoulli baseline) — `model.py`

- **Class:** `model.TypologicalMF`
- **Constructor:** `TypologicalMF(n_languages: int, n_features: int, embed_dim: int = 64)`
  - `n_features` here is the number of **binary columns** (i.e. one per
    one-hot expanded value), NOT the number of original WALS/Grambank
    features. This is the size of `binarise_features`'s output.
- **Forward:** `forward(lang_idx, feat_idx) -> (preds, batch_l2)`
  - `preds` is sigmoid probability per (lang, binary-col) cell
  - `batch_l2` is per-batch L2 over batch-accessed embeddings only
    (mean over `B`)
- **Embedding attributes:**
  - Language matrix: `model.lang_embeddings.weight`  shape `(n_languages, embed_dim)`
  - Per-binary-column matrix: `model.feat_embeddings.weight` shape `(n_features, embed_dim)`
- **L2 mechanism:** in-`forward`, returned alongside preds. Trainer adds
  `(l2_reg / 2.0) * batch_l2` to BCE. `Adam(weight_decay=0)` is mandatory
  (see giant comment in `model.py` lines 99–112 and 243–251).
- **Index convention at training time:** triples are
  `(lang_idx, binary_col_idx, binary_value)`. The "feature" axis is the
  binary-column axis. Built by `WALSDataset(lang_indices, feat_indices, values)`.

### 1b. Learned (softmax + cross-entropy) — `model_learned.py`

- **Class:** `model_learned.TypologicalMF_Learned`
- **Constructor:**
  `TypologicalMF_Learned(n_languages: int, n_total_values: int,
                         feat_to_global_ids: Dict[int, List[int]],
                         embed_dim: int = 64)`
  - `n_total_values` is the count of unique (feature, value) pairs across
    all features — i.e. the size of the value embedding table.
  - `feat_to_global_ids[feat_idx]` is a list of global value IDs for that
    feature. The model uses this to build padded-batched softmax tensors
    as `register_buffer`s `padded_ids` / `valid_mask`.
- **Forward:** `forward(lang_idx, feat_idx, value_idx) -> (log_probs, batch_l2)`
  - `value_idx` is the **local** value index within the feature
    (0..K_f − 1), NOT the global ID.
  - `log_probs` is the log-softmax probability of the true value.
  - Trainer's loss is `-log_probs.mean() + (l2_reg / 2.0) * batch_l2`.
- **Embedding attributes:**
  - Language matrix: `model.lang_embeddings.weight`  shape `(n_languages, embed_dim)`
  - **Value matrix: `model.value_embeddings.weight` shape `(n_total_values, embed_dim)`** —
    note: `value_embeddings`, not `featvalue_embeddings` (the prompt's guess).
  - Per-feature value-emb extractor: `model.get_value_embeddings_for_feature(feat_idx)`
    returns the K_f × d slice for one feature.
- **Index convention at training time:** triples are
  `(lang_idx, original_feat_idx, local_value_idx)`. Built by
  `CategoricalTypDataset(lang_indices, feat_indices, value_indices)`.

### 1c. SemiSup variant — `model.py:TypologicalMF_SemiSup`

Out of scope for Phase 1–7 (no fixed pretrained char-LM embeddings used in
canonical full-data training). Documented for completeness.

---

## 2. Data preparation

### 2a. WALS loader

- `data_preparation.load_wals_cldf(wals_repo_dir, min_lang_features=10,
                                    min_feature_value_langs=10)
   -> (df: pd.DataFrame, feature_cols: List[str])`
- `df` is one row per language, with metadata columns plus one column per
  feature named `feat_<ParameterID>` (e.g. `feat_81A`).
- **Language identifier column: `wals_code`** (a WALS internal code such as
  `eng`, `mlt`). NOT a Glottocode. There is also an `iso639p3code` column
  if WALS metadata exposes it. **Translation step required** for cross-DB
  comparison (Phase 5).

### 2b. Grambank loader

- `data_preparation.load_grambank_cldf(grambank_repo_dir,
                                       min_lang_features=10,
                                       min_feature_value_langs=10,
                                       genus_source="glottolog",
                                       glottolog_repo_dir=None,
                                       glottolog_genus_level=2)
   -> (df: pd.DataFrame, feature_cols: List[str])`
- `df` has metadata + `feat_<ParameterID>` columns (e.g. `feat_GB020`).
- **Language identifier column: `glottocode`** (matches Grambank's native
  Language_ID, which IS a Glottocode). Phase 5 join key — no translation
  needed on this side.

### 2c. Categorical encoding (Learned)

- `model_learned.prepare_categorical(df, feature_cols)
   -> (cat_matrix, kept_feature_names, feat_to_global_ids, feat_to_value_names)`
  - `cat_matrix`: `(n_langs, n_features)` int, `-1` for missing, otherwise
    the local value index (0..K_f − 1).
  - `kept_feature_names`: list of feature column names actually used
    (drops single-value features).
  - `feat_to_global_ids`: `{feat_idx (int): [global_value_id, ...]}`, used
    to build the model's `padded_ids` buffer.
  - `feat_to_value_names`: `{feat_idx (int): [value_label_str, ...]}` in
    the same order as `feat_to_global_ids[feat_idx]`. Value labels are
    the human-readable strings from `codes.csv` (e.g. `"SOV"`,
    `"Postpositions"`, `"Numeral classifiers"`).
- The "global value ID" is just an integer — there is no built-in string
  label like `"81A=SOV"`. The Phase 4 geometry probe must construct labels
  itself by zipping `kept_feature_names` × `feat_to_value_names` to recover
  `"feat_81A=SOV"` (or strip the `feat_` prefix → `"81A=SOV"`).

### 2d. Binarisation (T-CF)

- `data_preparation.binarise_features(df, feature_cols)
   -> (binary_matrix, binary_col_names, feature_groups, feature_value_names)`
  - `binary_matrix`: `(n_langs, n_binary_cols)` float, NaN for missing.
  - `binary_col_names`: e.g. `"feat_81A=SOV"` for ≥3-valued features, or
    `"feat_GB020"` for already-binary features (no `=value` suffix).
  - `feature_groups`: `{original_feat_name: [binary_col_indices]}`.
  - `feature_value_names`: `{original_feat_name: [val0, val1, ...]}`.
- Important quirk: 2-valued features become a **single** binary column
  (no `=value` suffix). For T-CF geometry probes, the embedding for that
  column represents the "1" value (alphabetically-second). A column for
  the "0" value does not exist.
- Backwards-compat alias: `binarise_wals` → `binarise_features`.

### 2e. Full-data loaders (no held-out branch)

There is no separate "load everything" function — `load_wals_cldf` and
`load_grambank_cldf` already return the full database as a single DataFrame.
Branch splitting happens later in
`evaluation_pipeline.split_by_branch` (binary) and
`model_learned.split_by_branch_categorical`. For canonical training we
simply skip those splitters and emit triples directly from `binary_matrix`
or `cat_matrix` with no branch held out.

---

## 3. L2 / regularisation

For BOTH architectures: the model returns `(preds_or_logprobs, batch_l2)`
where `batch_l2` is computed only over batch-accessed embeddings. The
trainer adds `(l2_reg / 2.0) * batch_l2` to the prediction loss. Adam is
constructed with `weight_decay=0`. The existing pipeline uses
`l2_reg=0.1` (default in `evaluation_pipeline.py`, `compare_models.py`,
`run_all.py`). Canonical training will keep the same default, expose it
as `--l2_coef`.

---

## 4. Glottocode normalisation plan (for Phase 5 cross-DB join)

- **Grambank side:** `df["glottocode"]` is already a Glottocode. Use as-is.
- **WALS side:** `df["wals_code"]` is a WALS internal code. To convert:
  the WALS CLDF `cldf/languages.csv` includes a `Glottocode` column. Add a
  helper `canonical/utils.py::wals_code_to_glottocode(wals_repo_dir) -> Dict[str, str]`
  that reads `languages.csv` and maps `ID` → `Glottocode`. Apply this
  mapping in `train_canonical.py` AFTER `load_wals_cldf` produces `df`,
  so that the dumped `lang2id.json` is keyed by Glottocode.

  The WALS `Glottocode` column is occasionally empty (some WALS languages
  predate full Glottolog coverage). Languages without a Glottocode will
  be EXCLUDED from the dumped lang2id.json — this guarantees the
  intersection-based Phase 5 logic is well-defined. The full lang2id (incl.
  WALS-code-only languages) will also be saved as `lang2id_full.json` for
  bookkeeping, but the canonical join uses the Glottocode-keyed file.

---

## 5. Featvalue label convention (for Phase 4 geometry probes)

The codebase nowhere materialises a single canonical string ID like
`"81A=SOV"`. The closest ready-made artifact is:

- **For Learned** (`prepare_categorical` outputs): we have
  `kept_feature_names[fi]` (e.g. `"feat_81A"`) plus
  `feat_to_value_names[fi]` (e.g. `["OS", "OSV", "OVS", "SOV", "SVO", "VOS", "VSO"]`).
  Canonical training will emit `featvalue2id.json` with keys built as
  `f"{feat_name.removeprefix('feat_')}={value_label}"`, e.g. `"81A=SOV"`.

- **For T-CF** (`binarise_features` outputs): we have `binary_col_names`
  already in the form `"feat_81A=SOV"` for ≥3-valued features and
  `"feat_GB020"` for 2-valued features. Canonical training will emit
  `binarycol2id.json` with `feat_` stripped, so keys are `"81A=SOV"` or
  `"GB020"`. Phase 4 probes for T-CF must be aware that 2-valued features
  have only one column representing the "1" value.

---

## 6. Sanity items / deviations from the prompt's assumptions

- The prompt's `lang_emb` / `language_embeddings` guess is wrong; the
  actual attribute is `lang_embeddings.weight`.
- The prompt's `feat_to_value_names` returns dict keyed by **feat index**
  (an int), not by feature name. Translating to feature-name keys is done
  via `kept_feature_names`.
- The prompt asks for `feat2values.json`. We will dump it with feature-name
  keys (e.g. `"81A"`) and value-label list values, so it is human-readable
  and joinable across runs.
- WALS `lang2id` keys must be remapped from WALS code → Glottocode for
  Phase 5 to work. Documented above.
