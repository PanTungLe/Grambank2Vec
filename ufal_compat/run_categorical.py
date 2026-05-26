#!/usr/bin/env python3
"""
UFAL replication with categorical cross-entropy instead of binary BCE.

Architecture: IDENTICAL to run_replication.py (same embedding dot-product model,
same cosine normalization, same model weights/dims). Only the training objective:
  - Binary:      sample (lang, fv, label=0/1)  → BCE over sigmoid
  - Categorical: sample (lang, feature)         → softmax over all K values → sparse CE

Two fixes vs naive implementation:
  1. Dense layer (W,b) is fixed to (1, 0): it's not trained by categorical CE so its
     random init sign could invert cosine rankings.  With W=1,b=0 the Filler's
     sigmoid(cos_sim) remains monotone → argmax is correct.
  2. MAX_K_CAP (default 20): the 3 geographic cluster features have K=300, making
     padded tensors 15× bigger than needed.  Features with K > cap use sampled
     softmax (1 correct + cap-1 random negatives), which is standard practice and
     unbiased in expectation.

Usage (from lang_embedding/ directory):
  PYTHONPATH=$(dirname $(pwd)) python3 run_categorical.py \\
    --epochs 200 --steps 1000 --embed 512 --clusters 300
"""
import sys
import os
import argparse
import time
import numpy as np

_here = os.path.dirname(os.path.abspath(__file__))
_scripts = os.path.dirname(_here)
sys.path.insert(0, _scripts)
sys.path.insert(0, _here)

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',        type=int,   default=200)
parser.add_argument('--steps',         type=int,   default=1000)
parser.add_argument('--embed',         type=int,   default=512)
parser.add_argument('--clusters',      type=int,   default=300)
parser.add_argument('--batch-size',    type=int,   default=512)
parser.add_argument('--max-k-cap',     type=int,   default=20,
                    help='Max feature values per batch item. Features with K>cap '
                         'use sampled softmax (default 20 covers all WALS features).')
parser.add_argument('--train-on-test', action='store_true',
                    help='Use test_x in training (trains test embeddings inline)')
args = parser.parse_args()

print("=" * 60)
print(f"UFAL categorical CE: embed={args.embed}, epochs={args.epochs}, "
      f"steps={args.steps}, max_k_cap={args.max_k_cap}")
print(f"  train_on: {'test_x' if args.train_on_test else 'dev_x (original)'}")
print("=" * 60)

# ── Patch Dataset (same as run_replication.py) ────────────────────────────────
import lang_embedding.dataset as _ds_mod

def _patched_init(self, clusters):
    import pandas as pd
    import sklearn
    # Use dataset.py's __file__ to locate the data directory correctly
    _data = os.path.join(
        os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(_ds_mod.__file__)))),
        'data')
    self.train_x = pd.read_csv(os.path.join(_data, 'train_x.csv'))
    self.train_y = pd.read_csv(os.path.join(_data, 'train_y.csv'))
    self.dev_x   = pd.read_csv(os.path.join(_data, 'dev_x.csv'))
    self.dev_y   = pd.read_csv(os.path.join(_data, 'dev_y.csv'))
    self.test_x  = pd.read_csv(os.path.join(_data, 'test_x.csv'))
    self.kmeans  = self.create_kmeans(
        self.train_x['latitude'].to_numpy(),
        self.train_x['longitude'].to_numpy(), clusters)
    self.train_x = self.preprocess(self.train_x)
    self.train_y = self.preprocess(self.train_y)
    self.dev_x   = self.preprocess(self.dev_x)
    self.dev_y   = self.preprocess(self.dev_y)
    self.test_x  = self.preprocess(self.test_x)
    self.lang_to_int          = {}
    self.int_to_lang          = {}
    self.feature_maps         = [{} for _ in range(self.train_x.shape[1])]
    self.feature_maps_int     = [{} for _ in range(self.train_x.shape[1])]
    self.feature_id_to_column_id = {}
    self.all_features         = []
    self.global_feature_id    = 0
    concat_df = pd.concat([self.train_y,
                           self.test_x if args.train_on_test else self.dev_x])
    self.train_dataset = self.create_dataset(concat_df)
    self.all_features  = np.array(self.all_features)
    self.class_weights = sklearn.utils.class_weight.compute_class_weight(
        'balanced', classes=np.unique(self.all_features), y=self.all_features)

_ds_mod.Dataset.__init__ = _patched_init

# ── Load data ─────────────────────────────────────────────────────────────────
t0 = time.time()
print("\n[1] Building dataset...")
from lang_embedding.dataset import Dataset
from lang_embedding.callbacks import Filler
import lang_embedding.models as models

dataset = Dataset(args.clusters)
langs_num       = dataset.train_x.shape[0] + dataset.dev_x.shape[0] + dataset.test_x.shape[0]
feature_val_num = dataset.global_feature_id
print(f"  langs_num={langs_num}, feature_val_num={feature_val_num}")

# Per-column fv_id arrays (global ids in insertion order) — still needed to
# look up the correct global fv_id from (col_id, local_idx).
col_fvids = [
    np.array(list(fmap.values()), dtype=np.int32) if fmap else None
    for fmap in dataset.feature_maps
]
cap = args.max_k_cap

# Flat list of (lang_id, col_id, local_correct_idx) training triplets
lang_feat_pairs = []
for row in dataset.train_dataset:
    lang_id = int(row[0])
    for pair in row[1:]:
        col_id, gfv = int(pair[0]), int(pair[1])
        fv_list = list(dataset.feature_maps[col_id].values())
        lang_feat_pairs.append((lang_id, col_id, fv_list.index(gfv)))
lang_feat_pairs = np.array(lang_feat_pairs, dtype=np.int32)
print(f"  Training triplets: {len(lang_feat_pairs)}")

# Per-lang set of global fv_ids — for O(1) rejection during negative sampling.
# Mirrors binary BCE's self._lang_pairs (dataset.py) used for the same purpose.
_lang_fv_sets = {}
for row in dataset.train_dataset:
    lid = int(row[0])
    _lang_fv_sets[lid] = set(int(pair[1]) for pair in row[1:])

print(f"  Global negative sampling: 1 correct + {cap-1} global negatives per item "
      f"(cap={cap}) — categorical equivalent of binary BCE's 50/50 pos/neg sampling")
print(f"  Dataset built in {time.time()-t0:.1f}s")

# ── Categorical batch generator ───────────────────────────────────────────────
def categorical_batch_generator(batch_size=512):
    """
    Direct categorical equivalent of binary BCE's batch_generator.

    Binary BCE (dataset.py):
      50% positive: (lang, correct_fv, label=1)  — known feature value for lang
      50% negative: (lang, random_fv,  label=0)  — globally-sampled wrong value
      → 2-way BCE loss

    Categorical (here):
      Each item: (lang, correct_fv) + (cap-1) globally-sampled wrong fv values
      → cap-way softmax CE loss

    Negatives are drawn from the GLOBAL fv pool (any column), exactly as in the
    binary run — this is what prevented lang-embedding collapse in binary BCE.
    Without global negatives the within-column gradients are too correlated and
    embeddings collapse to ~most-common-value accuracy (~0.60).

    Yields: lang_ids (B,), fv_padded (B, cap), correct_idx (B,)
    No masking needed: every batch item has exactly cap valid candidates.
    """
    n          = len(lang_feat_pairs)
    global_max = dataset.global_feature_id   # fv_ids in [1, global_max-1] (matching binary)
    while True:
        idxs        = np.random.randint(0, n, size=batch_size)
        batch       = lang_feat_pairs[idxs]
        lang_ids    = batch[:, 0]
        fv_padded   = np.zeros((batch_size, cap), dtype=np.int32)
        correct_idx = np.zeros(batch_size, dtype=np.int32)

        for i in range(batch_size):
            lang_id     = int(batch[i, 0])
            col_id      = int(batch[i, 1])
            local_idx   = int(batch[i, 2])
            correct_gfv = int(col_fvids[col_id][local_idx])
            lang_fv_set = _lang_fv_sets[lang_id]

            # Vectorised negative sampling: oversample then filter, with fallback loop.
            n_needed = cap - 1
            candidates = np.random.randint(1, global_max, size=n_needed * 4)
            valid = candidates[~np.isin(candidates, list(lang_fv_set))]
            if len(valid) < n_needed:          # very rare: keep sampling
                extras = []
                while len(extras) < n_needed - len(valid):
                    c = np.random.randint(1, global_max)
                    if c not in lang_fv_set:
                        extras.append(c)
                valid = np.concatenate([valid, extras])
            negs = valid[:n_needed]

            sampled = np.concatenate([[correct_gfv], negs]).astype(np.int32)
            np.random.shuffle(sampled)
            correct_idx[i]  = int(np.where(sampled == correct_gfv)[0][0])
            fv_padded[i]    = sampled

        yield (lang_ids, fv_padded, correct_idx)

# ── Build model ───────────────────────────────────────────────────────────────
print("\n[2] Building Keras model (same arch as binary)...")
keras_model = models.get_model_embedding(langs_num, feature_val_num, 0.5, args.embed)
keras_model.summary()

# FIX 1: Pin dense layer to (W=1, b=0) so sigmoid(cos_sim) stays monotone.
# The dense layer is NOT updated by categorical CE (no gradient path through it),
# so its random init sign would otherwise invert cosine rankings at eval time.
dense_layer = keras_model.get_layer('dense')
dense_layer.set_weights([np.array([[1.0]]), np.array([0.0])])
dense_layer.trainable = False   # freeze it — not needed for categorical CE
print("  Dense layer fixed to W=1, b=0 and frozen.")

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

lang_emb_layer = keras_model.get_layer('langs')
fv_emb_layer   = keras_model.get_layer('feature_value')
dropout_lang   = keras_model.get_layer('dropout')
dropout_fv     = keras_model.get_layer('dropout_1')

# ── Training step ─────────────────────────────────────────────────────────────
@tf.function
def train_step(lang_ids, fv_padded, correct_idx):
    with tf.GradientTape() as tape:
        # (B, 1, d) lang embedding with dropout
        lang_e = lang_emb_layer(tf.expand_dims(lang_ids, 1))
        lang_e = dropout_lang(lang_e, training=True)

        # (B, cap, d) feature-value embeddings (1 correct + cap-1 global negatives)
        fv_e = fv_emb_layer(fv_padded)
        fv_e = dropout_fv(fv_e, training=True)

        # Cosine normalization (matches UFAL's Dot(normalize=True))
        lang_n = tf.nn.l2_normalize(lang_e, axis=-1)   # (B, 1, d)
        fv_n   = tf.nn.l2_normalize(fv_e,   axis=-1)   # (B, cap, d)

        # Logits: cosine similarity with each candidate fv  (B, cap)
        logits = tf.squeeze(
            tf.matmul(lang_n, tf.transpose(fv_n, [0, 2, 1])), axis=1)

        # No masking needed: all cap positions are valid global negatives.
        # Unweighted CE — binary BCE also uses no class weights (they are computed
        # but silently dropped by dataset.py's generator yield on line 120).
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                correct_idx, logits, from_logits=True))

    # Only update embedding layers (dense is frozen)
    train_vars = [v for v in keras_model.trainable_variables]
    grads = tape.gradient(loss, train_vars)
    optimizer.apply_gradients(zip(grads, train_vars))
    return loss

# ── Filler callback ───────────────────────────────────────────────────────────
print("\n[3] Setting up Filler callback...")
filler = Filler(dataset.feature_maps, dataset.feature_maps_int)
filler.set_model(keras_model)   # Keras 3: .model is read-only property; use set_model()

# ── Training loop ─────────────────────────────────────────────────────────────
print(f"\n[4] Training {args.epochs} epochs × {args.steps} steps (categorical CE) ...")
print("    (First ~30-120s: silent while TF compiles the train_step graph)")
gen = categorical_batch_generator(args.batch_size)

t1 = time.time()
for epoch in range(1, args.epochs + 1):
    losses = []
    t_ep = time.time()
    for step in range(1, args.steps + 1):
        li, fp, ci = next(gen)
        loss = train_step(
            tf.constant(li,  dtype=tf.int32),
            tf.constant(fp,  dtype=tf.int32),
            tf.constant(ci,  dtype=tf.int32))
        losses.append(float(loss))
        # Progress within epoch every 100 steps
        if step % 100 == 0:
            print(f"  ep {epoch:3d} step {step:4d}/{args.steps}  "
                  f"loss={np.mean(losses[-100:]):.4f}  "
                  f"({time.time()-t_ep:.0f}s)", flush=True)

    filler_acc = filler.fill(
        np.array(filler.x_to_predict, copy=True), filler.golden)
    best_mark  = "  ← best" if filler_acc >= filler.best else ""
    elapsed    = time.time() - t1
    eta        = elapsed / epoch * (args.epochs - epoch)
    print(f"Epoch {epoch:3d}/{args.epochs}  loss={np.mean(losses):.4f}  "
          f"filler={filler_acc:.4f}  best={filler.best:.4f}{best_mark}  "
          f"{elapsed/60:.1f}m  ETA={eta/60:.0f}m")

    if epoch % 20 == 0:
        try:
            keras_model.save_weights(f'cat_ckpt_e{epoch:03d}.weights.h5')
        except Exception:
            keras_model.save_weights(f'cat_ckpt_e{epoch:03d}.h5')

elapsed = time.time() - t1
print("\n" + "=" * 60)
print("RESULTS (Categorical CE)")
print("=" * 60)
print(f"  Training time:        {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"  Best dev Filler acc:  {filler.best:.4f}  [binary BCE target: 0.739]")
print(f"  Δ vs binary target:   {filler.best - 0.739:+.4f}")
print("=" * 60)
