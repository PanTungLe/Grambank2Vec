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

# Per-column fv_id arrays (global ids in insertion order)
col_fvids = [
    np.array(list(fmap.values()), dtype=np.int32) if fmap else None
    for fmap in dataset.feature_maps
]
cap = args.max_k_cap
max_k = min(cap, max(len(v) for v in col_fvids if v is not None))
n_sampled = sum(1 for v in col_fvids if v is not None and len(v) > cap)
print(f"  max_k (capped at {cap}): {max_k}  |  "
      f"features using sampled softmax (K>{cap}): {n_sampled}")

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
print(f"  Dataset built in {time.time()-t0:.1f}s")

# ── Categorical batch generator ───────────────────────────────────────────────
def categorical_batch_generator(batch_size=512):
    """
    Yields: lang_ids (B,), fv_padded (B,max_k), correct_idx (B,), mask (B,max_k)
    For features with K > cap: sampled softmax (1 correct + cap-1 random negatives).
    """
    n = len(lang_feat_pairs)
    while True:
        idxs        = np.random.randint(0, n, size=batch_size)
        batch       = lang_feat_pairs[idxs]
        lang_ids    = batch[:, 0]
        fv_padded   = np.zeros((batch_size, max_k), dtype=np.int32)
        correct_idx = np.zeros(batch_size, dtype=np.int32)
        mask        = np.zeros((batch_size, max_k), dtype=np.float32)

        for i in range(batch_size):
            col_id    = batch[i, 1]
            local_idx = batch[i, 2]
            fvs       = col_fvids[col_id]
            k         = len(fvs)

            if k <= cap:
                # Full categorical CE
                fv_padded[i, :k] = fvs
                mask[i, :k]      = 1.0
                correct_idx[i]   = local_idx
            else:
                # Sampled softmax: correct + (cap-1) random negatives
                correct_gfv = fvs[local_idx]
                neg_pool    = np.concatenate([fvs[:local_idx], fvs[local_idx+1:]])
                neg_sampled = neg_pool[np.random.choice(len(neg_pool), cap-1, replace=False)]
                sampled     = np.concatenate([[correct_gfv], neg_sampled])
                np.random.shuffle(sampled)
                new_local   = int(np.where(sampled == correct_gfv)[0][0])
                fv_padded[i, :cap] = sampled
                mask[i, :cap]      = 1.0
                correct_idx[i]     = new_local

        yield (lang_ids, fv_padded, correct_idx, mask)

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
def train_step(lang_ids, fv_padded, correct_idx, mask):
    with tf.GradientTape() as tape:
        # (B,1,d) lang embedding with dropout
        lang_e = lang_emb_layer(tf.expand_dims(lang_ids, 1))
        lang_e = dropout_lang(lang_e, training=True)

        # (B, max_k, d) feature-value embeddings with dropout
        fv_e = fv_emb_layer(fv_padded)
        fv_e = dropout_fv(fv_e, training=True)

        # Cosine normalization (matches UFAL's Dot(normalize=True))
        lang_n = tf.nn.l2_normalize(lang_e, axis=-1)   # (B, 1, d)
        fv_n   = tf.nn.l2_normalize(fv_e,   axis=-1)   # (B, max_k, d)

        # Logits: cosine similarity with each feature value  (B, max_k)
        logits = tf.squeeze(
            tf.matmul(lang_n, tf.transpose(fv_n, [0, 2, 1])), axis=1)

        # Mask padding positions to -inf before softmax
        logits = logits + (1.0 - mask) * -1e9

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
gen = categorical_batch_generator(args.batch_size)

t1 = time.time()
for epoch in range(1, args.epochs + 1):
    losses = []
    for _ in range(args.steps):
        li, fp, ci, mk = next(gen)
        loss = train_step(
            tf.constant(li,  dtype=tf.int32),
            tf.constant(fp,  dtype=tf.int32),
            tf.constant(ci,  dtype=tf.int32),
            tf.constant(mk,  dtype=tf.float32))
        losses.append(float(loss))

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
