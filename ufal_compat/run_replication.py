#!/usr/bin/env python3
"""
UFAL replication runner (compat-fixed for numpy 1.26 / sklearn 1.x / pandas 2.x).

Key changes vs original run.py:
  - Absolute data paths (works from any cwd)
  - train_dataset uses test_x (commented-out line in original) OR dev_x (original default)
  - numpy object dtype for ragged array in create_dataset
  - sklearn compute_class_weight keyword args
  - KNN callback optionally disabled (fast path)
  - Calls keras model.fit() directly to avoid Model.train()'s hardcoded callbacks
  - Reports best dev Filler accuracy cleanly

Usage:
  python run_replication.py [--epochs 200] [--steps 1000] [--embed 512] [--clusters 300]
                            [--no-knn] [--train-on-test]
"""
import sys
import os
import argparse
import time
import importlib

# Make imports work from any working directory
_here = os.path.dirname(os.path.abspath(__file__))
_scripts = os.path.dirname(_here)
sys.path.insert(0, _scripts)
sys.path.insert(0, _here)

import tensorflow as tf

parser = argparse.ArgumentParser()
parser.add_argument('--epochs',         type=int, default=200)
parser.add_argument('--steps',          type=int, default=1000)
parser.add_argument('--embed',          type=int, default=512)
parser.add_argument('--clusters',       type=int, default=300)
parser.add_argument('--no-knn',         action='store_true', help='Skip KNN callback (~25s saved per epoch)')
parser.add_argument('--train-on-test',  action='store_true', help='Use test_x in training (trains test embeddings inline)')
args = parser.parse_args()

print("=" * 60)
print(f"UFAL replication: embed={args.embed}, epochs={args.epochs}, steps={args.steps}")
print(f"  train_on: {'test_x (test embeddings trained inline)' if args.train_on_test else 'dev_x (original; replicates published 0.755 dev)'}")
print(f"  KNN: {'disabled' if args.no_knn else 'enabled'}")
print("=" * 60)

# Patch Dataset to use the right training split before importing
import lang_embedding.dataset as _ds_mod
_orig_init = _ds_mod.Dataset.__init__

def _patched_init(self, clusters):
    import pandas as pd
    import numpy as np
    import sklearn

    _data = os.path.join(os.path.dirname(os.path.dirname(_here)), 'data')

    self.train_x = pd.read_csv(os.path.join(_data, 'train_x.csv'))
    self.train_y = pd.read_csv(os.path.join(_data, 'train_y.csv'))
    self.dev_x   = pd.read_csv(os.path.join(_data, 'dev_x.csv'))
    self.dev_y   = pd.read_csv(os.path.join(_data, 'dev_y.csv'))
    self.test_x  = pd.read_csv(os.path.join(_data, 'test_x.csv'))

    self.kmeans = self.create_kmeans(
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

    if args.train_on_test:
        print("  Using pd.concat([train_y, test_x]) for training")
        concat_df = pd.concat([self.train_y, self.test_x])
    else:
        print("  Using pd.concat([train_y, dev_x]) for training (original UFAL)")
        concat_df = pd.concat([self.train_y, self.dev_x])

    self.train_dataset = self.create_dataset(concat_df)
    self.all_features  = np.array(self.all_features)
    self.class_weights = sklearn.utils.class_weight.compute_class_weight(
        'balanced', classes=np.unique(self.all_features), y=self.all_features)
    print(f"  class_weights shape: {self.class_weights.shape}")

_ds_mod.Dataset.__init__ = _patched_init

t0 = time.time()
print("\n[1] Building dataset...")
from lang_embedding.dataset import Dataset
from lang_embedding.callbacks import Filler, KNN
import lang_embedding.models as models

dataset = Dataset(args.clusters)
langs_num = dataset.train_x.shape[0] + dataset.dev_x.shape[0] + dataset.test_x.shape[0]
feature_val_num = dataset.global_feature_id
print(f"  langs_num={langs_num}, feature_val_num={feature_val_num}")
print(f"  train_dataset shape: {dataset.train_dataset.shape}")
print(f"  Dataset built in {time.time()-t0:.1f}s")

print("\n[2] Building Keras model...")
keras_model = models.get_model_embedding(langs_num, feature_val_num, 0.5, args.embed)
keras_model.summary()
optimizer = tf.optimizers.Adam()
keras_model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)

print("\n[3] Setting up callbacks...")
filler = Filler(dataset.feature_maps, dataset.feature_maps_int)
callback_list = [filler]
if not args.no_knn:
    knn_ks = [1, 5, 10, 20, 50, 100, 200]  # reduced set (original: 1-500, 72 values)
    callback_list.append(KNN(knn_ks))
    print(f"  KNN callback enabled with k={knn_ks}")
else:
    print("  KNN callback disabled")

print(f"\n[4] Training {args.epochs} epochs × {args.steps} steps...")
t1 = time.time()

# Periodic checkpoint every 20 epochs (safeguard against interruption)
ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath='ckpt_epoch{epoch:03d}.h5',
    save_weights_only=True,
    save_freq=20 * args.steps,   # every 20 epochs (steps-based)
    verbose=0
)
callback_list.append(ckpt_cb)

keras_model.fit(
    dataset.batch_generator(),
    steps_per_epoch=args.steps,
    epochs=args.epochs,
    callbacks=callback_list
)
elapsed = time.time() - t1

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"  Training time:          {elapsed:.0f}s ({elapsed/60:.1f} min)")
print(f"  Best dev Filler acc:    {filler.best:.4f}  [UFAL paper target: 0.755]")
print(f"  Δ vs target:            {filler.best - 0.755:+.4f}")
print(f"  Weights saved:          best.h5 (in {os.getcwd()})")
print("=" * 60)
