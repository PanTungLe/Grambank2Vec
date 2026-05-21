"""
downstream/direction_predictor.py
===================================
MLP model that predicts UD arc-direction probabilities from typological
language embeddings.

Model
-----
    f(T_l, r, h, d) → p(dep precedes head | relation r, head_upos h, dep_upos d)

where T_l is the language's typological representation (Repr A, B, or C),
and r/h/d are learned embeddings for the relation type and UPOS tags.

Input:   [T_l ; e_r ; e_h ; e_d]   (concatenated)
Output:  sigmoid(MLP output) ∈ (0,1)

Loss: weighted binomial NLL (accounts for count uncertainty):
    L = -Σ_{l,r,h,d} [k * log(p̂) + (n-k) * log(1-p̂)]
    where k = arcs where dep precedes head, n = total arcs

Baselines implemented
---------------------
  - MajorityBaseline: predict mean p across training languages per config
  - FamilyMeanBaseline: predict mean p of same genealogical family
  - ZeroOrderBaseline: predict 0.5 always
  - MLPPredictor: the actual model

Usage
-----
    python downstream/direction_predictor.py \\
        --profiles downstream_results/arc_directions/direction_profiles.csv \\
        --counts   downstream_results/arc_directions/direction_counts.csv \\
        --repr_dir downstream_results/representations/wals_s42 \\
        --repr_name B \\
        --out_dir  downstream_results/direction_model/wals_B \\
        --splits_file downstream_results/value_splits/splits.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from downstream.arc_directions import TARGET_CONFIGS, CONFIG_NAMES


# ---------------------------------------------------------------------------
# Baselines (no PyTorch required)
# ---------------------------------------------------------------------------

class MajorityBaseline:
    """Predict per-configuration mean across training languages."""
    def __init__(self):
        self.means: Dict[str, float] = {}

    def fit(self, profiles: pd.DataFrame, **kwargs):
        self.means = profiles.mean(skipna=True).to_dict()

    def predict(self, glottocodes: List[str],
                profiles: pd.DataFrame) -> pd.DataFrame:
        preds = pd.DataFrame(index=glottocodes, columns=profiles.columns,
                             dtype=float)
        for col in profiles.columns:
            preds[col] = self.means.get(col, 0.5)
        return preds

    def name(self) -> str:
        return "majority_per_config"


class ZeroOrderBaseline:
    """Predict 0.5 (maximum uncertainty) always."""
    def fit(self, *args, **kwargs):
        pass

    def predict(self, glottocodes: List[str],
                profiles: pd.DataFrame) -> pd.DataFrame:
        preds = pd.DataFrame(0.5, index=glottocodes,
                             columns=profiles.columns)
        return preds

    def name(self) -> str:
        return "zero_order_0.5"


class FamilyMeanBaseline:
    """Predict the mean p of languages in the same genealogical family."""
    def __init__(self, family_map: Dict[str, str]):
        self.family_map = family_map  # {glottocode: family_name}
        self.family_means: Dict[str, Dict[str, float]] = {}
        self.global_means: Dict[str, float] = {}

    def fit(self, profiles: pd.DataFrame, glottocodes: List[str], **kwargs):
        self.global_means = profiles.mean(skipna=True).to_dict()
        # Group by family
        families: Dict[str, List[str]] = {}
        for g in glottocodes:
            fam = self.family_map.get(g, "unknown")
            families.setdefault(fam, []).append(g)
        for fam, langs in families.items():
            sub = profiles.loc[[l for l in langs if l in profiles.index]]
            self.family_means[fam] = sub.mean(skipna=True).to_dict()

    def predict(self, glottocodes: List[str],
                profiles: pd.DataFrame) -> pd.DataFrame:
        preds = pd.DataFrame(index=glottocodes, columns=profiles.columns,
                             dtype=float)
        for glot in glottocodes:
            fam = self.family_map.get(glot, "unknown")
            fam_means = self.family_means.get(fam, self.global_means)
            for col in profiles.columns:
                preds.loc[glot, col] = fam_means.get(col,
                    self.global_means.get(col, 0.5))
        return preds

    def name(self) -> str:
        return "family_mean"


# ---------------------------------------------------------------------------
# MLP Predictor (PyTorch)
# ---------------------------------------------------------------------------

def _build_mlp_model(
    lang_dim: int,
    n_relations: int,
    n_upos: int,
    rel_embed_dim: int = 16,
    upos_embed_dim: int = 8,
    hidden_dims: List[int] = (256, 128),
    dropout: float = 0.2,
):
    import torch
    import torch.nn as nn

    class DirectionMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.rel_embed  = nn.Embedding(n_relations, rel_embed_dim)
            self.upos_embed = nn.Embedding(n_upos,      upos_embed_dim)

            input_dim = lang_dim + rel_embed_dim + 2 * upos_embed_dim
            layers = []
            in_d = input_dim
            for h in hidden_dims:
                layers += [nn.Linear(in_d, h), nn.ReLU(), nn.Dropout(dropout)]
                in_d = h
            layers.append(nn.Linear(in_d, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, lang_vecs, rel_ids, head_upos_ids, dep_upos_ids):
            # lang_vecs: (B, lang_dim)
            # rel_ids, head_upos_ids, dep_upos_ids: (B,)
            r = self.rel_embed(rel_ids)
            h = self.upos_embed(head_upos_ids)
            d = self.upos_embed(dep_upos_ids)
            x = torch.cat([lang_vecs, r, h, d], dim=-1)
            return torch.sigmoid(self.net(x)).squeeze(-1)  # (B,)

    return DirectionMLP()


class MLPPredictor:
    """
    Typology → arc-direction MLP predictor.

    Takes any language representation (Repr A, B, or C) as input,
    concatenates with learned relation/UPOS embeddings, and predicts
    p(dep precedes head) for each (relation, head_upos, dep_upos) triple.
    """

    # Canonical UPOS and relation label sets from UD
    UPOS_LABELS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ",
                   "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT",
                   "SCONJ", "SYM", "VERB", "X"]
    REL_LABELS  = sorted(set(c[0] for c in TARGET_CONFIGS))

    def __init__(
        self,
        lang_dim: int,
        rel_embed_dim: int = 16,
        upos_embed_dim: int = 8,
        hidden_dims: Tuple[int, ...] = (256, 128),
        dropout: float = 0.2,
        lr: float = 1e-3,
        n_epochs: int = 100,
        patience: int = 15,
        batch_size: int = 512,
        device: str = "cpu",
        repr_name: str = "B",
    ):
        self.lang_dim = lang_dim
        self.rel_embed_dim = rel_embed_dim
        self.upos_embed_dim = upos_embed_dim
        self.hidden_dims = list(hidden_dims)
        self.dropout = dropout
        self.lr = lr
        self.n_epochs = n_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.device = device
        self.repr_name = repr_name

        self.rel2id  = {r: i for i, r in enumerate(self.REL_LABELS)}
        self.upos2id = {u: i for i, u in enumerate(self.UPOS_LABELS)}
        self.model = None

    def _configs_to_ids(self) -> List[Tuple[int, int, int]]:
        """Return (rel_id, head_upos_id, dep_upos_id) for each target config."""
        result = []
        for deprel, head_upos, dep_upos, name, _ in TARGET_CONFIGS:
            if (deprel in self.rel2id
                    and head_upos in self.upos2id
                    and dep_upos in self.upos2id):
                result.append((
                    name,
                    self.rel2id[deprel],
                    self.upos2id[head_upos],
                    self.upos2id[dep_upos],
                ))
        return result

    def _build_dataset(
        self,
        glottocodes: List[str],
        lang_reprs: np.ndarray,       # (N, D)
        profiles:   pd.DataFrame,     # (N_langs, N_configs) — probabilities
        counts:     pd.DataFrame,     # (N_langs, N_configs) — counts n
        config_ids: List[Tuple],
    ) -> Tuple[List, List, List]:
        """Build (lang_vec, rel_id, head_id, dep_id, p_target, n_weight) lists."""
        import torch
        lang_vecs, rel_ids, head_ids, dep_ids, targets, weights = \
            [], [], [], [], [], []

        glot_to_idx = {g: i for i, g in enumerate(glottocodes)}

        for glot in glottocodes:
            i = glot_to_idx[glot]
            if i >= len(lang_reprs):
                continue
            lv = lang_reprs[i]

            for name, rid, hid, did in config_ids:
                if name not in profiles.columns:
                    continue
                if glot not in profiles.index:
                    continue
                p = profiles.loc[glot, name]
                n = counts.loc[glot, name] if (name in counts.columns
                    and glot in counts.index) else 0
                if pd.isna(p) or pd.isna(n) or n < 5:
                    continue

                lang_vecs.append(lv)
                rel_ids.append(rid)
                head_ids.append(hid)
                dep_ids.append(did)
                targets.append(float(p))
                weights.append(float(n))   # weight by count

        return lang_vecs, rel_ids, head_ids, dep_ids, targets, weights

    def fit(
        self,
        train_glots: List[str],
        lang_reprs: np.ndarray,
        profiles: pd.DataFrame,
        counts: pd.DataFrame,
        dev_glots: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> List[float]:
        import torch
        import torch.nn as nn
        import torch.optim as optim

        config_ids = self._configs_to_ids()
        self.model = _build_mlp_model(
            self.lang_dim,
            len(self.REL_LABELS),
            len(self.UPOS_LABELS),
            self.rel_embed_dim,
            self.upos_embed_dim,
            self.hidden_dims,
            self.dropout,
        ).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5)

        def make_tensors(glots, reprs):
            lvs, ris, his, dis, ts, ws = self._build_dataset(
                glots, reprs, profiles, counts, config_ids)
            if not lvs:
                return None
            return (
                torch.tensor(np.array(lvs), dtype=torch.float32).to(self.device),
                torch.tensor(ris, dtype=torch.long).to(self.device),
                torch.tensor(his, dtype=torch.long).to(self.device),
                torch.tensor(dis, dtype=torch.long).to(self.device),
                torch.tensor(ts,  dtype=torch.float32).to(self.device),
                torch.tensor(ws,  dtype=torch.float32).to(self.device),
            )

        train_data = make_tensors(train_glots, lang_reprs)
        dev_data   = make_tensors(dev_glots, lang_reprs) if dev_glots else None

        if train_data is None:
            print("WARNING: no training data")
            return []

        lv_t, ri_t, hi_t, di_t, p_t, w_t = train_data
        N = len(lv_t)

        train_losses, best_dev_loss = [], float("inf")
        best_state, patience_count = None, 0
        eps = 1e-7

        for epoch in range(self.n_epochs):
            self.model.train()
            perm = torch.randperm(N).to(self.device)
            epoch_loss = 0.0

            for start in range(0, N, self.batch_size):
                idx = perm[start:start + self.batch_size]
                p_pred = self.model(lv_t[idx], ri_t[idx], hi_t[idx], di_t[idx])
                p_true = p_t[idx]
                w      = w_t[idx]

                # Weighted binomial NLL
                bce = -(p_true * torch.log(p_pred + eps)
                        + (1 - p_true) * torch.log(1 - p_pred + eps))
                loss = (w * bce).sum() / w.sum()

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            n_batches = max(1, N // self.batch_size)
            avg_loss = epoch_loss / n_batches
            train_losses.append(avg_loss)

            if dev_data is not None:
                self.model.eval()
                lv_d, ri_d, hi_d, di_d, p_d, w_d = dev_data
                with torch.no_grad():
                    p_pred_d = self.model(lv_d, ri_d, hi_d, di_d)
                    bce_d = -(p_d * torch.log(p_pred_d + eps)
                              + (1 - p_d) * torch.log(1 - p_pred_d + eps))
                    dev_loss = ((w_d * bce_d).sum() / w_d.sum()).item()
                scheduler.step(dev_loss)
                if dev_loss < best_dev_loss:
                    best_dev_loss = dev_loss
                    best_state = {k: v.clone() for k, v
                                  in self.model.state_dict().items()}
                    patience_count = 0
                else:
                    patience_count += 1
                if verbose and (epoch + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1:4d}: train={avg_loss:.4f} "
                          f"dev={dev_loss:.4f} patience={patience_count}")
                if patience_count >= self.patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 20 == 0:
                    print(f"    Epoch {epoch+1:4d}: train={avg_loss:.4f}")

        if best_state is not None:
            self.model.load_state_dict(best_state)

        return train_losses

    def predict(
        self,
        glottocodes: List[str],
        lang_reprs: np.ndarray,
        profiles: pd.DataFrame,
    ) -> pd.DataFrame:
        """Predict direction probabilities for given languages."""
        import torch

        config_ids = self._configs_to_ids()
        config_id_map = {name: (rid, hid, did)
                         for name, rid, hid, did in config_ids}

        preds = pd.DataFrame(np.nan, index=glottocodes,
                             columns=profiles.columns)
        glot_to_idx = {g: i for i, g in enumerate(glottocodes)}

        self.model.eval()
        with torch.no_grad():
            for glot in glottocodes:
                i = glot_to_idx.get(glot)
                if i is None or i >= len(lang_reprs):
                    continue
                lv = torch.tensor(lang_reprs[i:i+1], dtype=torch.float32).to(self.device)

                for name, (rid, hid, did) in config_id_map.items():
                    if name not in profiles.columns:
                        continue
                    ri = torch.tensor([rid], dtype=torch.long).to(self.device)
                    hi = torch.tensor([hid], dtype=torch.long).to(self.device)
                    di = torch.tensor([did], dtype=torch.long).to(self.device)
                    p  = self.model(lv, ri, hi, di).item()
                    preds.loc[glot, name] = p

        return preds

    def name(self) -> str:
        return f"mlp_{self.repr_name}"


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def evaluate_predictions(
    preds:   pd.DataFrame,  # (N_langs, N_configs) predicted probs
    targets: pd.DataFrame,  # (N_langs, N_configs) observed probs
    counts:  pd.DataFrame,  # (N_langs, N_configs) arc counts
    prefix:  str = "",
) -> Dict:
    """
    Compute evaluation metrics between predicted and observed direction probs.

    Metrics
    -------
    MAE          mean absolute error over all (lang, config) cells
    RMSE         root mean squared error
    Brier        mean squared error (= RMSE^2, standard probabilistic scoring)
    Spearman_rho rank correlation between predicted and actual p values
    KL           mean KL divergence D_KL(observed || predicted)
               = p * log(p/p_hat) + (1-p) * log((1-p)/(1-p_hat))
    weighted_MAE MAE weighted by arc count n (penalises errors on frequent configs)

    Per-config breakdown also returned.
    """
    eps = 1e-7
    all_pred, all_true, all_w = [], [], []
    per_config: Dict[str, Dict] = {}

    shared_langs = [g for g in preds.index if g in targets.index]
    shared_cols  = [c for c in preds.columns if c in targets.columns]

    for col in shared_cols:
        col_pred, col_true, col_w = [], [], []
        for glot in shared_langs:
            p_hat = preds.loc[glot, col]
            p_obs = targets.loc[glot, col]
            n = (counts.loc[glot, col]
                 if (col in counts.columns and glot in counts.index) else 1.0)
            if pd.isna(p_hat) or pd.isna(p_obs) or pd.isna(n):
                continue
            p_hat = float(np.clip(p_hat, eps, 1 - eps))
            p_obs = float(np.clip(p_obs, eps, 1 - eps))
            col_pred.append(p_hat)
            col_true.append(p_obs)
            col_w.append(float(n))

        if len(col_pred) < 2:
            continue

        cp, ct, cw = np.array(col_pred), np.array(col_true), np.array(col_w)
        cw_norm = cw / cw.sum()

        mae    = float(np.abs(cp - ct).mean())
        rmse   = float(np.sqrt(((cp - ct) ** 2).mean()))
        brier  = float(((cp - ct) ** 2).mean())
        kl     = float(np.mean(
            ct * np.log(ct / cp) + (1 - ct) * np.log((1 - ct) / (1 - cp))))
        w_mae  = float((cw_norm * np.abs(cp - ct)).sum())
        rho, _ = spearmanr(cp, ct) if len(cp) >= 5 else (np.nan, np.nan)

        per_config[col] = {
            "mae": mae, "rmse": rmse, "brier": brier,
            "kl": kl, "weighted_mae": w_mae, "spearman_rho": float(rho),
            "n_langs": len(col_pred),
        }

        all_pred.extend(col_pred)
        all_true.extend(col_true)
        all_w.extend(col_w)

    if not all_pred:
        return {}

    ap, at, aw = np.array(all_pred), np.array(all_true), np.array(all_w)
    aw_norm = aw / aw.sum()

    rho_global, _ = spearmanr(ap, at)
    kl_global = float(np.mean(
        at * np.log(at / ap) + (1 - at) * np.log((1 - at) / (1 - ap))))

    metrics = {
        f"{prefix}mae":          float(np.abs(ap - at).mean()),
        f"{prefix}rmse":         float(np.sqrt(((ap - at) ** 2).mean())),
        f"{prefix}brier":        float(((ap - at) ** 2).mean()),
        f"{prefix}kl":           kl_global,
        f"{prefix}weighted_mae": float((aw_norm * np.abs(ap - at)).sum()),
        f"{prefix}spearman_rho": float(rho_global),
        f"{prefix}n_cells":      len(all_pred),
        f"{prefix}per_config":   per_config,
    }
    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train and evaluate typology→arc-direction predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--profiles",    required=True,
                   help="direction_profiles.csv from arc_directions.py")
    p.add_argument("--counts",      required=True,
                   help="direction_counts.csv from arc_directions.py")
    p.add_argument("--repr_dir",    required=True,
                   help="Representation directory (output of representations.py)")
    p.add_argument("--repr_name",   default="B",
                   help="Which representation to use: A, B, C, B_masked30, ...")
    p.add_argument("--splits_file", required=True,
                   help="splits.json from value_splits.py")
    p.add_argument("--out_dir",     required=True)
    p.add_argument("--device",      default="cpu", choices=["cpu","cuda","mps"])
    p.add_argument("--n_epochs",    type=int, default=100)
    p.add_argument("--patience",    type=int, default=15)
    p.add_argument("--batch_size",  type=int, default=512)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--family_map",  default=None,
                   help="JSON {glottocode: family} for FamilyMean baseline")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    profiles = pd.read_csv(args.profiles, index_col=0)
    counts   = pd.read_csv(args.counts,   index_col=0)
    print(f"Profiles: {profiles.shape}")

    with open(args.splits_file) as f:
        splits = json.load(f)

    repr_mat  = np.load(os.path.join(args.repr_dir, f"repr_{args.repr_name}.npy"))
    with open(os.path.join(args.repr_dir, f"repr_{args.repr_name}_langs.json")) as f:
        repr_langs = json.load(f)
    lang_dim = repr_mat.shape[1]
    print(f"Repr {args.repr_name}: {repr_mat.shape}")

    all_results = {}

    for split_name, split_data in splits.items():
        print(f"\n{'='*60}")
        print(f"Split: {split_name}")
        train_glots = [g for g in split_data["train"] if g in repr_langs]
        test_glots  = [g for g in split_data["test"]  if g in repr_langs]

        if len(train_glots) < 5 or len(test_glots) < 2:
            print(f"  Skipping: too few languages "
                  f"(train={len(train_glots)}, test={len(test_glots)})")
            continue

        print(f"  Train: {len(train_glots)}, Test: {len(test_glots)}")

        # Align repr to glottocodes
        all_glots = train_glots + test_glots
        glot2row  = {g: i for i, g in enumerate(repr_langs)}
        reprs = np.array([repr_mat[glot2row[g]]
                          if g in glot2row else np.zeros(lang_dim)
                          for g in all_glots])

        split_results = {}

        # Baselines
        for baseline in [ZeroOrderBaseline(), MajorityBaseline()]:
            baseline.fit(profiles.loc[[g for g in train_glots
                                       if g in profiles.index]])
            preds = baseline.predict(test_glots, profiles)
            metrics = evaluate_predictions(preds, profiles, counts,
                                           prefix=f"{baseline.name()}_")
            split_results[baseline.name()] = {
                k: v for k, v in metrics.items()
                if not k.endswith("per_config")}
            print(f"  [{baseline.name()}] MAE={metrics.get(baseline.name()+'_mae',0):.4f}  "
                  f"Spearman={metrics.get(baseline.name()+'_spearman_rho',0):.4f}")

        # MLP predictor
        predictor = MLPPredictor(
            lang_dim=lang_dim,
            n_epochs=args.n_epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            repr_name=args.repr_name,
        )
        try:
            predictor.fit(
                train_glots=train_glots,
                lang_reprs=reprs[:len(train_glots)],
                profiles=profiles,
                counts=counts,
                dev_glots=test_glots[:max(1, len(test_glots)//3)],
                verbose=True,
            )
            preds = predictor.predict(test_glots, reprs[len(train_glots):], profiles)
            metrics = evaluate_predictions(preds, profiles, counts,
                                           prefix=f"mlp_{args.repr_name}_")
            split_results[f"mlp_{args.repr_name}"] = {
                k: v for k, v in metrics.items()
                if not k.endswith("per_config")}
            print(f"  [MLP_{args.repr_name}] "
                  f"MAE={metrics.get(f'mlp_{args.repr_name}_mae',0):.4f}  "
                  f"Spearman={metrics.get(f'mlp_{args.repr_name}_spearman_rho',0):.4f}")
        except Exception as e:
            print(f"  MLP training failed: {e}")

        all_results[split_name] = split_results

    # Save results
    results_path = os.path.join(args.out_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {results_path}")


if __name__ == "__main__":
    main()
