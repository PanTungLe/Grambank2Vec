"""
Canonical full-data training for typological matrix factorisation.

Trains either the T-CF (binary Bernoulli) or Learned (softmax categorical)
model on the FULL dataset with 5% held-out validation used only for early
stopping.  Dumps stable, reproducible embeddings for downstream geometry
analysis (Phase 4) and cross-database comparison (Phase 5).

Usage
-----
python canonical/train_canonical.py \\
    --database wals \\
    --architecture learned \\
    --data_path /path/to/wals \\
    --out_dir checkpoints/wals_learned_s42 \\
    --seed 42

NOTE: --data_path accepts the cloned CLDF repo directory, NOT a
prepared CSV.  The upstream loaders (load_wals_cldf / load_grambank_cldf)
read directly from the CLDF files.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# --- project root on the path so upstream modules are importable ---
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # canonical/

from data_preparation import (
    load_wals_cldf,
    load_grambank_cldf,
    binarise_features,
)
from model import TypologicalMF, WALSDataset
from model_learned import (
    TypologicalMF_Learned,
    CategoricalTypDataset,
    prepare_categorical,
)
from utils import (
    seed_everything,
    wals_code_to_glottocode,
    build_all_triples_binary,
    build_all_triples_categorical,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a canonical (full-data) typological model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--database", required=True, choices=["wals", "grambank"])
    p.add_argument("--architecture", required=True, choices=["tcf", "learned"])
    p.add_argument(
        "--data_path", required=True,
        help="Path to the cloned CLDF repo (WALS or Grambank). "
             "DEVIATION from prompt: accepts a repo dir, not a prepared CSV, "
             "because the upstream loaders read CLDF files directly.")
    p.add_argument(
        "--glottolog_repo", default=None,
        help="Path to cloned glottolog-cldf repo (Grambank genus; "
             "omit to fall back to Family column).")
    p.add_argument("--out_dir", required=True,
                   help="Output directory for all artifacts.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--embed_dim", type=int, default=64)
    p.add_argument("--n_epochs", type=int, default=200,
                   help="Hard epoch cap.")
    p.add_argument("--patience", type=int, default=10,
                   help="Early-stopping patience (epochs without val improvement).")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--l2_coef", type=float, default=0.1,
        help="L2 regularisation coefficient. Matches existing pipeline "
             "default (l2_reg=0.1). Applied only to batch-accessed embeddings "
             "via the model's internal per-batch L2; Adam weight_decay=0.")
    p.add_argument(
        "--val_frac", type=float, default=0.05,
        help="Fraction of triples held out for validation/early-stopping.")
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, list[str], dict[str, Any]]:
    """
    Load and encode the full dataset for the specified database/architecture.

    Returns (df, feature_cols, arch_data) where arch_data is a dict whose
    keys depend on architecture (see inline comments).
    """
    if args.database == "wals":
        df, feature_cols = load_wals_cldf(args.data_path)
    else:
        df, feature_cols = load_grambank_cldf(
            args.data_path,
            genus_source="glottolog" if args.glottolog_repo else "family",
            glottolog_repo_dir=args.glottolog_repo,
        )

    if args.architecture == "tcf":
        binary_matrix, binary_col_names, feature_groups, feature_value_names = \
            binarise_features(df, feature_cols)
        return df, feature_cols, {
            "binary_matrix": binary_matrix,
            "binary_col_names": binary_col_names,
            "feature_groups": feature_groups,
            "feature_value_names": feature_value_names,
        }

    # architecture == "learned"
    cat_matrix, kept_feature_names, feat_to_global_ids, feat_to_value_names = \
        prepare_categorical(df, feature_cols)
    return df, feature_cols, {
        "cat_matrix": cat_matrix,
        "kept_feature_names": kept_feature_names,
        "feat_to_global_ids": feat_to_global_ids,
        "feat_to_value_names": feat_to_value_names,
    }


# ---------------------------------------------------------------------------
# lang2id construction (Glottocode-keyed)
# ---------------------------------------------------------------------------

def build_lang2id(
    df: pd.DataFrame,
    database: str,
    data_path: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """
    Build language-to-index dicts keyed by Glottocode and by native ID.

    Returns (glottocode_keyed, full_keyed).
    - Grambank: both dicts are identical (native IDs are already Glottocodes).
    - WALS: glottocode_keyed excludes languages whose WALS code has no
      corresponding Glottocode in the CLDF languages.csv.  full_keyed is
      keyed by WALS internal code (e.g. "eng") and covers all languages.
    """
    if database == "grambank":
        gc2id = {str(gc): int(i) for i, gc in enumerate(df["glottocode"])}
        return gc2id, gc2id

    # WALS: look up Glottocode via the CLDF languages.csv
    wals2glot = wals_code_to_glottocode(data_path)
    full: dict[str, int] = {str(wc): int(i)
                            for i, wc in enumerate(df["wals_code"])}
    glot_keyed: dict[str, int] = {}
    for wc, idx in full.items():
        gc = wals2glot.get(wc)
        if gc:
            glot_keyed[gc] = idx

    n_missing = len(full) - len(glot_keyed)
    if n_missing:
        print(f"  WALS Glottocode map: {len(glot_keyed)} languages have "
              f"Glottocodes; {n_missing} lack one and are excluded from "
              f"lang2id.json (kept in lang2id_full.json).")
    return glot_keyed, full


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    l2_coef: float,
    architecture: str,
    device: str,
) -> float:
    """Run one training epoch; return mean prediction loss (no L2 term)."""
    model.train()
    total_pred_loss = 0.0
    n_batches = 0

    for batch in train_loader:
        if architecture == "tcf":
            lang_idx, feat_idx, vals = batch
            lang_idx = lang_idx.to(device)
            feat_idx = feat_idx.to(device)
            vals = vals.to(device)
            preds, batch_l2 = model(lang_idx, feat_idx)
            pred_loss = F.binary_cross_entropy(preds, vals)
        else:
            lang_idx, feat_idx, val_idx = batch
            lang_idx = lang_idx.to(device)
            feat_idx = feat_idx.to(device)
            val_idx = val_idx.to(device)
            log_probs, batch_l2 = model(lang_idx, feat_idx, val_idx)
            pred_loss = -log_probs.mean()

        loss = pred_loss + (l2_coef / 2.0) * batch_l2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_pred_loss += pred_loss.item()
        n_batches += 1

    return total_pred_loss / n_batches if n_batches else float("nan")


def compute_val_loss(
    model: torch.nn.Module,
    val_loader: DataLoader,
    architecture: str,
    device: str,
) -> float:
    """Compute mean prediction loss on the validation set (no L2 term)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            if architecture == "tcf":
                lang_idx, feat_idx, vals = batch
                lang_idx = lang_idx.to(device)
                feat_idx = feat_idx.to(device)
                vals = vals.to(device)
                preds, _ = model(lang_idx, feat_idx)
                loss = F.binary_cross_entropy(preds, vals)
            else:
                lang_idx, feat_idx, val_idx = batch
                lang_idx = lang_idx.to(device)
                feat_idx = feat_idx.to(device)
                val_idx = val_idx.to(device)
                log_probs, _ = model(lang_idx, feat_idx, val_idx)
                loss = -log_probs.mean()

            total_loss += loss.item()
            n_batches += 1

    return total_loss / n_batches if n_batches else float("inf")


def get_emb_norms(
    model: torch.nn.Module,
    architecture: str,
) -> tuple[float, float]:
    """Return (mean_lang_emb_norm, mean_featvalue_emb_norm)."""
    with torch.no_grad():
        lang_norm = model.lang_embeddings.weight.norm(dim=1).mean().item()
        if architecture == "tcf":
            fv_norm = model.feat_embeddings.weight.norm(dim=1).mean().item()
        else:
            fv_norm = model.value_embeddings.weight.norm(dim=1).mean().item()
    return lang_norm, fv_norm


# ---------------------------------------------------------------------------
# Artifact dumping
# ---------------------------------------------------------------------------

def dump_artifacts(
    model: torch.nn.Module,
    args: argparse.Namespace,
    data: dict[str, Any],
    lang2id_glot: dict[str, int],
    lang2id_full: dict[str, int],
    out_dir: str,
) -> None:
    """Save all embedding arrays and index JSONs to out_dir."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Language embeddings — always present
    lang_emb = model.lang_embeddings.weight.detach().cpu().numpy()
    np.save(os.path.join(out_dir, "lang_embeddings.npy"), lang_emb)

    if args.architecture == "tcf":
        # Binary-column embeddings
        bin_emb = model.feat_embeddings.weight.detach().cpu().numpy()
        np.save(os.path.join(out_dir, "binarycol_embeddings.npy"), bin_emb)

        # binarycol2id: strip "feat_" prefix from column names
        binarycol2id: dict[str, int] = {
            name.removeprefix("feat_"): i
            for i, name in enumerate(data["binary_col_names"])
        }
        with open(os.path.join(out_dir, "binarycol2id.json"), "w") as fh:
            json.dump(binarycol2id, fh, indent=2)

        # feat2values: "81A" → ["OS", "OSV", ...]
        feat2values: dict[str, list[str]] = {
            feat_name.removeprefix("feat_"): list(val_names)
            for feat_name, val_names in data["feature_value_names"].items()
        }

    else:  # learned
        val_emb = model.value_embeddings.weight.detach().cpu().numpy()
        np.save(os.path.join(out_dir, "featvalue_embeddings.npy"), val_emb)

        # featvalue2id: "81A=SOV" → global_embedding_row_index
        kept = data["kept_feature_names"]
        f2gids = data["feat_to_global_ids"]
        f2vnames = data["feat_to_value_names"]
        featvalue2id: dict[str, int] = {}
        for fi, feat_name in enumerate(kept):
            bare = feat_name.removeprefix("feat_")
            for gid, vname in zip(f2gids[fi], f2vnames[fi]):
                featvalue2id[f"{bare}={vname}"] = int(gid)
        with open(os.path.join(out_dir, "featvalue2id.json"), "w") as fh:
            json.dump(featvalue2id, fh, indent=2)

        # feat2values: "81A" → ["OS", "OSV", ...]
        feat2values = {
            feat_name.removeprefix("feat_"): list(f2vnames[fi])
            for fi, feat_name in enumerate(kept)
        }

    with open(os.path.join(out_dir, "feat2values.json"), "w") as fh:
        json.dump(feat2values, fh, indent=2)

    # lang2id (Glottocode-keyed — canonical join key for Phase 5)
    with open(os.path.join(out_dir, "lang2id.json"), "w") as fh:
        json.dump(lang2id_glot, fh, indent=2)

    # lang2id_full (native-ID-keyed: wals_code for WALS, glottocode for Grambank)
    with open(os.path.join(out_dir, "lang2id_full.json"), "w") as fh:
        json.dump(lang2id_full, fh, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    t0 = time.time()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # 1. Seed everything FIRST — before any stochastic ops
    seed_everything(args.seed)

    print(f"\n{'='*60}")
    print(f"Canonical training: {args.database} / {args.architecture}")
    print(f"  data_path : {args.data_path}")
    print(f"  out_dir   : {args.out_dir}")
    print(f"  seed={args.seed}, embed_dim={args.embed_dim}, "
          f"n_epochs={args.n_epochs}, patience={args.patience}")
    print(f"  lr={args.lr}, l2_coef={args.l2_coef}, batch_size={args.batch_size}")
    print(f"{'='*60}")

    # 2. Load data (deterministic given inputs — no RNG consumed here)
    df, feature_cols, data = load_data(args)
    n_langs = len(df)
    print(f"Dataset: {n_langs} languages, {len(feature_cols)} features")

    # 3. Build full triple arrays (deterministic)
    if args.architecture == "tcf":
        all_langs, all_feats, all_vals = build_all_triples_binary(
            data["binary_matrix"])
    else:
        all_langs, all_feats, all_vals = build_all_triples_categorical(
            data["cat_matrix"])
    n_total = len(all_langs)
    print(f"Observed triples: {n_total:,}")

    # 4. Val split — use a separate seeded numpy RNG so the split is
    #    reproducible independently of any other numpy RNG state
    split_rng = np.random.default_rng(args.seed)
    idx_all = np.arange(n_total, dtype=np.int64)
    split_rng.shuffle(idx_all)
    n_val = max(1, int(n_total * args.val_frac))
    val_idx = idx_all[:n_val]
    train_idx = idx_all[n_val:]
    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}")

    # 5. Build datasets
    if args.architecture == "tcf":
        train_ds = WALSDataset(
            all_langs[train_idx], all_feats[train_idx],
            all_vals[train_idx].astype(np.float32))
        val_ds = WALSDataset(
            all_langs[val_idx], all_feats[val_idx],
            all_vals[val_idx].astype(np.float32))
    else:
        train_ds = CategoricalTypDataset(
            all_langs[train_idx], all_feats[train_idx], all_vals[train_idx])
        val_ds = CategoricalTypDataset(
            all_langs[val_idx], all_feats[val_idx], all_vals[val_idx])

    # Seeded generator for shuffle reproducibility in the DataLoader
    dl_gen = torch.Generator()
    dl_gen.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, drop_last=False, generator=dl_gen)
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 4,
        shuffle=False, drop_last=False)

    # 6. Init model (uses torch RNG seeded in step 1)
    if args.architecture == "tcf":
        n_bfeat = data["binary_matrix"].shape[1]
        model: torch.nn.Module = TypologicalMF(n_langs, n_bfeat, args.embed_dim)
        print(f"Model: TypologicalMF  n_langs={n_langs}  "
              f"n_bfeat={n_bfeat}  embed_dim={args.embed_dim}")
    else:
        n_total_values = (
            max(max(gids) for gids in data["feat_to_global_ids"].values()) + 1)
        model = TypologicalMF_Learned(
            n_langs, n_total_values,
            data["feat_to_global_ids"], args.embed_dim)
        print(f"Model: TypologicalMF_Learned  n_langs={n_langs}  "
              f"n_total_values={n_total_values}  embed_dim={args.embed_dim}")

    model = model.to(args.device)

    # Adam with weight_decay=0 — CRITICAL.  Global weight_decay crushes sparse
    # embeddings via Adam's adaptive denominator (see model.py lines 99-112).
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr, weight_decay=0)

    # 7. Training loop with early stopping
    log_rows: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0

    header = (f"{'Epoch':>6}  {'TrainLoss':>10}  {'ValLoss':>10}  "
              f"{'LangNorm':>9}  {'FVNorm':>9}")
    print(f"\n{header}")
    print("-" * len(header))

    for epoch in range(1, args.n_epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer,
            args.l2_coef, args.architecture, args.device)
        val_loss = compute_val_loss(
            model, val_loader, args.architecture, args.device)
        lang_norm, fv_norm = get_emb_norms(model, args.architecture)

        print(f"{epoch:>6}  {train_loss:>10.4f}  {val_loss:>10.4f}  "
              f"{lang_norm:>9.4f}  {fv_norm:>9.4f}")

        log_rows.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "mean_lang_emb_norm": lang_norm,
            "mean_featvalue_emb_norm": fv_norm,
        })

        # --- Collapse detection ---
        # Norms collapsing to ~0.0002 with loss stuck at ln(2)≈0.6931
        # is the Adam+weight_decay embedding-collapse failure mode.
        if epoch >= 5 and (lang_norm < 1e-3 or fv_norm < 1e-3):
            print(f"\nFATAL: Embedding collapse detected at epoch {epoch}! "
                  f"lang_norm={lang_norm:.6f}, fv_norm={fv_norm:.6f}. "
                  f"This is the Adam weight_decay=0 violation.  Halting.")
            break
        # For T-CF (binary CE) the random baseline is ln(2)≈0.693.
        # For Learned (multiclass CE) the random baseline is ln(K) where K
        # is the avg number of values per feature — always > ln(2), so we
        # don't flag Learned with the binary threshold.
        if epoch >= 10 and args.architecture == "tcf" and train_loss > 0.693:
            print(f"\nWARNING: T-CF train_loss≈ln(2) at epoch {epoch} — "
                  f"model may not be learning.  Check data and seeding.")

        # --- Early stopping ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            no_improve = 0
            torch.save(
                {k: v.cpu().clone() for k, v in model.state_dict().items()},
                os.path.join(args.out_dir, "model_best.pt"))
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"\nEarly stop at epoch {epoch}  "
                      f"(best val={best_val_loss:.4f} at epoch {best_epoch})")
                break

    # 8. Save training log
    pd.DataFrame(log_rows).to_csv(
        os.path.join(args.out_dir, "training_log.csv"), index=False)

    # 9. Reload best checkpoint and dump artifacts
    print(f"\nReloading best checkpoint (epoch {best_epoch}, "
          f"val_loss={best_val_loss:.4f})")
    best_state = torch.load(
        os.path.join(args.out_dir, "model_best.pt"),
        map_location="cpu")
    model.load_state_dict(best_state)
    model = model.to(args.device)

    lang2id_glot, lang2id_full = build_lang2id(
        df, args.database, args.data_path)
    dump_artifacts(model, args, data, lang2id_glot, lang2id_full, args.out_dir)

    # 10. Config dump
    config: dict[str, Any] = vars(args).copy()
    config["n_epochs_run"] = len(log_rows)
    config["best_epoch"] = best_epoch
    config["best_val_loss"] = best_val_loss
    with open(os.path.join(args.out_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)

    elapsed = time.time() - t0
    final_lang_norm, final_fv_norm = get_emb_norms(model, args.architecture)
    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Best val loss : {best_val_loss:.4f} (epoch {best_epoch})")
    print(f"  Final lang emb norm  : {final_lang_norm:.4f}")
    print(f"  Final featvalue norm : {final_fv_norm:.4f}")
    print(f"  Artifacts in : {args.out_dir}/")


if __name__ == "__main__":
    main()
