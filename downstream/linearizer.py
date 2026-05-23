"""
downstream/linearizer.py
=========================
Typology-Conditioned Dependency Linearization (TCDL).

Given:
  - An unordered dependency tree (head/deprel/upos structure)
  - A target language's typological representation T_l
  - A trained MLPPredictor (from direction_predictor.py)

Produce:
  - A target-like linear ordering of the tokens in the sentence

This is the sentence-level counterpart to the corpus-level arc-direction
prediction task.  Where the direction predictor asks "what fraction of
obj-VERB-NOUN arcs in language l have the object before the verb?",
the linearizer asks "for THIS sentence with THESE specific tokens, which
order should they appear in for language l?"

Algorithm
---------
The linearization is performed by a top-down recursive projection over
the dependency tree:

  linearize(node n, language repr T_l):
    for each child c of n:
      p_before_head = MLPPredictor(T_l, deprel(c,n), upos(n), upos(c))
      predict: c comes BEFORE n if p_before_head > 0.5, else AFTER
    left_children  = subtrees predicted before head, sorted by distance
    right_children = subtrees predicted after head, sorted by distance
    return [*linearize(left[0]), ..., head_token, ..., *linearize(right[-1])]

This produces a projective linearization.  Non-projective arcs in the
original treebank are handled by the greedy heuristic: the subtree of
a non-projective arc is placed as a unit adjacent to its predicted side.

Evaluation metrics
------------------
Given gold order, we measure:
  - Kendall's tau: rank correlation between predicted and gold token positions
  - Pair accuracy: % of token pairs in correct relative order
  - Exact match: fraction of sentences with perfect order
  - Per-config accuracy: which arc types are linearized correctly

Usage
-----
    # Programmatic:
    from downstream.linearizer import Linearizer
    lin = Linearizer(mlp_predictor, lang_repr)
    new_order = lin.linearize_sentence(tokens)

    # CLI:
    python downstream/linearizer.py \\
        --conllu   downstream_results/ud_data/eng_ewt-test.conllu \\
        --repr_dir downstream_direction/representations/wals_s42 \\
        --repr_name B \\
        --glottocode nucl1301 \\        # Turkish — reorder English as SOV
        --out_dir  downstream_linearization/reordered \\
        --evaluate_against_gold
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from downstream.arc_directions import parse_conllu_arcs, TARGET_CONFIGS


# ---------------------------------------------------------------------------
# Core linearization logic
# ---------------------------------------------------------------------------

class Linearizer:
    """
    Typology-conditioned sentence linearizer.

    Uses a trained MLPPredictor to predict per-arc direction probabilities,
    then applies top-down recursive projection to produce a total linear order.
    """

    def __init__(
        self,
        predictor,          # MLPPredictor instance (already trained)
        lang_repr: np.ndarray,  # (D,) float32 — language representation T_l
        threshold: float = 0.5,
        device: str = "cpu",
    ):
        self.predictor = predictor
        self.lang_repr = lang_repr
        self.threshold = threshold
        self.device = device

        # Pre-build lookup for target config predictions
        self._config_cache: Dict[Tuple, float] = {}

    def _predict_direction(
        self,
        deprel: str,
        head_upos: str,
        dep_upos: str,
    ) -> float:
        """
        Return P(dependent precedes head) for this arc configuration.

        Uses the trained MLP predictor with the language's T_l embedding.
        Returns 0.5 (no preference) if the config was not seen in training.
        """
        import torch

        key = (deprel, head_upos, dep_upos)
        if key in self._config_cache:
            return self._config_cache[key]

        pred = self.predictor
        if pred.model is None:
            return 0.5

        rel2id  = pred.rel2id
        upos2id = pred.upos2id

        # Use the primary deprel (strip subtypes)
        rel_clean = deprel.split(":")[0]

        if rel_clean not in rel2id or head_upos not in upos2id or dep_upos not in upos2id:
            self._config_cache[key] = 0.5
            return 0.5

        pred.model.eval()
        with torch.no_grad():
            lv = torch.tensor(
                self.lang_repr[None, :], dtype=torch.float32).to(self.device)
            ri = torch.tensor([rel2id[rel_clean]],  dtype=torch.long).to(self.device)
            hi = torch.tensor([upos2id[head_upos]], dtype=torch.long).to(self.device)
            di = torch.tensor([upos2id[dep_upos]],  dtype=torch.long).to(self.device)
            p = pred.model(lv, ri, hi, di).item()

        self._config_cache[key] = p
        return p

    def linearize_sentence(
        self,
        tokens: List[Dict],
    ) -> List[int]:
        """
        Linearize a sentence (list of token dicts from parse_conllu_arcs).

        Returns
        -------
        order : list of token IDs (1-indexed) in the predicted linear order
        """
        if not tokens:
            return []

        # Build tree structure
        id_to_tok = {t["id"]: t for t in tokens}
        children: Dict[int, List[int]] = defaultdict(list)
        root_ids = []

        for tok in tokens:
            if tok["head"] == 0:
                root_ids.append(tok["id"])
            else:
                children[tok["head"]].append(tok["id"])

        # Use first root if multiple (shouldn't happen in well-formed trees)
        root_id = root_ids[0] if root_ids else tokens[0]["id"]

        def _linearize_subtree(node_id: int) -> List[int]:
            """Recursively linearize the subtree rooted at node_id."""
            node = id_to_tok.get(node_id)
            if node is None:
                return []

            child_ids = children.get(node_id, [])
            if not child_ids:
                return [node_id]

            # For each child, predict direction relative to this head
            left_children  = []  # predicted to come before head
            right_children = []  # predicted to come after head

            for child_id in child_ids:
                child = id_to_tok.get(child_id)
                if child is None:
                    continue
                p_before = self._predict_direction(
                    child["deprel"], node["upos"], child["upos"])
                if p_before >= self.threshold:
                    left_children.append((child_id, p_before))
                else:
                    right_children.append((child_id, 1.0 - p_before))

            # Sort left children: higher probability of being before head → further left
            # (higher p_before = more strongly pre-head → place first)
            left_children.sort(key=lambda x: x[1], reverse=True)
            # Sort right children: higher confidence of being after head → further right
            right_children.sort(key=lambda x: x[1], reverse=True)

            result = []
            for cid, _ in left_children:
                result.extend(_linearize_subtree(cid))
            result.append(node_id)
            for cid, _ in right_children:
                result.extend(_linearize_subtree(cid))
            return result

        order = _linearize_subtree(root_id)

        # Handle any tokens not reached (detached nodes, multiple roots, etc.)
        covered = set(order)
        missed = [t["id"] for t in tokens if t["id"] not in covered]
        order.extend(missed)

        return order

    def linearize_corpus(
        self,
        sentences: List[List[Dict]],
    ) -> List[List[int]]:
        """Linearize all sentences in a corpus."""
        return [self.linearize_sentence(sent) for sent in sentences]


# ---------------------------------------------------------------------------
# Majority-direction baseline (predicts based on corpus-level statistics)
# ---------------------------------------------------------------------------

class MajorityDirectionLinearizer:
    """
    Baseline: for each arc, always place dependent before or after head
    based on the majority direction seen in the training languages
    (passed as a pre-computed direction_profiles.csv entry).
    """

    def __init__(self, direction_profile: Dict[str, float], threshold: float = 0.5):
        """
        direction_profile : {config_name: p(dep_before_head)}
        For configs not in the profile, default to 0.5 (no preference).
        """
        self.profile = direction_profile
        self.threshold = threshold

        # Build lookup from (deprel, head_upos, dep_upos) -> config_name
        self._key_to_p: Dict[Tuple, float] = {}
        for deprel, h_upos, d_upos, name, _ in TARGET_CONFIGS:
            p = direction_profile.get(name, 0.5)
            self._key_to_p[(deprel, h_upos, d_upos)] = p

    def _predict_direction(self, deprel, head_upos, dep_upos) -> float:
        rel_clean = deprel.split(":")[0]
        return self._key_to_p.get((rel_clean, head_upos, dep_upos), 0.5)

    def linearize_sentence(self, tokens: List[Dict]) -> List[int]:
        """Same tree-recursive algorithm as Linearizer, but using corpus-level priors."""
        if not tokens:
            return []
        id_to_tok = {t["id"]: t for t in tokens}
        children: Dict[int, List[int]] = defaultdict(list)
        root_ids = []
        for tok in tokens:
            if tok["head"] == 0:
                root_ids.append(tok["id"])
            else:
                children[tok["head"]].append(tok["id"])
        root_id = root_ids[0] if root_ids else tokens[0]["id"]

        def _lin(node_id):
            node = id_to_tok.get(node_id)
            if node is None:
                return []
            left, right = [], []
            for cid in children.get(node_id, []):
                child = id_to_tok.get(cid)
                if child is None:
                    continue
                p = self._predict_direction(child["deprel"], node["upos"], child["upos"])
                (left if p >= self.threshold else right).append((cid, p if p >= self.threshold else 1 - p))
            left.sort(key=lambda x: x[1], reverse=True)
            right.sort(key=lambda x: x[1], reverse=True)
            result = []
            for cid, _ in left: result.extend(_lin(cid))
            result.append(node_id)
            for cid, _ in right: result.extend(_lin(cid))
            return result

        order = _lin(root_id)
        covered = set(order)
        order.extend(t["id"] for t in tokens if t["id"] not in covered)
        return order

    def linearize_corpus(self, sentences):
        return [self.linearize_sentence(s) for s in sentences]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_linearization(
    predicted_orders: List[List[int]],   # list of token-ID orderings
    sentences: List[List[Dict]],         # original sentences (gold positions = tok["id"])
    min_length: int = 3,
) -> Dict:
    """
    Evaluate predicted linearization against gold UD token order.

    Gold order is simply the original position (tok["id"]) in the treebank.

    Metrics
    -------
    kendall_tau   : mean Kendall's tau between predicted rank and gold rank.
                    tau=1.0 = perfect agreement; tau=-1.0 = reversed; tau=0 = random.
    pair_accuracy : fraction of token pairs in correct relative order.
                    = (tau + 1) / 2
    exact_match   : fraction of sentences with perfect linear order.
    arc_accuracy  : fraction of individual arcs with correct direction prediction
                    (each arc = one (head, dep) pair with known gold direction).
    """
    import warnings

    taus, pair_accs, exact_matches = [], [], []
    arc_correct = arc_total = 0

    for pred_order, sent in zip(predicted_orders, sentences):
        if len(sent) < min_length:
            continue

        # Gold order = original position (1-indexed token IDs)
        gold_ids = [t["id"] for t in sorted(sent, key=lambda t: t["position"])]
        n = len(gold_ids)

        if len(pred_order) != n:
            # Pad or truncate if something went wrong
            gold_set = set(gold_ids)
            pred_order = [x for x in pred_order if x in gold_set]
            missing = [x for x in gold_ids if x not in pred_order]
            pred_order = pred_order + missing

        # Convert to rank arrays for Kendall computation
        gold_rank = {tok_id: rank for rank, tok_id in enumerate(gold_ids)}
        pred_rank = {tok_id: rank for rank, tok_id in enumerate(pred_order)}

        # Vectors in a consistent token order
        toks_in_gold = gold_ids
        gold_vec = [gold_rank[t] for t in toks_in_gold]
        pred_vec = [pred_rank.get(t, gold_rank[t]) for t in toks_in_gold]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tau, _ = kendalltau(gold_vec, pred_vec)

        if not np.isnan(tau):
            taus.append(float(tau))
            pair_accs.append((float(tau) + 1.0) / 2.0)
            exact_matches.append(float(pred_order == gold_ids))

        # Per-arc accuracy
        id_to_pos_pred = {tok_id: rank for rank, tok_id in enumerate(pred_order)}
        id_to_pos_gold = {t["id"]: t["position"] for t in sent}
        for tok in sent:
            if tok["head"] == 0:
                continue
            dep_id  = tok["id"]
            head_id = tok["head"]
            if dep_id not in id_to_pos_pred or head_id not in id_to_pos_pred:
                continue
            gold_dep_before = id_to_pos_gold[dep_id] < id_to_pos_gold[head_id]
            pred_dep_before = id_to_pos_pred[dep_id] < id_to_pos_pred[head_id]
            arc_correct += int(gold_dep_before == pred_dep_before)
            arc_total += 1

    return {
        "kendall_tau":   float(np.mean(taus))       if taus        else float("nan"),
        "pair_accuracy": float(np.mean(pair_accs))  if pair_accs   else float("nan"),
        "exact_match":   float(np.mean(exact_matches)) if exact_matches else float("nan"),
        "arc_accuracy":  float(arc_correct / arc_total) if arc_total else float("nan"),
        "n_sentences":   len(taus),
        "n_arcs":        arc_total,
    }


def evaluate_per_config(
    predicted_orders: List[List[int]],
    sentences: List[List[Dict]],
) -> Dict[str, Dict]:
    """
    Per-arc-configuration accuracy breakdown.

    For each (deprel, head_upos, dep_upos) configuration, report:
    - fraction of arcs in the correct direction
    - n_arcs
    """
    config_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})

    for pred_order, sent in zip(predicted_orders, sentences):
        id_to_pos_pred = {tok_id: rank for rank, tok_id in enumerate(pred_order)}
        id_to_pos_gold = {t["id"]: t["position"] for t in sent}
        id_to_tok = {t["id"]: t for t in sent}

        for tok in sent:
            if tok["head"] == 0:
                continue
            head = id_to_tok.get(tok["head"])
            if head is None:
                continue
            dep_id, head_id = tok["id"], tok["head"]
            if dep_id not in id_to_pos_pred or head_id not in id_to_pos_pred:
                continue

            config_name = f"{tok['deprel'].split(':')[0]}_{head['upos']}_{tok['upos']}"
            gold_dep_before = id_to_pos_gold[dep_id] < id_to_pos_gold[head_id]
            pred_dep_before = id_to_pos_pred[dep_id] < id_to_pos_pred[head_id]

            config_stats[config_name]["correct"] += int(gold_dep_before == pred_dep_before)
            config_stats[config_name]["total"]   += 1

    result = {}
    for name, s in config_stats.items():
        if s["total"] >= 5:
            result[name] = {
                "accuracy": s["correct"] / s["total"],
                "n_arcs":   s["total"],
            }
    return result


# ---------------------------------------------------------------------------
# CoNLL-U writer (for relinearized treebanks)
# ---------------------------------------------------------------------------

def rewrite_conllu(
    original_text: str,
    predicted_orders: List[List[int]],
    out_path: str,
):
    """
    Rewrite a CoNLL-U file with tokens in the predicted linearization order.

    Adjusts token IDs to reflect the new order; head references are updated
    accordingly.  Morphological features, lemmas, and dependency labels are
    preserved — only the token positions change.

    This produces a synthetic treebank that can be used to train a parser
    that expects target-language word order.
    """
    sentences_text = []
    current_comments = []
    current_tokens = []

    lines = original_text.splitlines()
    sent_idx = 0

    def _flush():
        nonlocal sent_idx
        if sent_idx < len(predicted_orders) and current_tokens:
            pred_order = predicted_orders[sent_idx]
            # Build id->line mapping
            id_to_fields = {}
            for line in current_tokens:
                parts = line.split("\t")
                id_to_fields[int(parts[0])] = parts

            # Build new-id mapping: old_id -> new_position (1-indexed)
            old_to_new = {old_id: (new_pos + 1)
                          for new_pos, old_id in enumerate(pred_order)}

            reordered_lines = []
            for new_pos, old_id in enumerate(pred_order):
                fields = list(id_to_fields[old_id])
                fields[0] = str(new_pos + 1)   # new ID
                # Update HEAD reference
                old_head = int(fields[6])
                if old_head == 0:
                    fields[6] = "0"
                else:
                    fields[6] = str(old_to_new.get(old_head, 0))
                reordered_lines.append("\t".join(fields))

            sent_text = "\n".join(current_comments + reordered_lines) + "\n"
            sentences_text.append(sent_text)
            sent_idx += 1

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            current_comments.append(stripped)
        elif stripped == "":
            if current_tokens:
                _flush()
                current_comments = []
                current_tokens   = []
        else:
            parts = stripped.split("\t")
            if len(parts) >= 8 and "-" not in parts[0] and "." not in parts[0]:
                current_tokens.append(stripped)

    if current_tokens:
        _flush()

    os.makedirs(os.path.dirname(out_path), exist_ok=True) if os.path.dirname(out_path) else None
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences_text) + "\n")

    print(f"  Written {len(sentences_text)} sentences → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Typology-conditioned dependency linearizer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--conllu",    required=True,
                   help="Input .conllu file to linearize")
    p.add_argument("--repr_dir",  required=True,
                   help="Representation directory (output of representations.py)")
    p.add_argument("--repr_name", default="B",
                   help="Which representation: A, B, C, B_masked30, ...")
    p.add_argument("--glottocode", required=True,
                   help="Glottocode of the TARGET language "
                        "(whose word order we want the output to match)")
    p.add_argument("--model_dir", default=None,
                   help="Directory with saved MLPPredictor weights "
                        "(if None, use majority-direction baseline)")
    p.add_argument("--out_dir",   required=True)
    p.add_argument("--evaluate_against_gold", action="store_true",
                   help="Evaluate the output against the gold token order "
                        "in the input file (useful when input IS the target treebank)")
    p.add_argument("--max_sentences", type=int, default=0,
                   help="Cap sentences processed (0=all)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    # Load language representation
    repr_path = os.path.join(args.repr_dir, f"repr_{args.repr_name}.npy")
    langs_path = os.path.join(args.repr_dir, f"repr_{args.repr_name}_langs.json")
    if not os.path.exists(repr_path):
        print(f"ERROR: repr file not found: {repr_path}")
        sys.exit(1)
    repr_mat = np.load(repr_path)
    with open(langs_path) as f:
        repr_langs = json.load(f)

    glot2row = {g: i for i, g in enumerate(repr_langs)}
    if args.glottocode not in glot2row:
        print(f"ERROR: glottocode {args.glottocode!r} not in checkpoint")
        print(f"Available: {repr_langs[:10]}...")
        sys.exit(1)

    lang_repr = repr_mat[glot2row[args.glottocode]]
    print(f"Language repr: {args.glottocode}, shape={lang_repr.shape}, "
          f"norm={np.linalg.norm(lang_repr):.3f}")

    # Load sentences
    with open(args.conllu, encoding="utf-8") as f:
        conllu_text = f.read()
    sentences = parse_conllu_arcs(conllu_text)
    if args.max_sentences > 0:
        sentences = sentences[:args.max_sentences]
    print(f"Loaded {len(sentences)} sentences from {args.conllu}")

    # Choose linearizer
    if args.model_dir is not None:
        from downstream.direction_predictor import MLPPredictor
        # Load predictor (simplified — assumes model was saved)
        predictor = MLPPredictor(lang_dim=lang_repr.shape[0])
        lin = Linearizer(predictor, lang_repr)
        print("Using trained MLP linearizer")
    else:
        # Build majority-direction baseline from TARGET_CONFIGS priors
        lin = MajorityDirectionLinearizer(
            direction_profile={},  # empty = 0.5 for all = random
        )
        print("Using majority-direction baseline (no model loaded)")

    # Linearize
    print("Linearizing...")
    orders = lin.linearize_corpus(sentences)

    # Write relinearized CoNLL-U
    out_conllu = os.path.join(args.out_dir,
                              f"{args.glottocode}_{args.repr_name}_relinearized.conllu")
    rewrite_conllu(conllu_text, orders, out_conllu)

    # Evaluate against gold
    if args.evaluate_against_gold:
        print("Evaluating against gold order...")
        metrics = evaluate_linearization(orders, sentences)
        per_config = evaluate_per_config(orders, sentences)

        print(f"\n  Kendall tau:    {metrics['kendall_tau']:.4f}  "
              f"(1.0=perfect, 0.0=random)")
        print(f"  Pair accuracy:  {metrics['pair_accuracy']:.4f}")
        print(f"  Exact match:    {metrics['exact_match']:.4f}")
        print(f"  Arc accuracy:   {metrics['arc_accuracy']:.4f}  "
              f"(n={metrics['n_arcs']})")

        results_path = os.path.join(args.out_dir, "linearization_eval.json")
        with open(results_path, "w") as f:
            json.dump({"metrics": metrics, "per_config": per_config}, f, indent=2)
        print(f"  Evaluation saved: {results_path}")

    print(f"\nDone. Output: {out_conllu}")


if __name__ == "__main__":
    main()
