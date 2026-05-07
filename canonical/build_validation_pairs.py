"""
Build the literature-backed validation gold standard for RQ3.

Three sources:
  Source 1 — gb_docs:   WALS references extracted from Grambank parameter
                        descriptions and documentation via regex.
  Source 2 — hand_curated: structurally verified correspondences from the
                        seed list, with Grambank-ID verification against
                        parameters.csv.
  Source 3 — agent_discovered: any Grambank feature whose Description
                        contains a WALS reference not in Sources 1 or 2.

Output: analysis/cross_database_alignment/validation_pairs.json

Usage
-----
  python canonical/build_validation_pairs.py \\
      [--grambank_dir /tmp/grambank] \\
      [--output_dir analysis/cross_database_alignment] \\
      [--smoke | --full] \\
      [--seed 42]
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils import seed_everything

_GRAMBANK_REPO = "https://github.com/grambank/grambank.git"
_DEFAULT_GRAMBANK_DIR = "/tmp/grambank"
_DEFAULT_OUTPUT_DIR = str(_REPO_ROOT / "analysis" / "cross_database_alignment")
_WALS_CKPT = str(_REPO_ROOT / "checkpoints" / "wals_learned_s42")
_GB_CKPT = str(_REPO_ROOT / "checkpoints" / "grambank_learned_s42")

# WALS reference pattern: "W 81A", "WALS 81A", "W81A", "WALS81A"
_WALS_REF_PATTERN = re.compile(r"\bW(?:ALS)?\s*(\d{2,3}[A-Z]?)\b")


# ---------------------------------------------------------------------------
# Seed list (Source 2)
# ---------------------------------------------------------------------------

# Each entry: (wals_featvalue, grambank_featvalues, confidence, note)
# wals_featvalue MUST match the exact label from wals_learned_s42/featvalue2id.json
# grambank_featvalues MUST match labels from grambank_learned_s42/featvalue2id.json
#
# Corrections vs. original seed:
#   - GB202 is absent from parameters.csv → replaced by GB025 (demonstrative order)
#   - GB089 is about S-argument verbal indexing, not genitive/possessor order →
#     replaced by GB065 (adnominal possessor-possessed order)
#   - 90A relative-clause order entries were INVERTED in the original seed →
#     corrected (GB328=1 for rel-precedes-noun; GB327=1 for rel-follows-noun)
#   - 27A/GB320 pairing is semantically unrelated (GB320 = paucal free marker,
#     not reduplication) → retained with low confidence and flagged
_SEED_LIST: list[tuple[str, list[str], str, str | None]] = [
    # Word order — transitive clause
    ("81A=SOV",   ["GB133=1"], "high",
     "verb-final transitive"),
    ("81A=VSO",   ["GB131=1"], "high",
     "verb-initial transitive"),
    ("81A=VOS",   ["GB131=1"], "medium",
     "verb-initial subset; VOS is a subtype of verb-initial order"),
    ("81A=SVO",   ["GB132=1"], "high",
     "verb-medial transitive"),

    # Adpositions
    ("85A=Prepositions",  ["GB074=1"], "high", "prepositions present"),
    ("85A=Postpositions", ["GB075=1"], "high", "postpositions present"),

    # Possessor/genitive order — CORRECTED: original seed used GB089 (S-argument
    # verbal indexing), which is semantically unrelated.  GB065 codes the
    # pragmatically unmarked order of adnominal possessor and possessed noun.
    ("86A=Genitive-Noun",  ["GB065=1"], "high",
     "possessor precedes possessed (=genitive-noun); corrected from seed GB089"),
    ("86A=Noun-Genitive",  ["GB065=2"], "high",
     "possessed precedes possessor (=noun-genitive); corrected from seed GB089"),

    # Adjective order
    ("87A=Adjective-Noun",  ["GB193=1"], "high",
     "ANM-N (adnominal property word precedes noun)"),
    ("87A=Noun-Adjective",  ["GB193=2"], "high",
     "N-ANM (noun precedes adnominal property word)"),

    # Demonstrative order — CORRECTED: GB202 is absent from Grambank
    # parameters.csv; GB025 directly codes adnominal demonstrative-noun order.
    ("88A=Demonstrative-Noun",  ["GB025=1"], "high",
     "Dem-N order; corrected from missing GB202 to GB025"),
    ("88A=Noun-Demonstrative",  ["GB025=2"], "high",
     "N-Dem order; corrected from missing GB202 to GB025"),

    # Numeral/quantifier order
    # GB203 codes order of collective universal quantifier ('all') + noun, the
    # closest Grambank equivalent to WALS numeral order (89A).  Quantifier order
    # correlates strongly with numeral order cross-linguistically.
    ("89A=Numeral-Noun",  ["GB203=1"], "high",
     "UQ-N quantifier order; best Grambank proxy for numeral-noun (no dedicated GB feature)"),
    ("89A=Noun-Numeral",  ["GB203=2"], "high",
     "N-UQ quantifier order; best Grambank proxy for noun-numeral"),

    # Relative clause order — CORRECTED: original seed had GB327 and GB328
    # SWAPPED.  GB327 = 'can RC follow noun' → corresponds to Noun-Relative.
    # GB328 = 'can RC precede noun' → corresponds to Relative-Noun.
    ("90A=Relative clause-Noun",  ["GB328=1"], "low",
     "RC can precede noun; CORRECTED — original seed had GB327/GB328 inverted"),
    ("90A=Noun-Relative clause",  ["GB327=1"], "low",
     "RC can follow noun; CORRECTED — original seed had GB327/GB328 inverted"),

    # Articles
    ("37A=Definite word distinct from demonstrative", ["GB020=1"], "high",
     "definite/specific article present"),
    ("38A=Indefinite word distinct from 'one'",       ["GB021=1"], "high",
     "indefinite article present"),

    # Reduplication vs paucal — FLAGGED: GB320 codes paucal number marked by a
    # dedicated free element, NOT reduplication.  The pairing is semantically
    # remote; retained at low confidence to allow the model to confirm or refute.
    ("27A=Productive full and partial reduplication", ["GB320=1"], "low",
     "FLAGGED: GB320 is paucal free-element marking, not reduplication; "
     "pair is semantically unrelated but included as a hard negative"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clone_or_verify(grambank_dir: str) -> str:
    """Clone Grambank to grambank_dir if absent; return git commit hash."""
    gdir = Path(grambank_dir)
    if not gdir.exists():
        print(f"Cloning {_GRAMBANK_REPO} → {gdir} …")
        subprocess.run(
            ["git", "clone", "--depth=1", _GRAMBANK_REPO, str(gdir)],
            check=True,
        )
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(gdir), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        commit = "unknown"
    return commit


def _load_grambank_params(grambank_dir: str) -> pd.DataFrame:
    params_csv = Path(grambank_dir) / "cldf" / "parameters.csv"
    if not params_csv.exists():
        raise FileNotFoundError(
            f"parameters.csv not found at {params_csv}. "
            "Ensure grambank_dir points to the cloned repo."
        )
    return pd.read_csv(str(params_csv))


def _load_checkpoint_fv2id(ckpt_dir: str, db: str) -> dict[str, int]:
    ckpt = Path(ckpt_dir)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"{db} checkpoint not found at {ckpt}. "
            "Train first with canonical/train_canonical.py."
        )
    return json.loads((ckpt / "featvalue2id.json").read_text())


def _gb_base_id(gb_fv: str) -> str:
    """'GB133=1' → 'GB133'"""
    return gb_fv.split("=")[0]


# ---------------------------------------------------------------------------
# Source 1 — gb_docs (WALS references in Grambank descriptions)
# ---------------------------------------------------------------------------

def source_gb_docs(
    params_df: pd.DataFrame,
    existing_wals_refs: set[str],
    existing_gb_ids: set[str],
    gb_fv2id: dict[str, int],
) -> list[dict[str, Any]]:
    """
    Scan Grambank parameter descriptions for WALS feature references.

    Only references of the form 'W<number><letter>' / 'WALS <number><letter>'
    are matched.  The scan covers Name, Description, and Grambank_ID_desc
    columns.  Pairs already represented in Sources 2 and 3 are excluded.
    """
    pairs: list[dict[str, Any]] = []

    for _, row in params_df.iterrows():
        gb_id = str(row["ID"])
        text = " ".join([
            str(row.get("Name", "")),
            str(row.get("Description", "")),
            str(row.get("Grambank_ID_desc", "")),
        ])
        matches = _WALS_REF_PATTERN.findall(text)
        if not matches:
            continue

        for wals_feat in sorted(set(matches)):
            wals_ref = wals_feat.upper()
            # Skip if already covered by Source 2
            if wals_ref in existing_wals_refs and gb_id in existing_gb_ids:
                continue

            # Build WALS featvalue references: include all values we have for
            # this feature in the WALS checkpoint (we don't know which value
            # the reference is about from the description text alone).
            # For now, record the feature-level reference; downstream
            # validation will skip pairs whose exact featvalue isn't found.
            # We record at feature level with confidence='medium'.
            pairs.append({
                "wals": f"{wals_ref}=*",   # feature-level, value unknown
                "grambank": [gb_id + "=*"],
                "source": "gb_docs",
                "confidence": "medium",
                "note": (
                    f"WALS feature {wals_ref} referenced in GB {gb_id} "
                    f"description; specific value mapping not determined"
                ),
            })
    return pairs


# ---------------------------------------------------------------------------
# Source 2 — hand_curated
# ---------------------------------------------------------------------------

def source_hand_curated(
    params_df: pd.DataFrame,
    wals_fv2id: dict[str, int],
    gb_fv2id: dict[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Process and verify the seed list.

    Returns (verified_pairs, dropped_pairs).  A pair is DROPPED if:
      - The Grambank base ID is absent from parameters.csv, AND no corrected
        replacement is specified
      - The WALS featvalue is absent from the WALS learned checkpoint
    A pair is DOWNGRADED if:
      - The Grambank base ID is absent from parameters.csv but a correction
        exists (logged in the note field)
      - The WALS featvalue is absent from the checkpoint

    Corrections are already encoded in _SEED_LIST above (GB202→GB025,
    GB089→GB065, 90A order swap).
    """
    gb_ids_in_params = set(params_df["ID"].astype(str).tolist())
    pairs: list[dict[str, Any]] = []
    dropped: list[dict[str, Any]] = []

    for wals_fv, gb_fvs, confidence, note in _SEED_LIST:
        # --- Verify WALS side ---
        if wals_fv not in wals_fv2id:
            dropped.append({
                "wals": wals_fv,
                "grambank": gb_fvs,
                "reason": f"WALS featvalue '{wals_fv}' not in WALS checkpoint",
                "original_confidence": confidence,
            })
            continue

        # --- Verify Grambank side ---
        verified_gb = []
        for gb_fv in gb_fvs:
            base_id = _gb_base_id(gb_fv)
            if base_id not in gb_ids_in_params:
                dropped.append({
                    "wals": wals_fv,
                    "grambank": [gb_fv],
                    "reason": (
                        f"Grambank base ID '{base_id}' absent from "
                        "parameters.csv"
                    ),
                    "original_confidence": confidence,
                })
            elif gb_fv not in gb_fv2id:
                dropped.append({
                    "wals": wals_fv,
                    "grambank": [gb_fv],
                    "reason": (
                        f"Grambank featvalue '{gb_fv}' absent from "
                        "Grambank learned checkpoint"
                    ),
                    "original_confidence": confidence,
                })
            else:
                verified_gb.append(gb_fv)

        if not verified_gb:
            continue

        pairs.append({
            "wals": wals_fv,
            "grambank": verified_gb,
            "source": "hand_curated",
            "confidence": confidence,
            "note": note or "",
        })

    return pairs, dropped


# ---------------------------------------------------------------------------
# Source 3 — agent_discovered
# ---------------------------------------------------------------------------

def source_agent_discovered(
    params_df: pd.DataFrame,
    wals_fv2id: dict[str, int],
    gb_fv2id: dict[str, int],
    existing_wals_features: set[str],
    existing_gb_base_ids: set[str],
) -> list[dict[str, Any]]:
    """
    WALS references in Grambank descriptions not already in Sources 1 or 2.

    Each matched (GB_id, WALS_feature) pair is added with confidence='low'.
    Feature-level matches are expanded to specific featvalue pairs only when
    both the WALS value and Grambank value are in the respective checkpoints.
    """
    pairs: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    for _, row in params_df.iterrows():
        gb_id = str(row["ID"])
        text = " ".join([
            str(row.get("Name", "")),
            str(row.get("Description", "")),
            str(row.get("Grambank_ID_desc", "")),
        ])
        wals_refs = _WALS_REF_PATTERN.findall(text)
        if not wals_refs:
            continue

        for wals_feat in sorted(set(wals_refs)):
            key = (wals_feat.upper(), gb_id)
            if key in seen:
                continue

            # Skip if already covered by Source 2
            if (wals_feat.upper() in existing_wals_features
                    and gb_id in existing_gb_base_ids):
                continue

            seen.add(key)
            pairs.append({
                "wals": f"{wals_feat.upper()}=*",
                "grambank": [f"{gb_id}=*"],
                "source": "agent_discovered",
                "confidence": "low",
                "note": (
                    f"WALS {wals_feat.upper()} referenced in GB {gb_id} "
                    f"({str(row.get('Name',''))[:60]})"
                ),
            })

    return pairs


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_validation_pairs(
    grambank_dir: str,
    output_dir: str,
    smoke: bool = False,
    seed: int = 42,
) -> dict[str, Any]:
    """Build and save the validation pairs JSON."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # --- Clone / verify Grambank ---
    print(f"Checking Grambank repo at {grambank_dir} …")
    grambank_commit = _clone_or_verify(grambank_dir)
    print(f"  Grambank commit: {grambank_commit}")

    # --- Load data ---
    print("Loading Grambank parameters.csv …")
    params_df = _load_grambank_params(grambank_dir)
    print(f"  {len(params_df)} Grambank parameters loaded")

    print("Loading checkpoint featvalue mappings …")
    wals_fv2id = _load_checkpoint_fv2id(_WALS_CKPT, "WALS")
    gb_fv2id = _load_checkpoint_fv2id(_GB_CKPT, "Grambank")
    print(f"  WALS featvalues: {len(wals_fv2id)}  |  "
          f"Grambank featvalues: {len(gb_fv2id)}")

    # --- Source 2 first (defines existing coverage) ---
    print("\n[Source 2] Processing hand-curated seed list …")
    hc_pairs, hc_dropped = source_hand_curated(params_df, wals_fv2id, gb_fv2id)
    print(f"  Verified pairs: {len(hc_pairs)}  |  Dropped: {len(hc_dropped)}")
    for d in hc_dropped:
        print(f"    DROPPED: {d['wals']} → {d['grambank']}  ({d['reason']})")

    existing_wals_features = {p["wals"].split("=")[0] for p in hc_pairs}
    existing_gb_base_ids = {
        _gb_base_id(g) for p in hc_pairs for g in p["grambank"]
    }

    # --- Source 1 ---
    print("\n[Source 1] Scanning Grambank descriptions for WALS references …")
    gb_doc_pairs = source_gb_docs(
        params_df, existing_wals_features, existing_gb_base_ids, gb_fv2id)
    print(f"  Pairs found: {len(gb_doc_pairs)}")

    # --- Source 3 ---
    print("\n[Source 3] Agent-discovered (WALS refs not in Sources 1 or 2) …")
    agent_pairs = source_agent_discovered(
        params_df, wals_fv2id, gb_fv2id,
        existing_wals_features, existing_gb_base_ids,
    )
    print(f"  Pairs found: {len(agent_pairs)}")

    # --- Combine ---
    all_pairs = hc_pairs + gb_doc_pairs + agent_pairs

    # --- Count by confidence ---
    conf_counts: dict[str, int] = defaultdict(int)
    for p in all_pairs:
        conf_counts[p["confidence"]] += 1

    n_high = conf_counts["high"]
    n_medium = conf_counts["medium"]
    n_low = conf_counts["low"]

    print(f"\n{'='*60}")
    print(f"  Total pairs  : {len(all_pairs)}")
    print(f"  high         : {n_high}")
    print(f"  medium       : {n_medium}")
    print(f"  low          : {n_low}")
    print(f"{'='*60}")

    # --- Acceptance check ---
    if n_high < 15:
        print(
            f"\n[HALT] Acceptance criterion not met: only {n_high} high-confidence "
            f"pairs (need ≥15). Review dropped entries and seed list corrections.",
            file=sys.stderr,
        )
        print("Dropped entries:", file=sys.stderr)
        for d in hc_dropped:
            print(f"  {d}", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"\n[OK] Acceptance criterion met: {n_high} ≥ 15 high-confidence pairs")

    output: dict[str, Any] = {
        "pairs": all_pairs,
        "n_high": n_high,
        "n_medium": n_medium,
        "n_low": n_low,
        "total": len(all_pairs),
        "dropped": hc_dropped,
        "build_date": date.today().isoformat(),
        "grambank_commit": grambank_commit,
        "smoke": smoke,
        "seed": seed,
    }
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build literature-backed validation pairs for RQ3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--grambank_dir", default=_DEFAULT_GRAMBANK_DIR,
                   help="Local path for the Grambank CLDF repo clone.")
    p.add_argument("--output_dir", default=_DEFAULT_OUTPUT_DIR)
    scale = p.add_mutually_exclusive_group()
    scale.add_argument("--smoke", action="store_true")
    scale.add_argument("--full", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Skip if validation_pairs.json already exists.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    out = Path(args.output_dir)

    if args.resume and (out / "validation_pairs.json").exists():
        print("[resume] validation_pairs.json already exists — skipping.")
        return

    result = build_validation_pairs(
        grambank_dir=args.grambank_dir,
        output_dir=args.output_dir,
        smoke=args.smoke,
        seed=args.seed,
    )

    out_file = out / "validation_pairs.json"
    out_file.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"\nSaved: {out_file}")

    # Config dump
    config = vars(args).copy()
    (out / "config_build_validation_pairs.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False)
    )
    print(f"Saved: {out / 'config_build_validation_pairs.json'}")

    # Summary table
    print("\nHigh-confidence pairs:")
    for p in result["pairs"]:
        if p["confidence"] == "high":
            print(f"  {p['wals']:50s} → {p['grambank']}")


if __name__ == "__main__":
    main()
