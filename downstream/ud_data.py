"""
downstream/ud_data.py
=====================
Download UD v2.14 treebanks for a target set of languages (identified
by Glottocode) and provide lightweight POS-tagging datasets.

Design decisions
----------------
1. Downloads only the treebanks for our 59 target languages — avoids
   fetching the full 2 GB UD release.
2. Uses the GitHub raw content API to fetch individual .conllu files
   from the UniversalDependencies organisation.
3. For each language, selects the treebank with the largest training set.
4. Exposes a simple Dataset API (word sequences + UPOS tag sequences).

Usage
-----
    python downstream/ud_data.py \\
        --out_dir downstream/ud_data \\
        --languages eng,fra,deu,jpn,...  (ISO 639-3 codes)
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Glottocode ↔ ISO 639-3 mapping and UD treebank catalogue
# ---------------------------------------------------------------------------

# Mapping: ISO 639-3 → UD treebank names (all known treebanks per language,
# ordered by approximate training size descending).
# Source: UD v2.14 release (2024).
# Each entry is (iso, glottocode, [treebank_name, ...])
# treebank_name is the part after "UD_" in the GitHub repo name.
UD_TREEBANKS: List[Tuple[str, str, str, List[str]]] = [
    # (language_name, iso, glottocode, [treebank_ids])
    ("Arabic",        "arb", "stan1318", ["Arabic-PADT", "Arabic-PUD"]),
    ("Armenian",      "hye", "nucl1235", ["Armenian-ArmTDP", "Armenian-BSUT"]),
    ("Bambara",       "bam", "bamb1269", ["Bambara-CRB"]),
    ("Beja",          "bej", "beja1238", ["Beja-NSC"]),
    ("Belarusian",    "bel", "bela1254", ["Belarusian-HSE"]),
    ("Bhojpuri",      "bho", "bhoj1244", ["Bhojpuri-BHTB"]),
    ("Breton",        "bre", "bret1244", ["Breton-KEB"]),
    ("Cantonese",     "yue", "yuec1235", ["Cantonese-HK"]),
    ("Catalan",       "cat", "stan1289", ["Catalan-AnCora"]),
    ("Chinese",       "cmn", "mand1415", ["Chinese-GSDSimp", "Chinese-GSD",
                                           "Chinese-PUD", "Chinese-CFL",
                                           "Chinese-HK", "Chinese-PatentChar"]),
    ("Chukchi",       "ckt", "chuk1273", ["Chukchi-HSE"]),
    ("Coptic",        "cop", "copt1239", ["Coptic-Scriptorium"]),
    ("Czech",         "ces", "czec1258", ["Czech-PDT", "Czech-FicTree",
                                           "Czech-CAC", "Czech-PUD", "Czech-CLTT"]),
    ("Danish",        "dan", "dani1285", ["Danish-DDT"]),
    ("Dutch",         "nld", "dutc1256", ["Dutch-Alpino", "Dutch-LassySmall"]),
    ("English",       "eng", "stan1293", ["English-EWT", "English-GUM",
                                           "English-LinES", "English-PUD",
                                           "English-ParTUT", "English-Atis",
                                           "English-GUMReddit"]),
    ("Evenki",        "evn", "even1259", ["Evenki-PSU"]),
    ("Faroese",       "fao", "faro1244", ["Faroese-FarPaHC", "Faroese-OFT"]),
    ("Finnish",       "fin", "finn1318", ["Finnish-TDT", "Finnish-FTB",
                                           "Finnish-PUD", "Finnish-OOD"]),
    ("French",        "fra", "stan1290", ["French-GSD", "French-Sequoia",
                                           "French-ParTUT", "French-PUD",
                                           "French-FQB", "French-Rhapsodie"]),
    ("Frisian",       "fry", "west2354", ["Frisian_Dutch-Fame"]),
    ("Greek",         "ell", "mode1248", ["Greek-GDT"]),
    ("Guarani",       "gug", "para1311", ["Guarani-OldTuDeT"]),
    ("Hebrew",        "heb", "hebr1245", ["Hebrew-HTB", "Hebrew-IAHLTwiki"]),
    ("Hindi",         "hin", "hind1269", ["Hindi-HDTB", "Hindi-PUD"]),
    ("Hungarian",     "hun", "hung1274", ["Hungarian-Szeged"]),
    ("Icelandic",     "isl", "icel1247", ["Icelandic-IcePaHC", "Icelandic-Modern",
                                           "Icelandic-PUD"]),
    ("Irish",         "gle", "iris1253", ["Irish-IDT", "Irish-TwittIrish"]),
    ("Italian",       "ita", "ital1282", ["Italian-ISDT", "Italian-VIT",
                                           "Italian-PUD", "Italian-ParTUT",
                                           "Italian-PoSTWITA", "Italian-TWITTIRO",
                                           "Italian-MARKIT", "Italian-Old"]),
    ("Japanese",      "jpn", "nucl1643", ["Japanese-GSD", "Japanese-BCCWJ",
                                           "Japanese-PUD", "Japanese-GSDLUW"]),
    ("Komi",          "kpv", "komi1268", ["Komi_Zyrian-IKDP", "Komi_Zyrian-Lattice"]),
    ("Korean",        "kor", "kore1280", ["Korean-GSD", "Korean-Kaist",
                                           "Korean-PUD"]),
    ("Latvian",       "lav", "latv1249", ["Latvian-LVTB"]),
    ("Lithuanian",    "lit", "lith1251", ["Lithuanian-ALKSNIS", "Lithuanian-HSE"]),
    ("Maltese",       "mlt", "malt1254", ["Maltese-MUDT"]),
    ("Marathi",       "mar", "mara1378", ["Marathi-UFAL"]),
    ("Munduruku",     "myu", "mund1330", ["Munduruku-TuDeT"]),
    ("North Sami",    "sme", "nort2671", ["North_Sami-Giella"]),
    ("Olo",           "ong", "oloo1241", ["Olo-PUD"]),
    ("Persian",       "pes", "west2369", ["Persian-PerDT", "Persian-Seraji"]),
    ("Polish",        "pol", "poli1260", ["Polish-PDB", "Polish-LFG",
                                           "Polish-PUD"]),
    ("Portuguese",    "por", "port1283", ["Portuguese-Bosque", "Portuguese-GSD",
                                           "Portuguese-PUD", "Portuguese-CINTIL",
                                           "Portuguese-PORTTINARI"]),
    ("Russian",       "rus", "russ1263", ["Russian-SynTagRus", "Russian-GSD",
                                           "Russian-PUD", "Russian-Taiga"]),
    ("Slovenian",     "slv", "slov1268", ["Slovenian-SSJ", "Slovenian-SST"]),
    ("Somali",        "som", "soma1255", ["Somali-SUD"]),
    ("Swahili",       "swh", "swah1253", ["Swahili-Jumanjieng"]),
    ("Swedish",       "swe", "swed1254", ["Swedish-Talbanken", "Swedish-LinES",
                                           "Swedish-PUD"]),
    ("Tagalog",       "tgl", "taga1270", ["Tagalog-TRG", "Tagalog-Ugnayan"]),
    ("Tamil",         "tam", "tami1289", ["Tamil-TTB", "Tamil-MWTT", "Tamil-PUD"]),
    ("Telugu",        "tel", "telu1262", ["Telugu-MTG"]),
    ("Thai",          "tha", "thai1261", ["Thai-PUD", "Thai-ORCHID"]),
    ("Turkish",       "tur", "nucl1301", ["Turkish-BOUN", "Turkish-Kenet",
                                           "Turkish-Penn", "Turkish-PUD",
                                           "Turkish-Tourism", "Turkish-FrameNet",
                                           "Turkish-BOUNTREEBANK"]),
    ("Ukrainian",     "ukr", "ukra1253", ["Ukrainian-IU"]),
    ("Vietnamese",    "vie", "viet1252", ["Vietnamese-VTB"]),
    ("Warlpiri",      "wbp", "warl1254", ["Warlpiri-UFAL"]),
    ("Welsh",         "cym", "wels1247", ["Welsh-CCG"]),
    ("Wolof",         "wol", "nucl1347", ["Wolof-WTB"]),
    ("Yakut",         "sah", "yaku1245", ["Yakut-YKTDT"]),
    ("Zulu",          "zul", "zulu1248", ["Zulu-NCHLT"]),
]

# Build lookup tables
_ISO_TO_ENTRY = {iso: (name, iso, glot, tbs) for name, iso, glot, tbs in UD_TREEBANKS}
_GLOT_TO_ENTRY = {glot: (name, iso, glot, tbs) for name, iso, glot, tbs in UD_TREEBANKS}
_GLOT_TO_ISO = {glot: iso for _, iso, glot, _ in UD_TREEBANKS}
_ISO_TO_GLOT = {iso: glot for _, iso, glot, _ in UD_TREEBANKS}

GLOTTOCODES_IN_UD = list(_GLOT_TO_ENTRY.keys())
UD_BASE_URL = ("https://raw.githubusercontent.com/UniversalDependencies/"
               "{treebank}/master/{iso_short}_{corpus}-ud-{split}.conllu")

# ---------------------------------------------------------------------------
# CoNLL-U parser (minimal, no external dependency)
# ---------------------------------------------------------------------------

def parse_conllu(text: str) -> List[Tuple[List[str], List[str]]]:
    """
    Parse CoNLL-U text into (word_list, upos_list) pairs.

    Filters out:
    - Multi-word tokens (ID contains '-')
    - Empty nodes (ID contains '.')
    - Punctuation tokens (UPOS == 'PUNCT') — kept by default
    """
    sentences = []
    words, tags = [], []

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        if not line:
            if words:
                sentences.append((words, tags))
                words, tags = [], []
            continue
        parts = line.split("\t")
        if len(parts) < 10:
            continue
        tok_id = parts[0]
        if "-" in tok_id or "." in tok_id:
            continue  # skip MWTs and empty nodes
        word = parts[1]
        upos = parts[3]
        if upos == "_":
            continue
        words.append(word)
        tags.append(upos)

    if words:
        sentences.append((words, tags))
    return sentences


def fetch_conllu(treebank: str, iso_short: str, corpus: str,
                  split: str = "train",
                  retries: int = 3, sleep: float = 1.5) -> Optional[str]:
    """
    Fetch a .conllu file from GitHub raw content.

    Parameters
    ----------
    treebank    e.g. "UD_English-EWT"
    iso_short   e.g. "en"  (UD uses ISO 639-1 or 2-letter codes for file names)
    corpus      e.g. "ewt"
    split       "train", "dev", or "test"
    """
    url = UD_BASE_URL.format(
        treebank=treebank,
        iso_short=iso_short,
        corpus=corpus.lower(),
        split=split,
    )
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "GrambankVec/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.read().decode("utf-8", errors="replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(sleep * (attempt + 1))
        except Exception:
            time.sleep(sleep * (attempt + 1))
    return None


# ---------------------------------------------------------------------------
# ISO 639-3 → UD file prefix (2-letter or 3-letter)
# ---------------------------------------------------------------------------

# UD uses language codes in filenames; these often differ from ISO 639-3.
# The mapping below is derived from the UD treebank README files.
# format: iso639-3 → ud_file_prefix  (used in conllu file names)
_ISO3_TO_UD_FILE_PREFIX = {
    "arb": "ar",   "hye": "hy",   "bam": "bm",   "bej": "bej",
    "bel": "be",   "bho": "bho",  "bre": "br",   "yue": "yue",
    "cat": "ca",   "cmn": "zh",   "ckt": "ckt",  "cop": "cop",
    "ces": "cs",   "dan": "da",   "nld": "nl",   "eng": "en",
    "evn": "evn",  "fao": "fo",   "fin": "fi",   "fra": "fr",
    "fry": "qfn",  "ell": "el",   "gug": "gun",  "heb": "he",
    "hin": "hi",   "hun": "hu",   "isl": "is",   "gle": "ga",
    "ita": "it",   "jpn": "ja",   "kpv": "kpv",  "kor": "ko",
    "lav": "lv",   "lit": "lt",   "mlt": "mt",   "mar": "mr",
    "myu": "myu",  "sme": "sme",  "ong": "ong",  "pes": "fa",
    "pol": "pl",   "por": "pt",   "rus": "ru",   "slv": "sl",
    "som": "so",   "swh": "swl",  "swe": "sv",   "tgl": "tl",
    "tam": "ta",   "tel": "te",   "tha": "th",   "tur": "tr",
    "ukr": "uk",   "vie": "vi",   "wbp": "wbp",  "cym": "cy",
    "wol": "wo",   "sah": "sah",  "zul": "zu",
}


def _corpus_from_treebank(treebank: str) -> str:
    """Extract corpus name from 'UD_Language-Corpus' string."""
    return treebank.split("-", 1)[1]


def download_treebank(
    language_entry: Tuple,
    out_dir: str,
    split: str = "train",
    overwrite: bool = False,
) -> Optional[str]:
    """
    Download the best (largest) treebank for a given language.

    Tries treebanks in the order given (largest first by convention)
    until one is found.  Returns the local path to the downloaded file,
    or None on failure.
    """
    name, iso, glot, treebanks = language_entry
    os.makedirs(out_dir, exist_ok=True)

    iso_prefix = _ISO3_TO_UD_FILE_PREFIX.get(iso, iso[:2])

    for tb in treebanks:
        corpus = _corpus_from_treebank(tb)
        corpus_lower = corpus.lower()
        local_path = os.path.join(out_dir, f"{iso}_{corpus_lower}-{split}.conllu")

        if os.path.exists(local_path) and not overwrite:
            return local_path

        print(f"    Trying {tb} ({split})...", end=" ", flush=True)
        # fetch_conllu expects the full GitHub repo name (with 'UD_' prefix)
        tb_full = f"UD_{tb}" if not tb.startswith("UD_") else tb
        content = fetch_conllu(tb_full, iso_prefix, corpus_lower, split)
        if content is None:
            # Some treebanks use merged corpus names (e.g. ParTUT → partut)
            content = fetch_conllu(tb_full, iso_prefix, corpus_lower.replace("_", ""), split)
        if content is None:
            print("not found")
            continue

        sents = parse_conllu(content)
        if len(sents) < 5:
            print(f"too small ({len(sents)} sentences)")
            continue

        with open(local_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"OK ({len(sents)} sentences)")
        return local_path

    return None


def load_pos_dataset(
    conllu_path: str,
    max_sentences: Optional[int] = None,
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Load a .conllu file and return (word_seqs, tag_seqs).

    Parameters
    ----------
    max_sentences   if set, truncate to this many sentences (for speed)
    """
    with open(conllu_path, "r", encoding="utf-8") as f:
        text = f.read()
    sents = parse_conllu(text)
    if max_sentences:
        sents = sents[:max_sentences]
    word_seqs = [w for w, t in sents]
    tag_seqs  = [t for w, t in sents]
    return word_seqs, tag_seqs


# ---------------------------------------------------------------------------
# Download pipeline
# ---------------------------------------------------------------------------

def download_all(
    glottocodes: List[str],
    out_dir: str,
    splits: List[str] = ("train", "test"),
    overwrite: bool = False,
) -> Dict[str, Dict[str, str]]:
    """
    Download UD treebanks for all requested glottocodes.

    Returns
    -------
    manifest : {glottocode -> {split -> local_path}}
    """
    manifest: Dict[str, Dict[str, str]] = {}
    missing = []

    for glot in glottocodes:
        entry = _GLOT_TO_ENTRY.get(glot)
        if entry is None:
            print(f"  WARNING: no UD treebank entry for glottocode {glot!r}")
            missing.append(glot)
            continue

        name = entry[0]
        print(f"\n  [{name} / {glot}]")
        manifest[glot] = {}

        for split in splits:
            path = download_treebank(entry, out_dir, split=split,
                                      overwrite=overwrite)
            if path:
                manifest[glot][split] = path
            else:
                print(f"    WARNING: could not download {split} split for {name}")

    if missing:
        print(f"\nMissing entries for {len(missing)} glottocodes: {missing}")

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Download UD treebanks for typological transfer experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir", required=True,
                   help="Directory to store downloaded .conllu files")
    p.add_argument("--glottocodes", default=None,
                   help="Comma-separated glottocodes to download "
                        "(default: all 59 UD-compatible languages)")
    p.add_argument("--splits", default="train,test",
                   help="Comma-separated splits to download")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--smoke", action="store_true",
                   help="Download only 3 languages as a smoke test")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    if args.glottocodes:
        glottocodes = [g.strip() for g in args.glottocodes.split(",")]
    elif args.smoke:
        glottocodes = ["stan1293", "nucl1301", "tami1289"]  # English, Turkish, Tamil
    else:
        glottocodes = GLOTTOCODES_IN_UD

    splits = [s.strip() for s in args.splits.split(",")]
    print(f"Downloading UD treebanks for {len(glottocodes)} languages "
          f"({', '.join(splits)} splits)")
    print(f"Output directory: {args.out_dir}")

    manifest = download_all(glottocodes, args.out_dir, splits=splits,
                             overwrite=args.overwrite)

    # Save manifest
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest saved to: {manifest_path}")

    n_train = sum(1 for v in manifest.values() if "train" in v)
    n_test  = sum(1 for v in manifest.values() if "test" in v)
    print(f"Downloaded: {n_train} train splits, {n_test} test splits")
    print(f"Languages with both train+test: "
          f"{sum(1 for v in manifest.values() if 'train' in v and 'test' in v)}")


if __name__ == "__main__":
    main()
