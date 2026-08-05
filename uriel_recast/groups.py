"""39 categorical groups over 98 of the 289 original-URIEL feature flags.

Provenance
----------
The original URIEL feature block (``original_uriel/features.npz``, 289 flags) stores
mutually exclusive typological values as independent binary flags. ``S_SVO``,
``S_SOV``, ``S_VSO``, ``S_VOS``, ``S_OVS`` and ``S_OSV`` are six separate columns
even though a language's dominant order in WALS 81A is a single categorical choice.
Nothing in the representation prevents a profile with ``S_SVO = S_SOV = 1``.

Each group below bundles flags that are alternative values of one comparative
concept in the source database. The ``wals`` field records the WALS chapter whose
value set the group is read against, where one exists; ``None`` marks a group whose
exclusivity follows from the feature names themselves (e.g. ``S_SUBJECT_BEFORE_OBJECT``
vs ``S_SUBJECT_AFTER_OBJECT``) rather than from a numbered WALS chapter.

``kind`` is ``exclusive`` for 38 groups: at most one member may be active, and the
"no member active" case is a legitimate state (the concept does not apply, or the
language has no dominant value). ``vowel_quality_inventory`` is ``ordinal``: its
eight members are ordered points on the vowel-inventory-size scale carried in
``scale``, which the Phase 5 ordinal probe correlates against embedding geometry.

Caveats carried from the experiment design, all standing:

  * This mapping is one reading of URIEL feature names against WALS chapter titles.
    ``alignment``, ``object_marking_locus`` and ``tendency_affix`` are the shakiest.
    WALS 98A/99A carry neutral and tripartite alignment values that URIEL's two
    flags cannot express, so a "neither flag active" profile conflates "neutral
    alignment" with "unknown". Locus-of-marking (23A-25A) is arguably multi-label
    in the source, since double-marking is an attested value.
  * ``basic_order``, ``adposition``, ``adjective``, ``possessor`` and
    ``vowel_quality_inventory`` are safe: the source concepts are genuinely
    single-valued and the flag sets are complete.
"""

GROUPS = {
    # ---- Word order and constituent order -------------------------------------
    "basic_order": {
        "kind": "exclusive",
        "wals": "81A",
        "feats": ["S_SVO", "S_SOV", "S_VSO", "S_VOS", "S_OVS", "S_OSV"],
    },
    "subj_verb": {
        "kind": "exclusive",
        "wals": "82A",
        "feats": ["S_SUBJECT_BEFORE_VERB", "S_SUBJECT_AFTER_VERB"],
    },
    "obj_verb": {
        "kind": "exclusive",
        "wals": "83A",
        "feats": ["S_OBJECT_BEFORE_VERB", "S_OBJECT_AFTER_VERB"],
    },
    "subj_obj": {
        "kind": "exclusive",
        "wals": None,
        "feats": ["S_SUBJECT_BEFORE_OBJECT", "S_SUBJECT_AFTER_OBJECT"],
    },
    "oblique_order": {
        "kind": "exclusive",
        "wals": "84A",
        "feats": ["S_VOX", "S_XVO", "S_XOV", "S_OXV", "S_OVX"],
    },
    "oblique_v": {
        "kind": "exclusive",
        "wals": None,
        "feats": ["S_OBLIQUE_BEFORE_VERB", "S_OBLIQUE_AFTER_VERB"],
    },
    "oblique_o": {
        "kind": "exclusive",
        "wals": None,
        "feats": ["S_OBLIQUE_BEFORE_OBJECT", "S_OBLIQUE_AFTER_OBJECT"],
    },
    "adposition": {
        "kind": "exclusive",
        "wals": "85A",
        "feats": ["S_ADPOSITION_BEFORE_NOUN", "S_ADPOSITION_AFTER_NOUN"],
    },
    "possessor": {
        "kind": "exclusive",
        "wals": "86A",
        "feats": ["S_POSSESSOR_BEFORE_NOUN", "S_POSSESSOR_AFTER_NOUN"],
    },
    "adjective": {
        "kind": "exclusive",
        "wals": "87A",
        "feats": ["S_ADJECTIVE_BEFORE_NOUN", "S_ADJECTIVE_AFTER_NOUN"],
    },
    "demonstrative_word": {
        "kind": "exclusive",
        "wals": "88A",
        "feats": ["S_DEMONSTRATIVE_WORD_BEFORE_NOUN", "S_DEMONSTRATIVE_WORD_AFTER_NOUN"],
    },
    "numeral": {
        "kind": "exclusive",
        "wals": "89A",
        "feats": ["S_NUMERAL_BEFORE_NOUN", "S_NUMERAL_AFTER_NOUN"],
    },
    "relative": {
        "kind": "exclusive",
        "wals": "90A",
        "feats": ["S_RELATIVE_BEFORE_NOUN", "S_RELATIVE_AFTER_NOUN", "S_RELATIVE_AROUND_NOUN"],
    },
    "degree_word": {
        "kind": "exclusive",
        "wals": "91A",
        "feats": ["S_DEGREE_WORD_BEFORE_ADJECTIVE", "S_DEGREE_WORD_AFTER_ADJECTIVE"],
    },
    "article_word": {
        "kind": "exclusive",
        "wals": None,
        "feats": ["S_ARTICLE_WORD_BEFORE_NOUN", "S_ARTICLE_WORD_AFTER_NOUN"],
    },
    "subordinator": {
        "kind": "exclusive",
        "wals": "94A",
        "feats": [
            "S_SUBORDINATOR_WORD_BEFORE_CLAUSE",
            "S_SUBORDINATOR_WORD_AFTER_CLAUSE",
            "S_SUBORDINATOR_SUFFIX",
        ],
    },
    "complementizer": {
        "kind": "exclusive",
        "wals": None,
        "feats": ["S_COMPLEMENTIZER_WORD_BEFORE_CLAUSE", "S_COMPLEMENTIZER_WORD_AFTER_CLAUSE"],
    },
    "polarq_pos": {
        "kind": "exclusive",
        "wals": "92A",
        "feats": ["S_POLARQ_MARK_INITIAL", "S_POLARQ_MARK_FINAL", "S_POLARQ_MARK_SECOND"],
    },
    # ---- Exponence choices -----------------------------------------------------
    "possessive_affix": {
        "kind": "exclusive",
        "wals": None,
        "feats": ["S_POSSESSIVE_PREFIX", "S_POSSESSIVE_SUFFIX"],
    },
    "demonstrative_affix": {
        "kind": "exclusive",
        "wals": None,
        "feats": ["S_DEMONSTRATIVE_PREFIX", "S_DEMONSTRATIVE_SUFFIX"],
    },
    "case_affix": {
        "kind": "exclusive",
        "wals": "51A",
        "feats": ["S_CASE_PREFIX", "S_CASE_SUFFIX", "S_CASE_PROCLITIC", "S_CASE_ENCLITIC"],
    },
    "plural_exponence": {
        "kind": "exclusive",
        "wals": "33A/34A",
        "feats": ["S_PLURAL_PREFIX", "S_PLURAL_SUFFIX", "S_PLURAL_WORD"],
    },
    "tam_affix": {
        "kind": "exclusive",
        "wals": "69A",
        "feats": ["S_TAM_PREFIX", "S_TAM_SUFFIX"],
    },
    "definite_exponence": {
        "kind": "exclusive",
        "wals": "37A",
        "feats": ["S_DEFINITE_AFFIX", "S_DEFINITE_WORD"],
    },
    "indefinite_exponence": {
        "kind": "exclusive",
        "wals": "38A",
        "feats": ["S_INDEFINITE_AFFIX", "S_INDEFINITE_WORD"],
    },
    "prosubject_exponence": {
        "kind": "exclusive",
        "wals": "101A",
        "feats": ["S_PROSUBJECT_WORD", "S_PROSUBJECT_AFFIX", "S_PROSUBJECT_CLITIC"],
    },
    # ---- Negation --------------------------------------------------------------
    "negation_exponence": {
        "kind": "exclusive",
        "wals": "112A",
        "feats": ["S_NEGATIVE_AFFIX", "S_NEGATIVE_WORD"],
    },
    "negation_affix_pos": {
        "kind": "exclusive",
        "wals": None,
        "feats": ["S_NEGATIVE_PREFIX", "S_NEGATIVE_SUFFIX"],
    },
    "negword_verb_pos": {
        "kind": "exclusive",
        "wals": "143A",
        "feats": ["S_NEGATIVE_WORD_BEFORE_VERB", "S_NEGATIVE_WORD_AFTER_VERB"],
    },
    "negword_subj_pos": {
        "kind": "exclusive",
        "wals": None,
        "feats": ["S_NEGATIVE_WORD_BEFORE_SUBJECT", "S_NEGATIVE_WORD_AFTER_SUBJECT"],
    },
    "negword_obj_pos": {
        "kind": "exclusive",
        "wals": None,
        "feats": ["S_NEGATIVE_WORD_BEFORE_OBJECT", "S_NEGATIVE_WORD_AFTER_OBJECT"],
    },
    "negword_clause_pos": {
        "kind": "exclusive",
        "wals": None,
        "feats": ["S_NEGATIVE_WORD_INITIAL", "S_NEGATIVE_WORD_FINAL"],
    },
    "negword_adjacency": {
        "kind": "exclusive",
        "wals": None,
        "feats": [
            "S_NEGATIVE_WORD_ADJACENT_BEFORE_VERB",
            "S_NEGATIVE_WORD_ADJACENT_AFTER_VERB",
        ],
    },
    # ---- Alignment and locus of marking ---------------------------------------
    "alignment": {
        "kind": "exclusive",
        "wals": "98A/99A",
        "feats": ["S_NOMINATIVE_VS_ACCUSATIVE_MARK", "S_ERGATIVE_VS_ABSOLUTIVE_MARK"],
    },
    "object_marking_locus": {
        "kind": "exclusive",
        "wals": "23A",
        "feats": ["S_OBJECT_HEADMARK", "S_OBJECT_DEPMARK"],
    },
    "possessive_marking_locus": {
        "kind": "exclusive",
        "wals": "24A",
        "feats": ["S_POSSESSIVE_HEADMARK", "S_POSSESSIVE_DEPMARK"],
    },
    "tendency_marking_locus": {
        "kind": "exclusive",
        "wals": "25A",
        "feats": ["S_TEND_HEADMARK", "S_TEND_DEPMARK"],
    },
    "tendency_affix": {
        "kind": "exclusive",
        "wals": "26A",
        "feats": ["S_TEND_PREFIX", "S_TEND_SUFFIX"],
    },
    # ---- Ordinal (the only one) ------------------------------------------------
    "vowel_quality_inventory": {
        "kind": "ordinal",
        "wals": "2A",
        "scale": [3, 4, 5, 6, 7, 8, 9, 10],
        "feats": [
            "INV_VOWEL_3",
            "INV_VOWEL_4",
            "INV_VOWEL_5",
            "INV_VOWEL_6",
            "INV_VOWEL_7",
            "INV_VOW_8",
            "INV_VOW_9",
            "INV_VOW_10_MORE",
        ],
    },
}

# Flat list of the 98 grouped feature names, in group order.
GROUPED_FEATS = [f for g in GROUPS.values() for f in g["feats"]]


def resolve(feats):
    """Map group definitions onto column indices of ``feats`` (the 289 names).

    Returns ``(group_cols, grouped_idx, ungrouped_idx)`` where ``group_cols`` is an
    ordered dict of group name -> list of column indices. Raises ``KeyError`` if any
    of the 98 names fails to resolve, and ``ValueError`` if a column is claimed by
    more than one group.
    """
    pos = {name: i for i, name in enumerate(feats)}
    missing = [f for f in GROUPED_FEATS if f not in pos]
    if missing:
        raise KeyError(f"{len(missing)} feature name(s) not in feats: {missing}")

    group_cols = {}
    seen = {}
    for gname, g in GROUPS.items():
        cols = [pos[f] for f in g["feats"]]
        for c in cols:
            if c in seen:
                raise ValueError(f"column {feats[c]} claimed by {seen[c]} and {gname}")
            seen[c] = gname
        group_cols[gname] = cols

    grouped_idx = sorted(seen)
    ungrouped_idx = [i for i in range(len(feats)) if i not in seen]
    return group_cols, grouped_idx, ungrouped_idx
