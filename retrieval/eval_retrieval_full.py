"""
eval_retrieval_full.py  —  10-method retrieval, corrected gold, paper-matched methodology
=========================================================================================

Fixes applied relative to previous version:
  1. Gold set: 74-pair v2 (two logical inversions removed: 39A=No-incl/excl→GB028=0,
               118A=Nonverbal→GB068=0).
  2. Observed methods: skip WALS values with n_shared_observed < MIN_OBS (= 10).
  3. Family-prevalence baseline: uses WALS Genus column restricted to genera with
               >= 5 training-set languages (84 qualifying genera), matching the paper's
               81-branch taxonomy. Isolates and small families are excluded.
  4. T-CF method: uses Grambank T-CF model (not Learned) for the GB side of the
               predicted-profile cosine, giving a matched T-CF × T-CF comparison.
"""
from __future__ import annotations
import json, re, time, warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.metrics import matthews_corrcoef

warnings.filterwarnings("ignore", category=RuntimeWarning)

_REPO   = Path(__file__).resolve().parent.parent
_CKPTS  = _REPO / "checkpoints"
# Overridden at runtime by --wals_dir / --grambank_dir / --uriel_parquet
_CLDF_W = Path("/path/to/wals/cldf")
_CLDF_G = Path("/path/to/grambank/cldf")
_URIEL  = _REPO / "analysis/conditioning_uriel_plus" / "uriel_plus_vectors_familymean.parquet"
SEEDS   = [42, 43, 44, 45, 46]
MIN_OBS = 10   # minimum co-attested languages for observed statistics

# Pairs evaluable by the Learned model (most restrictive — sets the shared n=70 denominator).
# All methods use this same set so every number in the table has an identical denominator.
_LEARNED_SKIP = {"90A=Correlative", "41A=Four-way contrast",
                 "44A=3rd person non-singular only", "59A=More than five classes"}


# ── 0. Data loading ───────────────────────────────────────────────────────────

def load_cldf():
    wl  = pd.read_csv(_CLDF_W / "languages.csv")
    gl  = pd.read_csv(_CLDF_G / "languages.csv")
    wv  = pd.read_csv(_CLDF_W / "values.csv")
    gv  = pd.read_csv(_CLDF_G / "values.csv")
    shared = set(wl["Glottocode"].dropna()) & set(gl["Glottocode"].dropna())
    all_gc = sorted(shared)
    gc_ord = {gc: i for i, gc in enumerate(all_gc)}
    w_gc2lid: dict[str, list[str]] = {}
    for _, r in wl.iterrows():
        gc = r["Glottocode"]
        if pd.notna(gc) and gc in shared:
            w_gc2lid.setdefault(gc, []).append(r["ID"])
    g_gc2lid = gl[gl["Glottocode"].isin(shared)].set_index("Glottocode")["ID"].to_dict()
    g_lid2gc = gl[gl["Glottocode"].isin(shared)].set_index("ID")["Glottocode"].to_dict()
    wals_idx = wv.set_index(["Language_ID","Parameter_ID"])["Value"].to_dict()
    # Fix 3: build qualifying-genus taxonomy (WALS Genus col, >=5 training-set langs)
    with open(_CKPTS / "wals_learned_s42" / "lang2id.json") as f:
        training_l2id = json.load(f)
    wals_model = wl[wl["Glottocode"].isin(training_l2id.keys())]
    genus_sizes   = wals_model.groupby("Genus").size()
    qualifying_genera = set(genus_sizes[genus_sizes >= 5].index)
    gc2genus = wl[wl["Glottocode"].isin(shared)].drop_duplicates("Glottocode")\
                  .set_index("Glottocode")["Genus"].to_dict()
    return dict(shared_gc=shared, all_gc=all_gc, gc_ord=gc_ord,
                w_gc2lid=w_gc2lid, g_gc2lid=g_gc2lid, g_lid2gc=g_lid2gc,
                wals_idx=wals_idx, gb_vals=gv,
                qualifying_genera=qualifying_genera, gc2genus=gc2genus)


def build_gb_matrix(gv, all_gc, gc_ord, all_gb_forms, g_lid2gc):
    form_ord = {f: i for i, f in enumerate(all_gb_forms)}
    G = np.full((len(all_gc), len(all_gb_forms)), np.nan, dtype=np.float32)
    sub = gv[gv["Language_ID"].isin(g_lid2gc)].copy()
    sub["gc"] = sub["Language_ID"].map(g_lid2gc)
    sub = sub[sub["gc"].isin(gc_ord)]
    for _, row in sub.iterrows():
        gc  = row["gc"]
        val = str(row["Value"]).strip()
        if val == "?": continue
        gi   = gc_ord[gc]
        feat = row["Parameter_ID"]
        for v in ("0","1","2","3"):
            fi = form_ord.get(f"{feat}={v}")
            if fi is not None and np.isnan(G[gi, fi]):
                G[gi, fi] = 1.0 if val == v else 0.0
    return G


def build_wals_matrix(all_gc, gc_ord, w_gc2lid, wals_idx, gold_pairs):
    unique  = list({(p["wals_feat"], p["wals_val_num"]) for p in gold_pairs})
    col_ord = {fv: i for i, fv in enumerate(unique)}
    W = np.full((len(all_gc), len(unique)), np.nan, dtype=np.float32)
    for (feat, val_num), ci in col_ord.items():
        for gc, lids in w_gc2lid.items():
            gi = gc_ord[gc]
            for lid in lids:
                v = wals_idx.get((lid, feat))
                if v is not None:
                    W[gi, ci] = 1.0 if int(v) == val_num else 0.0
                    break
    return W, col_ord


def load_ckpt(db, arch, seed):
    d = _CKPTS / f"{db}_{arch}_s{seed}"
    if arch == "learned":
        fv_emb = np.load(d/"featvalue_embeddings.npy")
        fv2id  = json.loads((d/"featvalue2id.json").read_text())
    else:
        fv_emb = np.load(d/"binarycol_embeddings.npy")
        fv2id  = json.loads((d/"binarycol2id.json").read_text())
    return dict(fv_emb=fv_emb, fv2id=fv2id,
                feat2vals=json.loads((d/"feat2values.json").read_text()),
                lang_emb=np.load(d/"lang_embeddings.npy"),
                lang2id=json.loads((d/"lang2id.json").read_text()))


def load_gold(path):
    pairs = []
    n_skipped = 0
    for p in json.loads(Path(path).read_text())["pairs"]:
        wals_fv = f"{p['wals_feature_id']}={p['wals_value_label']}"
        if wals_fv in _LEARNED_SKIP:
            n_skipped += 1
            continue
        pairs.append(dict(
            wals_fv=wals_fv,
            wals_feat=p["wals_feature_id"], wals_val_num=p["wals_value_number"],
            wals_val_label=p["wals_value_label"], gb_formula=p["gb_formula"],
            relation_type=p["relation_type"], scope_caveat=p["scope_caveat"]))
    if n_skipped:
        print(f"  Restricted to shared n=70 intersection "
              f"(dropped {n_skipped} pairs not in Learned vocab: {_LEARNED_SKIP})")
    return pairs


def load_uriel(all_gc):
    df = pd.read_parquet(_URIEL).set_index("glottocode")
    dim = len(df.iloc[0]["phylo_vec"])
    U   = np.zeros((len(all_gc), dim), dtype=np.float32)
    prs = np.zeros(len(all_gc), dtype=bool)
    gc_ord = {gc: i for i, gc in enumerate(all_gc)}
    for gc in all_gc:
        if gc in df.index:
            pv = df.loc[gc, "phylo_vec"]
            if isinstance(pv, np.ndarray) and len(pv) == dim:
                U[gc_ord[gc]] = pv; prs[gc_ord[gc]] = True
    print(f"  URIEL+: {prs.sum()}/{len(all_gc)} shared GCs with phylo vectors")
    return U, prs


# ── 1. Vectorised observed statistics ─────────────────────────────────────────

def score_all_observed(w_col, G, min_n=MIN_OBS):
    """
    Returns (cos, phi, jac, ppmi, lor) each of shape (n_gb,).
    Fix 2: pairs where n < min_n are assigned sentinel -2.0 / -99.0.
    """
    n_gc, n_gb = G.shape
    w_obs = ~np.isnan(w_col)
    g_obs = ~np.isnan(G)
    both  = w_obs[:, None] & g_obs
    w     = np.where(w_obs, w_col, 0.0)
    Gf    = np.where(g_obs, G, 0.0)
    n     = both.sum(axis=0).astype(float)
    n11   = (w[:, None] * Gf * both).sum(axis=0)
    n10   = (w[:, None] * (1-Gf) * both).sum(axis=0)
    n01   = ((1-w[:, None]) * Gf * both).sum(axis=0)
    n00   = ((1-w[:, None]) * (1-Gf) * both).sum(axis=0)
    n_pos_w = n11 + n10; n_pos_g = n11 + n01
    # cosine
    dot_wg = (w[:, None]*Gf*both).sum(axis=0)
    nw = np.sqrt((w[:,None]**2*both).sum(axis=0)); ng = np.sqrt((Gf**2*both).sum(axis=0))
    cos = np.where((nw>1e-12)&(ng>1e-12), dot_wg/(nw*ng), 0.0)
    cos = np.where(n>=min_n, cos, -2.0)
    # phi
    dp = np.sqrt(n_pos_w*(n-n_pos_w)*n_pos_g*(n-n_pos_g))
    phi = np.where(dp>1e-12, (n11*n00-n10*n01)/dp, 0.0)
    phi = np.where(n>=min_n, phi, -2.0)
    # jaccard
    union = n11+n10+n01
    jac = np.where(union>0, n11/union, 0.0)
    jac = np.where(n>=min_n, jac, -2.0)
    # ppmi
    p_x = np.where(n>0, n_pos_w/n, 0.0); p_y = np.where(n>0, n_pos_g/n, 0.0)
    p_xy = np.where(n>0, n11/n, 0.0)
    safe = (p_x>1e-12)&(p_y>1e-12)&(p_xy>1e-12)
    ppmi = np.maximum(np.where(safe, np.log(np.where(safe, p_xy/(p_x*p_y+1e-30), 1.0)), 0.0), 0.0)
    ppmi = np.where(n>=min_n, ppmi, -2.0)
    # log-odds
    h11=n11+.5; h10=n10+.5; h01=n01+.5; h00=n00+.5
    lor = np.log(h11*h00/(h10*h01))
    lor = np.where(n>=min_n, lor, -99.0)
    return cos, phi, jac, ppmi, lor


def score_family_cos(w_col, G, gc2genus, qualifying_genera, all_gc):
    """
    Fix 3: family-prevalence cosine using WALS Genus taxonomy,
    restricted to genera with >= 5 training-set languages.
    Languages in non-qualifying genera are excluded.
    """
    # Assign each GC to a branch (or None if non-qualifying)
    gc_branch = [gc2genus.get(gc,"") if gc2genus.get(gc,"") in qualifying_genera
                 else None for gc in all_gc]
    branches = sorted(qualifying_genera)
    b_ord = {b: i for i, b in enumerate(branches)}
    n_b   = len(branches)
    n_gb  = G.shape[1]
    # WALS prevalence per branch
    wp_pos = np.zeros(n_b); wp_tot = np.zeros(n_b)
    for i, (obs, val, br) in enumerate(zip(~np.isnan(w_col), np.where(~np.isnan(w_col), w_col, 0.0), gc_branch)):
        if obs and br is not None:
            bi = b_ord[br]; wp_tot[bi] += 1; wp_pos[bi] += val
    wf = np.where(wp_tot>0, wp_pos/wp_tot, 0.0)   # (n_b,)
    # GB prevalence per branch per formula
    g_obs = ~np.isnan(G); Gf = np.where(g_obs, G, 0.0)
    gp_pos = np.zeros((n_b, n_gb)); gp_tot = np.zeros((n_b, n_gb))
    for i, br in enumerate(gc_branch):
        if br is None: continue
        bi = b_ord[br]
        gp_tot[bi] += g_obs[i].astype(float)
        gp_pos[bi] += np.where(g_obs[i], Gf[i], 0.0)
    gf = np.where(gp_tot>0, gp_pos/gp_tot, 0.0)   # (n_b, n_gb)
    both_b = (wp_tot>0)[:,None] & (gp_tot>0)
    wv = np.where(both_b, wf[:,None], 0.0)
    gv = np.where(both_b, gf,         0.0)
    wn = np.sqrt((wv**2).sum(axis=0)); gn = np.sqrt((gv**2).sum(axis=0))
    dot = (wv*gv).sum(axis=0)
    fc = np.where((wn>1e-12)&(gn>1e-12), dot/(wn*gn), 0.0)
    return np.where(both_b.sum(axis=0)>=5, fc, -2.0)


def score_uriel_centroid(w_col, G, U, U_present):
    w_mask = (~np.isnan(w_col)) & U_present & (w_col>0.5)
    if w_mask.sum()<3: return np.full(G.shape[1], -2.0)
    c_w = U[w_mask].mean(axis=0); cn_w = np.linalg.norm(c_w)
    if cn_w<1e-12: return np.full(G.shape[1], 0.0)
    c_w_u = c_w/cn_w
    n_gb = G.shape[1]; scores = np.full(n_gb, -2.0)
    G_obs = ~np.isnan(G); Gf2 = np.where(G_obs, G, 0.0)
    for j in range(n_gb):
        g_mask = G_obs[:,j] & U_present & (Gf2[:,j]>0.5)
        if g_mask.sum()<3: continue
        c_g = U[g_mask].mean(axis=0); cn_g = np.linalg.norm(c_g)
        scores[j] = float(np.dot(c_w_u, c_g/cn_g)) if cn_g>1e-12 else 0.0
    return scores


# ── 2. Predicted-profile helpers ──────────────────────────────────────────────

def build_all_pred_profiles(ckpt, lang_idx, model):
    feat2vals=ckpt["feat2vals"]; fv2id=ckpt["fv2id"]
    fv_emb=ckpt["fv_emb"]; L=ckpt["lang_emb"][lang_idx]
    profiles={}
    if model=="learned":
        for feat,vals in feat2vals.items():
            rows=[fv2id.get(f"{feat}={v}") for v in vals]
            if any(r is None for r in rows): continue
            V=fv_emb[rows]; sc=L@V.T
            sc-=sc.max(axis=1,keepdims=True)
            probs=np.exp(sc); probs/=probs.sum(axis=1,keepdims=True)
            for k,v in enumerate([str(vv) for vv in vals]):
                profiles[f"{feat}={v}"]=probs[:,k]
    else:   # T-CF sigmoid
        for feat,vals in feat2vals.items():
            for v in vals:
                key=f"{feat}={v}"; row=fv2id.get(key) or fv2id.get(feat)
                if row is None: continue
                profiles[key]=1.0/(1.0+np.exp(-(L@fv_emb[row])))
    return profiles


def pred_profile_cos_vec(wp, w_gcs, gb_P, g_gcs):
    common=sorted(set(w_gcs)&set(g_gcs))
    if len(common)<5: return np.full(gb_P.shape[1],-2.0)
    wi={gc:i for i,gc in enumerate(w_gcs)}; gi={gc:i for i,gc in enumerate(g_gcs)}
    wv=wp[[wi[gc] for gc in common]][:,None]
    gv=gb_P[[gi[gc] for gc in common],:]
    wn=np.linalg.norm(wv); gn=np.linalg.norm(gv,axis=0)
    if wn<1e-12: return np.zeros(gb_P.shape[1])
    return np.where(gn>1e-12, (wv*gv).sum(axis=0)/(wn*gn), 0.0)


# ── 3. Ranking helpers ─────────────────────────────────────────────────────────

def rank_of_vec(score_vec, formula, form_ord):
    fi = form_ord.get(formula)
    if fi is None: return None
    return int(1 + (score_vec > score_vec[fi]).sum())


def mrr_recall(ranks, cutoffs=(1,3,5,10)):
    n=len(ranks)
    if n==0: return {}
    rrs=[1.0/r if r else 0.0 for r in ranks]
    out={"n":n,"mrr":round(float(np.mean(rrs)),4)}
    for k in cutoffs: out[f"r@{k}"]=round(sum(1 for r in ranks if r and r<=k)/n,4)
    valid=[r for r in ranks if r]; out["med"]=float(np.median(valid)) if valid else None
    return out


# ── 4. Main evaluation ─────────────────────────────────────────────────────────

def run(crosswalk_path, out_dir):
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    t0=time.time()

    print("Loading CLDF…"); cldf=load_cldf()
    pairs=load_gold(crosswalk_path)
    all_gc=cldf["all_gc"]; gc_ord=cldf["gc_ord"]
    print(f"  {len(pairs)} pairs — shared n=70 intersection (all methods identical denominator)")

    # GB formula universe
    gb_codes=pd.read_csv(_CLDF_G/"codes.csv")
    all_gb_forms=sorted(set(
        f"{r['Parameter_ID']}={r['Name']}"
        for _,r in gb_codes.iterrows()
        if str(r["Name"]).strip() in ("0","1","2","3"))
        | {p["gb_formula"] for p in pairs})
    n_gb=len(all_gb_forms); form_ord={f:i for i,f in enumerate(all_gb_forms)}
    print(f"  {n_gb} GB formulas")

    print("Loading URIEL+…"); U,U_prs=load_uriel(all_gc)

    print("Loading model checkpoints…")
    ckpts_tcf_w     =[load_ckpt("wals",    "tcf",    s) for s in SEEDS]
    ckpts_learned_w =[load_ckpt("wals",    "learned",s) for s in SEEDS]
    ckpts_learned_g =[load_ckpt("grambank","learned",s) for s in SEEDS]
    # Fix 4: use grambank_tcf for the GB side of T-CF comparison
    ckpts_tcf_g     =[load_ckpt("grambank","tcf",    s) for s in SEEDS]
    ckpts_proc_w    =[load_ckpt("wals",    "learned",s) for s in SEEDS]
    ckpts_proc_g    =[load_ckpt("grambank","learned",s) for s in SEEDS]

    # ── Pre-filter: restrict to pairs evaluable by ALL 10 methods ───────────
    # Excludes WALS values absent from T-CF or Learned vocab (< 10 training
    # instances), so every method uses the same denominator n.
    tcf_fv2id_s42 = ckpts_tcf_w[0]["fv2id"]
    lea_fv2id_s42 = ckpts_learned_w[0]["fv2id"]

    def in_tcf_vocab(feat, label):
        return f"{feat}={label}" in tcf_fv2id_s42 or feat in tcf_fv2id_s42

    def in_lea_vocab(feat, label):
        return f"{feat}={label}" in lea_fv2id_s42

    pairs_ok  = [p for p in pairs
                 if in_tcf_vocab(p["wals_feat"],p["wals_val_label"])
                 and in_lea_vocab(p["wals_feat"],p["wals_val_label"])]
    pairs_drop = [p for p in pairs if p not in pairs_ok]
    print(f"  Pre-filter: {len(pairs_drop)} pairs dropped (absent from ≥1 model vocab):")
    for p in pairs_drop:
        reasons=[]
        if not in_tcf_vocab(p["wals_feat"],p["wals_val_label"]): reasons.append("no_tcf")
        if not in_lea_vocab(p["wals_feat"],p["wals_val_label"]): reasons.append("no_lea")
        print(f"    {p['wals_fv'][:50]:52s} ({','.join(reasons)})")
    pairs = pairs_ok
    print(f"  All 10 methods will use n={len(pairs)}")

    print("Building matrices…"); t1=time.time()
    G=build_gb_matrix(cldf["gb_vals"],all_gc,gc_ord,all_gb_forms,cldf["g_lid2gc"])
    W,w_col_ord=build_wals_matrix(all_gc,gc_ord,cldf["w_gc2lid"],cldf["wals_idx"],pairs)
    print(f"  done ({time.time()-t1:.1f}s)")

    print("Pre-computing GB predicted profiles (5 seeds × Learned)…"); t1=time.time()
    gb_pred_lea=[]
    for g_ckpt in ckpts_learned_g:
        g_idx=np.array([g_ckpt["lang2id"][gc] for gc in all_gc if gc in g_ckpt["lang2id"]])
        g_gcs=[gc for gc in all_gc if gc in g_ckpt["lang2id"]]
        profs=build_all_pred_profiles(g_ckpt,g_idx,"learned")
        P=np.zeros((len(g_gcs),n_gb),dtype=np.float32)
        for fi,f in enumerate(all_gb_forms):
            atoms=[a.strip() for a in f.split("&")]
            gp=None
            for a in atoms:
                ap=profs.get(a)
                if ap is None: gp=None; break
                gp=ap if gp is None else gp*ap
            if gp is not None: P[:,fi]=gp
        gb_pred_lea.append((P,g_gcs))
    print(f"  done ({time.time()-t1:.1f}s)")

    # Fix 4: GB T-CF predicted profiles for T-CF method
    print("Pre-computing GB predicted profiles (5 seeds × T-CF)…"); t1=time.time()
    gb_pred_tcf=[]
    for g_ckpt in ckpts_tcf_g:
        g_idx=np.array([g_ckpt["lang2id"][gc] for gc in all_gc if gc in g_ckpt["lang2id"]])
        g_gcs=[gc for gc in all_gc if gc in g_ckpt["lang2id"]]
        profs=build_all_pred_profiles(g_ckpt,g_idx,"tcf")
        P=np.zeros((len(g_gcs),n_gb),dtype=np.float32)
        for fi,f in enumerate(all_gb_forms):
            atoms=[a.strip() for a in f.split("&")]
            gp=None
            for a in atoms:
                ap=profs.get(a)
                if ap is None: gp=None; break
                gp=ap if gp is None else gp*ap
            if gp is not None: P[:,fi]=gp
        gb_pred_tcf.append((P,g_gcs))
    print(f"  done ({time.time()-t1:.1f}s)")

    print("Pre-computing Procrustes (5 seeds)…"); t1=time.time()
    proc_data=[]
    for pw,pg in zip(ckpts_proc_w,ckpts_proc_g):
        shared_p=set(pw["lang2id"])&set(pg["lang2id"])
        L_W=pw["lang_emb"][[pw["lang2id"][gc] for gc in shared_p]]
        L_G=pg["lang_emb"][[pg["lang2id"][gc] for gc in shared_p]]
        R,_=orthogonal_procrustes(L_G,L_W)
        def fvmean(formula,fv_emb,fv2id):
            vecs=[]
            for a in formula.split("&"):
                k=a.strip(); idx=fv2id.get(k)
                if idx is None and k.endswith("=1"): idx=fv2id.get(k[:-2])
                if idx is None: return None
                vecs.append(fv_emb[idx])
            return np.mean(np.stack(vecs),axis=0)
        gb_fvs={f:fvmean(f,pg["fv_emb"],pg["fv2id"]) for f in all_gb_forms}
        proc_data.append((R,pw,gb_fvs))
    print(f"  done ({time.time()-t1:.1f}s)")

    print("Evaluating pairs…"); t1=time.time()
    method_keys=["obs_cos","obs_phi","obs_jaccard","family_cos","uriel_centroid",
                 "ppmi","log_odds","procrustes_A","emb_tcf","emb_learned"]
    methods={m:[] for m in method_keys}
    per_pair=[]

    for i,pair in enumerate(pairs):
        if (i+1)%20==0: print(f"  pair {i+1}/{len(pairs)}  ({time.time()-t1:.0f}s)")
        wf=pair["wals_fv"]; wfeat=pair["wals_feat"]
        wnum=pair["wals_val_num"]; wlbl=pair["wals_val_label"]
        formula=pair["gb_formula"]
        row={"wals_fv":wf,"gb_formula":formula,
             "relation_type":pair["relation_type"],"scope_caveat":pair["scope_caveat"]}
        ci=w_col_ord.get((wfeat,wnum))
        if ci is None: per_pair.append(row); continue
        w_col=W[:,ci]

        # Fix 2: check if this WALS value has enough co-attested observations
        # (done implicitly via MIN_OBS sentinel in score_all_observed)

        cos_v,phi_v,jac_v,ppmi_v,lor_v=score_all_observed(w_col,G)
        row["rank_cos"]    =rank_of_vec(cos_v,  formula,form_ord)
        row["rank_phi"]    =rank_of_vec(phi_v,  formula,form_ord)
        row["rank_jaccard"]=rank_of_vec(jac_v,  formula,form_ord)
        row["rank_ppmi"]   =rank_of_vec(ppmi_v, formula,form_ord)
        row["rank_logodds"]=rank_of_vec(lor_v,  formula,form_ord)
        row["phi_val"]     =float(phi_v[form_ord[formula]]) if formula in form_ord else None

        for m,v in [("obs_cos","rank_cos"),("obs_phi","rank_phi"),
                    ("obs_jaccard","rank_jaccard"),("ppmi","rank_ppmi"),
                    ("log_odds","rank_logodds")]:
            methods[m].append(row.get(v))

        # Fix 3: family-prevalence with qualifying-genus taxonomy
        fam_v=score_family_cos(w_col,G,cldf["gc2genus"],cldf["qualifying_genera"],all_gc)
        row["rank_family"]=rank_of_vec(fam_v,formula,form_ord)
        methods["family_cos"].append(row["rank_family"])

        uri_v=score_uriel_centroid(w_col,G,U,U_prs)
        row["rank_uriel"]=rank_of_vec(uri_v,formula,form_ord)
        methods["uriel_centroid"].append(row["rank_uriel"])

        # Procrustes A (5-seed mean)
        proc_ranks=[]
        for R,pw,gb_fvs in proc_data:
            k=wf; idx=pw["fv2id"].get(k)
            if idx is None and "=1" in k: idx=pw["fv2id"].get(k.rsplit("=",1)[0])
            if idx is None: continue
            wrot=pw["fv_emb"][idx]@R; wn=np.linalg.norm(wrot)
            if wn<1e-12: continue
            wu=wrot/wn
            psc=np.array([float(np.dot(wu,gb_fvs[f]/np.linalg.norm(gb_fvs[f])))
                          if gb_fvs[f] is not None and np.linalg.norm(gb_fvs[f])>1e-12
                          else -2.0 for f in all_gb_forms])
            r2=rank_of_vec(psc,formula,form_ord)
            if r2: proc_ranks.append(r2)
        rp=int(round(np.mean(proc_ranks))) if proc_ranks else None
        row["rank_proc"]=rp; methods["procrustes_A"].append(rp)

        # Fix 4: T-CF WALS × T-CF Grambank predicted profiles
        tcf_ranks=[]
        for si,wt in enumerate(ckpts_tcf_w):
            key=f"{wfeat}={wlbl}"; ridx=wt["fv2id"].get(key) or wt["fv2id"].get(wfeat)
            if ridx is None: continue
            w_idx=np.array([wt["lang2id"][gc] for gc in all_gc if gc in wt["lang2id"]])
            w_gcs=[gc for gc in all_gc if gc in wt["lang2id"]]
            wp=1.0/(1.0+np.exp(-(wt["lang_emb"][w_idx]@wt["fv_emb"][ridx])))
            P,g_gcs=gb_pred_tcf[si]
            sc=pred_profile_cos_vec(wp,w_gcs,P,g_gcs)
            r2=rank_of_vec(sc,formula,form_ord)
            if r2: tcf_ranks.append(r2)
        rt=int(round(np.mean(tcf_ranks))) if tcf_ranks else None
        row["rank_tcf"]=rt; methods["emb_tcf"].append(rt)

        # Learned WALS × Learned Grambank predicted profiles
        lea_ranks=[]
        for si,wl_ckpt in enumerate(ckpts_learned_w):
            vals=wl_ckpt["feat2vals"].get(wfeat)
            if not vals: continue
            vals_s=[str(v) for v in vals]
            if wlbl not in vals_s: continue
            tk=vals_s.index(wlbl)
            rows_k=[wl_ckpt["fv2id"].get(f"{wfeat}={v}") for v in vals]
            if any(r is None for r in rows_k): continue
            w_idx=np.array([wl_ckpt["lang2id"][gc] for gc in all_gc if gc in wl_ckpt["lang2id"]])
            w_gcs=[gc for gc in all_gc if gc in wl_ckpt["lang2id"]]
            V=wl_ckpt["fv_emb"][rows_k]
            sc2=wl_ckpt["lang_emb"][w_idx]@V.T
            sc2-=sc2.max(axis=1,keepdims=True)
            probs=np.exp(sc2); probs/=probs.sum(axis=1,keepdims=True)
            wp=probs[:,tk]
            P,g_gcs=gb_pred_lea[si]
            sc=pred_profile_cos_vec(wp,w_gcs,P,g_gcs)
            r2=rank_of_vec(sc,formula,form_ord)
            if r2: lea_ranks.append(r2)
        rl=int(round(np.mean(lea_ranks))) if lea_ranks else None
        row["rank_learned"]=rl; methods["emb_learned"].append(rl)

        per_pair.append(row)

    print(f"  Done in {time.time()-t1:.1f}s")

    # ── aggregate + print ─────────────────────────────────────────────────────
    metrics={m:mrr_recall(ranks) for m,ranks in methods.items()}
    LABELS={"obs_cos":"Observed cos","obs_phi":"Observed ϕ",
            "obs_jaccard":"Observed Jaccard","family_cos":"Family-prevalence cos",
            "uriel_centroid":"URIEL+ centroid","ppmi":"PPMI","log_odds":"Log-odds",
            "procrustes_A":"Procrustes A","emb_tcf":"T-CF (5-seed)",
            "emb_learned":"Learned (5-seed)"}

    # Count valid n per method
    n_valid={m:sum(1 for r in ranks if r is not None) for m,ranks in methods.items()}

    print(f"\n{'='*76}")
    print(f"  RETRIEVAL — {len(pairs)} Tier-A pairs (v2) — {n_gb} GB formulas")
    print(f"  Seeds: {SEEDS}  |  MIN_OBS={MIN_OBS}  |  Family: WALS Genus ≥5")
    print(f"{'='*76}")
    print(f"  {'Method':24s}   @1    @3    @5   @10    MRR   med    n")
    print(f"  {'─'*70}")
    for m,lbl in LABELS.items():
        mm=metrics[m]; med=mm.get("med")
        print(f"  {lbl:24s}  "
              f"{mm.get('r@1',0):.3f}  {mm.get('r@3',0):.3f}  "
              f"{mm.get('r@5',0):.3f}  {mm.get('r@10',0):.3f}  "
              f"{mm.get('mrr',0):.4f}  {int(med) if med else '—':>5}  {n_valid[m]:>3}")
    print(f"{'='*76}")
    print(f"\n  Paper Table 3/11 (74-pair v2) for comparison:")
    for lbl,vals in [("Observed cos",".514 .736 .764 .613 n=72"),
                     ("Observed ϕ",  ".556 .722 .778 .641 n=72"),
                     ("Observed Jac",".597 .750 .750 .657 n=72"),
                     ("Family-prev.", ".194 .375 .444 .294 n=72"),
                     ("URIEL+ centrd",".264 .472 .528 .368 n=72"),
                     ("T-CF",        ".034 .099 .169 .075 n=71"),
                     ("Learned",     ".097 .283 .343 .180 n=70")]:
        print(f"    {lbl:20s}: {vals}")

    # domain breakdown
    _AREA={str(n):a for r,a in [(range(1,15),"Phonology"),(range(15,36),"Morphology"),
        (range(36,56),"Nominal Cat"),(range(56,66),"Nominal Syn"),
        (range(66,81),"Verbal Cat"),(range(81,98),"Word Order"),
        (range(98,113),"Simple Cls"),(range(113,145),"Lexicon")] for n in r}
    def dom(wf):
        m=re.match(r"^(\d+)",wf); return _AREA.get(m.group(1),"Other") if m else "Other"
    dg=defaultdict(list)
    for r in per_pair: dg[dom(r["wals_fv"])].append(r)
    print(f"\n  @10 by domain:")
    COLS=["rank_cos","rank_phi","rank_jaccard","rank_family","rank_uriel",
          "rank_ppmi","rank_logodds","rank_proc","rank_tcf","rank_learned"]
    print(f"  {'Domain':12s}  n  cos  ϕ    Jac  Fam  URI  PPMI LOR  Proc TCF  Lea")
    print(f"  {'─'*80}")
    for d in sorted(dg):
        rows=dg[d]; n=len(rows)
        def r10(col):
            rks=[r.get(col) for r in rows if r.get(col) is not None]
            return f"{sum(1 for r in rks if r<=10)/n:.2f}" if rks else "— "
        print(f"  {d:12s}  {n:2d}  "+"  ".join(r10(c) for c in COLS))
    print(f"\n  Total wall time: {time.time()-t0:.0f}s")

    out={"n_pairs":len(pairs),"n_gb_formulas":n_gb,"seeds":SEEDS,
         "min_obs":MIN_OBS,"family_taxonomy":"WALS Genus >= 5 training-set langs",
         "fixes":["logical_inversions_removed","min_obs_threshold",
                  "family_genus_taxonomy","tcf_uses_grambank_tcf"],
         "metrics":metrics,"per_pair":per_pair}
    (out_dir/"retrieval_full.json").write_text(json.dumps(out,indent=2))
    pd.DataFrame(per_pair).to_csv(out_dir/"retrieval_full_per_pair.csv",index=False)
    print(f"\nSaved → {out_dir}/retrieval_full.json")
    return out


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(
        description="10-method WALS–Grambank retrieval evaluation (Table 1).")
    p.add_argument("--wals_dir",   required=True,
                   help="Path to wals/cldf/ directory (WALS CLDF v2020.4).")
    p.add_argument("--grambank_dir", required=True,
                   help="Path to grambank/cldf/ directory (Grambank CLDF v1.0).")
    p.add_argument("--crosswalk",
                   default=str(_REPO/"data/crosswalk_gold.json"),
                   help="Path to crosswalk_gold.json.")
    p.add_argument("--uriel_parquet",
                   default=str(_URIEL),
                   help="Path to URIEL+ parquet file.")
    p.add_argument("--out_dir",
                   default=str(_REPO/"analysis/crosswalk_eval"))
    args=p.parse_args()
    # Patch module-level paths from CLI before load_cldf() runs
    import eval_retrieval_full as _self
    _self._CLDF_W = Path(args.wals_dir)
    _self._CLDF_G = Path(args.grambank_dir)
    _self._URIEL  = Path(args.uriel_parquet)
    run(args.crosswalk, args.out_dir)
