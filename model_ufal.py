#!/usr/bin/env python3
"""
PyTorch reimplementation of UFAL 2020 neural system + ablation upgrades.

UFALNeural     : cosine similarity → Dense(1) → sigmoid  (binary BCE)
CategoricalNeural    : cosine similarity → softmax over K values  (CE)
CategoricalGeomNeural: shared value geometry → softmax over K values (CE)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


# ===========================================================================
# Model 1: UFALNeural
# ===========================================================================

class UFALNeural(nn.Module):
    """
    PyTorch reimplementation of UFAL 2020 neural system.
    Architecture: cos_similarity(lang_emb, fv_emb) → Dense(1) → sigmoid

    Key distinction: feature value IDs are GLOBAL across all features
    (including metadata encoded as feature values). This matches UFAL's
    global_feature_id scheme in dataset.py.
    """

    def __init__(self, n_languages, n_global_fv_ids, embed_dim=64, dropout_rate=0.5):
        super().__init__()
        self.lang_embeddings = nn.Embedding(n_languages, embed_dim)
        self.fv_embeddings   = nn.Embedding(n_global_fv_ids, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(1, 1)  # w + b on top of cosine similarity
        nn.init.normal_(self.lang_embeddings.weight, std=0.01)
        nn.init.normal_(self.fv_embeddings.weight,   std=0.01)
        nn.init.normal_(self.output_layer.weight,    std=0.01)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, lang_ids, fv_ids):
        """
        lang_ids : LongTensor (B,)
        fv_ids   : LongTensor (B,)
        Returns  : sigmoid probs (B,)
        """
        le = F.normalize(self.dropout(self.lang_embeddings(lang_ids)), dim=-1)  # (B, d)
        fe = F.normalize(self.fv_embeddings(fv_ids), dim=-1)                    # (B, d)
        cos_sim = (le * fe).sum(dim=-1, keepdim=True)                           # (B, 1)
        return torch.sigmoid(self.output_layer(cos_sim)).squeeze(-1)            # (B,)

    def predict_feature_probs(self, lang_id_tensor, fv_ids_for_feature, device):
        """
        Given a language embedding index, return sigmoid probabilities for all
        K values of a feature.
        fv_ids_for_feature: list of global fv IDs for the K values of this feature.
        Returns: np.array (K,) of sigmoid probabilities.
        """
        li = lang_id_tensor.expand(len(fv_ids_for_feature))  # (K,)
        fi = torch.tensor(fv_ids_for_feature, dtype=torch.long, device=device)
        with torch.no_grad():
            return self(li, fi).cpu().numpy()

    def predict_feature_from_emb(self, raw_lang_emb, fv_ids_for_feature, device):
        """
        Predict using a RAW embedding vector (1, d) instead of a table index.
        Used for transductive fine-tuning where we have a learned new_emb.
        """
        fi  = torch.tensor(fv_ids_for_feature, dtype=torch.long, device=device)
        fe  = F.normalize(self.fv_embeddings(fi), dim=-1)              # (K, d)
        le  = F.normalize(raw_lang_emb.expand(len(fi), -1), dim=-1)    # (K, d)
        cos = (le * fe).sum(dim=-1, keepdim=True)                      # (K, 1)
        with torch.no_grad():
            probs = torch.sigmoid(self.output_layer(cos)).squeeze(-1)
        return probs.cpu().numpy()


# ===========================================================================
# Model 2: CategoricalNeural
# ===========================================================================

class CategoricalNeural(nn.Module):
    """
    Ablation step 1: UFAL architecture with softmax over K mutually exclusive
    values per feature, replacing binary sigmoid prediction.

    Same global fv_id embedding table as UFALNeural.
    Same cosine similarity backbone.
    Change: for each feature, score K values via cos_sim → softmax (not sigmoid).
    """

    def __init__(self, n_languages, n_global_fv_ids, feat_to_fv_ids,
                 embed_dim=64, dropout_rate=0.5):
        """
        feat_to_fv_ids: dict feat_idx -> list of global fv IDs (length K_f)
        """
        super().__init__()
        self.feat_to_fv_ids = feat_to_fv_ids
        self.lang_embeddings = nn.Embedding(n_languages, embed_dim)
        self.fv_embeddings   = nn.Embedding(n_global_fv_ids, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.normal_(self.lang_embeddings.weight, std=0.01)
        nn.init.normal_(self.fv_embeddings.weight,   std=0.01)

    def logits_for_feature(self, lang_emb_normed, feat_idx, device):
        """
        lang_emb_normed: (1, d) normalized embedding
        Returns: (K_f,) cosine similarity scores (raw logits for softmax)
        """
        fv_ids = self.feat_to_fv_ids[feat_idx]
        fi_t   = torch.tensor(fv_ids, dtype=torch.long, device=device)
        fe     = F.normalize(self.fv_embeddings(fi_t), dim=-1)  # (K, d)
        return (lang_emb_normed @ fe.T).squeeze(0)              # (K,)

    def forward(self, lang_ids, feat_indices, val_local_indices, device):
        """Training forward: CE over K values per (lang, feat) cell."""
        total_nll = torch.tensor(0.0, device=device)
        acc_embs, acc_langs = [], []
        for i in range(len(lang_ids)):
            le_raw = self.dropout(self.lang_embeddings(lang_ids[i:i+1]))
            le     = F.normalize(le_raw, dim=-1)
            logits = self.logits_for_feature(le, feat_indices[i], device)
            total_nll -= F.log_softmax(logits, dim=0)[val_local_indices[i]]
            acc_embs.append(self.fv_embeddings(
                torch.tensor(self.feat_to_fv_ids[feat_indices[i]],
                             dtype=torch.long, device=device)))
            acc_langs.append(le_raw)
        loss = total_nll / max(len(lang_ids), 1)
        all_acc = torch.cat(acc_embs + acc_langs, dim=0)
        return loss, all_acc.pow(2).mean()  # (loss, l2)

    def predict_from_emb(self, raw_lang_emb, feat_idx, device, temperature=1.0):
        """Predict from raw (1, d) embedding."""
        le = F.normalize(raw_lang_emb, dim=-1)
        logits = self.logits_for_feature(le, feat_idx, device)
        lp = F.log_softmax(logits / temperature, dim=0)
        return lp.argmax().item(), lp.exp().cpu().numpy()


# ===========================================================================
# Model 3: CategoricalGeomNeural
# ===========================================================================

class CategoricalGeomNeural(nn.Module):
    """
    Ablation step 2: Categorical prediction with SHARED DENSE VALUE GEOMETRY.

    Key change from CategoricalNeural:
    - fv_embeddings indexed by GLOBAL VALUE ID (feat_to_global_ids, same as
      SetEncoder) rather than feature:value string ID (feat_to_fv_ids).
    - Values of DIFFERENT features share the same embedding space.
    - SOV and OVS can be geometrically close even though they belong to
      different features — this is the thesis contribution.

    Same cosine similarity backbone. Same softmax prediction.
    """

    def __init__(self, n_languages, n_total_values, feat_to_global_ids,
                 embed_dim=64, dropout_rate=0.5):
        super().__init__()
        self.feat_to_global_ids = feat_to_global_ids
        self.lang_embeddings = nn.Embedding(n_languages, embed_dim)
        self.val_embeddings  = nn.Embedding(n_total_values, embed_dim)
        self.dropout = nn.Dropout(dropout_rate)
        nn.init.normal_(self.lang_embeddings.weight, std=0.01)
        nn.init.normal_(self.val_embeddings.weight,  std=0.01)

    def logits_for_feature(self, lang_emb_normed, feat_idx, device):
        gids = self.feat_to_global_ids[feat_idx]
        gi_t = torch.tensor(gids, dtype=torch.long, device=device)
        ve   = F.normalize(self.val_embeddings(gi_t), dim=-1)   # (K, d)
        return (lang_emb_normed @ ve.T).squeeze(0)              # (K,)

    def forward(self, lang_ids, feat_indices, val_local_indices, device):
        total_nll = torch.tensor(0.0, device=device)
        acc_embs, acc_langs = [], []
        for i in range(len(lang_ids)):
            le_raw = self.dropout(self.lang_embeddings(lang_ids[i:i+1]))
            le     = F.normalize(le_raw, dim=-1)
            logits = self.logits_for_feature(le, feat_indices[i], device)
            total_nll -= F.log_softmax(logits, dim=0)[val_local_indices[i]]
            acc_embs.append(self.val_embeddings(
                torch.tensor(self.feat_to_global_ids[feat_indices[i]],
                             dtype=torch.long, device=device)))
            acc_langs.append(le_raw)
        loss = total_nll / max(len(lang_ids), 1)
        all_acc = torch.cat(acc_embs + acc_langs, dim=0)
        return loss, all_acc.pow(2).mean()

    def predict_from_emb(self, raw_lang_emb, feat_idx, device, temperature=1.0):
        le = F.normalize(raw_lang_emb, dim=-1)
        logits = self.logits_for_feature(le, feat_idx, device)
        lp = F.log_softmax(logits / temperature, dim=0)
        return lp.argmax().item(), lp.exp().cpu().numpy()


# ===========================================================================
# Vocabulary helpers (UFAL global fv ID scheme)
# ===========================================================================

def build_ufal_vocab(cat_matrix, kept_feature_names, feat_to_value_names,
                     train_meta_df, kmeans_clusters=300):
    """
    Replicate UFAL's global feature value ID scheme from dataset.py.

    Encodes ALL feature values (WALS + metadata) into a single global ID table.
    Metadata features: genus, family, k-Means geographic cluster.

    Returns:
      feat_to_fv_ids    : dict feat_idx -> list of global fv IDs (for WALS features)
      metadata_fv_maps  : dict column_name -> {value_str: global_fv_id}
      n_global_fv_ids   : int total size of embedding table
      km                : fitted KMeans model
    """
    from sklearn.cluster import KMeans

    global_id = 0
    feat_to_fv_ids = {}

    # WALS feature values
    for fi in range(len(kept_feature_names)):
        ids = []
        for vi in range(len(feat_to_value_names[fi])):
            ids.append(global_id)
            global_id += 1
        feat_to_fv_ids[fi] = ids

    # Metadata: genus, family
    metadata_fv_maps = {}
    for col in ['genus', 'family']:
        unique_vals = sorted(train_meta_df[col].dropna().unique())
        metadata_fv_maps[col] = {v: global_id + i for i, v in enumerate(unique_vals)}
        global_id += len(unique_vals)

    # Geographic clusters (fit k-Means on training lat/lon)
    lats = train_meta_df['latitude'].astype(float).values
    lons = train_meta_df['longitude'].astype(float).values
    coords = np.stack([lats, lons], axis=1)
    km = KMeans(n_clusters=kmeans_clusters, random_state=42, n_init=10)
    km.fit(coords)
    metadata_fv_maps['kmeans'] = km
    metadata_fv_maps['cluster_id_offset'] = global_id
    global_id += kmeans_clusters

    return feat_to_fv_ids, metadata_fv_maps, global_id, km


def build_training_data_ufal(cat_matrix, train_meta_df, feat_to_fv_ids,
                              metadata_fv_maps, km, cluster_id_offset,
                              feat_to_value_names):
    """
    Build (lang_idx, fv_global_id) positive pairs from training data,
    mirroring UFAL's create_dataset().
    Also encodes genus, family, k-Means cluster as additional observed features.
    Returns list of (lang_idx, fv_global_id) positive pairs.
    """
    positives = []
    n_langs, n_feats = cat_matrix.shape
    lats = train_meta_df['latitude'].astype(float).values
    lons = train_meta_df['longitude'].astype(float).values

    for li in range(n_langs):
        # WALS features
        for fi in range(n_feats):
            v = int(cat_matrix[li, fi])
            if v >= 0:
                positives.append((li, feat_to_fv_ids[fi][v]))

        # Metadata features
        genus  = train_meta_df.iloc[li].get('genus', '')
        family = train_meta_df.iloc[li].get('family', '')
        if genus in metadata_fv_maps.get('genus', {}):
            positives.append((li, metadata_fv_maps['genus'][genus]))
        if family in metadata_fv_maps.get('family', {}):
            positives.append((li, metadata_fv_maps['family'][family]))

        # Cluster
        coord = [[lats[li], lons[li]]]
        cluster_local = km.predict(coord)[0]
        positives.append((li, cluster_id_offset + cluster_local))

    return positives


def build_training_data_wals_only(cat_matrix, feat_to_global_ids):
    """
    Build positive (lang_idx, global_val_id) pairs using the WALS-only vocab
    from prepare_categorical (feat_to_global_ids). No metadata features.
    Use with UFALNeural(n_total_values, ...) for the binary+WALS-only
    factorial condition.
    """
    positives = []
    n_langs, n_feats = cat_matrix.shape
    for li in range(n_langs):
        for fi in range(n_feats):
            v = int(cat_matrix[li, fi])
            if v >= 0:
                positives.append((li, feat_to_global_ids[fi][v]))
    return positives


# ===========================================================================
# Training functions
# ===========================================================================

def train_ufal_model(model, positives, n_global_fv_ids, n_epochs=200,
                     batch_size=512, steps_per_epoch=1000,
                     lr=1e-3, l2_reg=0.1, device='cpu', seed=42):
    """
    Train UFALNeural mirroring UFAL's batch_generator:
    50% positive (observed fv_id for that language),
    50% negative (random fv_id NOT in that language's observed set).

    Uses steps_per_epoch (not full epoch) matching UFAL's 1000 steps/epoch.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)
    criterion = nn.BCELoss()

    lang_positives = defaultdict(set)
    for li, fv_id in positives:
        lang_positives[li].add(fv_id)
    pos_array = np.array(positives)
    n_pos = len(pos_array)

    losses = []
    for epoch in range(n_epochs):
        model.train()
        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            idxs = np.random.randint(0, n_pos, size=batch_size)
            batch_langs, batch_fvs, batch_labels = [], [], []

            for idx in idxs:
                li, fv_pos = pos_array[idx]
                if np.random.uniform() < 0.5:
                    batch_langs.append(li)
                    batch_fvs.append(fv_pos)
                    batch_labels.append(1.0)
                else:
                    for _ in range(20):
                        fv_neg = np.random.randint(1, n_global_fv_ids)
                        if fv_neg not in lang_positives[li]:
                            break
                    batch_langs.append(li)
                    batch_fvs.append(fv_neg)
                    batch_labels.append(0.0)

            li_t  = torch.tensor(batch_langs,  dtype=torch.long,  device=device)
            fv_t  = torch.tensor(batch_fvs,    dtype=torch.long,  device=device)
            lab_t = torch.tensor(batch_labels, dtype=torch.float, device=device)

            probs = model(li_t, fv_t)
            loss  = criterion(probs, lab_t)

            le = model.lang_embeddings(li_t)
            fe = model.fv_embeddings(fv_t)
            l2 = torch.cat([le, fe], dim=0).pow(2).mean()
            total = loss + (l2_reg / 2.0) * l2

            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg = epoch_loss / steps_per_epoch
        losses.append(avg)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  loss={avg:.4f}")
    return losses


def train_categorical_model(model, cat_matrix, n_epochs=200, batch_size=256,
                             lr=1e-3, l2_reg=0.1, device='cpu', seed=42):
    """Train CategoricalNeural or CategoricalGeomNeural with CE loss."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    triples = [(li, fi, int(cat_matrix[li, fi]))
               for li in range(cat_matrix.shape[0])
               for fi in range(cat_matrix.shape[1])
               if cat_matrix[li, fi] >= 0]

    losses = []
    for epoch in range(n_epochs):
        model.train()
        perm = np.random.permutation(len(triples))
        epoch_loss = 0.0
        for start in range(0, len(triples), batch_size):
            batch = [triples[perm[i]]
                     for i in range(start, min(start + batch_size, len(triples)))]
            li_t = torch.tensor([t[0] for t in batch], dtype=torch.long, device=device)
            fis  = [t[1] for t in batch]
            vi_t = torch.tensor([t[2] for t in batch], dtype=torch.long, device=device)
            loss, l2 = model(li_t, fis, vi_t, device)
            total = loss + (l2_reg / 2.0) * l2
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            epoch_loss += loss.item()
        avg = epoch_loss / max(1, len(triples) // batch_size)
        losses.append(avg)
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1}/{n_epochs}  loss={avg:.4f}")
    return losses


# ===========================================================================
# Transductive fine-tuning
# ===========================================================================

def transductive_finetune_emb(model, model_type, obs_feat_dict,
                               feat_name_to_idx, feat_to_value_names,
                               feat_to_fv_ids_or_gids, embed_dim,
                               device, n_steps=200, lr=0.05, temperature=1.0):
    """
    Transductive fine-tuning: freeze model, optimize a fresh lang embedding
    against observed features.

    Returns new_emb (1, d) detached.
    NOTE: UFAL does NOT do this for test languages (their test embeddings
    are random). We do this for ALL three models to provide a fair comparison
    where all rows use observed features equally.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    new_emb = nn.Parameter(torch.zeros(1, embed_dim, device=device))
    nn.init.normal_(new_emb, std=0.01)
    opt = torch.optim.Adam([new_emb], lr=lr, weight_decay=0)

    obs = []
    for fname, vstr in obs_feat_dict.items():
        fi = feat_name_to_idx.get(fname)
        if fi is None:
            continue
        vlist = feat_to_value_names.get(fi, [])
        if vstr not in vlist:
            continue
        obs.append((fi, vlist.index(vstr)))

    for step in range(n_steps):
        if not obs:
            break
        total_loss = torch.tensor(0.0, device=device)
        le_normed  = F.normalize(new_emb, dim=-1)

        for fi, vi_local in obs:
            ids  = feat_to_fv_ids_or_gids[fi]
            fi_t = torch.tensor(ids, dtype=torch.long, device=device)

            if model_type == 'ufal':
                fe  = F.normalize(model.fv_embeddings(fi_t), dim=-1)
                cos = (le_normed.expand(len(ids), -1) * fe).sum(-1, keepdim=True)
                probs = torch.sigmoid(model.output_layer(cos)).squeeze(-1)
                targets = torch.zeros(len(ids), device=device)
                targets[vi_local] = 1.0
                total_loss += F.binary_cross_entropy(probs, targets)
            elif model_type == 'categorical':
                fe     = F.normalize(model.fv_embeddings(fi_t), dim=-1)
                logits = (le_normed @ fe.T).squeeze(0)
                total_loss -= F.log_softmax(logits / temperature, dim=0)[vi_local]
            else:  # geom
                ve     = F.normalize(model.val_embeddings(fi_t), dim=-1)
                logits = (le_normed @ ve.T).squeeze(0)
                total_loss -= F.log_softmax(logits / temperature, dim=0)[vi_local]

        loss = total_loss / max(len(obs), 1)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if loss.item() < 0.005:
            break

    for p in model.parameters():
        p.requires_grad_(True)
    return new_emb.detach()
