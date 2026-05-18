"""
train_cached.py — Fast training loop using pre-cached VAE latents
==================================================================
Requires cache_latents.py to have been run first.

Instead of running the 33M-parameter SomaLogic VAE on every batch,
this script loads the pre-computed latents from disk and trains only
the temporal + fusion transformer (~3M parameters).

Expected speedup: ~20-30x vs the original train.py

Usage:
    # Single fold, 10-epoch test
    python3 train_cached.py ./ --fold 0 --epochs 10

    # Single fold, full training
    python3 train_cached.py ./ --fold 0

    # All 5 folds
    python3 train_cached.py ./

    # With permutation test
    python3 train_cached.py ./ --permute
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(__file__))

# ── Config ────────────────────────────────────────────────────────────────────

CFG = {
    "data_dir":       "./",
    "latent_dir":     "outputs/latents/",
    "ckpt_dir":       "outputs/checkpoints/",
    "results_dir":    "outputs/results/",
    "n_folds":        5,
    "epochs":         50,
    "batch_size":     32,
    "lr":             3e-4,
    "weight_decay":   1e-4,
    "dropout":        0.3,
    "n_permutations": 1000,
    "seed":           42,
}

RANDOM_SEED = 42


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Cached Dataset ────────────────────────────────────────────────────────────

class CachedLatentDataset(Dataset):
    """
    Lightweight dataset that serves pre-computed latents.

    Each item is a dict matching exactly what MultimodalTransformer.forward()
    expects — but somalogic/metabolon/etc. are now latent vectors, not raw omics.

    The model's encode_modality() is bypassed; we feed latents directly into
    the temporal encoders via a thin adapter in CachedMultimodalTransformer.
    """

    def __init__(self, patient_ids, soma_cache, met_cache, lip_cache,
                 lum_cache, clin_cache, label_cache,
                 permuted_labels=None):
        """
        Args:
            patient_ids:      list of patient_id strings for this split
            *_cache:          dicts loaded from outputs/latents/*.pt
            permuted_labels:  optional list of ints to override mortality labels
                              (used for permutation test)
        """
        self.patient_ids    = patient_ids
        self.soma_cache     = soma_cache
        self.met_cache      = met_cache
        self.lip_cache      = lip_cache
        self.lum_cache      = lum_cache
        self.clin_cache     = clin_cache
        self.label_cache    = label_cache
        self.permuted_labels = permuted_labels

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        pid  = self.patient_ids[idx]
        info = self.label_cache[pid]

        mortality = (self.permuted_labels[idx]
                     if self.permuted_labels is not None
                     else info["mortality"])

        return {
            # Latent sequences — (3, latent_dim) each
            "somalogic":        self.soma_cache[pid],     # (3, 128)
            "somalogic_mask":   info["somalogic_mask"],   # (3,)
            "metabolon":        self.met_cache[pid],      # (3, 64)
            "metabolon_mask":   info["metabolon_mask"],   # (3,)
            "lipidomics":       self.lip_cache[pid],      # (3, 64)
            "lipidomics_mask":  info["lipidomics_mask"],  # (3,)
            "luminex":          self.lum_cache[pid],      # (3, 32)
            "luminex_mask":     info["luminex_mask"],     # (3,)
            # Clinical latent — (32,)
            "clinical":         self.clin_cache[pid],     # (32,)
            # Labels
            "mortality":        torch.tensor(mortality, dtype=torch.long),
            "patient_id":       pid,
            "is_synthetic":     info["is_synthetic"],
        }


def collate_fn(batch):
    out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([b[key] for b in batch])
        else:
            out[key] = [b[key] for b in batch]
    return out


# ── Cached Model ──────────────────────────────────────────────────────────────
#
# Architecture: mean-pool each modality's latents across timepoints,
# concatenate all modalities, then a 3-layer MLP → mortality logit.
#
# Why simpler than the transformer version:
#   - 195 real patients is too few for deep multi-head attention to converge
#   - Direct gradient path: loss → head → fusion MLP → input projections
#   - LogReg on clinical alone gets 0.90 AUROC; the omics should push it higher
#   - We keep per-modality projection layers so the model learns modality-
#     specific representations before fusing — this is the key inductive bias
#
# Total trainable params: ~180K (vs 2M transformer) — much better ratio for n=195
#
# The full transformer can be restored later with more data or transfer learning.

class CachedMultimodalTransformer(nn.Module):
    """
    Multimodal MLP fusion model operating on pre-cached VAE latents.

    Pipeline per patient:
        somalogic  (3, 128) --mean_pool--> (128,) --Linear--> (64,)  ┐
        metabolon  (3,  64) --mean_pool--> ( 64,) --Linear--> (32,)  │
        lipidomics (3,  64) --mean_pool--> ( 64,) --Linear--> (32,)  ├→ cat → (192,) → MLP → logit
        luminex    (3,  32) --mean_pool--> ( 32,) --Linear--> (32,)  │
        clinical       (32,)              ( 32,) --Linear--> (32,)  ┘

    Masked mean-pool: only averages over present timepoints (mask=1).
    """

    def __init__(self, dropout=0.3):
        super().__init__()

        # Per-modality projection layers (latent → hidden)
        self.soma_proj  = nn.Sequential(
            nn.Linear(128, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout))
        self.met_proj   = nn.Sequential(
            nn.Linear(64,  32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout))
        self.lip_proj   = nn.Sequential(
            nn.Linear(64,  32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout))
        self.lum_proj   = nn.Sequential(
            nn.Linear(32,  32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout))
        self.clin_proj  = nn.Sequential(
            nn.Linear(32,  32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(dropout))

        # Fused dim: 64 + 32 + 32 + 32 + 32 = 192
        fused_dim = 64 + 32 + 32 + 32 + 32

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    @staticmethod
    def masked_mean(x, mask):
        """
        Mean-pool (B, T, D) over T dimension using mask (B, T).
        Falls back to full mean if all timepoints are missing.
        """
        mask_exp = mask.unsqueeze(-1).float()          # (B, T, 1)
        summed   = (x * mask_exp).sum(dim=1)           # (B, D)
        counts   = mask_exp.sum(dim=1).clamp(min=1.0)  # (B, 1)
        return summed / counts                          # (B, D)

    def forward(self, batch):
        # Mean-pool each modality across timepoints
        soma = self.masked_mean(batch["somalogic"],  batch["somalogic_mask"])   # (B,128)
        met  = self.masked_mean(batch["metabolon"],  batch["metabolon_mask"])   # (B, 64)
        lip  = self.masked_mean(batch["lipidomics"], batch["lipidomics_mask"])  # (B, 64)
        lum  = self.masked_mean(batch["luminex"],    batch["luminex_mask"])     # (B, 32)
        clin = batch["clinical"]                                                # (B, 32)

        # Per-modality projections
        soma = self.soma_proj(soma)   # (B, 64)
        met  = self.met_proj(met)     # (B, 32)
        lip  = self.lip_proj(lip)     # (B, 32)
        lum  = self.lum_proj(lum)     # (B, 32)
        clin = self.clin_proj(clin)   # (B, 32)

        # Concatenate and fuse
        fused  = torch.cat([soma, met, lip, lum, clin], dim=-1)  # (B, 192)
        logits = self.fusion(fused).squeeze(-1)                   # (B,)

        return logits, {}   # empty dict keeps interface compatible with train loop

    def predict_proba(self, batch):
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(batch)
            return torch.sigmoid(logits)


# ── Load cached latents ───────────────────────────────────────────────────────

def load_latent_cache(latent_dir):
    """Load all cached latent dicts from disk."""
    print(f"Loading cached latents from {latent_dir} ...")
    soma_cache  = torch.load(os.path.join(latent_dir, "somalogic_latents.pt"),
                             map_location="cpu")
    met_cache   = torch.load(os.path.join(latent_dir, "metabolon_latents.pt"),
                             map_location="cpu")
    lip_cache   = torch.load(os.path.join(latent_dir, "lipidomics_latents.pt"),
                             map_location="cpu")
    lum_cache   = torch.load(os.path.join(latent_dir, "luminex_latents.pt"),
                             map_location="cpu")
    clin_cache  = torch.load(os.path.join(latent_dir, "clinical_latents.pt"),
                             map_location="cpu")
    label_cache = torch.load(os.path.join(latent_dir, "labels.pt"),
                             map_location="cpu")

    # Separate real vs synthetic patient IDs
    real_ids = sorted([
        pid for pid, info in label_cache.items()
        if info["is_synthetic"] == 0
    ])
    syn_ids = sorted([
        pid for pid, info in label_cache.items()
        if info["is_synthetic"] == 1
    ])

    print(f"  Real patients : {len(real_ids)}")
    print(f"  Synthetic     : {len(syn_ids)}")
    print(f"  Total cached  : {len(label_cache)}")

    return (soma_cache, met_cache, lip_cache, lum_cache,
            clin_cache, label_cache, real_ids, syn_ids)


# ── CV split ──────────────────────────────────────────────────────────────────

def get_fold_splits(real_ids, label_cache, fold):
    """
    Replicate the exact same stratified split used in dataset.py.
    Returns train_ids (real+syn), val_ids (real only), test_ids (real only).
    """
    real_labels = np.array([label_cache[pid]["mortality"] for pid in real_ids])
    syn_ids_all = [
        pid for pid, info in label_cache.items() if info["is_synthetic"] == 1
    ]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    folds = list(skf.split(real_ids, real_labels))
    train_val_idx, test_idx = folds[fold]

    inner_skf  = StratifiedKFold(n_splits=5, shuffle=True,
                                  random_state=RANDOM_SEED + 1)
    tv_ids     = [real_ids[i] for i in train_val_idx]
    tv_labels  = real_labels[train_val_idx]
    inner_folds = list(inner_skf.split(tv_ids, tv_labels))
    train_inner_idx, val_inner_idx = inner_folds[fold % 5]

    train_real_ids = [tv_ids[i] for i in train_inner_idx]
    val_ids        = [tv_ids[i] for i in val_inner_idx]
    test_ids       = [real_ids[i] for i in test_idx]

    # Synthetic goes into training only
    train_ids = train_real_ids + syn_ids_all

    return train_ids, val_ids, test_ids


# ── Class weight ──────────────────────────────────────────────────────────────

def get_class_weight(train_ids, label_cache, device):
    labels = [label_cache[pid]["mortality"] for pid in train_ids]
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    weight = n_neg / max(n_pos, 1)
    print(f"  Class weight: {weight:.2f} (pos={n_pos}, neg={n_neg})")
    return torch.tensor(weight, dtype=torch.float32, device=device)


# ── NaN guard ─────────────────────────────────────────────────────────────────

def safe_logits(logits):
    return torch.nan_to_num(logits, nan=0.0, posinf=6.0, neginf=-6.0)


# ── Train / eval epochs ───────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for batch in loader:
        batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                  for k, v in batch.items()}
        labels = batch["mortality"].float()

        optimiser.zero_grad()
        logits, _ = model(batch)
        logits     = safe_logits(logits)
        loss       = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).detach().cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float("nan")
    return avg_loss, auroc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                      for k, v in batch.items()}
            labels = batch["mortality"].float()
            logits, _ = model(batch)
            logits     = safe_logits(logits)
            loss       = criterion(logits, labels)
            total_loss += loss.item()
            all_probs.extend(torch.sigmoid(logits).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float("nan")
    return avg_loss, auroc, all_probs, all_labels


# ── Baselines ─────────────────────────────────────────────────────────────────

def run_baselines(train_ids, test_ids, clin_cache, label_cache):
    """Logistic regression + random forest on clinical latents."""

    def get_Xy(ids):
        X = np.stack([clin_cache[pid].numpy() for pid in ids
                      if label_cache[pid]["is_synthetic"] == 0])
        y = np.array([label_cache[pid]["mortality"] for pid in ids
                      if label_cache[pid]["is_synthetic"] == 0])
        return X, y

    X_train, y_train = get_Xy(train_ids)
    X_test,  y_test  = get_Xy(test_ids)
    results = {}

    try:
        sc  = StandardScaler()
        Xtr = sc.fit_transform(X_train)
        Xte = sc.transform(X_test)
        lr  = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED,
                                  class_weight="balanced")
        lr.fit(Xtr, y_train)
        results["LogReg"] = roc_auc_score(y_test, lr.predict_proba(Xte)[:, 1])
    except Exception as e:
        results["LogReg"] = float("nan")
        print(f"    LogReg failed: {e}")

    try:
        rf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED,
                                     class_weight="balanced", n_jobs=-1)
        rf.fit(X_train, y_train)
        results["RandomForest"] = roc_auc_score(
            y_test, rf.predict_proba(X_test)[:, 1])
    except Exception as e:
        results["RandomForest"] = float("nan")
        print(f"    RF failed: {e}")

    return results


# ── Single fold ───────────────────────────────────────────────────────────────

def train_fold(fold, device, caches, permuted_labels=None, use_synthetic=True):
    """
    Train and evaluate one fold.

    Args:
        fold:            int 0-4
        device:          torch device
        caches:          tuple returned by load_latent_cache()
        permuted_labels: optional list for permutation test

    Returns:
        test_auroc, test_probs, test_labels, baseline_aurocs, history
    """
    (soma_cache, met_cache, lip_cache, lum_cache,
     clin_cache, label_cache, real_ids, syn_ids) = caches

    print(f"\n{'='*55}")
    print(f"FOLD {fold}")
    print(f"{'='*55}")

    train_ids, val_ids, test_ids = get_fold_splits(
        real_ids, label_cache, fold)

    if not use_synthetic:
        train_ids = [pid for pid in train_ids
                     if label_cache[pid]["is_synthetic"] == 0]

    print(f"  Train: {len(train_ids)} ({len(train_ids)-len(syn_ids)} real "
          f"+ {len(syn_ids)} synthetic) | "
          f"Val: {len(val_ids)} | Test: {len(test_ids)}")

    def make_ds(ids, perm_labels=None):
        return CachedLatentDataset(
            patient_ids=ids,
            soma_cache=soma_cache, met_cache=met_cache,
            lip_cache=lip_cache,   lum_cache=lum_cache,
            clin_cache=clin_cache, label_cache=label_cache,
            permuted_labels=perm_labels,
        )

    train_ds = make_ds(train_ids, permuted_labels)
    val_ds   = make_ds(val_ids)
    test_ds  = make_ds(test_ids)

    train_loader = DataLoader(train_ds, batch_size=CFG["batch_size"],
                              shuffle=True, collate_fn=collate_fn,
                              num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=CFG["batch_size"],
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=CFG["batch_size"],
                              shuffle=False, collate_fn=collate_fn,
                              num_workers=0)

    # ── Model ──
    model = CachedMultimodalTransformer(
        dropout=CFG["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable parameters: {n_params:,}")

    # ── Loss ──
    pos_weight = get_class_weight(train_ids, label_cache, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # ── Optimiser ──
    optimiser = AdamW(model.parameters(),
                      lr=CFG["lr"],
                      weight_decay=CFG["weight_decay"])
    scheduler = CosineAnnealingLR(optimiser,
                                   T_max=CFG["epochs"],
                                   eta_min=1e-6)

    # ── Training loop ──
    best_val_auroc = 0.0
    best_state     = None
    history        = []

    for epoch in range(1, CFG["epochs"] + 1):
        train_loss, train_auroc = train_epoch(
            model, train_loader, optimiser, criterion, device)
        val_loss, val_auroc, _, _ = evaluate(
            model, val_loader, criterion, device)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_auroc": train_auroc,
            "val_loss":   val_loss,   "val_auroc":   val_auroc,
        })

        if val_auroc > best_val_auroc and not np.isnan(val_auroc):
            best_val_auroc = val_auroc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{CFG['epochs']} | "
                  f"Train Loss={train_loss:.4f} AUROC={train_auroc:.3f} | "
                  f"Val Loss={val_loss:.4f} AUROC={val_auroc:.3f}")

    # ── Evaluate on test set ──
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_auroc, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device)

    print(f"\n  Best Val AUROC : {best_val_auroc:.4f}")
    print(f"  Test AUROC     : {test_auroc:.4f}")

    # Save checkpoint
    if permuted_labels is None:
        ckpt_path = os.path.join(CFG["ckpt_dir"], f"model_cached_fold{fold}.pt")
        torch.save(best_state, ckpt_path)
        print(f"  Saved to {ckpt_path}")

    # Baselines
    if permuted_labels is None:
        print("\n  Running baselines...")
        baseline_aurocs = run_baselines(
            train_ids, test_ids, clin_cache, label_cache)
        for name, auroc in baseline_aurocs.items():
            print(f"    {name:20s}: AUROC={auroc:.4f}")
    else:
        baseline_aurocs = {}

    return test_auroc, test_probs, test_labels, baseline_aurocs, history


# ── Nested CV ─────────────────────────────────────────────────────────────────

def run_nested_cv(device, caches, folds=None, use_synthetic=True):
    folds = folds or list(range(CFG["n_folds"]))
    fold_aurocs      = []
    all_probs        = []
    all_labels       = []
    baseline_results = {"LogReg": [], "RandomForest": []}
    all_history      = []

    for fold in folds:
        auroc, probs, labels, baselines, history = train_fold(
            fold, device, caches, use_synthetic=use_synthetic)
        fold_aurocs.append(auroc)
        all_probs.extend(probs)
        all_labels.extend(labels)
        all_history.append(history)
        for name, val in baselines.items():
            if name in baseline_results:
                baseline_results[name].append(val)

    mean_auroc = np.nanmean(fold_aurocs)
    std_auroc  = np.nanstd(fold_aurocs)

    print(f"\n{'='*55}")
    print("NESTED CV RESULTS")
    print(f"{'='*55}")
    label = "With synthetic" if use_synthetic else "Real data only"
    print(f"  [{label}] AUROC = {mean_auroc:.4f} ± {std_auroc:.4f}")
    print(f"  Per-fold AUROCs: {[round(a, 4) for a in fold_aurocs]}")
    for name, vals in baseline_results.items():
        if vals:
            print(f"  {name:20s}: AUROC = "
                  f"{np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")

    # ── Wilcoxon signed-rank test vs baselines ──
    from scipy.stats import wilcoxon
    print(f"\n  Wilcoxon signed-rank test (transformer vs baselines):")
    for name, vals in baseline_results.items():
        if len(vals) == len(fold_aurocs):
            try:
                stat, p = wilcoxon(fold_aurocs, vals)
                direction = "better" if np.mean(fold_aurocs) > np.mean(vals) else "worse"
                print(f"    vs {name:20s}: p={p:.4f}  ({direction})")
            except Exception as e:
                print(f"    vs {name:20s}: could not compute ({e})")

    # Bootstrap 95% CI
    boot = [np.mean([fold_aurocs[i]
                     for i in np.random.choice(len(fold_aurocs),
                                               len(fold_aurocs), replace=True)])
            for _ in range(10000)]
    ci_low, ci_high = np.percentile(boot, [2.5, 97.5])
    print(f"\n  95% CI (bootstrap): [{ci_low:.4f}, {ci_high:.4f}]")

    os.makedirs(CFG["results_dir"], exist_ok=True)
    results = {
        "fold_aurocs":      fold_aurocs,
        "mean_auroc":       mean_auroc,
        "std_auroc":        std_auroc,
        "ci_95":            [ci_low, ci_high],
        "baseline_results": {k: list(v) for k, v in baseline_results.items()},
    }
    with open(os.path.join(CFG["results_dir"], "cv_results_cached.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {CFG['results_dir']}cv_results_cached.json")

    return mean_auroc, fold_aurocs, baseline_results


# ── Permutation test ──────────────────────────────────────────────────────────

def run_permutation_test(real_auroc, device, caches, n_permutations=None):
    n_permutations = n_permutations or CFG["n_permutations"]
    print(f"\n{'='*55}")
    print(f"PERMUTATION TEST ({n_permutations} permutations, fold 0)")
    print(f"{'='*55}")

    (_, _, _, _, _, label_cache, real_ids, _) = caches
    train_ids, _, _ = get_fold_splits(real_ids, label_cache, fold=0)
    real_labels = [label_cache[pid]["mortality"] for pid in train_ids]
    n = len(real_labels)

    null_aurocs = []
    for perm_i in range(n_permutations):
        shuffled = np.random.permutation(real_labels).tolist()
        perm_auroc, _, _, _, _ = train_fold(
            fold=0, device=device, caches=caches,
            permuted_labels=shuffled)
        null_aurocs.append(perm_auroc)

        if (perm_i + 1) % 50 == 0:
            p_val = np.mean(np.array(null_aurocs) >= real_auroc)
            print(f"  Permutation {perm_i+1:4d}/{n_permutations} | "
                  f"Null mean={np.mean(null_aurocs):.4f} | p={p_val:.4f}")

    p_value = np.mean(np.array(null_aurocs) >= real_auroc)
    print(f"\n  Null AUROC: {np.mean(null_aurocs):.4f} ± {np.std(null_aurocs):.4f}")
    print(f"  Real AUROC: {real_auroc:.4f}")
    print(f"  p-value:    {p_value:.4f} "
          f"({'SIGNIFICANT' if p_value < 0.05 else 'not significant'} at α=0.05)")

    os.makedirs(CFG["results_dir"], exist_ok=True)
    with open(os.path.join(CFG["results_dir"], "permutation_test.json"), "w") as f:
        json.dump({"real_auroc": real_auroc, "null_aurocs": null_aurocs,
                   "p_value": p_value}, f, indent=2)
    return p_value, null_aurocs


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", default="./")
    parser.add_argument("--fold",    type=int, default=None)
    parser.add_argument("--epochs",  type=int, default=CFG["epochs"])
    parser.add_argument("--permute", action="store_true")
    args = parser.parse_args()

    CFG["data_dir"] = args.data_dir
    CFG["epochs"]   = args.epochs

    device = get_device()
    print(f"Using device: {device}")
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    # Check cache exists
    required = ["somalogic_latents.pt", "metabolon_latents.pt",
                "lipidomics_latents.pt", "luminex_latents.pt",
                "clinical_latents.pt",  "labels.pt"]
    missing = [f for f in required
               if not os.path.exists(os.path.join(CFG["latent_dir"], f))]
    if missing:
        print(f"\nERROR: Missing cache files: {missing}")
        print(f"Run first:  python3 cache_latents.py {args.data_dir}")
        sys.exit(1)

    os.makedirs(CFG["ckpt_dir"],    exist_ok=True)
    os.makedirs(CFG["results_dir"], exist_ok=True)

    caches = load_latent_cache(CFG["latent_dir"])

    if args.fold is not None:
        auroc, probs, labels, baselines, history = train_fold(
            args.fold, device, caches)
        print(f"\nFold {args.fold} Test AUROC: {auroc:.4f}")
        if args.permute:
            run_permutation_test(auroc, device, caches)
    else:
        results = {}

        # ── Synthetic volume sweep ──
        syn_fractions = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
        # 0.0  = real only
        # 1.0  = 100% synthetic (195 synthetic = 195 real)
        # 2.0  = 200% synthetic (390 synthetic patients)

        print("\n" + "="*55)
        print("SYNTHETIC AUGMENTATION VOLUME SWEEP")
        print("="*55)
        print(f"  {'Syn fraction':15s} {'N synthetic':12s} {'AUROC':10s} {'±std':8s} {'95% CI':20s}")
        print("  " + "-"*65)

        for frac in syn_fractions:
            fold_aurocs_frac = []

            for fold in range(CFG["n_folds"]):
                (soma_cache, met_cache, lip_cache, lum_cache,
                 clin_cache, label_cache, real_ids, syn_ids) = caches

                train_ids, val_ids, test_ids = get_fold_splits(
                    real_ids, label_cache, fold)

                # Subsample or expand synthetic pool
                all_syn = [pid for pid in train_ids
                           if label_cache[pid]["is_synthetic"] == 1]
                real_train = [pid for pid in train_ids
                              if label_cache[pid]["is_synthetic"] == 0]

                n_syn_target = int(len(real_train) * frac)

                if n_syn_target == 0:
                    final_train = real_train
                elif n_syn_target <= len(all_syn):
                    rng = np.random.default_rng(RANDOM_SEED)
                    chosen_syn = rng.choice(
                        all_syn, size=n_syn_target, replace=False).tolist()
                    final_train = real_train + chosen_syn
                else:
                    # Need more synthetic than available — sample with replacement
                    rng = np.random.default_rng(RANDOM_SEED)
                    chosen_syn = rng.choice(
                        all_syn, size=n_syn_target, replace=True).tolist()
                    final_train = real_train + chosen_syn

                # Build datasets manually for this fold
                (soma_cache, met_cache, lip_cache, lum_cache,
                 clin_cache, label_cache, real_ids, syn_ids) = caches

                train_ds = CachedLatentDataset(
                    patient_ids=final_train,
                    soma_cache=soma_cache, met_cache=met_cache,
                    lip_cache=lip_cache,   lum_cache=lum_cache,
                    clin_cache=clin_cache, label_cache=label_cache,
                )
                val_ds = CachedLatentDataset(
                    patient_ids=val_ids,
                    soma_cache=soma_cache, met_cache=met_cache,
                    lip_cache=lip_cache,   lum_cache=lum_cache,
                    clin_cache=clin_cache, label_cache=label_cache,
                )
                test_ds = CachedLatentDataset(
                    patient_ids=test_ids,
                    soma_cache=soma_cache, met_cache=met_cache,
                    lip_cache=lip_cache,   lum_cache=lum_cache,
                    clin_cache=clin_cache, label_cache=label_cache,
                )

                train_loader = DataLoader(
                    train_ds, batch_size=CFG["batch_size"],
                    shuffle=True, collate_fn=collate_fn, num_workers=0)
                val_loader = DataLoader(
                    val_ds, batch_size=CFG["batch_size"],
                    shuffle=False, collate_fn=collate_fn, num_workers=0)
                test_loader = DataLoader(
                    test_ds, batch_size=CFG["batch_size"],
                    shuffle=False, collate_fn=collate_fn, num_workers=0)

                model = CachedMultimodalTransformer(
                    dropout=CFG["dropout"]).to(device)
                pos_weight = get_class_weight(final_train, label_cache, device)
                criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
                optimiser  = AdamW(model.parameters(),
                                   lr=CFG["lr"],
                                   weight_decay=CFG["weight_decay"])
                scheduler  = CosineAnnealingLR(optimiser,
                                               T_max=CFG["epochs"],
                                               eta_min=1e-6)

                best_val   = 0.0
                best_state = None

                for epoch in range(1, CFG["epochs"] + 1):
                    train_epoch(model, train_loader, optimiser, criterion, device)
                    _, val_auroc, _, _ = evaluate(
                        model, val_loader, criterion, device)
                    scheduler.step()
                    if not np.isnan(val_auroc) and val_auroc > best_val:
                        best_val   = val_auroc
                        best_state = {k: v.clone()
                                      for k, v in model.state_dict().items()}

                if best_state:
                    model.load_state_dict(best_state)
                _, test_auroc, _, _ = evaluate(
                    model, test_loader, criterion, device)
                fold_aurocs_frac.append(test_auroc)

            mean_a = np.nanmean(fold_aurocs_frac)
            std_a  = np.nanstd(fold_aurocs_frac)
            boot   = [np.mean([fold_aurocs_frac[i]
                               for i in np.random.choice(
                                   len(fold_aurocs_frac),
                                   len(fold_aurocs_frac), replace=True)])
                      for _ in range(2000)]
            ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
            n_syn_avg    = int(len(real_ids) * 0.8 * frac)

            results[frac] = {
                "mean_auroc":    mean_a,
                "std":           std_a,
                "ci":            [ci_lo, ci_hi],
                "fold_aurocs":   fold_aurocs_frac,
            }

            print(f"  {frac*100:>5.0f}% synthetic  "
                  f"{n_syn_avg:>6d} pts    "
                  f"{mean_a:.4f}    "
                  f"±{std_a:.4f}  "
                  f"[{ci_lo:.4f}, {ci_hi:.4f}]")

        # Save results
        os.makedirs(CFG["results_dir"], exist_ok=True)
        with open(os.path.join(CFG["results_dir"],
                               "synthetic_volume_sweep.json"), "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f, indent=2)

        print(f"\n  Results saved to {CFG['results_dir']}synthetic_volume_sweep.json")
        print("\n  Interpretation:")
        aurocs = [results[f]["mean_auroc"] for f in syn_fractions]
        peak_frac = syn_fractions[np.argmax(aurocs)]
        print(f"  Peak performance at {peak_frac*100:.0f}% synthetic augmentation")
        print(f"  AUROC range: {min(aurocs):.4f} – {max(aurocs):.4f}")