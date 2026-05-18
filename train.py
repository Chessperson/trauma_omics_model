"""
train.py — Training loop for Multimodal Trauma Mortality Transformer
=====================================================================
Implements:
  - Nested 5-fold cross validation
  - Weighted BCE loss for class imbalance
  - AUROC evaluation on real patients only
  - Baseline comparisons (ISS, logistic regression, random forest)
  - Permutation test for statistical significance
  - Model checkpointing

Usage:
    # Train all 5 folds
    python3 train.py ./

    # Train a single fold
    python3 train.py ./ --fold 0

    # Run permutation test after training
    python3 train.py ./ --permute
"""

from email import encoders
import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import model
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from dataset import build_datasets
from encoders import load_pretrained_encoders, freeze_encoders
from model import MultimodalTransformer, D_MODEL

# ── Config ────────────────────────────────────────────────────────────────────

CFG = {
    "data_dir":    "./",
    "ckpt_dir":    "outputs/checkpoints/",
    "results_dir": "outputs/results/",
    "n_folds":     5,
    "epochs":      50,
    "batch_size":  16,
    "lr":          1e-4,
    "weight_decay": 1e-3,
    "d_model":     128,
    "n_heads":     4,
    "n_layers":    2,
    "dropout":     0.1,
    "n_permutations": 1000,
    "seed":        42,
}


# ── Device ────────────────────────────────────────────────────────────────────

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ── Collate function ──────────────────────────────────────────────────────────

def collate_fn(batch):
    out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([b[key] for b in batch])
        else:
            out[key] = [b[key] for b in batch]
    return out


# ── Class weight calculation ──────────────────────────────────────────────────

def get_class_weight(dataset, device):
    """
    Compute positive class weight for weighted BCE.
    weight = n_negative / n_positive
    """
    labels = [int(dataset[i]["mortality"]) for i in range(len(dataset))]
    n_pos  = sum(labels)
    n_neg  = len(labels) - n_pos
    weight = n_neg / max(n_pos, 1)
    print(f"  Class weight: {weight:.2f} "
          f"(pos={n_pos}, neg={n_neg})")
    return torch.tensor(weight, dtype=torch.float32, device=device)


# ── NaN guard ────────────────────────────────────────────────────────────────

def safe_logits(logits):
    """Replace NaN logits with 0 (sigmoid → 0.5, neutral prediction)."""
    return torch.nan_to_num(logits, nan=0.0, posinf=6.0, neginf=-6.0)


# ── Single epoch train ────────────────────────────────────────────────────────

def train_epoch(model, loader, optimiser, criterion, device):
    model.train()
    total_loss = 0.0
    all_probs, all_labels = [], []

    for batch in loader:
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
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
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float("nan")

    return avg_loss, auroc


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_probs, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            labels = batch["mortality"].float()
            logits, _ = model(batch)
            logits     = safe_logits(logits)
            loss       = criterion(logits, labels)
            total_loss += loss.item()
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader)
    try:
        auroc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auroc = float("nan")

    return avg_loss, auroc, all_probs, all_labels


# ── Baseline models ───────────────────────────────────────────────────────────

def run_baselines(train_ds, test_ds):
    """
    Run three baseline models on clinical features only:
      1. ISS score alone
      2. Logistic regression
      3. Random forest
    Returns dict of AUROC scores.
    """
    def get_features_labels(ds):
        rows, labels = [], []
        for i in range(len(ds)):
            item = ds[i]
            if int(item["is_synthetic"]) == 1:
                continue
            rows.append(item["clinical"].numpy())
            labels.append(int(item["mortality"]))
        return np.array(rows), np.array(labels)

    X_train, y_train = get_features_labels(train_ds)
    X_test,  y_test  = get_features_labels(test_ds)

    results = {}

    # ISS alone (first clinical feature that contains ISS)
    # ISS is typically the highest-value injury score in clinical cols
    # We use feature index 0 as a proxy — adjust if needed
    try:
        iss_probs = X_test[:, 0]
        results["ISS_alone"] = roc_auc_score(y_test, iss_probs)
    except Exception:
        results["ISS_alone"] = float("nan")

    # Logistic regression
    try:
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_train)
        X_te_s = scaler.transform(X_test)
        lr = LogisticRegression(max_iter=1000, random_state=CFG["seed"],
                                class_weight="balanced")
        lr.fit(X_tr_s, y_train)
        results["LogReg"] = roc_auc_score(
            y_test, lr.predict_proba(X_te_s)[:, 1])
    except Exception as e:
        results["LogReg"] = float("nan")
        print(f"    LogReg failed: {e}")

    # Random forest
    try:
        rf = RandomForestClassifier(
            n_estimators=200, random_state=CFG["seed"],
            class_weight="balanced", n_jobs=-1)
        rf.fit(X_train, y_train)
        results["RandomForest"] = roc_auc_score(
            y_test, rf.predict_proba(X_test)[:, 1])
    except Exception as e:
        results["RandomForest"] = float("nan")
        print(f"    RF failed: {e}")

    return results


# ── Single fold training ──────────────────────────────────────────────────────

def train_fold(fold, device, feature_dims, permuted_labels=None):
    """
    Train and evaluate one outer fold.

    Args:
        fold:             int 0-4
        device:           torch device
        feature_dims:     dict from build_datasets
        permuted_labels:  if set, replace training labels with these
                          (used for permutation test)

    Returns:
        test_auroc:   float
        test_probs:   list of predicted probabilities
        test_labels:  list of true labels
        baseline_aurocs: dict
    """
    print(f"\n{'='*55}")
    print(f"FOLD {fold}")
    print(f"{'='*55}")

    train_ds, val_ds, test_ds, _ = build_datasets(
        data_dir=CFG["data_dir"], fold=fold)

    # Permute labels if running permutation test
    if permuted_labels is not None:
        for i in range(len(train_ds.patients)):
            train_ds.patients.at[i, "mortality"] = int(permuted_labels[i])

    # Dataloaders
    train_loader = DataLoader(
        train_ds, batch_size=CFG["batch_size"],
        shuffle=True, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(
        val_ds, batch_size=CFG["batch_size"],
        shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(
        test_ds, batch_size=CFG["batch_size"],
        shuffle=False, collate_fn=collate_fn)

    # Load and freeze pretrained encoders
    encoders = load_pretrained_encoders(
        CFG["ckpt_dir"], feature_dims, device)
    # Don't freeze — fine-tune encoders end-to-end with lower lr
    for name, enc in encoders.items():
        for param in enc.parameters():
            param.requires_grad = True

    # Build model
    model = MultimodalTransformer(
        encoders,
        d_model=CFG["d_model"],
        n_heads=CFG["n_heads"],
        n_layers=CFG["n_layers"],
        dropout=CFG["dropout"],
    ).to(device)

    # Class-weighted BCE loss
    pos_weight = get_class_weight(train_ds, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Optimiser + scheduler (only train non-frozen params)
    encoder_params     = list(model.encoders.parameters())
    encoder_param_ids  = set(id(p) for p in encoder_params)
    transformer_params = [p for p in model.parameters()
                        if id(p) not in encoder_param_ids]

    optimiser = AdamW([
        {"params": transformer_params, "lr": CFG["lr"]},
        {"params": encoder_params,     "lr": CFG["lr"] * 0.1},
    ], weight_decay=CFG["weight_decay"])
    scheduler  = CosineAnnealingLR(
        optimiser, T_max=CFG["epochs"], eta_min=1e-6)

    # Training loop
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

    # Restore best model and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_auroc, test_probs, test_labels = evaluate(
        model, test_loader, criterion, device)

    print(f"\n  Best Val AUROC : {best_val_auroc:.4f}")
    print(f"  Test AUROC     : {test_auroc:.4f}")

    # Save fold checkpoint
    if permuted_labels is None:
        fold_ckpt = os.path.join(
            CFG["ckpt_dir"], f"model_fold{fold}.pt")
        torch.save(best_state, fold_ckpt)
        print(f"  Saved to {fold_ckpt}")

    # Baselines
    if permuted_labels is None:
        print("\n  Running baselines...")
        baseline_aurocs = run_baselines(train_ds, test_ds)
        for name, auroc in baseline_aurocs.items():
            print(f"    {name:20s}: AUROC={auroc:.4f}")
    else:
        baseline_aurocs = {}

    return test_auroc, test_probs, test_labels, baseline_aurocs, history


# ── Full nested CV ────────────────────────────────────────────────────────────

def run_nested_cv(device, feature_dims, folds=None):
    """
    Run nested 5-fold CV across all folds.
    Reports mean ± std AUROC and comparison vs baselines.
    """
    folds = folds or list(range(CFG["n_folds"]))
    fold_aurocs     = []
    all_probs       = []
    all_labels      = []
    baseline_results = {
        "ISS_alone": [], "LogReg": [], "RandomForest": []}
    all_history     = []

    for fold in folds:
        auroc, probs, labels, baselines, history = train_fold(
            fold, device, feature_dims)
        fold_aurocs.append(auroc)
        all_probs.extend(probs)
        all_labels.extend(labels)
        all_history.append(history)

        for name, val in baselines.items():
            if name in baseline_results:
                baseline_results[name].append(val)

    # Summary
    mean_auroc = np.nanmean(fold_aurocs)
    std_auroc  = np.nanstd(fold_aurocs)

    print(f"\n{'='*55}")
    print("NESTED CV RESULTS")
    print(f"{'='*55}")
    print(f"  Multimodal Transformer: "
          f"AUROC = {mean_auroc:.4f} ± {std_auroc:.4f}")
    print(f"  Per-fold AUROCs: {[round(a, 4) for a in fold_aurocs]}")
    print()
    for name, vals in baseline_results.items():
        if vals:
            print(f"  {name:20s}: AUROC = "
                  f"{np.nanmean(vals):.4f} ± {np.nanstd(vals):.4f}")

    # Bootstrap 95% CI on mean AUROC
    bootstrap_aurocs = []
    for _ in range(10000):
        idx = np.random.choice(len(fold_aurocs), len(fold_aurocs), replace=True)
        bootstrap_aurocs.append(np.mean([fold_aurocs[i] for i in idx]))
    ci_low  = np.percentile(bootstrap_aurocs, 2.5)
    ci_high = np.percentile(bootstrap_aurocs, 97.5)
    print(f"\n  95% CI (bootstrap): [{ci_low:.4f}, {ci_high:.4f}]")

    # Save results
    os.makedirs(CFG["results_dir"], exist_ok=True)
    results = {
        "fold_aurocs":       fold_aurocs,
        "mean_auroc":        mean_auroc,
        "std_auroc":         std_auroc,
        "ci_95":             [ci_low, ci_high],
        "baseline_results":  {k: list(v) for k, v in baseline_results.items()},
    }
    with open(os.path.join(CFG["results_dir"], "cv_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {CFG['results_dir']}cv_results.json")

    return mean_auroc, fold_aurocs, baseline_results


# ── Permutation test ──────────────────────────────────────────────────────────

def run_permutation_test(real_auroc, device, feature_dims,
                         n_permutations=None):
    """
    Shuffle mortality labels 1000 times, retrain on fold 0 each time,
    build null distribution of AUROCs.
    p-value = fraction of permutations >= real_auroc.
    """
    n_permutations = n_permutations or CFG["n_permutations"]
    print(f"\n{'='*55}")
    print(f"PERMUTATION TEST ({n_permutations} permutations, fold 0)")
    print(f"{'='*55}")
    print(f"  Real AUROC: {real_auroc:.4f}")

    # Get training labels for fold 0
    train_ds, _, _, _ = build_datasets(
        data_dir=CFG["data_dir"], fold=0)
    real_labels = [int(train_ds[i]["mortality"])
                   for i in range(len(train_ds))]
    n = len(real_labels)

    null_aurocs = []
    for perm_i in range(n_permutations):
        shuffled = np.random.permutation(real_labels).tolist()
        perm_auroc, _, _, _, _ = train_fold(
            fold=0,
            device=device,
            feature_dims=feature_dims,
            permuted_labels=shuffled,
        )
        null_aurocs.append(perm_auroc)

        if (perm_i + 1) % 50 == 0:
            p_val = np.mean(
                np.array(null_aurocs) >= real_auroc)
            print(f"  Permutation {perm_i+1:4d}/{n_permutations} | "
                  f"Null mean={np.mean(null_aurocs):.4f} | "
                  f"p={p_val:.4f}")

    p_value = np.mean(np.array(null_aurocs) >= real_auroc)
    print(f"\n  Null distribution: "
          f"{np.mean(null_aurocs):.4f} ± {np.std(null_aurocs):.4f}")
    print(f"  Real AUROC:  {real_auroc:.4f}")
    print(f"  p-value:     {p_value:.4f} "
          f"({'SIGNIFICANT' if p_value < 0.05 else 'not significant'} "
          f"at alpha=0.05)")

    # Save
    perm_results = {
        "real_auroc":   real_auroc,
        "null_aurocs":  null_aurocs,
        "p_value":      p_value,
        "null_mean":    float(np.mean(null_aurocs)),
        "null_std":     float(np.std(null_aurocs)),
    }
    os.makedirs(CFG["results_dir"], exist_ok=True)
    with open(os.path.join(
            CFG["results_dir"], "permutation_test.json"), "w") as f:
        json.dump(perm_results, f, indent=2)
    print(f"  Saved to {CFG['results_dir']}permutation_test.json")

    return p_value, null_aurocs


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir",  nargs="?", default="./",
                        help="Path to data directory")
    parser.add_argument("--fold",    type=int,  default=None,
                        help="Run a single fold (0-4)")
    parser.add_argument("--permute", action="store_true",
                        help="Run permutation test after training")
    parser.add_argument("--epochs",  type=int,  default=CFG["epochs"],
                        help="Training epochs per fold")
    args = parser.parse_args()

    CFG["data_dir"] = args.data_dir
    CFG["epochs"]   = args.epochs

    device = get_device()
    print(f"Using device: {device}")
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    # Load data once to get feature_dims
    print("Loading data to get feature dimensions...")
    _, _, _, feature_dims = build_datasets(
        data_dir=CFG["data_dir"], fold=0)

    os.makedirs(CFG["ckpt_dir"],   exist_ok=True)
    os.makedirs(CFG["results_dir"], exist_ok=True)

    if args.fold is not None:
        # Single fold
        auroc, probs, labels, baselines, history = train_fold(
            args.fold, device, feature_dims)
        print(f"\nFold {args.fold} Test AUROC: {auroc:.4f}")

        if args.permute:
            run_permutation_test(auroc, device, feature_dims)

    else:
        # Full nested CV
        mean_auroc, fold_aurocs, baselines = run_nested_cv(
            device, feature_dims)

        if args.permute:
            run_permutation_test(mean_auroc, device, feature_dims)