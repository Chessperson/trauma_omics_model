"""
repeated_cv.py — 10x5-fold repeated cross-validation on PAMPer
===============================================================
Addresses fold-partition variance by running 5-fold CV with 10
different random seeds and reporting the grand mean AUROC with CI.

Usage:
    python3 repeated_cv.py ./
"""

import os, sys, json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from train_cached import (
    CachedMultimodalTransformer, get_device,
    load_latent_cache, train_epoch, evaluate,
    get_class_weight, CFG, CachedLatentDataset, collate_fn,
)

N_REPEATS  = 10
N_FOLDS    = 5
SYN_FRAC   = 0.5
SEEDS      = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]
OUTPUT_DIR = "outputs/results/"


def get_fold_splits_seeded(real_ids, label_cache, syn_ids,
                            fold, seed):
    """Like get_fold_splits but with variable random seed."""
    real_labels = np.array(
        [label_cache[pid]["mortality"] for pid in real_ids])

    skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                             random_state=seed)
    folds = list(skf.split(real_ids, real_labels))
    train_val_idx, test_idx = folds[fold]

    inner_skf   = StratifiedKFold(n_splits=N_FOLDS, shuffle=True,
                                   random_state=seed + 1)
    tv_ids      = [real_ids[i] for i in train_val_idx]
    tv_labels   = real_labels[train_val_idx]
    inner_folds = list(inner_skf.split(tv_ids, tv_labels))
    train_inner_idx, val_inner_idx = inner_folds[fold % N_FOLDS]

    train_real = [tv_ids[i] for i in train_inner_idx]
    val_ids    = [tv_ids[i] for i in val_inner_idx]
    test_ids   = [real_ids[i] for i in test_idx]

    # Subsample synthetic at SYN_FRAC
    n_syn      = int(len(train_real) * SYN_FRAC)
    rng        = np.random.default_rng(seed + fold * 13)
    chosen_syn = rng.choice(syn_ids,
                             size=min(n_syn, len(syn_ids)),
                             replace=False).tolist()
    train_ids  = train_real + chosen_syn

    return train_ids, val_ids, test_ids


def run_one_fold(caches, fold, seed, device):
    (soma_cache, met_cache, lip_cache, lum_cache,
     clin_cache, label_cache, real_ids, syn_ids) = caches

    train_ids, val_ids, test_ids = get_fold_splits_seeded(
        real_ids, label_cache, syn_ids, fold, seed)

    train_ds = CachedLatentDataset(
        train_ids, soma_cache, met_cache, lip_cache,
        lum_cache, clin_cache, label_cache)
    val_ds   = CachedLatentDataset(
        val_ids, soma_cache, met_cache, lip_cache,
        lum_cache, clin_cache, label_cache)
    test_ds  = CachedLatentDataset(
        test_ids, soma_cache, met_cache, lip_cache,
        lum_cache, clin_cache, label_cache)

    train_loader = DataLoader(
        train_ds, batch_size=CFG['batch_size'],
        shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(
        val_ds, batch_size=CFG['batch_size'],
        shuffle=False, collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(
        test_ds, batch_size=CFG['batch_size'],
        shuffle=False, collate_fn=collate_fn, num_workers=0)

    model      = CachedMultimodalTransformer(
        dropout=CFG['dropout']).to(device)
    pos_weight = get_class_weight(train_ids, label_cache, device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser  = AdamW(model.parameters(),
                       lr=CFG['lr'],
                       weight_decay=CFG['weight_decay'])
    scheduler  = CosineAnnealingLR(
        optimiser, T_max=CFG['epochs'], eta_min=1e-6)

    best_val   = 0.0
    best_state = None

    for epoch in range(1, CFG['epochs'] + 1):
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

    return test_auroc


def main(data_dir="./"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    print("\nLoading cached latents...")
    caches = load_latent_cache(CFG['latent_dir'])
    (_, _, _, _, _, label_cache, real_ids, syn_ids) = caches
    real_labels = np.array(
        [label_cache[p]["mortality"] for p in real_ids])
    print(f"  Real: {len(real_ids)} | Synthetic: {len(syn_ids)}")
    print(f"  Mortality rate: {real_labels.mean():.3f}")

    print(f"\n{N_REPEATS}×{N_FOLDS}-fold CV "
          f"({N_REPEATS * N_FOLDS * CFG['epochs']} total epochs)")
    print("="*60)
    print(f"  {'Rep':5s} {'Mean':8s} {'Std':8s} {'Folds'}")
    print("  " + "-"*58)

    all_means   = []
    all_results = []

    for rep_idx, seed in enumerate(SEEDS):
        torch.manual_seed(seed)
        np.random.seed(seed)

        fold_aurocs = []
        for fold in range(N_FOLDS):
            auroc = run_one_fold(caches, fold, seed, device)
            fold_aurocs.append(auroc)

        mean_a = float(np.nanmean(fold_aurocs))
        std_a  = float(np.nanstd(fold_aurocs))
        all_means.append(mean_a)
        all_results.append({
            'rep':        rep_idx + 1,
            'seed':       seed,
            'mean_auroc': mean_a,
            'std':        std_a,
            'folds':      [round(float(x), 4) for x in fold_aurocs],
        })

        print(f"  Rep {rep_idx+1:2d}  "
              f"{mean_a:.4f}   "
              f"±{std_a:.4f}   "
              f"{[round(x,3) for x in fold_aurocs]}")

        # Save intermediate results after each rep
        with open(os.path.join(OUTPUT_DIR, 'repeated_cv.json'), 'w') as f:
            json.dump({'reps': all_results,
                       'grand_mean': float(np.mean(all_means)),
                       'grand_std':  float(np.std(all_means))},
                      f, indent=2)

    grand_mean = float(np.mean(all_means))
    grand_std  = float(np.std(all_means))
    ci_lo = grand_mean - 1.96 * grand_std / np.sqrt(N_REPEATS)
    ci_hi = grand_mean + 1.96 * grand_std / np.sqrt(N_REPEATS)

    print(f"\n{'='*60}")
    print(f"  Grand mean : {grand_mean:.4f} ± {grand_std:.4f}")
    print(f"  95% CI     : [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  Min rep    : {min(all_means):.4f}")
    print(f"  Max rep    : {max(all_means):.4f}")
    print(f"  Single-run : 0.8686  (seed=42 reference)")

    final = {
        'reps':        all_results,
        'grand_mean':  grand_mean,
        'grand_std':   grand_std,
        'ci_lo':       ci_lo,
        'ci_hi':       ci_hi,
        'n_repeats':   N_REPEATS,
        'n_folds':     N_FOLDS,
        'syn_frac':    SYN_FRAC,
    }
    with open(os.path.join(OUTPUT_DIR, 'repeated_cv.json'), 'w') as f:
        json.dump(final, f, indent=2)
    print(f"\n  Saved: {OUTPUT_DIR}repeated_cv.json")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    main(data_dir)
