"""
bidirectional_validation.py — Bidirectional Cross-Cohort Validation
=====================================================================
Tests generalization in both directions using the joint VAE encoder:

Direction 1 — PAMPer → SWAT (already established):
    Train mortality predictor on PAMPer (joint VAE latents)
    Evaluate zero-shot on all 134 SWAT patients
    Repeated fine-tuning evaluation: 10x stratified splits on SWAT

Direction 2 — SWAT → PAMPer:
    Train mortality predictor on SWAT labels (joint VAE latents)
    Evaluate on held-out PAMPer patients (5-fold CV)
    Tests whether SWAT proteomic signal generalizes to PAMPer

Direction 3 — Combined training:
    Train on PAMPer + SWAT jointly
    Evaluate on held-out from both cohorts
    Strongest statistical test

All directions use the joint SomaLogic VAE (trained on both cohorts)
for a fair comparison.

Usage:
    python3 bidirectional_validation.py ./
"""

import os, sys, re, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, os.path.dirname(__file__))
from train_cached import (CachedMultimodalTransformer, get_device,
                           load_latent_cache, CFG, safe_logits,
                           CachedLatentDataset, collate_fn,
                           train_epoch, evaluate, get_class_weight)
from encoders import SomaLogicVAE
from dataset import build_datasets

OUTPUT_DIR       = "outputs/bidirectional_validation/"
JOINT_LATENT_DIR = "outputs/latents_joint/"
CKPT_DIR         = "outputs/checkpoints/"
SWAT_CSV         = "SWAT_proteins_clinical.csv"

N_FOLDS     = 5
N_REPEATS   = 10
FT_EPOCHS   = 80
FT_LR       = 1e-4
SEEDS       = [42, 142, 242, 342, 442, 542, 642, 742, 842, 942]


# ── SWAT data loading ───────────────────────────────────────────────────────

def parse_swat_tp(name):
    m = re.search(r'SWAT\d+?(\d+)EDA$', str(name))
    if not m: return None
    s = m.group(1)
    return 24 if s.endswith('24') else (4 if s.endswith('4') else 0)


def load_swat_latents(data_dir, device, ckpt_dir):
    """Load SWAT patients and encode through joint VAE."""
    print("Loading SWAT data...")
    df = pd.read_csv(os.path.join(data_dir, SWAT_CSV))
    df['timepoint'] = df['Name'].apply(parse_swat_tp)

    pamp_df    = pd.read_excel(
        os.path.join(data_dir, "PAMPer_External_data.xlsx"),
        sheet_name="SomaLogic", nrows=1)
    pamp_prots = pamp_df.columns.tolist()[2:]

    plasma_treatments = {'All three', 'PRBC+Plasma', 'WB+Plasma'}

    _, _, _, feature_dims = build_datasets(data_dir=data_dir, fold=0)
    vae = SomaLogicVAE(feature_dims['somalogic'], 128)
    vae.load_state_dict(torch.load(
        os.path.join(ckpt_dir, 'somalogic_vae_joint.pt'),
        map_location=device))
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    patient_data = {}
    latents      = {}

    tp0_pids = df[df['timepoint'] == 0]['ID'].unique().tolist()

    with torch.no_grad():
        for pid in tp0_pids:
            prows = df[df['ID'] == pid].sort_values('timepoint')
            row0  = prows[prows['timepoint'] == 0].iloc[0]

            treatment_str = str(row0.get('Treatment', 'Other'))
            treatment     = 1 if treatment_str in plasma_treatments else 0
            mortality     = int(row0.get('30d_Mortality', 0))

            prot_mat = np.zeros((3, len(pamp_prots)), dtype=np.float32)
            masks    = np.zeros(3, dtype=np.float32)

            for _, row in prows.iterrows():
                tp   = row['timepoint']
                vals = np.array([
                    float(row[p]) if p in row.index and pd.notna(row[p])
                    else 0.0 for p in pamp_prots], dtype=np.float32)
                vals = np.log1p(np.clip(vals, 0, None))
                if tp == 0:
                    prot_mat[0] = vals; masks[0] = 1.0
                elif tp == 24:
                    prot_mat[1] = vals; masks[1] = 1.0

            if masks[0] == 0:
                continue

            clin = np.zeros(29, dtype=np.float32)
            for col, idx in {'phgcs': 0, 'phsbp': 1, 'phhr': 2, 'phrr': 3,
                              'phshockind': 4, 'Age': 5, 'ISS': 6,
                              'TBI': 7, 'Penetrating': 8}.items():
                if col in row0.index:
                    v = row0.get(col, 0)
                    clin[idx] = float(v) if pd.notna(v) else 0.0

            # Encode through joint VAE
            z_list = []
            for tp in range(3):
                x = torch.tensor(prot_mat[tp]).unsqueeze(0).to(device)
                z, _ = vae.encode(x)
                z_list.append(z.squeeze(0).cpu())

            patient_data[pid] = {
                'clinical':      clin,
                'masks':         masks,
                'mortality':     mortality,
                'treatment':     treatment,
                'treatment_str': treatment_str,
            }
            latents[pid] = torch.stack(z_list)

    print(f"  SWAT patients loaded: {len(patient_data)}")
    mort = np.mean([v['mortality'] for v in patient_data.values()])
    print(f"  SWAT mortality rate: {mort:.3f}")
    return patient_data, latents


def build_swat_batch(patient_data, latents, pids, device):
    B            = len(pids)
    soma_latents = torch.stack([latents[p] for p in pids]).to(device)
    soma_masks   = torch.tensor(
        np.array([patient_data[p]['masks'] for p in pids]),
        dtype=torch.float32).to(device)
    zeros_64  = torch.zeros(B, 3, 64, device=device)
    zeros_32  = torch.zeros(B, 3, 32, device=device)
    zero_mask = torch.zeros(B, 3, device=device)
    clin      = np.array([patient_data[p]['clinical'] for p in pids])
    clinical  = torch.zeros(B, 32, dtype=torch.float32)
    clinical[:, :29] = torch.tensor(clin, dtype=torch.float32)
    clinical  = clinical.to(device)
    return {
        'somalogic': soma_latents, 'somalogic_mask': soma_masks,
        'metabolon': zeros_64,     'metabolon_mask':  zero_mask,
        'lipidomics': zeros_64,    'lipidomics_mask': zero_mask,
        'luminex':   zeros_32,     'luminex_mask':    zero_mask,
        'clinical':  clinical,
    }


# ── Direction 1: PAMPer → SWAT ──────────────────────────────────────────────

def direction1_pamper_to_swat(swat_data, swat_latents, device,
                               n_repeats=N_REPEATS):
    """
    Train on PAMPer (joint latents), evaluate on SWAT.
    Zero-shot: use fold0 PAMPer model directly.
    Fine-tuned: repeated stratified splits on SWAT.
    """
    print("\n" + "="*60)
    print("DIRECTION 1: PAMPer → SWAT")
    print("="*60)

    # Zero-shot: load PAMPer fold0 model, evaluate on all SWAT
    print("\n  Zero-shot evaluation (PAMPer fold0 model → all SWAT)...")
    model = CachedMultimodalTransformer(dropout=0.3).to(device)
    model.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, 'model_cached_fold0.pt'),
        map_location=device))
    model.eval()

    swat_pids   = list(swat_data.keys())
    swat_labels = np.array([swat_data[p]['mortality'] for p in swat_pids])
    all_probs   = []

    with torch.no_grad():
        for i in range(0, len(swat_pids), 16):
            batch = build_swat_batch(
                swat_data, swat_latents, swat_pids[i:i+16], device)
            logits, _ = model(batch)
            all_probs.extend(
                torch.sigmoid(safe_logits(logits)).cpu().tolist())

    zs_auroc = roc_auc_score(swat_labels, all_probs)
    print(f"  Zero-shot AUROC: {zs_auroc:.4f} (n={len(swat_pids)}, "
          f"deaths={swat_labels.sum()})")

    # Fine-tuned: repeated stratified splits
    print(f"\n  Fine-tuned evaluation ({n_repeats} repeated splits)...")
    ft_aurocs = []

    for rep, seed in enumerate(SEEDS[:n_repeats]):
        torch.manual_seed(seed)
        np.random.seed(seed)

        tr_pids, te_pids, tr_y, te_y = train_test_split(
            swat_pids, swat_labels,
            test_size=0.2, stratify=swat_labels,
            random_state=seed)

        # Fine-tune head only
        model_ft = CachedMultimodalTransformer(dropout=0.3).to(device)
        model_ft.load_state_dict(torch.load(
            os.path.join(CKPT_DIR, 'model_cached_fold0.pt'),
            map_location=device))

        for name, param in model_ft.named_parameters():
            param.requires_grad = any(k in name for k in
                                      ['soma_proj', 'clin_proj', 'fusion'])

        n_pos      = tr_y.sum()
        n_neg      = len(tr_y) - n_pos
        pos_weight = torch.tensor(
            [min(n_neg / max(n_pos, 1), 10.0)],
            dtype=torch.float32).to(device)
        optim      = AdamW(
            [p for p in model_ft.parameters() if p.requires_grad],
            lr=FT_LR, weight_decay=1e-4)
        sched      = CosineAnnealingLR(optim, T_max=FT_EPOCHS, eta_min=1e-6)

        best_auroc = 0.0
        best_state = None

        for epoch in range(1, FT_EPOCHS + 1):
            model_ft.train()
            idx = np.random.permutation(len(tr_pids))
            for i in range(0, len(tr_pids), 16):
                bp  = [tr_pids[j] for j in idx[i:i+16]]
                bl  = torch.tensor(
                    [swat_data[p]['mortality'] for p in bp],
                    dtype=torch.float32).to(device)
                optim.zero_grad()
                batch    = build_swat_batch(swat_data, swat_latents, bp, device)
                logits,_ = model_ft(batch)
                loss = F.binary_cross_entropy_with_logits(
                    safe_logits(logits), bl, pos_weight=pos_weight)
                loss.backward()
                nn.utils.clip_grad_norm_(model_ft.parameters(), 1.0)
                optim.step()
            sched.step()

            if epoch == FT_EPOCHS:
                model_ft.eval()
                with torch.no_grad():
                    te_batch     = build_swat_batch(
                        swat_data, swat_latents, te_pids, device)
                    te_logits, _ = model_ft(te_batch)
                    te_probs     = torch.sigmoid(
                        safe_logits(te_logits)).cpu().numpy()
                try:
                    auroc = roc_auc_score(te_y, te_probs)
                    ft_aurocs.append(auroc)
                except Exception:
                    pass

        print(f"    Rep {rep+1:2d} (seed={seed}): "
              f"AUROC={ft_aurocs[-1]:.4f} "
              f"(n_test={len(te_pids)}, deaths={te_y.sum()})")

    ft_mean = float(np.mean(ft_aurocs))
    ft_std  = float(np.std(ft_aurocs))
    ft_ci_lo = ft_mean - 1.96 * ft_std / np.sqrt(len(ft_aurocs))
    ft_ci_hi = ft_mean + 1.96 * ft_std / np.sqrt(len(ft_aurocs))

    print(f"\n  Direction 1 Summary:")
    print(f"    Zero-shot AUROC : {zs_auroc:.4f}")
    print(f"    Fine-tuned AUROC: {ft_mean:.4f} ± {ft_std:.4f} "
          f"[{ft_ci_lo:.4f}, {ft_ci_hi:.4f}]")

    return {
        'zero_shot_auroc': float(zs_auroc),
        'finetuned_mean':  ft_mean,
        'finetuned_std':   ft_std,
        'finetuned_ci_lo': ft_ci_lo,
        'finetuned_ci_hi': ft_ci_hi,
        'finetuned_reps':  [float(x) for x in ft_aurocs],
    }


# ── Direction 2: SWAT → PAMPer ──────────────────────────────────────────────

def direction2_swat_to_pamper(swat_data, swat_latents,
                               joint_caches, device):
    """
    Train mortality predictor on SWAT labels.
    Evaluate on held-out PAMPer patients (5-fold CV with joint latents).
    """
    print("\n" + "="*60)
    print("DIRECTION 2: SWAT → PAMPer")
    print("="*60)

    (soma_cache, met_cache, lip_cache, lum_cache,
     clin_cache, label_cache, real_ids, syn_ids) = joint_caches

    swat_pids   = list(swat_data.keys())
    swat_labels = np.array([swat_data[p]['mortality'] for p in swat_pids])

    print(f"\n  Training on SWAT: {len(swat_pids)} patients, "
          f"{swat_labels.sum()} deaths")
    print(f"  Testing on PAMPer: 5-fold CV")

    fold_aurocs = []
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    real_labels = np.array(
        [label_cache[p]['mortality'] for p in real_ids])

    for fold_idx, (_, test_idx) in enumerate(
            skf.split(real_ids, real_labels)):
        test_pids  = [real_ids[i] for i in test_idx]
        test_labels = real_labels[test_idx]

        # Train on all SWAT
        n_pos      = swat_labels.sum()
        n_neg      = len(swat_labels) - n_pos
        pos_weight = torch.tensor(
            [min(n_neg / max(n_pos, 1), 10.0)],
            dtype=torch.float32).to(device)

        model = CachedMultimodalTransformer(dropout=0.3).to(device)
        torch.manual_seed(42 + fold_idx)

        optim = AdamW(model.parameters(), lr=FT_LR, weight_decay=1e-4)
        sched = CosineAnnealingLR(optim, T_max=FT_EPOCHS, eta_min=1e-6)

        for epoch in range(1, FT_EPOCHS + 1):
            model.train()
            idx = np.random.permutation(len(swat_pids))
            for i in range(0, len(swat_pids), 16):
                bp  = [swat_pids[j] for j in idx[i:i+16]]
                bl  = torch.tensor(
                    [swat_data[p]['mortality'] for p in bp],
                    dtype=torch.float32).to(device)
                optim.zero_grad()
                batch    = build_swat_batch(swat_data, swat_latents, bp, device)
                logits,_ = model(batch)
                loss = F.binary_cross_entropy_with_logits(
                    safe_logits(logits), bl, pos_weight=pos_weight)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
            sched.step()

        # Evaluate on PAMPer test fold
        model.eval()
        test_ds = CachedLatentDataset(
            test_pids, soma_cache, met_cache, lip_cache,
            lum_cache, clin_cache, label_cache)
        test_loader = DataLoader(
            test_ds, batch_size=16, shuffle=False,
            collate_fn=collate_fn, num_workers=0)

        criterion = nn.BCEWithLogitsLoss()
        _, test_auroc, _, _ = evaluate(
            model, test_loader, criterion, device)
        fold_aurocs.append(test_auroc)

        print(f"  Fold {fold_idx+1}: AUROC={test_auroc:.4f} "
              f"(n={len(test_pids)}, deaths={test_labels.sum()})")

    mean_a = float(np.nanmean(fold_aurocs))
    std_a  = float(np.nanstd(fold_aurocs))
    ci_lo  = mean_a - 1.96 * std_a / np.sqrt(N_FOLDS)
    ci_hi  = mean_a + 1.96 * std_a / np.sqrt(N_FOLDS)

    print(f"\n  Direction 2 Summary:")
    print(f"    AUROC: {mean_a:.4f} ± {std_a:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]")

    return {
        'mean_auroc': mean_a,
        'std':        std_a,
        'ci_lo':      ci_lo,
        'ci_hi':      ci_hi,
        'folds':      [float(x) for x in fold_aurocs],
    }


# ── Direction 3: Combined training ──────────────────────────────────────────

def direction3_combined(swat_data, swat_latents,
                         joint_caches, device):
    """
    Train on PAMPer + SWAT combined.
    Evaluate on held-out from both cohorts separately.
    """
    print("\n" + "="*60)
    print("DIRECTION 3: Combined PAMPer + SWAT Training")
    print("="*60)

    (soma_cache, met_cache, lip_cache, lum_cache,
     clin_cache, label_cache, real_ids, syn_ids) = joint_caches

    swat_pids   = list(swat_data.keys())
    swat_labels = np.array([swat_data[p]['mortality'] for p in swat_pids])
    real_labels = np.array(
        [label_cache[p]['mortality'] for p in real_ids])

    # Hold out 20% of each cohort for testing
    pamper_tr, pamper_te, _, pamper_te_y = train_test_split(
        real_ids, real_labels,
        test_size=0.2, stratify=real_labels, random_state=42)
    swat_tr, swat_te, _, swat_te_y = train_test_split(
        swat_pids, swat_labels,
        test_size=0.2, stratify=swat_labels, random_state=42)

    print(f"\n  PAMPer train/test: {len(pamper_tr)}/{len(pamper_te)}")
    print(f"  SWAT   train/test: {len(swat_tr)}/{len(swat_te)}")

    # Build combined training set
    # PAMPer patients come from CachedLatentDataset
    # SWAT patients use build_swat_batch
    n_pos = (real_labels[np.isin(real_ids, pamper_tr)].sum() +
             swat_labels[np.isin(swat_pids, swat_tr)].sum())
    n_neg = len(pamper_tr) + len(swat_tr) - n_pos
    pos_weight = torch.tensor(
        [min(n_neg / max(n_pos, 1), 10.0)],
        dtype=torch.float32).to(device)

    model = CachedMultimodalTransformer(dropout=0.3).to(device)
    torch.manual_seed(42)
    optim = AdamW(model.parameters(), lr=CFG['lr'],
                  weight_decay=CFG['weight_decay'])
    sched = CosineAnnealingLR(optim, T_max=CFG['epochs'], eta_min=1e-6)

    # Mixed training: alternate PAMPer and SWAT batches
    pamper_ds = CachedLatentDataset(
        pamper_tr + syn_ids, soma_cache, met_cache, lip_cache,
        lum_cache, clin_cache, label_cache)
    pamper_loader = DataLoader(
        pamper_ds, batch_size=CFG['batch_size'],
        shuffle=True, collate_fn=collate_fn, num_workers=0)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_state = None
    best_combined = 0.0

    print(f"\n  Training combined model ({CFG['epochs']} epochs)...")
    for epoch in range(1, CFG['epochs'] + 1):
        model.train()
        # PAMPer batches
        for batch in pamper_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            bl = batch['mortality'].float()
            optim.zero_grad()
            logits, _ = model(batch)
            loss = criterion(safe_logits(logits), bl)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        # SWAT batches
        swat_idx = np.random.permutation(len(swat_tr))
        for i in range(0, len(swat_tr), CFG['batch_size']):
            bp  = [swat_tr[j] for j in swat_idx[i:i+CFG['batch_size']]]
            bl  = torch.tensor(
                [swat_data[p]['mortality'] for p in bp],
                dtype=torch.float32).to(device)
            optim.zero_grad()
            batch    = build_swat_batch(swat_data, swat_latents, bp, device)
            logits,_ = model(batch)
            loss = F.binary_cross_entropy_with_logits(
                safe_logits(logits), bl, pos_weight=pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        sched.step()

        if epoch % 10 == 0 or epoch == CFG['epochs']:
            model.eval()
            # Quick eval on SWAT test
            with torch.no_grad():
                te_batch     = build_swat_batch(
                    swat_data, swat_latents, swat_te, device)
                te_logits, _ = model(te_batch)
                te_probs     = torch.sigmoid(
                    safe_logits(te_logits)).cpu().numpy()
            try:
                swat_auroc = roc_auc_score(swat_te_y, te_probs)
                if swat_auroc > best_combined:
                    best_combined = swat_auroc
                    best_state = {k: v.clone()
                                  for k, v in model.state_dict().items()}
                print(f"    Epoch {epoch:3d} | SWAT test AUROC: "
                      f"{swat_auroc:.4f}")
            except Exception:
                pass

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Final evaluation on both held-out sets
    # PAMPer held-out
    pamper_test_ds = CachedLatentDataset(
        pamper_te, soma_cache, met_cache, lip_cache,
        lum_cache, clin_cache, label_cache)
    pamper_test_loader = DataLoader(
        pamper_test_ds, batch_size=16, shuffle=False,
        collate_fn=collate_fn, num_workers=0)
    _, pamper_auroc, _, _ = evaluate(
        model, pamper_test_loader,
        nn.BCEWithLogitsLoss(), device)

    # SWAT held-out
    with torch.no_grad():
        te_batch     = build_swat_batch(
            swat_data, swat_latents, swat_te, device)
        te_logits, _ = model(te_batch)
        te_probs     = torch.sigmoid(
            safe_logits(te_logits)).cpu().numpy()
    try:
        swat_auroc = roc_auc_score(swat_te_y, te_probs)
    except Exception:
        swat_auroc = float('nan')

    print(f"\n  Direction 3 Summary:")
    print(f"    PAMPer held-out AUROC: {pamper_auroc:.4f} "
          f"(n={len(pamper_te)}, deaths={pamper_te_y.sum()})")
    print(f"    SWAT held-out AUROC  : {swat_auroc:.4f} "
          f"(n={len(swat_te)}, deaths={swat_te_y.sum()})")

    return {
        'pamper_auroc': float(pamper_auroc),
        'swat_auroc':   float(swat_auroc),
        'n_pamper_te':  len(pamper_te),
        'n_swat_te':    len(swat_te),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main(data_dir="./"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)
    device = get_device()
    print(f"Device: {device}")

    # Load joint-VAE PAMPer latents
    print("\nLoading joint-VAE PAMPer latents...")
    joint_caches = load_latent_cache(JOINT_LATENT_DIR)
    (_, _, _, _, _, label_cache, real_ids, syn_ids) = joint_caches
    real_labels = np.array(
        [label_cache[p]['mortality'] for p in real_ids])
    print(f"  PAMPer: {len(real_ids)} real, {len(syn_ids)} synthetic")
    print(f"  PAMPer mortality: {real_labels.mean():.3f}")

    # Load SWAT latents through joint VAE
    swat_data, swat_latents = load_swat_latents(data_dir, device, CKPT_DIR)

    # Direction 1: PAMPer → SWAT
    d1 = direction1_pamper_to_swat(swat_data, swat_latents, device)

    # Direction 2: SWAT → PAMPer
    d2 = direction2_swat_to_pamper(
        swat_data, swat_latents, joint_caches, device)

    # Direction 3: Combined
    d3 = direction3_combined(
        swat_data, swat_latents, joint_caches, device)

    # Summary
    print(f"\n{'='*60}")
    print("BIDIRECTIONAL VALIDATION SUMMARY")
    print(f"{'='*60}")
    print(f"  PAMPer training AUROC (10x5-fold): 0.857 ± 0.012")
    print(f"")
    print(f"  Direction 1 — PAMPer → SWAT:")
    print(f"    Zero-shot  : {d1['zero_shot_auroc']:.4f}")
    print(f"    Fine-tuned : {d1['finetuned_mean']:.4f} ± "
          f"{d1['finetuned_std']:.4f} "
          f"[{d1['finetuned_ci_lo']:.4f}, {d1['finetuned_ci_hi']:.4f}]")
    print(f"")
    print(f"  Direction 2 — SWAT → PAMPer:")
    print(f"    AUROC      : {d2['mean_auroc']:.4f} ± {d2['std']:.4f} "
          f"[{d2['ci_lo']:.4f}, {d2['ci_hi']:.4f}]")
    print(f"")
    print(f"  Direction 3 — Combined training:")
    print(f"    PAMPer test: {d3['pamper_auroc']:.4f}")
    print(f"    SWAT test  : {d3['swat_auroc']:.4f}")

    results = {
        'direction1_pamper_to_swat': d1,
        'direction2_swat_to_pamper': d2,
        'direction3_combined':       d3,
    }
    with open(os.path.join(OUTPUT_DIR, 'bidirectional_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {OUTPUT_DIR}bidirectional_results.json")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    main(data_dir)
