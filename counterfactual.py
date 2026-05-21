"""
counterfactual.py — Counterfactual Proteomics for Therapeutic Target Identification
====================================================================================
"If this non-survivor's proteome had looked different at admission,
 would they have survived? Which proteins would need to change?"

This module answers that question using gradient-based counterfactual
optimization in the VAE latent space.

Pipeline:
    1. Encode each non-survivor's tp0 proteome → latent vector z
    2. Find the "survivor centroid" in latent space
    3. For each non-survivor: find minimal perturbation δ such that
       the mortality predictor flips from high-risk to low-risk
    4. Decode (z + δ) back to protein space
    5. Δprotein = decoded_counterfactual - original_proteome
    6. Rank proteins by |Δprotein| — these are the therapeutic candidates
    7. Cross-reference with known druggable targets

Outputs (saved to outputs/counterfactual/):
    counterfactual_proteins.csv    — ranked therapeutic candidates
    patient_counterfactuals.csv    — per-patient risk shift + top proteins
    latent_space_map.csv           — 2D UMAP of latent space (survivors vs non)
    counterfactual_summary.json    — aggregate statistics

Usage:
    python3 counterfactual.py ./
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

from encoders import (SomaLogicVAE, load_pretrained_encoders)
from dataset  import build_datasets
from train_cached import (
    CachedMultimodalTransformer,
    load_latent_cache,
    get_fold_splits,
    safe_logits,
    get_device,
)

# ── Config ────────────────────────────────────────────────────────────────────

CKPT_DIR   = "outputs/checkpoints/"
LATENT_DIR = "outputs/latents/"
OUTPUT_DIR = "outputs/counterfactual/"
DATA_DIR   = "./"
FOLD       = 0        # use fold 0 trained model

# Counterfactual optimization settings
CF_LR           = 0.05    # step size for latent perturbation
CF_STEPS        = 500     # max gradient steps
CF_TARGET_PROB  = 0.30    # push non-survivors below this mortality probability
CF_LAMBDA_DIST  = 0.5     # regularization: penalize large latent perturbations
CF_PATIENCE     = 50      # stop early if loss hasn't improved


# ── Load protein names ────────────────────────────────────────────────────────

def load_protein_names(data_dir):
    """Load SomaLogic protein column names from protein_importance.csv if available,
    otherwise fall back to reading the raw Excel file."""

    imp_path = "outputs/interpretation/protein_importance.csv"
    if os.path.exists(imp_path):
        df = pd.read_csv(imp_path)
        return df["feature"].tolist()

    # Fallback: read from raw Excel
    soma_path = os.path.join(data_dir, "PAMPer_External_data.xlsx")
    df = pd.read_excel(soma_path, sheet_name="SomaLogic", nrows=1)
    skip = {"patient_id", "Subject_ID", "Time", "Timepoint",
            "timepoint", "GenID", "Time point", "Study_ID"}
    cols = [c for c in df.columns if c not in skip and not any(
        kw in str(c).lower() for kw in
        ["age", "gender", "race", "iss", "ais", "death",
         "icu", "blood", "shock", "plasma", "injury", "id", "study"]
    )]
    return cols


# ── Build model input dict from caches ───────────────────────────────────────

def build_patient_batch(pid, caches, device):
    """Build a single-patient batch dict from cached latents."""
    (soma_cache, met_cache, lip_cache, lum_cache,
     clin_cache, label_cache, _, _) = caches

    info = label_cache[pid]
    return {
        "somalogic":        soma_cache[pid].unsqueeze(0).to(device),
        "somalogic_mask":   info["somalogic_mask"].unsqueeze(0).to(device),
        "metabolon":        met_cache[pid].unsqueeze(0).to(device),
        "metabolon_mask":   info["metabolon_mask"].unsqueeze(0).to(device),
        "lipidomics":       lip_cache[pid].unsqueeze(0).to(device),
        "lipidomics_mask":  info["lipidomics_mask"].unsqueeze(0).to(device),
        "luminex":          lum_cache[pid].unsqueeze(0).to(device),
        "luminex_mask":     info["luminex_mask"].unsqueeze(0).to(device),
        "clinical":         clin_cache[pid].unsqueeze(0).to(device),
    }


# ── Compute survivor centroid in SomaLogic latent space ──────────────────────

def compute_survivor_centroid(caches, real_ids, label_cache, device):
    """
    Compute mean latent vector (tp0 only) for survivors and non-survivors.
    Returns (survivor_centroid, nonsurvivor_centroid) as tensors.
    """
    (soma_cache, _, _, _, _, _, _, _) = caches

    survivor_latents     = []
    nonsurvivor_latents  = []

    for pid in real_ids:
        if label_cache[pid]["is_synthetic"] == 1:
            continue
        z_tp0 = soma_cache[pid][0]   # tp0 latent, shape (128,)
        if label_cache[pid]["mortality"] == 0:
            survivor_latents.append(z_tp0)
        else:
            nonsurvivor_latents.append(z_tp0)

    surv_centroid    = torch.stack(survivor_latents).mean(dim=0).to(device)
    nonsurv_centroid = torch.stack(nonsurvivor_latents).mean(dim=0).to(device)

    print(f"  Survivor centroid    : {len(survivor_latents)} patients")
    print(f"  Non-survivor centroid: {len(nonsurvivor_latents)} patients")

    # Euclidean distance between centroids
    dist = torch.norm(surv_centroid - nonsurv_centroid).item()
    print(f"  Centroid separation  : {dist:.4f} (Euclidean in latent space)")

    return surv_centroid, nonsurv_centroid


# ── Counterfactual optimization ───────────────────────────────────────────────

def find_counterfactual(
    pid,
    caches,
    model,
    soma_vae,
    survivor_centroid,
    device,
    target_prob=CF_TARGET_PROB,
    lr=CF_LR,
    max_steps=CF_STEPS,
    lambda_dist=CF_LAMBDA_DIST,
    patience=CF_PATIENCE,
):
    """
    Find the minimal perturbation to a patient's SomaLogic latent vector
    that flips their predicted mortality from high to low risk.

    The optimization problem:
        minimize  ||δ||²                    (minimal perturbation)
        subject to  σ(f(z + δ)) ≤ target   (flip the prediction)

    Implemented as unconstrained optimization:
        L = prediction_loss + λ * ||δ||²

    where prediction_loss pushes the predicted probability below target_prob.

    Args:
        pid:               patient ID string
        caches:            tuple from load_latent_cache()
        model:             trained CachedMultimodalTransformer
        soma_vae:          trained SomaLogicVAE (for decoding)
        survivor_centroid: mean latent of survivors (128,) tensor
        device:            torch device
        target_prob:       target mortality probability (default 0.30)
        lr:                gradient step size
        max_steps:         maximum optimization steps
        lambda_dist:       regularization weight
        patience:          early stopping patience

    Returns dict with:
        original_prob:      float — baseline mortality probability
        cf_prob:            float — counterfactual mortality probability
        delta_z:            tensor (128,) — latent perturbation
        cf_proteins:        tensor (7596,) — counterfactual protein values
        original_proteins:  tensor (7596,) — original protein values
        protein_delta:      tensor (7596,) — change needed per protein
        converged:          bool — did optimization reach target?
        n_steps:            int — steps taken
    """
    (soma_cache, _, _, _, _, label_cache, _, _) = caches

    model.eval()
    soma_vae.eval()

    # ── Get original latent and prediction ──
    z_original = soma_cache[pid][0].clone().to(device)  # tp0 latent (128,)

    batch = build_patient_batch(pid, caches, device)

    with torch.no_grad():
        logits, _ = model(batch)
        original_prob = torch.sigmoid(safe_logits(logits)).item()

    # ── If already low risk, no counterfactual needed ──
    if original_prob <= target_prob:
        with torch.no_grad():
            orig_proteins = soma_vae.decode(z_original.unsqueeze(0)).squeeze(0)
        return {
            "original_prob":     original_prob,
            "cf_prob":           original_prob,
            "delta_z":           torch.zeros_like(z_original),
            "cf_proteins":       orig_proteins,
            "original_proteins": orig_proteins,
            "protein_delta":     torch.zeros_like(orig_proteins),
            "converged":         True,
            "n_steps":           0,
            "already_low_risk":  True,
        }

    # ── Initialize perturbation δ toward survivor centroid ──
    # Warm start: initialize δ as a small step toward survivor centroid
    direction = F.normalize(
        (survivor_centroid - z_original).unsqueeze(0), dim=1).squeeze(0)
    delta_z = (direction * 0.1).detach().clone().requires_grad_(True)

    optimizer = torch.optim.Adam([delta_z], lr=lr)

    best_prob     = original_prob
    best_delta    = delta_z.detach().clone()
    best_step     = 0
    no_improve    = 0
    converged     = False

    for step in range(max_steps):
        optimizer.zero_grad()

        # Build perturbed batch — only somalogic latent changes
        z_perturbed = z_original + delta_z   # (128,)

        # Replace somalogic tp0 in the batch
        perturbed_batch = {k: v.clone() for k, v in batch.items()}

        # Reconstruct full temporal sequence with perturbed tp0
        soma_seq = soma_cache[pid].clone().to(device)   # (3, 128)
        soma_seq[0] = z_perturbed                        # perturb tp0 only
        perturbed_batch["somalogic"] = soma_seq.unsqueeze(0)  # (1, 3, 128)

        logits, _ = model(perturbed_batch)
        prob = torch.sigmoid(safe_logits(logits))        # (1,)

        # Loss: push probability below target + regularize perturbation size
        pred_loss = F.relu(prob - target_prob)           # 0 if already below target
        reg_loss  = lambda_dist * (delta_z ** 2).mean()
        loss      = pred_loss + reg_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_([delta_z], max_norm=2.0)
        optimizer.step()

        current_prob = prob.item()

        # Track best
        if current_prob < best_prob:
            best_prob  = current_prob
            best_delta = delta_z.detach().clone()
            best_step  = step
            no_improve = 0
        else:
            no_improve += 1

        # Check convergence
        if current_prob <= target_prob:
            converged  = True
            best_delta = delta_z.detach().clone()
            best_prob  = current_prob
            break

        # Early stopping
        if no_improve >= patience:
            break

    # ── Decode counterfactual back to protein space ──
    with torch.no_grad():
        z_cf          = (z_original + best_delta).unsqueeze(0)  # (1, 128)
        cf_proteins   = soma_vae.decode(z_cf).squeeze(0)         # (7596,)
        orig_proteins = soma_vae.decode(
            z_original.unsqueeze(0)).squeeze(0)                  # (7596,)
        protein_delta = cf_proteins - orig_proteins               # (7596,)

    return {
        "original_prob":     original_prob,
        "cf_prob":           best_prob,
        "delta_z":           best_delta.cpu(),
        "cf_proteins":       cf_proteins.cpu(),
        "original_proteins": orig_proteins.cpu(),
        "protein_delta":     protein_delta.cpu(),
        "converged":         converged,
        "n_steps":           best_step,
        "already_low_risk":  False,
    }


# ── Aggregate across all non-survivors ───────────────────────────────────────

def run_population_counterfactuals(
    model,
    soma_vae,
    caches,
    test_ids,
    survivor_centroid,
    protein_names,
    device,
):
    """
    Run counterfactual optimization for all non-survivors in test set.
    Aggregate protein deltas to find population-level therapeutic candidates.
    """
    (_, _, _, _, _, label_cache, _, _) = caches

    # Only run on non-survivors (the patients we want to "rescue")
    nonsurvivor_ids = [
        pid for pid in test_ids
        if label_cache[pid]["mortality"] == 1
        and label_cache[pid]["is_synthetic"] == 0
    ]
    survivor_ids = [
        pid for pid in test_ids
        if label_cache[pid]["mortality"] == 0
        and label_cache[pid]["is_synthetic"] == 0
    ]

    print(f"\n  Non-survivors to analyze: {len(nonsurvivor_ids)}")
    print(f"  Survivors (reference):    {len(survivor_ids)}")

    all_deltas      = []   # protein-space deltas across patients
    patient_results = []

    for i, pid in enumerate(nonsurvivor_ids):
        result = find_counterfactual(
            pid=pid,
            caches=caches,
            model=model,
            soma_vae=soma_vae,
            survivor_centroid=survivor_centroid,
            device=device,
        )

        # Store per-patient result
        prob_shift = result["original_prob"] - result["cf_prob"]
        top_idx    = result["protein_delta"].abs().argsort(descending=True)[:5]
        top_proteins = [
            f"{protein_names[j] if j < len(protein_names) else f'protein_{j}'}"
            f"({result['protein_delta'][j]:+.3f})"
            for j in top_idx.tolist()
        ]

        patient_results.append({
            "patient_id":        pid,
            "original_prob":     round(result["original_prob"], 4),
            "cf_prob":           round(result["cf_prob"], 4),
            "prob_shift":        round(prob_shift, 4),
            "converged":         result["converged"],
            "n_steps":           result["n_steps"],
            "latent_perturbation_norm": round(
                result["delta_z"].norm().item(), 4),
            "top_5_proteins":    " | ".join(top_proteins),
            "already_low_risk":  result.get("already_low_risk", False),
        })

        if not result.get("already_low_risk", False):
            all_deltas.append(result["protein_delta"])

        status = "✓ converged" if result["converged"] else f"best={result['cf_prob']:.3f}"
        print(f"  [{i+1:2d}/{len(nonsurvivor_ids)}] {pid} | "
              f"risk {result['original_prob']:.3f} → {result['cf_prob']:.3f} | "
              f"{status}")

    return patient_results, all_deltas


# ── Population-level protein ranking ─────────────────────────────────────────

def rank_therapeutic_candidates(all_deltas, protein_names):
    """
    Aggregate protein deltas across all non-survivors to find
    population-level therapeutic candidates.

    For each protein:
        - mean_delta:     average change needed (sign = direction)
        - abs_mean_delta: average magnitude of change needed
        - consistency:    fraction of patients where this protein needs to change
                         in the same direction (↑ or ↓)
        - importance:     abs_mean_delta × consistency (composite score)
    """
    if not all_deltas:
        print("  No counterfactual deltas to aggregate.")
        return pd.DataFrame()

    delta_matrix = torch.stack(all_deltas).numpy()   # (n_patients, 7596)
    n_patients   = delta_matrix.shape[0]
    n_proteins   = delta_matrix.shape[1]
    n_names      = min(len(protein_names), n_proteins)

    mean_delta     = delta_matrix.mean(axis=0)          # (7596,)
    abs_mean_delta = np.abs(delta_matrix).mean(axis=0)  # (7596,)

    # Consistency: what fraction of patients need this protein to go UP?
    frac_up   = (delta_matrix > 0).mean(axis=0)
    frac_down = (delta_matrix < 0).mean(axis=0)
    consistency = np.maximum(frac_up, frac_down)        # (7596,)

    # Direction: UP if majority need increase, DOWN if majority need decrease
    direction = np.where(frac_up > frac_down, "↑ INCREASE", "↓ DECREASE")

    # Composite importance score
    importance = abs_mean_delta * consistency

    df = pd.DataFrame({
        "feature":         protein_names[:n_names],
        "mean_delta":      mean_delta[:n_names],
        "abs_mean_delta":  abs_mean_delta[:n_names],
        "consistency":     consistency[:n_names],
        "direction":       direction[:n_names],
        "importance_score": importance[:n_names],
        "frac_patients_up":   frac_up[:n_names],
        "frac_patients_down": frac_down[:n_names],
    })

    df = df.sort_values("importance_score", ascending=False).reset_index(drop=True)
    df["rank"] = df.index + 1

    return df


# ── UMAP latent space visualization ──────────────────────────────────────────

def compute_latent_umap(caches, real_ids, label_cache):
    """
    Compute 2D UMAP of tp0 SomaLogic latent space.
    Returns DataFrame with umap_x, umap_y, mortality, patient_id.
    """
    try:
        import umap
    except ImportError:
        print("  UMAP not installed — skipping latent map.")
        print("  Install with: pip3 install umap-learn")
        return None

    (soma_cache, _, _, _, _, _, _, _) = caches

    real_pids = [p for p in real_ids
                 if label_cache[p]["is_synthetic"] == 0]
    latents   = torch.stack([soma_cache[p][0] for p in real_pids]).numpy()
    labels    = [label_cache[p]["mortality"] for p in real_pids]

    print(f"  Running UMAP on {len(real_pids)} patients × 128 latent dims...")
    reducer   = umap.UMAP(n_components=2, random_state=42, n_neighbors=15)
    embedding = reducer.fit_transform(latents)

    df = pd.DataFrame({
        "patient_id": real_pids,
        "umap_x":     embedding[:, 0],
        "umap_y":     embedding[:, 1],
        "mortality":  labels,
        "outcome":    ["Non-survivor" if m == 1 else "Survivor" for m in labels],
    })
    return df


# ── Plot counterfactual results ───────────────────────────────────────────────

def plot_counterfactual_results(df_proteins, patient_results, output_dir):
    """Generate publication-ready figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return

    # ── Figure 1: Top 20 therapeutic candidates ──
    top20 = df_proteins.head(20)
    colors = ["#ef4444" if d == "↑ INCREASE" else "#3b82f6"
              for d in top20["direction"]]

    fig, ax = plt.subplots(figsize=(11, 7))
    bars = ax.barh(range(len(top20)),
                   top20["importance_score"].values,
                   color=colors, alpha=0.85)
    ax.set_yticks(range(len(top20)))
    ax.set_yticklabels(top20["feature"].values, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Therapeutic Importance Score\n(|mean Δprotein| × consistency)",
                  fontsize=10)
    ax.set_title("Top 20 Counterfactual Therapeutic Candidates\n"
                 "Proteins whose change would most reduce predicted mortality",
                 fontsize=11, fontweight="bold")

    red_patch  = mpatches.Patch(color="#ef4444", alpha=0.85,
                                label="Needs to INCREASE for survival")
    blue_patch = mpatches.Patch(color="#3b82f6", alpha=0.85,
                                label="Needs to DECREASE for survival")
    ax.legend(handles=[red_patch, blue_patch], fontsize=9,
              loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "top20_therapeutic_candidates.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: top20_therapeutic_candidates.png")

    # ── Figure 2: Risk shift per patient ──
    df_pat = pd.DataFrame(patient_results)
    df_pat = df_pat[~df_pat["already_low_risk"]].copy()

    if len(df_pat) > 0:
        df_pat = df_pat.sort_values("original_prob", ascending=False)
        x = range(len(df_pat))

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(x, df_pat["original_prob"].values,
               color="#ef4444", alpha=0.6, label="Original risk")
        ax.bar(x, df_pat["cf_prob"].values,
               color="#22c55e", alpha=0.8, label="Counterfactual risk")
        ax.axhline(y=0.5, color="black", linestyle="--",
                   linewidth=1, alpha=0.5, label="Decision threshold (0.5)")
        ax.axhline(y=CF_TARGET_PROB, color="gray", linestyle=":",
                   linewidth=1, alpha=0.7,
                   label=f"CF target ({CF_TARGET_PROB})")
        ax.set_xlabel("Non-survivor patients (sorted by original risk)",
                      fontsize=10)
        ax.set_ylabel("Predicted mortality probability", fontsize=10)
        ax.set_title("Counterfactual Risk Reduction per Patient\n"
                     "Original vs. counterfactual predicted mortality",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "patient_risk_shift.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: patient_risk_shift.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir=DATA_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    # ── Load caches ──
    caches = load_latent_cache(LATENT_DIR)
    (soma_cache, met_cache, lip_cache, lum_cache,
     clin_cache, label_cache, real_ids, syn_ids) = caches

    # ── Get test split ──
    train_ids, val_ids, test_ids = get_fold_splits(real_ids, label_cache, FOLD)
    test_ids = [p for p in test_ids if label_cache[p]["is_synthetic"] == 0]
    print(f"Test patients (fold {FOLD}): {len(test_ids)}")

    # ── Load trained mortality predictor ──
    print(f"\nLoading trained model (fold {FOLD})...")
    model = CachedMultimodalTransformer(dropout=0.3).to(device)
    ckpt  = torch.load(
        os.path.join(CKPT_DIR, f"model_cached_fold{FOLD}.pt"),
        map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print("  Model loaded.")

    # ── Load pretrained SomaLogic VAE ──
    print("\nLoading SomaLogic VAE...")
    _, _, _, feature_dims = build_datasets(data_dir=data_dir, fold=0)
    soma_vae = SomaLogicVAE(feature_dims["somalogic"], 128)
    soma_vae.load_state_dict(
        torch.load(os.path.join(CKPT_DIR, "somalogic_vae.pt"),
                   map_location=device))
    soma_vae = soma_vae.to(device)
    soma_vae.eval()
    print("  SomaLogic VAE loaded.")

    # ── Load protein names ──
    print("\nLoading protein names...")
    protein_names = load_protein_names(data_dir)
    print(f"  {len(protein_names)} protein names loaded.")

    # ── Compute survivor/non-survivor centroids ──
    print("\n" + "="*55)
    print("LATENT SPACE ANALYSIS")
    print("="*55)
    survivor_centroid, nonsurvivor_centroid = compute_survivor_centroid(
        caches, real_ids, label_cache, device)

    # ── UMAP visualization ──
    print("\nComputing latent space UMAP...")
    df_umap = compute_latent_umap(caches, real_ids, label_cache)
    if df_umap is not None:
        df_umap.to_csv(
            os.path.join(OUTPUT_DIR, "latent_space_umap.csv"), index=False)
        print(f"  Saved: latent_space_umap.csv")

    # ── Run counterfactual optimization ──
    print("\n" + "="*55)
    print("COUNTERFACTUAL OPTIMIZATION")
    print("="*55)
    print(f"  Target: reduce mortality probability to ≤ {CF_TARGET_PROB}")
    print(f"  Method: gradient descent in SomaLogic latent space")
    print(f"  Max steps per patient: {CF_STEPS}")

    patient_results, all_deltas = run_population_counterfactuals(
        model=model,
        soma_vae=soma_vae,
        caches=caches,
        test_ids=test_ids,
        survivor_centroid=survivor_centroid,
        protein_names=protein_names,
        device=device,
    )

    # ── Save per-patient results ──
    df_patients = pd.DataFrame(patient_results)
    df_patients.to_csv(
        os.path.join(OUTPUT_DIR, "patient_counterfactuals.csv"), index=False)

    # Summary stats
    converged     = df_patients["converged"].sum()
    already_low   = df_patients["already_low_risk"].sum()
    mean_shift    = df_patients[~df_patients["already_low_risk"]]["prob_shift"].mean()
    n_nonsurv     = (~df_patients["already_low_risk"]).sum()

    print(f"\n  Results:")
    print(f"  Patients already low-risk : {already_low}")
    print(f"  Non-survivors analyzed    : {n_nonsurv}")
    print(f"  Counterfactuals converged : {converged}/{n_nonsurv}")
    print(f"  Mean risk reduction       : {mean_shift:.4f} "
          f"({mean_shift*100:.1f} percentage points)")

    # ── Aggregate therapeutic candidates ──
    print("\n" + "="*55)
    print("THERAPEUTIC CANDIDATE RANKING")
    print("="*55)

    df_proteins = rank_therapeutic_candidates(all_deltas, protein_names)

    if len(df_proteins) > 0:
        df_proteins.to_csv(
            os.path.join(OUTPUT_DIR, "counterfactual_proteins.csv"), index=False)

        print(f"\n  Top 20 therapeutic candidates:")
        print(f"  {'Rank':4s} {'Protein':50s} {'Direction':12s} "
              f"{'Score':8s} {'Consistency':12s}")
        print("  " + "-"*90)
        for _, row in df_proteins.head(20).iterrows():
            name = str(row["feature"])[:48]
            print(f"  {int(row['rank']):4d} {name:50s} {row['direction']:12s} "
                  f"{row['importance_score']:8.4f} {row['consistency']:12.1%}")

    # ── Generate plots ──
    print("\nGenerating figures...")
    plot_counterfactual_results(df_proteins, patient_results, OUTPUT_DIR)

    # ── Save summary JSON ──
    summary = {
        "fold":                  FOLD,
        "n_test_patients":       len(test_ids),
        "n_nonsurvivor_analyzed": int(n_nonsurv),
        "n_converged":           int(converged),
        "convergence_rate":      float(converged / max(n_nonsurv, 1)),
        "mean_risk_reduction":   float(mean_shift) if not np.isnan(mean_shift) else 0,
        "cf_target_prob":        CF_TARGET_PROB,
        "top_10_candidates": df_proteins.head(10)[
            ["rank", "feature", "direction", "importance_score", "consistency"]
        ].to_dict("records") if len(df_proteins) > 0 else [],
    }

    with open(os.path.join(OUTPUT_DIR, "counterfactual_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print("COUNTERFACTUAL ANALYSIS COMPLETE")
    print(f"{'='*55}")
    print(f"  Outputs saved to: {OUTPUT_DIR}")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
        print(f"    {fname:45s} {size/1024:.1f} KB")

    print(f"\n  KEY RESULT:")
    print(f"  The model identifies proteins that, if modulated, would")
    print(f"  computationally reduce mortality risk in non-surviving patients.")
    print(f"  These are your in-silico therapeutic candidates.")
    if len(df_proteins) > 0:
        top3 = df_proteins.head(3)
        for _, row in top3.iterrows():
            print(f"    #{int(row['rank'])}: {row['feature']} "
                  f"— needs to {row['direction']} "
                  f"(consistent in {row['consistency']:.0%} of patients)")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    main(data_dir=data_dir)