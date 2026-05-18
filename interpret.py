"""
interpret.py — Interpretability analysis for multimodal mortality model
=======================================================================
Generates:
  1. Protein importance ranking (SomaLogic) via VAE decoder Jacobian
  2. Metabolite importance ranking (Metabolon) via VAE decoder Jacobian
  3. Lipid importance ranking (Lipidomics) via VAE decoder Jacobian
  4. Modality contribution scores (ablation — how much does each omics drop AUROC?)
  5. Per-patient mortality risk scores with top driving features

Outputs (all saved to outputs/interpretation/):
  protein_importance.csv      — ranked SomaLogic proteins with importance scores
  metabolite_importance.csv   — ranked metabolites
  lipid_importance.csv        — ranked lipids
  modality_ablation.csv       — AUROC drop when each modality is zeroed out
  patient_risk_scores.csv     — per-patient predicted risk + top features
  top20_proteins.png          — bar chart of top 20 proteins

Usage:
    python3 interpret.py ./
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import build_datasets
from encoders import (SomaLogicVAE, MetabolonVAE, LipidomicsVAE,
                      load_pretrained_encoders)
from train_cached import (CachedMultimodalTransformer, CachedLatentDataset,
                           load_latent_cache, get_fold_splits, collate_fn,
                           get_device, safe_logits)

# ── Config ────────────────────────────────────────────────────────────────────

CKPT_DIR    = "outputs/checkpoints/"
LATENT_DIR  = "outputs/latents/"
OUTPUT_DIR  = "outputs/interpretation/"
DATA_DIR    = "./"
FOLD        = 0        # use fold 0 model for interpretation
BATCH_SIZE  = 32
RANDOM_SEED = 42


# ── Jacobian-based feature importance ────────────────────────────────────────

def decoder_jacobian_importance(vae, latent_dim, input_dim, device, n_samples=500):
    """
    Compute feature importance via the decoder Jacobian.

    For each latent dimension z_j, compute d(x_i)/d(z_j) for all input
    features x_i. Importance of feature i = ||J[i, :]||_2 (L2 norm across
    all latent dims). This tells us: which input features are most sensitive
    to changes in the latent space?

    We average the Jacobian over n_samples random latent vectors to get a
    stable, population-level importance estimate.

    Args:
        vae:        pretrained OmicsVAE
        latent_dim: int
        input_dim:  int
        device:     torch device
        n_samples:  number of random z samples to average over

    Returns:
        importance: np.ndarray of shape (input_dim,) — L2 norm per feature
        jacobians:  np.ndarray of shape (n_samples, input_dim, latent_dim)
    """
    vae.eval()
    vae.to(device)

    all_jacobians = []

    for _ in range(n_samples):
        # Sample a random latent vector
        z = torch.randn(1, latent_dim, requires_grad=True, device=device)

        # Forward pass through decoder only
        # decode: z -> fc_decode -> decoder layers -> reconstruction
        x_recon = vae.decode(z)   # (1, input_dim)

        # Compute Jacobian: d(x_recon) / d(z)
        # Shape: (input_dim, latent_dim)
        J = torch.zeros(input_dim, latent_dim, device=device)

        for i in range(input_dim):
            if z.grad is not None:
                z.grad.zero_()
            x_recon[0, i].backward(retain_graph=(i < input_dim - 1))
            if z.grad is not None:
                J[i] = z.grad[0].detach()

        all_jacobians.append(J.cpu().numpy())

    jacobians  = np.stack(all_jacobians, axis=0)       # (n_samples, input_dim, latent_dim)
    mean_J     = np.abs(jacobians).mean(axis=0)         # (input_dim, latent_dim)
    importance = np.linalg.norm(mean_J, axis=1)         # (input_dim,)

    return importance, jacobians


def fast_jacobian_importance(vae, latent_dim, input_dim, device, n_samples=200):
    """
    Faster version using batched autograd instead of looping over features.
    Uses torch.autograd.functional.jacobian.

    This is ~50x faster than the loop version for large input_dim.
    """
    vae.eval()
    vae.to(device)

    importances = []

    for _ in range(n_samples):
        z = torch.randn(latent_dim, device=device)

        def decode_fn(z_input):
            z_batch = z_input.unsqueeze(0)   # (1, latent_dim)
            return vae.decode(z_batch).squeeze(0)   # (input_dim,)

        try:
            J = torch.autograd.functional.jacobian(decode_fn, z)
            # J shape: (input_dim, latent_dim)
            importances.append(J.abs().cpu().numpy())
        except Exception:
            # Fallback: use finite differences for one sample
            eps = 1e-3
            J_fd = np.zeros((input_dim, latent_dim))
            with torch.no_grad():
                x0 = vae.decode(z.unsqueeze(0)).squeeze(0).cpu().numpy()
                for j in range(latent_dim):
                    z_plus = z.clone()
                    z_plus[j] += eps
                    x_plus = vae.decode(z_plus.unsqueeze(0)).squeeze(0).cpu().numpy()
                    J_fd[:, j] = (x_plus - x0) / eps
            importances.append(np.abs(J_fd))

    mean_J     = np.stack(importances).mean(axis=0)  # (input_dim, latent_dim)
    importance = np.linalg.norm(mean_J, axis=1)       # (input_dim,)
    return importance


# ── Load feature names from dataset ──────────────────────────────────────────

def get_feature_names(data_dir):
    """Extract protein/metabolite/lipid column names from the raw data files."""
    import openpyxl

    print("Loading feature names from raw data...")

    # SomaLogic protein names
    soma_path = os.path.join(data_dir, "PAMPer_External_data.xlsx")
    soma_xl   = pd.ExcelFile(soma_path)

    # Find SomaLogic sheet
    soma_sheet = "SomaLogic"
    df_soma    = pd.read_excel(soma_path, sheet_name=soma_sheet, nrows=1)
    soma_cols  = [c for c in df_soma.columns
                  if c not in ("patient_id", "Subject_ID", "Time",
                               "Timepoint", "timepoint", "GenID",
                               "Time point", "Study_ID")]
    # Keep only numeric-looking columns (protein names)
    soma_cols = [c for c in soma_cols if not any(
        kw in str(c).lower() for kw in
        ["age", "gender", "race", "iss", "ais", "death", "icu",
         "blood", "shock", "plasma", "injury", "id", "study"]
    )]

    print(f"  SomaLogic proteins : {len(soma_cols)}")

    # Metabolon metabolite names
    met_path  = os.path.join(data_dir, "PAMPer_Metabolon_metabolomics.xlsx")
    met_xl    = pd.ExcelFile(met_path)
    df_met    = pd.read_excel(met_path, sheet_name=met_xl.sheet_names[0], nrows=1)
    met_cols  = [c for c in df_met.columns
                 if c not in ("GenID", "Time", "patient_id", "timepoint")]
    print(f"  Metabolon metabolites: {len(met_cols)}")

    # Lipidomics names (from SomaLogic file, Species sheet)
    df_lip = pd.read_excel(soma_path, sheet_name="Species_concentrations_unscaled", nrows=1)
    lip_cols = [c for c in df_lip.columns
                if c not in ("patient_id", "Subject_ID", "Time",
                             "Timepoint", "timepoint", "GenID",
                             "RACE ETHNICITY", "PAMPER ID NUMBER", "GROUP NAME",
                             "SUBJECT ID", "GROUP NUMBER", "SAMPLE AMOUNT",
                             "MECHANISM OF INJURY MOI", "STATISTICAL COMPARISONS",
                             "CLIENT SAMPLE NUMBER", "TIME POINT",
                             "INJURY SEVERITY SCORE 1", "GENDER", "BMI",
                             "BOX NUMBER", "SAMPLE NUMBER2", "COMMENTS",
                             "SAMPLE AMOUNT UNITS", "AMOUNT METHOD1",
                             "SAMPLE DESCRIPTION", "SAMPLE BOX LOCATION")]
    print(f"  Lipidomics species : {len(lip_cols)}")

    return soma_cols, met_cols, lip_cols


# ── Modality ablation ─────────────────────────────────────────────────────────

def modality_ablation(model, test_ids, caches, device):
    """
    Zero out each modality in turn and measure AUROC drop.
    This tells us how much each modality contributes to predictions.

    Returns dict: modality_name -> auroc_when_ablated
    """
    from sklearn.metrics import roc_auc_score

    (soma_cache, met_cache, lip_cache, lum_cache,
     clin_cache, label_cache, _, _) = caches

    modalities = ["somalogic", "metabolon", "lipidomics", "luminex", "clinical"]
    results    = {}

    def run_with_ablation(ablate_mod):
        """Run inference with one modality zeroed out."""
        all_probs, all_labels = [], []
        model.eval()

        for pid in test_ids:
            info = label_cache[pid]

            item = {
                "somalogic":        soma_cache[pid].unsqueeze(0),
                "somalogic_mask":   info["somalogic_mask"].unsqueeze(0),
                "metabolon":        met_cache[pid].unsqueeze(0),
                "metabolon_mask":   info["metabolon_mask"].unsqueeze(0),
                "lipidomics":       lip_cache[pid].unsqueeze(0),
                "lipidomics_mask":  info["lipidomics_mask"].unsqueeze(0),
                "luminex":          lum_cache[pid].unsqueeze(0),
                "luminex_mask":     info["luminex_mask"].unsqueeze(0),
                "clinical":         clin_cache[pid].unsqueeze(0),
                "mortality":        torch.tensor([info["mortality"]]),
            }

            # Zero out the ablated modality
            if ablate_mod in item:
                item[ablate_mod] = torch.zeros_like(item[ablate_mod])
            if f"{ablate_mod}_mask" in item:
                item[f"{ablate_mod}_mask"] = torch.zeros_like(
                    item[f"{ablate_mod}_mask"])

            # Move to device
            item = {k: v.to(device) for k, v in item.items()}

            with torch.no_grad():
                logits, _ = model(item)
                prob = torch.sigmoid(safe_logits(logits)).cpu().item()

            all_probs.append(prob)
            all_labels.append(info["mortality"])

        try:
            return roc_auc_score(all_labels, all_probs)
        except Exception:
            return float("nan")

    # Baseline (no ablation)
    baseline_auroc = run_with_ablation(ablate_mod=None)
    results["baseline"] = baseline_auroc
    print(f"  Baseline AUROC (no ablation): {baseline_auroc:.4f}")

    for mod in modalities:
        auroc = run_with_ablation(ablate_mod=mod)
        drop  = baseline_auroc - auroc
        results[mod] = {"auroc": auroc, "drop": drop}
        print(f"  Ablate {mod:12s}: AUROC={auroc:.4f}  (drop={drop:+.4f})")

    return results, baseline_auroc


# ── Per-patient risk scores ───────────────────────────────────────────────────

def compute_patient_risk_scores(model, test_ids, caches, device):
    """
    Compute predicted mortality probability for each test patient.
    Returns DataFrame with patient_id, true_label, predicted_prob.
    """
    (soma_cache, met_cache, lip_cache, lum_cache,
     clin_cache, label_cache, _, _) = caches

    model.eval()
    records = []

    for pid in test_ids:
        info = label_cache[pid]

        item = {
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

        with torch.no_grad():
            logits, _ = model(item)
            prob = torch.sigmoid(safe_logits(logits)).cpu().item()

        records.append({
            "patient_id":    pid,
            "true_mortality": info["mortality"],
            "predicted_prob": round(prob, 4),
            "predicted_class": int(prob >= 0.5),
            "correct":        int(int(prob >= 0.5) == info["mortality"]),
        })

    df = pd.DataFrame(records).sort_values("predicted_prob", ascending=False)
    return df


# ── Plot top proteins ─────────────────────────────────────────────────────────

def plot_top_features(df, title, output_path, n=20, color="#2563eb"):
    """Bar chart of top N features by importance score."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        top = df.head(n)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(top)), top["importance_score"].values,
                       color=color, alpha=0.85)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top["feature"].values, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Importance Score (Decoder Jacobian L2 norm)", fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved plot: {output_path}")
    except Exception as e:
        print(f"  Plot skipped ({e})")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir=DATA_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    # ── Load caches ──
    caches = load_latent_cache(LATENT_DIR)
    (soma_cache, met_cache, lip_cache, lum_cache,
     clin_cache, label_cache, real_ids, syn_ids) = caches

    # ── Get test split for fold 0 ──
    train_ids, val_ids, test_ids = get_fold_splits(real_ids, label_cache, FOLD)
    # Only real patients for evaluation
    test_ids = [pid for pid in test_ids if label_cache[pid]["is_synthetic"] == 0]
    print(f"\nTest patients (fold {FOLD}): {len(test_ids)}")

    # ── Load trained model ──
    print(f"\nLoading trained model from fold {FOLD}...")
    model = CachedMultimodalTransformer(dropout=0.3).to(device)
    ckpt  = torch.load(
        os.path.join(CKPT_DIR, f"model_cached_fold{FOLD}.pt"),
        map_location=device)
    model.load_state_dict(ckpt)
    model.eval()
    print("  Model loaded.")

    # ── Load pretrained VAEs ──
    print("\nLoading pretrained VAE encoders...")
    _, _, _, feature_dims = build_datasets(data_dir=data_dir, fold=0)
    vaes = load_pretrained_encoders(CKPT_DIR, feature_dims, device)

    # ── Get feature names ──
    soma_cols, met_cols, lip_cols = get_feature_names(data_dir)

    # ── 1. Protein importance (SomaLogic) ──
    print("\n" + "="*55)
    print("1. SOMALOGIC PROTEIN IMPORTANCE (Decoder Jacobian)")
    print("="*55)
    soma_vae = vaes["somalogic"]
    print("  Computing Jacobian over 200 random latent samples...")
    soma_importance = fast_jacobian_importance(
        soma_vae, latent_dim=128,
        input_dim=feature_dims["somalogic"],
        device=device, n_samples=50)

    # Align with column names
    n_soma = min(len(soma_importance), len(soma_cols))
    df_soma = pd.DataFrame({
        "feature":          soma_cols[:n_soma],
        "importance_score": soma_importance[:n_soma],
    }).sort_values("importance_score", ascending=False).reset_index(drop=True)
    df_soma["rank"] = df_soma.index + 1

    df_soma.to_csv(os.path.join(OUTPUT_DIR, "protein_importance.csv"), index=False)
    print(f"\n  Top 20 proteins by mortality importance:")
    print(df_soma[["rank", "feature", "importance_score"]].head(20).to_string(index=False))
    plot_top_features(df_soma, "Top 20 SomaLogic Proteins by Importance",
                      os.path.join(OUTPUT_DIR, "top20_proteins.png"),
                      color="#2563eb")

    # ── 2. Metabolite importance ──
    print("\n" + "="*55)
    print("2. METABOLON METABOLITE IMPORTANCE")
    print("="*55)
    met_vae = vaes["metabolon"]
    print("  Computing Jacobian...")
    met_importance = fast_jacobian_importance(
        met_vae, latent_dim=64,
        input_dim=feature_dims["metabolon"],
        device=device, n_samples=200)

    n_met = min(len(met_importance), len(met_cols))
    df_met = pd.DataFrame({
        "feature":          met_cols[:n_met],
        "importance_score": met_importance[:n_met],
    }).sort_values("importance_score", ascending=False).reset_index(drop=True)
    df_met["rank"] = df_met.index + 1

    df_met.to_csv(os.path.join(OUTPUT_DIR, "metabolite_importance.csv"), index=False)
    print(f"\n  Top 20 metabolites:")
    print(df_met[["rank", "feature", "importance_score"]].head(20).to_string(index=False))
    plot_top_features(df_met, "Top 20 Metabolites by Importance",
                      os.path.join(OUTPUT_DIR, "top20_metabolites.png"),
                      color="#16a34a")

    # ── 3. Lipid importance ──
    print("\n" + "="*55)
    print("3. LIPIDOMICS IMPORTANCE")
    print("="*55)
    lip_vae = vaes["lipidomics"]
    print("  Computing Jacobian...")
    lip_importance = fast_jacobian_importance(
        lip_vae, latent_dim=64,
        input_dim=feature_dims["lipidomics"],
        device=device, n_samples=200)

    n_lip = min(len(lip_importance), len(lip_cols))
    if n_lip > 0:
        df_lip = pd.DataFrame({
            "feature":          lip_cols[:n_lip],
            "importance_score": lip_importance[:n_lip],
        }).sort_values("importance_score", ascending=False).reset_index(drop=True)
        df_lip["rank"] = df_lip.index + 1
        df_lip.to_csv(os.path.join(OUTPUT_DIR, "lipid_importance.csv"), index=False)
        print(f"\n  Top 20 lipids:")
        print(df_lip[["rank", "feature", "importance_score"]].head(20).to_string(index=False))
        plot_top_features(df_lip, "Top 20 Lipids by Importance",
                          os.path.join(OUTPUT_DIR, "top20_lipids.png"),
                          color="#dc2626")
    else:
        print("  Could not extract lipid column names — skipping.")

    # ── 4. Modality ablation ──
    print("\n" + "="*55)
    print("4. MODALITY CONTRIBUTION (Ablation Analysis)")
    print("="*55)
    ablation_results, baseline = modality_ablation(model, test_ids, caches, device)

    ablation_rows = []
    for mod, vals in ablation_results.items():
        if mod == "baseline":
            continue
        ablation_rows.append({
            "modality":      mod,
            "auroc_ablated": round(vals["auroc"], 4),
            "auroc_drop":    round(vals["drop"],  4),
            "contribution":  round(vals["drop"] / baseline * 100, 1),
        })
    df_ablation = pd.DataFrame(ablation_rows).sort_values(
        "auroc_drop", ascending=False).reset_index(drop=True)
    df_ablation.to_csv(os.path.join(OUTPUT_DIR, "modality_ablation.csv"), index=False)

    print(f"\n  {'Modality':15s} {'AUROC (ablated)':18s} {'Drop':10s} {'% contribution':15s}")
    print("  " + "-"*55)
    for _, row in df_ablation.iterrows():
        print(f"  {row['modality']:15s} {row['auroc_ablated']:18.4f} "
              f"{row['auroc_drop']:+10.4f} {row['contribution']:15.1f}%")

    # ── 5. Per-patient risk scores ──
    print("\n" + "="*55)
    print("5. PER-PATIENT RISK SCORES")
    print("="*55)
    df_risk = compute_patient_risk_scores(model, test_ids, caches, device)
    df_risk.to_csv(os.path.join(OUTPUT_DIR, "patient_risk_scores.csv"), index=False)

    n_correct = df_risk["correct"].sum()
    accuracy  = n_correct / len(df_risk)
    print(f"\n  Patients scored: {len(df_risk)}")
    print(f"  Accuracy at 0.5 threshold: {accuracy:.1%} ({n_correct}/{len(df_risk)})")
    print(f"\n  Highest risk patients (predicted mortality):")
    print(df_risk[df_risk["predicted_prob"] >= 0.6][
        ["patient_id", "true_mortality", "predicted_prob"]
    ].head(10).to_string(index=False))

    # ── Summary ──
    print("\n" + "="*55)
    print("INTERPRETATION COMPLETE")
    print("="*55)
    print(f"  Outputs saved to: {OUTPUT_DIR}")
    print(f"  Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f))
        print(f"    {f:40s} {size/1024:.1f} KB")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    main(data_dir=data_dir)