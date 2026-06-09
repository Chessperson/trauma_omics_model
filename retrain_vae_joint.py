"""
retrain_vae_joint.py — Retrain SomaLogic VAE on PAMPer + SWAT proteins
========================================================================
Addresses domain shift by training the VAE on both cohorts jointly.
No mortality labels used for SWAT — purely unsupervised reconstruction.

The retrained VAE is saved to:
    outputs/checkpoints/somalogic_vae_joint.pt

Usage:
    python3 retrain_vae_joint.py ./
"""

import os, sys, re
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(__file__))
from encoders import SomaLogicVAE, pretrain_vae
from dataset import build_datasets

CKPT_DIR  = "outputs/checkpoints/"
SWAT_CSV  = "SWAT_proteins_clinical.csv"
EPOCHS    = 150
SEED      = 42


def parse_swat_tp(name):
    m = re.search(r'SWAT\d+?(\d+)EDA$', str(name))
    if not m: return None
    s = m.group(1)
    return 24 if s.endswith('24') else (4 if s.endswith('4') else 0)


def main(data_dir="./"):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = torch.device("mps" if torch.backends.mps.is_available()
                          else "cuda" if torch.cuda.is_available()
                          else "cpu")
    print(f"Device: {device}")

    # ── Load PAMPer protein data ──
    print("\nLoading PAMPer SomaLogic data...")
    soma = pd.read_excel(
        os.path.join(data_dir, "PAMPer_External_data.xlsx"),
        sheet_name="SomaLogic")
    protein_cols = soma.columns.tolist()[2:]
    pamper_vals  = soma[protein_cols].apply(
        pd.to_numeric, errors='coerce').clip(lower=0).values
    pamper_vals  = np.nan_to_num(pamper_vals, nan=0.0).astype(np.float32)
    pamper_log   = np.log1p(pamper_vals)
    print(f"  PAMPer rows: {len(pamper_log)} "
          f"(patients × timepoints)")
    print(f"  PAMPer norm std: "
          f"{torch.tensor(pamper_log).norm(dim=1).std():.4f}")

    # ── Load SWAT protein data ──
    print("\nLoading SWAT SomaLogic data...")
    df = pd.read_csv(os.path.join(data_dir, SWAT_CSV))
    df['timepoint'] = df['Name'].apply(parse_swat_tp)

    pamp_df    = pd.read_excel(
        os.path.join(data_dir, "PAMPer_External_data.xlsx"),
        sheet_name="SomaLogic", nrows=1)
    pamp_prots = pamp_df.columns.tolist()[2:]

    swat_rows = []
    for tp in [0, 24]:
        tp_df = df[df['timepoint'] == tp]
        for _, row in tp_df.iterrows():
            v = np.array([
                float(row[p]) if p in row.index and pd.notna(row[p]) else 0.0
                for p in pamp_prots], dtype=np.float32)
            v = np.log1p(np.clip(v, 0, None))
            swat_rows.append(v)

    swat_log = np.array(swat_rows, dtype=np.float32)
    print(f"  SWAT rows: {len(swat_log)} (patients × timepoints)")
    print(f"  SWAT norm std before joint training: "
          f"{torch.tensor(swat_log).norm(dim=1).std():.4f}")

    # ── Combine PAMPer + SWAT ──
    combined = np.vstack([pamper_log, swat_log])
    print(f"\nCombined dataset: {combined.shape}")
    data_tensor = torch.tensor(combined, dtype=torch.float32)

    # ── Load existing VAE and retrain ──
    print("\nLoading existing SomaLogic VAE...")
    _, _, _, feature_dims = build_datasets(data_dir=data_dir, fold=0)
    vae = SomaLogicVAE(feature_dims['somalogic'], 128)
    vae.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, 'somalogic_vae.pt'),
        map_location='cpu'))
    print(f"  VAE loaded (beta={vae.beta})")

    # ── Joint retraining ──
    save_path = os.path.join(CKPT_DIR, 'somalogic_vae_joint.pt')
    vae = pretrain_vae(
        vae, data_tensor, device,
        epochs=EPOCHS,
        batch_size=64,
        lr=2e-4,
        save_path=save_path,
        modality_name="SomaLogic (PAMPer+SWAT joint)")

    # ── Verify SWAT latent diversity improved ──
    print("\nVerifying SWAT latent diversity after joint training...")
    vae = vae.to(device).eval()
    z_swat = []
    with torch.no_grad():
        for i in range(min(20, len(swat_log))):
            x = torch.tensor(swat_log[i]).unsqueeze(0).to(device)
            z, _ = vae.encode(x)
            z_swat.append(z.squeeze(0).cpu())
    z_swat = torch.stack(z_swat)
    print(f"  SWAT latent norm std after joint training: "
          f"{z_swat.norm(dim=1).std():.4f}")
    print(f"  PAMPer reference std: 0.6428")
    print(f"\nSaved joint VAE to: {save_path}")
    print("Now update swat_validation.py to load somalogic_vae_joint.pt")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    main(data_dir)
