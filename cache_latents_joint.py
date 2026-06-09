"""
cache_latents_joint.py — Re-encode PAMPer through joint VAE
============================================================
Creates a separate latent cache using the joint PAMPer+SWAT VAE.
Saves to outputs/latents_joint/ to avoid overwriting original cache.

Usage:
    python3 cache_latents_joint.py ./
"""

import os, sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
from dataset import build_datasets, TraumaDataset
from encoders import (SomaLogicVAE, MetabolonVAE, LipidomicsVAE,
                      LuminexProjection, ClinicalEncoder,
                      load_pretrained_encoders)

CKPT_DIR       = "outputs/checkpoints/"
LATENT_DIR     = "outputs/latents/"
JOINT_LATENT_DIR = "outputs/latents_joint/"
BATCH_SIZE     = 64


def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")


def collate_fn(batch):
    out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([b[key] for b in batch])
        else:
            out[key] = [b[key] for b in batch]
    return out


def main(data_dir="./"):
    os.makedirs(JOINT_LATENT_DIR, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    print("\nLoading dataset...")
    train_ds, val_ds, test_ds, feature_dims = build_datasets(
        data_dir=data_dir, fold=0)
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | "
          f"Test: {len(test_ds)}")

    # Load all encoders using original checkpoints
    print("\nLoading encoders...")
    encoders = load_pretrained_encoders(CKPT_DIR, feature_dims, device)

    # Replace SomaLogic encoder with joint VAE
    print("  Replacing SomaLogic VAE with joint VAE...")
    soma_vae_joint = SomaLogicVAE(feature_dims['somalogic'], 128)
    soma_vae_joint.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, 'somalogic_vae_joint.pt'),
        map_location=device))
    soma_vae_joint = soma_vae_joint.to(device).eval()
    encoders['somalogic'] = soma_vae_joint

    for enc in encoders.values():
        enc.eval()

    print(f"  Joint VAE loaded")

    # Check latent diversity
    print("\nVerifying joint VAE latent diversity on PAMPer...")

    def encode_split(dataset, split_name):
        loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                            shuffle=False, collate_fn=collate_fn,
                            num_workers=0)
        soma_cache = {}
        met_cache  = {}
        lip_cache  = {}
        lum_cache  = {}
        clin_cache = {}
        label_cache = {}

        soma_enc = encoders["somalogic"]
        met_enc  = encoders["metabolon"]
        lip_enc  = encoders["lipidomics"]
        lum_enc  = encoders["luminex"]
        clin_enc = encoders["clinical"]

        n_encoded = 0
        with torch.no_grad():
            for batch in loader:
                pids       = batch["patient_id"]
                soma_data  = batch["somalogic"].to(device)
                met_data   = batch["metabolon"].to(device)
                lip_data   = batch["lipidomics"].to(device)
                lum_data   = batch["luminex"].to(device)
                clin_data  = batch["clinical"].to(device)
                soma_mask  = batch["somalogic_mask"].to(device)
                met_mask   = batch["metabolon_mask"].to(device)
                lip_mask   = batch["lipidomics_mask"].to(device)
                lum_mask   = batch["luminex_mask"].to(device)
                labels     = batch["mortality"]
                is_syn     = batch["is_synthetic"]
                gen_ids    = batch.get("gen_id", [""] * len(pids))

                B, T, D_s = soma_data.shape
                soma_z = torch.zeros(B, T, 128, device=device)
                for t in range(T):
                    z, _ = soma_enc.encode(soma_data[:, t, :])
                    soma_z[:, t, :] = z

                _, _, D_m = met_data.shape
                met_z = torch.zeros(B, T, 64, device=device)
                for t in range(T):
                    z, _ = met_enc.encode(met_data[:, t, :])
                    met_z[:, t, :] = z

                lip_z = torch.zeros(B, T, 64, device=device)
                for t in range(T):
                    z, _ = lip_enc.encode(lip_data[:, t, :])
                    lip_z[:, t, :] = z

                lum_z = torch.zeros(B, T, 32, device=device)
                for t in range(T):
                    lum_z[:, t, :] = lum_enc(lum_data[:, t, :])

                clin_z = clin_enc(clin_data)

                for i, pid in enumerate(pids):
                    soma_cache[pid]  = soma_z[i].cpu()
                    met_cache[pid]   = met_z[i].cpu()
                    lip_cache[pid]   = lip_z[i].cpu()
                    lum_cache[pid]   = lum_z[i].cpu()
                    clin_cache[pid]  = clin_z[i].cpu()
                    label_cache[pid] = {
                        "mortality":    int(labels[i]),
                        "is_synthetic": int(is_syn[i]),
                        "gen_id":       gen_ids[i] if isinstance(gen_ids, list) else "",
                    }
                n_encoded += len(pids)

        print(f"    {split_name}: {n_encoded} patients encoded")
        return (soma_cache, met_cache, lip_cache,
                lum_cache, clin_cache, label_cache)

    # Encode all splits
    all_soma = {}; all_met = {}; all_lip = {}
    all_lum  = {}; all_clin = {}; all_labels = {}

    for ds, name in [(train_ds, "train"),
                     (val_ds,   "val"),
                     (test_ds,  "test")]:
        s, m, l, lu, c, lb = encode_split(ds, name)
        all_soma.update(s);  all_met.update(m)
        all_lip.update(l);   all_lum.update(lu)
        all_clin.update(c);  all_labels.update(lb)

    # Verify latent diversity
    real_pids = [p for p, v in all_labels.items() if v['is_synthetic'] == 0]
    z_check   = torch.stack([all_soma[p][0] for p in real_pids[:30]])
    print(f"\n  Joint VAE PAMPer latents:")
    print(f"    norm mean: {z_check.norm(dim=1).mean():.4f}")
    print(f"    norm std:  {z_check.norm(dim=1).std():.4f}")
    print(f"    (original VAE: mean=1.3239, std=0.6428)")

    # Save
    print(f"\nSaving joint latents to {JOINT_LATENT_DIR}...")
    torch.save(all_soma,   os.path.join(JOINT_LATENT_DIR, 'somalogic_latents.pt'))
    torch.save(all_met,    os.path.join(JOINT_LATENT_DIR, 'metabolon_latents.pt'))
    torch.save(all_lip,    os.path.join(JOINT_LATENT_DIR, 'lipidomics_latents.pt'))
    torch.save(all_lum,    os.path.join(JOINT_LATENT_DIR, 'luminex_latents.pt'))
    torch.save(all_clin,   os.path.join(JOINT_LATENT_DIR, 'clinical_latents.pt'))
    torch.save(all_labels, os.path.join(JOINT_LATENT_DIR, 'labels.pt'))

    total = len(all_soma)
    real  = sum(1 for v in all_labels.values() if v['is_synthetic'] == 0)
    syn   = total - real
    print(f"  Saved {total} patients ({real} real + {syn} synthetic)")
    print(f"  Done. Run bidirectional_validation.py next.")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    main(data_dir)
