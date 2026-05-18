"""
cache_latents.py — Pre-compute VAE latents and save to disk
============================================================
Run this ONCE before training. It passes all data through the
pretrained VAE encoders and caches the latent vectors so that
train.py never has to run the heavy VAE forward pass during training.
 
Output:
    outputs/latents/somalogic_latents.pt   — dict: patient_id -> (3, 128) tensor
    outputs/latents/metabolon_latents.pt   — dict: patient_id -> (3, 64)  tensor
    outputs/latents/lipidomics_latents.pt  — dict: patient_id -> (3, 64)  tensor
    outputs/latents/luminex_latents.pt     — dict: patient_id -> (3, 32)  tensor
    outputs/latents/clinical_latents.pt    — dict: patient_id -> (32,)    tensor
    outputs/latents/labels.pt             — dict: patient_id -> {mortality, is_synthetic, gen_id}
 
Usage:
    python3 cache_latents.py ./
"""
 
import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader
 
sys.path.insert(0, os.path.dirname(__file__))
from dataset import build_datasets, TraumaDataset
from encoders import (
    SomaLogicVAE, MetabolonVAE, LipidomicsVAE,
    LuminexProjection, ClinicalEncoder,
    load_pretrained_encoders,
)
 
# ── Config ────────────────────────────────────────────────────────────────────
 
CKPT_DIR    = "outputs/checkpoints/"
LATENT_DIR  = "outputs/latents/"
BATCH_SIZE  = 64   # larger batch is fine here — no gradients
 
 
# ── Device ────────────────────────────────────────────────────────────────────
 
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
 
 
# ── Collate ───────────────────────────────────────────────────────────────────
 
def collate_fn(batch):
    out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([b[key] for b in batch])
        else:
            out[key] = [b[key] for b in batch]
    return out
 
 
# ── Main caching logic ────────────────────────────────────────────────────────
 
def cache_all_latents(data_dir="./", ckpt_dir=CKPT_DIR, latent_dir=LATENT_DIR):
    """
    Load all data (train + val + test across all folds = entire dataset),
    run every sample through the pretrained encoders, save latent dicts.
    """
    os.makedirs(latent_dir, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")
 
    # ── Load full dataset (fold=0 gives us all patients across splits) ──
    # We use fold=0 just to get feature_dims and the full patient list.
    # We'll iterate over ALL splits so every patient gets cached.
    print("\nLoading dataset...")
    train_ds, val_ds, test_ds, feature_dims = build_datasets(
        data_dir=data_dir, fold=0)
 
    print(f"  Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
 
    # ── Load pretrained encoders ──
    print("\nLoading pretrained encoders...")
    encoders = load_pretrained_encoders(ckpt_dir, feature_dims, device)
 
    # Set all encoders to eval mode
    for enc in encoders.values():
        enc.eval()
 
    # ── Helper: encode one dataset split into dicts ──
    def encode_split(dataset, split_name):
        """
        Returns five dicts keyed by patient_id:
            soma_cache    : patient_id -> (3, 128) tensor  [on CPU]
            met_cache     : patient_id -> (3, 64)  tensor
            lip_cache     : patient_id -> (3, 64)  tensor
            lum_cache     : patient_id -> (3, 32)  tensor
            clin_cache    : patient_id -> (32,)    tensor
            label_cache   : patient_id -> dict
        """
        loader = DataLoader(
            dataset, batch_size=BATCH_SIZE,
            shuffle=False, collate_fn=collate_fn,
            num_workers=0,
        )
 
        soma_cache  = {}
        met_cache   = {}
        lip_cache   = {}
        lum_cache   = {}
        clin_cache  = {}
        label_cache = {}
 
        soma_enc = encoders["somalogic"]
        met_enc  = encoders["metabolon"]
        lip_enc  = encoders["lipidomics"]
        lum_enc  = encoders["luminex"]
        clin_enc = encoders["clinical"]
 
        n_batches = len(loader)
        print(f"\n  Encoding {split_name} ({len(dataset)} patients, "
              f"{n_batches} batches)...")
 
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                # Move tensors to device
                b = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
 
                patient_ids = b["patient_id"]   # list of strings
                B = len(patient_ids)
 
                # ── SomaLogic: (B, 3, 7596) -> (B, 3, 128) ──
                soma_latents = []
                for t in range(3):
                    xt = b["somalogic"][:, t, :]          # (B, 7596)
                    mu, _ = soma_enc.encode(xt)            # (B, 128)
                    soma_latents.append(mu.cpu())
                soma_seq = torch.stack(soma_latents, dim=1)  # (B, 3, 128)
 
                # ── Metabolon: (B, 3, 898) -> (B, 3, 64) ──
                met_latents = []
                for t in range(3):
                    xt = b["metabolon"][:, t, :]
                    mu, _ = met_enc.encode(xt)
                    met_latents.append(mu.cpu())
                met_seq = torch.stack(met_latents, dim=1)    # (B, 3, 64)
 
                # ── Lipidomics: (B, 3, 997) -> (B, 3, 64) ──
                lip_latents = []
                for t in range(3):
                    xt = b["lipidomics"][:, t, :]
                    mu, _ = lip_enc.encode(xt)
                    lip_latents.append(mu.cpu())
                lip_seq = torch.stack(lip_latents, dim=1)    # (B, 3, 64)
 
                # ── Luminex: (B, 3, 21) -> (B, 3, 32) ──
                lum_latents = []
                for t in range(3):
                    xt = b["luminex"][:, t, :]
                    lt = lum_enc(xt)                          # (B, 32)
                    lum_latents.append(lt.cpu())
                lum_seq = torch.stack(lum_latents, dim=1)    # (B, 3, 32)
 
                # ── Clinical: (B, 29) -> (B, 32) ──
                clin_lat = clin_enc(b["clinical"]).cpu()      # (B, 32)
 
                # ── Store per patient ──
                for i, pid in enumerate(patient_ids):
                    soma_cache[pid]  = soma_seq[i]    # (3, 128)
                    met_cache[pid]   = met_seq[i]     # (3, 64)
                    lip_cache[pid]   = lip_seq[i]     # (3, 64)
                    lum_cache[pid]   = lum_seq[i]     # (3, 32)
                    clin_cache[pid]  = clin_lat[i]    # (32,)
                    label_cache[pid] = {
                        "mortality":     int(b["mortality"][i].item()),
                        "is_synthetic":  int(b["is_synthetic"][i]),
                        "somalogic_mask": b["somalogic_mask"][i].cpu(),
                        "luminex_mask":   b["luminex_mask"][i].cpu(),
                        "metabolon_mask": b["metabolon_mask"][i].cpu(),
                        "lipidomics_mask": b["lipidomics_mask"][i].cpu(),
                    }
 
                if (batch_idx + 1) % 5 == 0 or batch_idx == n_batches - 1:
                    print(f"    Batch {batch_idx+1}/{n_batches} done", end="\r")
 
        print(f"    Done — {len(soma_cache)} patients encoded.          ")
        return soma_cache, met_cache, lip_cache, lum_cache, clin_cache, label_cache
 
    # ── Encode all three splits ──
    # Note: some patients appear in multiple splits across folds,
    # but we only need each patient encoded once. We merge all splits.
    all_soma, all_met, all_lip, all_lum, all_clin, all_labels = {}, {}, {}, {}, {}, {}
 
    for ds, name in [(train_ds, "train"), (val_ds, "val"), (test_ds, "test")]:
        s, m, l, lu, c, lb = encode_split(ds, name)
        # dict.update: later splits overwrite — fine, same patient same latents
        all_soma.update(s)
        all_met.update(m)
        all_lip.update(l)
        all_lum.update(lu)
        all_clin.update(c)
        all_labels.update(lb)
 
    total = len(all_soma)
    print(f"\nTotal unique patients cached: {total}")
 
    # ── Save to disk ──
    print(f"\nSaving latents to {latent_dir} ...")
 
    torch.save(all_soma,   os.path.join(latent_dir, "somalogic_latents.pt"))
    print(f"  somalogic_latents.pt   — {total} patients, shape (3, 128) each")
 
    torch.save(all_met,    os.path.join(latent_dir, "metabolon_latents.pt"))
    print(f"  metabolon_latents.pt   — {total} patients, shape (3, 64) each")
 
    torch.save(all_lip,    os.path.join(latent_dir, "lipidomics_latents.pt"))
    print(f"  lipidomics_latents.pt  — {total} patients, shape (3, 64) each")
 
    torch.save(all_lum,    os.path.join(latent_dir, "luminex_latents.pt"))
    print(f"  luminex_latents.pt     — {total} patients, shape (3, 32) each")
 
    torch.save(all_clin,   os.path.join(latent_dir, "clinical_latents.pt"))
    print(f"  clinical_latents.pt    — {total} patients, shape (32,) each")
 
    torch.save(all_labels, os.path.join(latent_dir, "labels.pt"))
    print(f"  labels.pt              — mortality + masks per patient")
 
    # Quick size report
    total_mb = sum(
        os.path.getsize(os.path.join(latent_dir, f))
        for f in os.listdir(latent_dir)
    ) / 1e6
    print(f"\nTotal cache size: {total_mb:.1f} MB")
    print("\nDone. Run train_cached.py to start training.")
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    cache_all_latents(data_dir=data_dir)