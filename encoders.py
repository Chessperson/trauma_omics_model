"""
encoders.py — Per-modality VAE encoders for multimodal trauma transformer
==========================================================================
Implements β-VAE encoders for each omics modality:
  - SomaLogicVAE:   7596 → 128 dim latent
  - MetabolonVAE:   898  → 64  dim latent
  - LipidomicsVAE:  997  → 64  dim latent
  - LuminexProjection: 21 → 32  dim (linear, no VAE needed)
  - ClinicalEncoder:   29 → 32  dim (linear + LayerNorm)

Pretraining:
    from encoders import pretrain_all_vaes
    pretrain_all_vaes(train_dataset, device, save_dir="outputs/checkpoints/")

Loading pretrained weights:
    from encoders import load_pretrained_encoders
    encoders = load_pretrained_encoders("outputs/checkpoints/", feature_dims, device)
"""

import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# ── Building blocks ───────────────────────────────────────────────────────────

class EncoderBlock(nn.Module):
    """Linear → BatchNorm → ELU → Dropout"""
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class DecoderBlock(nn.Module):
    """Linear → BatchNorm → ELU → Dropout"""
    def __init__(self, in_dim, out_dim, dropout=0.2, last=False):
        super().__init__()
        if last:
            self.block = nn.Linear(in_dim, out_dim)
        else:
            self.block = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ELU(),
                nn.Dropout(dropout),
            )

    def forward(self, x):
        return self.block(x)


# ── β-VAE base class ──────────────────────────────────────────────────────────

class OmicsVAE(nn.Module):
    """
    Generic β-VAE for omics data.
    Subclasses define encoder_layers and decoder_layers.

    Args:
        input_dim:  number of input features
        latent_dim: size of latent space z
        beta:       KL weight (beta > 1 encourages disentanglement)
    """

    def __init__(self, input_dim, latent_dim, beta=4.0, dropout=0.2):
        super().__init__()
        self.input_dim  = input_dim
        self.latent_dim = latent_dim
        self.beta       = beta

        # Subclasses fill these
        self.encoder = None
        self.decoder = None

        # Latent projections
        self.fc_mu      = None
        self.fc_logvar  = None
        self.fc_decode  = None

    def encode(self, x):
        """Returns (mu, logvar)"""
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu, logvar):
        """z = mu + eps * std  (eps ~ N(0,I))"""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # deterministic at inference

    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z          = self.reparameterise(mu, logvar)
        recon      = self.decode(z)
        return recon, mu, logvar

    def loss(self, x, recon, mu, logvar):
        """β-VAE loss = reconstruction + β * KL divergence"""
        recon_loss = F.mse_loss(recon, x, reduction="mean")
        kl_loss    = -0.5 * torch.mean(
            1 + logvar - mu.pow(2) - logvar.exp()
        )
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    def get_latent(self, x):
        """Encode to latent space (deterministic — returns mu)."""
        self.eval()
        with torch.no_grad():
            mu, _ = self.encode(x)
        return mu


# ── SomaLogic VAE (7596 → 128) ───────────────────────────────────────────────

class SomaLogicVAE(OmicsVAE):
    """
    VAE for SomaLogic proteomics.
    Architecture: 7596 → 2048 → 512 → 256 → z(128)
    """

    def __init__(self, input_dim=7596, latent_dim=128, beta=4.0, dropout=0.2):
        super().__init__(input_dim, latent_dim, beta, dropout)

        self.encoder = nn.Sequential(
            EncoderBlock(input_dim, 2048, dropout),
            EncoderBlock(2048,      512,  dropout),
            EncoderBlock(512,       256,  dropout),
        )

        self.fc_mu     = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256)

        self.decoder = nn.Sequential(
            DecoderBlock(256,  512,       dropout),
            DecoderBlock(512,  2048,      dropout),
            DecoderBlock(2048, input_dim, dropout, last=True),
        )


# ── Metabolon VAE (898 → 64) ──────────────────────────────────────────────────

class MetabolonVAE(OmicsVAE):
    """
    VAE for Metabolon metabolomics.
    Architecture: 898 → 256 → 128 → z(64)
    """

    def __init__(self, input_dim=898, latent_dim=64, beta=4.0, dropout=0.2):
        super().__init__(input_dim, latent_dim, beta, dropout)

        self.encoder = nn.Sequential(
            EncoderBlock(input_dim, 256, dropout),
            EncoderBlock(256,       128, dropout),
        )

        self.fc_mu     = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128)

        self.decoder = nn.Sequential(
            DecoderBlock(128,       256,       dropout),
            DecoderBlock(256, input_dim, dropout, last=True),
        )


# ── Lipidomics VAE (997 → 64) ─────────────────────────────────────────────────

class LipidomicsVAE(OmicsVAE):
    """
    VAE for lipidomics species concentrations.
    Architecture: 997 → 256 → 128 → z(64)
    """

    def __init__(self, input_dim=997, latent_dim=64, beta=4.0, dropout=0.2):
        super().__init__(input_dim, latent_dim, beta, dropout)

        self.encoder = nn.Sequential(
            EncoderBlock(input_dim, 256, dropout),
            EncoderBlock(256,       128, dropout),
        )

        self.fc_mu     = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 128)

        self.decoder = nn.Sequential(
            DecoderBlock(128,       256,       dropout),
            DecoderBlock(256, input_dim, dropout, last=True),
        )


# ── Luminex linear projection (21 → 32) ──────────────────────────────────────

class LuminexProjection(nn.Module):
    """
    Simple linear projection for Luminex cytokines.
    21 features is too small for a VAE — linear projection suffices.
    """

    def __init__(self, input_dim=21, output_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def get_latent(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# ── Clinical encoder (29 → 32) ───────────────────────────────────────────────

class ClinicalEncoder(nn.Module):
    """
    Small MLP for clinical/demographic features.
    Already normalised by StandardScaler in dataset.py.
    """

    def __init__(self, input_dim=29, output_dim=32, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x):
        return self.net(x)

    def get_latent(self, x):
        self.eval()
        with torch.no_grad():
            return self.forward(x)


# ── Pretraining helpers ───────────────────────────────────────────────────────

def extract_modality_tensors(dataset, modality, timepoint_idx=None):
    """
    Pull all tensors for a given modality from a TraumaDataset.
    For temporal modalities (somalogic, luminex, metabolon), returns
    all timepoints flattened — one row per (patient × timepoint).
    Only includes rows where the mask is 1 (timepoint is present).

    Args:
        dataset:       TraumaDataset instance
        modality:      one of 'somalogic', 'luminex', 'metabolon', 'lipidomics'
        timepoint_idx: if set, only extract this timepoint index (0, 1, or 2)

    Returns:
        tensor of shape (n_valid_rows, n_features)
    """
    tensors = []
    mask_key = f"{modality}_mask"

    for i in range(len(dataset)):
        item = dataset[i]
        data = item[modality]     # (T, F) for temporal, (1, F) for lipidomics
        mask = item[mask_key]     # (T,) or (1,)

        for t in range(data.shape[0]):
            if mask[t].item() == 1.0:
                if timepoint_idx is None or t == timepoint_idx:
                    tensors.append(data[t])

    if len(tensors) == 0:
        raise ValueError(f"No valid data found for modality '{modality}'")

    return torch.stack(tensors)


def pretrain_vae(vae, data_tensor, device,
                 epochs=100, batch_size=64, lr=1e-3,
                 save_path=None, modality_name="VAE"):
    """
    Pretrain a single VAE on the given data tensor.

    Args:
        vae:          OmicsVAE instance
        data_tensor:  (N, input_dim) float tensor
        device:       torch device
        epochs:       number of training epochs
        batch_size:   mini-batch size
        lr:           initial learning rate
        save_path:    if set, saves best model weights here
        modality_name: name for logging
    """
    print(f"\n  Pretraining {modality_name} VAE...")
    print(f"    Data: {data_tensor.shape}, "
          f"Latent: {vae.latent_dim}, β={vae.beta}")

    vae = vae.to(device)
    dataset   = TensorDataset(data_tensor)
    loader    = DataLoader(dataset, batch_size=batch_size,
                           shuffle=True, drop_last=False)
    optimiser = AdamW(vae.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-5)

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        vae.train()
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_kl = 0.0
        n_batches = 0

        for (batch,) in loader:
            batch = batch.to(device)
            optimiser.zero_grad()
            recon, mu, logvar = vae(batch)
            loss, recon_l, kl_l = vae.loss(batch, recon, mu, logvar)
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=1.0)
            optimiser.step()

            epoch_loss  += loss.item()
            epoch_recon += recon_l.item()
            epoch_kl    += kl_l.item()
            n_batches   += 1

        scheduler.step()

        avg_loss  = epoch_loss  / n_batches
        avg_recon = epoch_recon / n_batches
        avg_kl    = epoch_kl    / n_batches

        if avg_loss < best_loss:
            best_loss  = avg_loss
            best_state = {k: v.clone() for k, v in vae.state_dict().items()}

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | "
                  f"Loss={avg_loss:.4f} | "
                  f"Recon={avg_recon:.4f} | "
                  f"KL={avg_kl:.4f}")

    # Restore best weights
    vae.load_state_dict(best_state)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(best_state, save_path)
        print(f"    Saved to {save_path}")

    print(f"    Best loss: {best_loss:.4f}")
    return vae


# ── Main pretraining entry point ──────────────────────────────────────────────

def pretrain_all_vaes(train_dataset, feature_dims, device,
                      save_dir="outputs/checkpoints/",
                      epochs=100, batch_size=64):
    """
    Pretrain all VAE encoders on training data (real + synthetic combined).
    Returns dict of pretrained encoder modules (encoders only, frozen).

    Args:
        train_dataset: TraumaDataset (training split)
        feature_dims:  dict from build_datasets()
        device:        torch device
        save_dir:      directory to save checkpoints
        epochs:        pretraining epochs per VAE
        batch_size:    batch size for pretraining

    Returns:
        dict of frozen encoder modules keyed by modality name
    """
    print("=" * 55)
    print("PRETRAINING VAE ENCODERS")
    print("=" * 55)

    os.makedirs(save_dir, exist_ok=True)
    encoders = {}

    # ── SomaLogic ──
    soma_data = extract_modality_tensors(train_dataset, "somalogic")
    soma_vae  = SomaLogicVAE(
        input_dim=feature_dims["somalogic"],
        latent_dim=128, beta=4.0
    )
    soma_vae = pretrain_vae(
        soma_vae, soma_data, device,
        epochs=epochs, batch_size=batch_size,
        save_path=os.path.join(save_dir, "somalogic_vae.pt"),
        modality_name="SomaLogic"
    )
    encoders["somalogic"] = soma_vae

    # ── Metabolon ──
    met_data = extract_modality_tensors(train_dataset, "metabolon")
    met_vae  = MetabolonVAE(
        input_dim=feature_dims["metabolon"],
        latent_dim=64, beta=4.0
    )
    met_vae = pretrain_vae(
        met_vae, met_data, device,
        epochs=epochs, batch_size=batch_size,
        save_path=os.path.join(save_dir, "metabolon_vae.pt"),
        modality_name="Metabolon"
    )
    encoders["metabolon"] = met_vae

    # ── Lipidomics ──
    lip_data = extract_modality_tensors(train_dataset, "lipidomics")
    lip_vae  = LipidomicsVAE(
        input_dim=feature_dims["lipidomics"],
        latent_dim=64, beta=4.0
    )
    lip_vae = pretrain_vae(
        lip_vae, lip_data, device,
        epochs=epochs, batch_size=batch_size,
        save_path=os.path.join(save_dir, "lipidomics_vae.pt"),
        modality_name="Lipidomics"
    )
    encoders["lipidomics"] = lip_vae

    # ── Luminex & Clinical (no pretraining needed — trained end-to-end) ──
    lum_proj = LuminexProjection(
        input_dim=feature_dims["luminex"],
        output_dim=32
    )
    clin_enc = ClinicalEncoder(
        input_dim=feature_dims["clinical"],
        output_dim=32
    )
    encoders["luminex"]  = lum_proj
    encoders["clinical"] = clin_enc

    print("\nAll encoders ready.")
    return encoders


def load_pretrained_encoders(save_dir, feature_dims, device):
    """
    Load pretrained VAE weights from disk.
    Call this instead of pretrain_all_vaes() if you already have checkpoints.
    """
    encoders = {}

    soma_vae = SomaLogicVAE(feature_dims["somalogic"], 128)
    soma_vae.load_state_dict(
        torch.load(os.path.join(save_dir, "somalogic_vae.pt"),
                   map_location=device))
    encoders["somalogic"] = soma_vae.to(device)

    met_vae = MetabolonVAE(feature_dims["metabolon"], 64)
    met_vae.load_state_dict(
        torch.load(os.path.join(save_dir, "metabolon_vae.pt"),
                   map_location=device))
    encoders["metabolon"] = met_vae.to(device)

    lip_vae = LipidomicsVAE(feature_dims["lipidomics"], 64)
    lip_vae.load_state_dict(
        torch.load(os.path.join(save_dir, "lipidomics_vae.pt"),
                   map_location=device))
    encoders["lipidomics"] = lip_vae.to(device)

    encoders["luminex"]  = LuminexProjection(feature_dims["luminex"],  32).to(device)
    encoders["clinical"] = ClinicalEncoder(feature_dims["clinical"], 32).to(device)

    print("Pretrained encoders loaded from", save_dir)
    return encoders


def freeze_encoders(encoders):
    """Freeze VAE encoder weights — call before fusion transformer training."""
    for name, enc in encoders.items():
        for param in enc.parameters():
            param.requires_grad = False
    print("All encoders frozen.")
    return encoders


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from dataset import build_datasets

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./"

    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load data
    train_ds, val_ds, test_ds, feature_dims = build_datasets(
        data_dir=data_dir, fold=0)

    print("\n" + "=" * 55)
    print("ENCODER ARCHITECTURE CHECK")
    print("=" * 55)

    # Instantiate each encoder and print param counts
    models = {
        "SomaLogicVAE":      SomaLogicVAE(feature_dims["somalogic"], 128),
        "MetabolonVAE":      MetabolonVAE(feature_dims["metabolon"],  64),
        "LipidomicsVAE":     LipidomicsVAE(feature_dims["lipidomics"], 64),
        "LuminexProjection": LuminexProjection(feature_dims["luminex"], 32),
        "ClinicalEncoder":   ClinicalEncoder(feature_dims["clinical"],  32),
    }

    for name, model in models.items():
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  {name:25s}: {n_params:>10,} parameters")

    # Quick forward pass test on dummy data
    print("\nForward pass check...")
    soma_vae  = SomaLogicVAE(feature_dims["somalogic"], 128).to(device)
    dummy     = torch.randn(4, feature_dims["somalogic"]).to(device)
    recon, mu, logvar = soma_vae(dummy)
    loss, _, _ = soma_vae.loss(dummy, recon, mu, logvar)
    print(f"  SomaLogic VAE forward pass: OK")
    print(f"  Input shape:  {dummy.shape}")
    print(f"  Latent shape: {mu.shape}")
    print(f"  Recon shape:  {recon.shape}")
    print(f"  Loss:         {loss.item():.4f}")

    # Run short pretraining (5 epochs just to verify it works)
    full_train = "--full" in sys.argv
    epochs = 100 if full_train else 5
    print(f"\nRunning {'full' if full_train else 'short test'} pretraining ({epochs} epochs)...")
    test_encoders = pretrain_all_vaes(
        train_ds, feature_dims, device,
        save_dir="outputs/checkpoints/",
        epochs=epochs,
        batch_size=32,
    )
    print("\nAll checks passed — ready for full pretraining.")
    print("\nTo run full pretraining (100 epochs), run:")
    print("  python3 encoders.py ./")