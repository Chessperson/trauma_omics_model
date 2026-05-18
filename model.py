"""
model.py — Multimodal Transformer for Trauma Mortality Prediction
=================================================================
Architecture:
  1. Frozen VAE encoders compress each omics modality to latent vectors
  2. Per-modality temporal transformer attends across 3 timepoints
  3. Cross-modal fusion transformer attends across 5 modality tokens
  4. [CLS] token → mortality prediction head

Usage:
    from model import MultimodalTransformer
    from encoders import load_pretrained_encoders

    encoders = load_pretrained_encoders("outputs/checkpoints/", feature_dims, device)
    model = MultimodalTransformer(encoders, d_model=128).to(device)

    batch = next(iter(train_loader))
    logits, attn_weights = model(batch)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

D_MODEL    = 128   # common dimension throughout transformer
N_HEADS    = 4     # attention heads (D_MODEL must be divisible by N_HEADS)
N_LAYERS   = 2     # transformer layers (temporal and fusion)
DROPOUT    = 0.1
TIMEPOINTS = 3

MODALITY_LATENT_DIMS = {
    "somalogic":  128,
    "metabolon":  64,
    "luminex":    32,
    "lipidomics": 64,
    "clinical":   32,
}


# ── Positional encoding for timepoints ───────────────────────────────────────

class TimePointEmbedding(nn.Module):
    """
    Learned positional embeddings for timepoints (0hr, 24hr, 72hr).
    Learned rather than sinusoidal because intervals are irregular.
    Also embeds a learned [MASK] token for missing timepoints.
    """

    def __init__(self, d_model=D_MODEL):
        super().__init__()
        # 3 timepoints + 1 mask token
        self.tp_embed  = nn.Embedding(3, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, d_model) * 0.02)

    def forward(self, x, mask):
        """
        Args:
            x:    (B, T, D) — latent vectors for each timepoint
            mask: (B, T)    — 1=present, 0=missing

        Returns:
            (B, T, D) — x + positional embeddings, missing TPs replaced by mask token
        """
        B, T, D = x.shape
        device  = x.device

        # Timepoint indices 0, 1, 2
        tp_idx = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pos_emb = self.tp_embed(tp_idx)  # (B, T, D)

        # Replace missing timepoints with learned mask token
        mask_expanded = mask.unsqueeze(-1).expand_as(x)  # (B, T, D)
        mask_tok      = self.mask_token.expand(B, T, D)
        x = torch.where(mask_expanded.bool(), x, mask_tok)

        return x + pos_emb


# ── Transformer encoder layer with attention capture ─────────────────────────

class TransformerLayer(nn.Module):
    """
    Single transformer encoder layer.
    Stores attention weights for interpretability.
    """

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS,
                 ff_dim=None, dropout=DROPOUT):
        super().__init__()
        ff_dim = ff_dim or d_model * 4
        self.attn    = nn.MultiheadAttention(d_model, n_heads,
                                              dropout=dropout,
                                              batch_first=True)
        self.norm1   = nn.LayerNorm(d_model)
        self.norm2   = nn.LayerNorm(d_model)
        self.ff      = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
            nn.Dropout(dropout),
        )
        self.attn_weights = None  # stored after forward pass

    def forward(self, x, key_padding_mask=None):
        """
        Args:
            x:                (B, T, D)
            key_padding_mask: (B, T) bool — True = ignore this position

        Returns:
            (B, T, D)
        """
        attn_out, attn_w = self.attn(
            x, x, x,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # keep per-head weights
        )
        self.attn_weights = attn_w.detach()  # (B, n_heads, T, T)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ff(x))
        return x


class TransformerEncoder(nn.Module):
    """Stack of N transformer layers."""

    def __init__(self, d_model=D_MODEL, n_heads=N_HEADS,
                 n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, dropout=dropout)
            for _ in range(n_layers)
        ])

    def forward(self, x, key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return x

    def get_attention_weights(self):
        """Returns list of attention weight tensors, one per layer."""
        return [layer.attn_weights for layer in self.layers]


# ── Per-modality temporal encoder ────────────────────────────────────────────

class ModalityTemporalEncoder(nn.Module):
    """
    For a single modality:
      1. Project latent dim → D_MODEL
      2. Add timepoint positional embeddings + mask tokens
      3. Apply temporal transformer (attends across timepoints)
      4. Mean-pool present timepoints → single modality token

    Args:
        latent_dim: input latent dimension from VAE
        d_model:    common transformer dimension
    """

    def __init__(self, latent_dim, d_model=D_MODEL,
                 n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()
        self.proj       = nn.Linear(latent_dim, d_model)
        self.tp_embed   = TimePointEmbedding(d_model)
        self.transformer = TransformerEncoder(d_model, n_heads, n_layers, dropout)
        self.norm        = nn.LayerNorm(d_model)

    def forward(self, x, mask):
        """
        Args:
            x:    (B, T, latent_dim) — VAE latents across timepoints
            mask: (B, T)             — 1=present, 0=missing

        Returns:
            token: (B, D_MODEL) — single modality token
        """
        B, T, _ = x.shape

        # Project to D_MODEL
        x = self.proj(x)                        # (B, T, D)

        # Add timepoint embeddings, fill missing with mask token
        x = self.tp_embed(x, mask)              # (B, T, D)

        # Key padding mask: True = ignore (missing timepoints)
        key_pad = (mask == 0)                   # (B, T) bool

        # Temporal self-attention
        x = self.transformer(x, key_padding_mask=key_pad)  # (B, T, D)

        # Mean pool over present timepoints only
        mask_exp = mask.unsqueeze(-1).float()   # (B, T, 1)
        token    = (x * mask_exp).sum(dim=1) / mask_exp.sum(dim=1).clamp(min=1)
        token    = self.norm(token)             # (B, D)

        return token


# ── Full multimodal transformer ───────────────────────────────────────────────

class MultimodalTransformer(nn.Module):
    """
    Full multimodal transformer for trauma mortality prediction.

    Pipeline:
        VAE encoders (frozen) → per-modality temporal transformers
        → cross-modal fusion transformer → mortality prediction head

    Args:
        encoders:    dict of pretrained encoder modules from encoders.py
        d_model:     transformer hidden dimension (default 128)
        n_heads:     attention heads for fusion transformer
        n_layers:    layers for fusion transformer
        dropout:     dropout rate
    """

    def __init__(self, encoders, d_model=D_MODEL,
                 n_heads=N_HEADS, n_layers=N_LAYERS, dropout=DROPOUT):
        super().__init__()

        self.d_model  = d_model
        self.encoders = nn.ModuleDict(encoders)

        # Per-modality temporal encoders
        self.temporal_encoders = nn.ModuleDict({
            mod: ModalityTemporalEncoder(
                latent_dim=MODALITY_LATENT_DIMS[mod],
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                dropout=dropout,
            )
            for mod in ["somalogic", "metabolon", "lipidomics", "luminex"]
        })

        # Clinical encoder projects directly to D_MODEL (no temporal axis)
        self.clinical_proj = nn.Sequential(
            nn.Linear(32, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        # Learned modality type embeddings (so fusion transformer knows which
        # token is proteomics vs metabolomics vs clinical etc.)
        n_modalities = 5  # somalogic, metabolon, lipidomics, luminex, clinical
        self.modality_embed = nn.Embedding(n_modalities, d_model)

        # [CLS] token for fusion
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # Cross-modal fusion transformer (attends across 6 tokens: CLS + 5 modalities)
        self.fusion_transformer = TransformerEncoder(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            dropout=dropout,
        )

        # Mortality prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "encoder" in name:
                continue  # don't reinitialise pretrained weights
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_modality(self, batch, modality):
        """
        Run VAE encoder + temporal encoder for one modality.

        Returns:
            token: (B, D_MODEL)
        """
        enc   = self.encoders[modality]
        x     = batch[modality]           # (B, T, input_dim)
        mask  = batch[f"{modality}_mask"] # (B, T)
        B, T, F = x.shape

        # Run VAE encoder timepoint by timepoint
        latents = []
        for t in range(T):
            xt = x[:, t, :]              # (B, input_dim)
            with torch.no_grad():
                if hasattr(enc, "encode"):
                    mu, _ = enc.encode(xt)
                else:
                    mu = enc(xt)         # linear projection (Luminex)
            latents.append(mu)

        latent_seq = torch.stack(latents, dim=1)  # (B, T, latent_dim)

        # Temporal transformer
        token = self.temporal_encoders[modality](latent_seq, mask)
        return token

    def forward(self, batch):
        """
        Args:
            batch: dict from TraumaDataset.__getitem__

        Returns:
            logits:       (B,) — raw logits (apply sigmoid for probability)
            attn_weights: dict of attention weight tensors for interpretability
        """
        device = batch["somalogic"].device

        # ── Encode each modality to a single token ──
        modality_order = ["somalogic", "metabolon", "lipidomics", "luminex"]
        tokens = []

        for i, mod in enumerate(modality_order):
            token    = self.encode_modality(batch, mod)  # (B, D)
            mod_emb  = self.modality_embed(
                torch.tensor(i, device=device))          # (D,)
            token    = token + mod_emb.unsqueeze(0)      # (B, D)
            tokens.append(token)

        # Clinical token
        clin_token = self.clinical_proj(
            self.encoders["clinical"](batch["clinical"]))  # (B, D)
        clin_emb   = self.modality_embed(
            torch.tensor(4, device=device))
        clin_token = clin_token + clin_emb.unsqueeze(0)
        tokens.append(clin_token)

        # Stack into sequence: (B, 5, D)
        token_seq = torch.stack(tokens, dim=1)

        # Prepend [CLS] token: (B, 6, D)
        B = token_seq.shape[0]
        cls = self.cls_token.expand(B, -1, -1)
        token_seq = torch.cat([cls, token_seq], dim=1)  # (B, 6, D)

        # ── Cross-modal fusion transformer ──
        fused = self.fusion_transformer(token_seq)       # (B, 6, D)

        # ── Mortality prediction from [CLS] token ──
        cls_out = fused[:, 0, :]                         # (B, D)
        logits  = self.head(cls_out).squeeze(-1)         # (B,)

        # ── Collect attention weights for interpretability ──
        attn_weights = {
            "fusion": self.fusion_transformer.get_attention_weights(),
            "temporal": {
                mod: self.temporal_encoders[mod].transformer.get_attention_weights()
                for mod in modality_order
            }
        }

        return logits, attn_weights

    def predict_proba(self, batch):
        """Returns mortality probabilities (0-1) for a batch."""
        self.eval()
        with torch.no_grad():
            logits, _ = self.forward(batch)
            return torch.sigmoid(logits)


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from dataset import build_datasets
    from encoders import load_pretrained_encoders, pretrain_all_vaes

    data_dir   = sys.argv[1] if len(sys.argv) > 1 else "./"
    ckpt_dir   = "outputs/checkpoints/"

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

    # Load pretrained encoders
    if all(os.path.exists(os.path.join(ckpt_dir, f"{m}_vae.pt"))
           for m in ["somalogic", "metabolon", "lipidomics"]):
        print("Loading pretrained encoders...")
        encoders = load_pretrained_encoders(ckpt_dir, feature_dims, device)
    else:
        print("No checkpoints found — run encoders.py --full first")
        sys.exit(1)

    # Build model
    print("\nBuilding MultimodalTransformer...")
    model = MultimodalTransformer(encoders, d_model=D_MODEL).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Trainable parameters : {n_params:>12,}")
    print(f"  Frozen (VAE encoders): {n_frozen:>12,}")

    # Forward pass on a single batch
    from torch.utils.data import DataLoader

    def collate(batch):
        out = {}
        for key in batch[0]:
            if isinstance(batch[0][key], torch.Tensor):
                out[key] = torch.stack([b[key] for b in batch])
            else:
                out[key] = [b[key] for b in batch]
        return out

    loader = DataLoader(train_ds, batch_size=8,
                        shuffle=False, collate_fn=collate)
    batch  = next(iter(loader))
    batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
              for k, v in batch.items()}

    print("\nRunning forward pass on batch of 8...")
    model.eval()
    with torch.no_grad():
        logits, attn_weights = model(batch)
        probs = torch.sigmoid(logits)

    print(f"  Logits shape : {logits.shape}")
    print(f"  Probs        : {probs.round(decimals=3).tolist()}")
    print(f"  True labels  : {batch['mortality'].tolist()}")

    print("\nAttention weight shapes:")
    for layer_w in attn_weights["fusion"]:
        print(f"  Fusion layer : {layer_w.shape}  (batch, heads, tokens, tokens)")
        break
    for mod, layers in attn_weights["temporal"].items():
        print(f"  Temporal [{mod}]: {layers[0].shape}")
        break

    print("\nModel ready. Proceed to train.py")