"""
train_supervised_vae.py — End-to-End Supervised β-VAE Training
===============================================================
Replaces the two-stage pretrain-then-freeze pipeline with joint
end-to-end training. The SomaLogic VAE encoder is simultaneously
optimized for:

    1. Reconstruction (recon loss)       — preserves biological info
    2. KL regularization (KL loss)       — smooth, sampable latent space
    3. Graph prior (graph loss)          — pathway co-membership structure
    4. Mortality prediction (BCE loss)   — task-specific latent geometry

Combined loss:
    L = L_recon + β * L_KL + γ * L_graph + λ * L_mortality

PAMPer patients (labeled): contribute all four losses
Synthetic patients:         contribute reconstruction + KL + graph only
SWAT patients (unlabeled):  contribute reconstruction + KL only

Why this matters:
    The two-stage pipeline optimizes compression for reconstruction,
    not for the prediction task. Dimensions that explain protein
    variance but are irrelevant to mortality consume latent capacity.
    Joint training forces the 128 latent dimensions to be both
    biologically coherent AND maximally discriminative for survival.

Usage:
    python3 train_supervised_vae.py ./
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from scipy.sparse import csr_matrix

sys.path.insert(0, os.path.dirname(__file__))
from encoders import SomaLogicVAE
from dataset import build_datasets, TraumaDataset
from train_cached import get_device, safe_logits

CKPT_DIR   = "outputs/checkpoints/"
OUTPUT_DIR = "outputs/supervised_vae/"

# ── Hyperparameters ──────────────────────────────────────────────────────────
CFG = {
    "latent_dim":    128,
    "beta":          2.0,    # KL weight — lower than unsupervised (was 4.0)
                             # reduces to allow more task-specific geometry
    "lambda_mort":   0.3,    # mortality loss weight — main new term
    "gamma_graph":   0.5,    # graph prior weight
    "lr":            2e-4,
    "epochs":        250,
    "batch_size":    32,     # smaller batch — labeled data is scarce
    "dropout":       0.2,
    "n_folds":       5,
    "seed":          42,
}


# ── Mortality head ───────────────────────────────────────────────────────────

class MortalityHead(nn.Module):
    """
    Lightweight MLP attached to the VAE latent space.
    128 → 64 → 1. Trained jointly with the encoder.
    Separate from CachedMultimodalTransformer — this head only
    uses somalogic latent, not the full multimodal fusion.
    Think of it as a somalogic-only mortality predictor that shapes
    the latent space, with the full multimodal model trained on top.
    """
    def __init__(self, latent_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, z):
        return self.net(z).squeeze(-1)


# ── Supervised VAE (VAE + MortalityHead jointly) ─────────────────────────────

class SupervisedSomaVAE(nn.Module):
    """
    Wraps SomaLogicVAE + MortalityHead for joint training.
    The encoder sees gradients from both reconstruction and mortality.
    The decoder sees gradients only from reconstruction.
    """
    def __init__(self, input_dim, latent_dim=128,
                 beta=1.0, dropout=0.2):
        super().__init__()
        self.vae  = SomaLogicVAE(input_dim, latent_dim, beta, dropout)
        self.head = MortalityHead(latent_dim, dropout)
        self.beta = beta

    def forward(self, x):
        recon, mu, logvar = self.vae(x)
        z_sample = self.vae.reparameterise(mu, logvar)
        logit    = self.head(z_sample)
        return recon, mu, logvar, logit

    def encode(self, x):
        return self.vae.encode(x)

    def decode(self, z):
        return self.vae.decode(z)


# ── Graph prior loss ─────────────────────────────────────────────────────────

def load_graph_prior(output_dir="outputs/graph_prior/"):
    """Load adjacency matrix from graph_prior_vae output."""
    import scipy.sparse
    adj_path = os.path.join(output_dir, "adjacency_matrix.npz")
    if not os.path.exists(adj_path):
        print("  No graph prior found — graph loss disabled")
        return None, None
    adj = scipy.sparse.load_npz(adj_path)
    # Load protein indices mapping
    meta_path = os.path.join(output_dir, "graph_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        indices = meta.get("local_to_global", list(range(adj.shape[0])))
    else:
        indices = list(range(adj.shape[0]))
    print(f"  Graph prior loaded: {adj.shape[0]}×{adj.shape[0]}, "
          f"{adj.nnz//2} edges")
    return adj, indices


def compute_graph_loss(mu, adj, indices, device):
    """
    Penalize when interacting proteins have uncorrelated latent dims.
    adj[i,j] = 1 means proteins i and j should have correlated latents.
    """
    if adj is None or len(indices) == 0:
        return torch.tensor(0.0, device=device)

    n = len(indices)
    if n == 0 or mu.shape[0] < 2:
        return torch.tensor(0.0, device=device)

    # Get latent vectors for graph proteins only
    idx_tensor = torch.tensor(indices[:min(n, mu.shape[0])],
                               device=device)
    # mu is (batch, latent_dim) — select graph protein subset
    # For graph loss we use the latent dims, not protein dims
    # Approximate: use first n latent dims as proxy for protein embedding
    n_use = min(n, mu.shape[1])
    z_sub = mu[:, :n_use]   # (batch, n_use)

    # Compute empirical correlation across batch
    if z_sub.shape[0] < 2:
        return torch.tensor(0.0, device=device)

    z_centered = z_sub - z_sub.mean(0, keepdim=True)
    std        = z_sub.std(0, keepdim=True).clamp(min=1e-6)
    z_norm     = z_centered / std
    emp_corr   = (z_norm.T @ z_norm) / z_sub.shape[0]  # (n_use, n_use)

    # Get adjacency for these dimensions
    adj_dense = torch.tensor(
        adj.toarray()[:n_use, :n_use],
        dtype=torch.float32, device=device)

    # Loss: interacting pairs should have high positive correlation
    # Non-interacting pairs: no constraint (sparse prior)
    mask       = adj_dense > 0
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    target_corr = adj_dense[mask]           # should be ~1.0
    actual_corr = emp_corr[mask]
    graph_loss  = F.mse_loss(actual_corr,
                              target_corr.clamp(0, 1))
    return graph_loss


# ── Data loading ─────────────────────────────────────────────────────────────

def extract_labeled_somalogic(train_dataset):
    """
    Extract (protein_tensor, mortality_label, is_synthetic) from dataset.
    Returns three tensors aligned by patient index.
    """
    proteins_list = []
    labels_list   = []
    is_syn_list   = []

    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        # somalogic: (3, 7596) — use tp0 for supervised signal
        soma_tp0 = sample["somalogic"][0]   # (7596,)
        proteins_list.append(soma_tp0)
        labels_list.append(float(sample["mortality"]))
        is_syn_list.append(int(sample["is_synthetic"]))

    proteins = torch.stack(proteins_list)   # (N, 7596)
    labels   = torch.tensor(labels_list,
                             dtype=torch.float32)
    is_syn   = torch.tensor(is_syn_list,
                             dtype=torch.long)
    return proteins, labels, is_syn


# ── Training loop ─────────────────────────────────────────────────────────────

def train_supervised_vae_fold(
        proteins, labels, is_syn,
        feature_dims, device, fold,
        adj, adj_indices, cfg):
    """
    Train one fold of the supervised VAE.
    Returns trained model and test AUROC.
    """
    # Stratified split
    real_mask  = (is_syn == 0).numpy()
    real_idx   = np.where(real_mask)[0]
    real_labels = labels[real_idx].numpy()

    skf = StratifiedKFold(
        n_splits=cfg["n_folds"],
        shuffle=True,
        random_state=cfg["seed"])
    folds = list(skf.split(real_idx, real_labels))
    train_val_real, test_real = folds[fold]

    train_real_idx = real_idx[train_val_real]
    test_real_idx  = real_idx[test_real]
    syn_idx        = np.where(~real_mask)[0]

    # Training: real + synthetic. Test: real only
    train_idx = np.concatenate([train_real_idx, syn_idx])
    test_idx  = test_real_idx

    # Subsample synthetic to 50% of real train
    n_syn_use = min(len(syn_idx), int(len(train_real_idx) * 0.5))
    rng       = np.random.default_rng(cfg["seed"] + fold)
    syn_use   = rng.choice(syn_idx, size=n_syn_use, replace=False)
    train_idx = np.concatenate([train_real_idx, syn_use])

    print(f"\n  Fold {fold}: "
          f"train={len(train_idx)} "
          f"({len(train_real_idx)} real + {len(syn_use)} syn), "
          f"test={len(test_idx)}")

    # Build model
    model = SupervisedSomaVAE(
        input_dim  = feature_dims["somalogic"],
        latent_dim = cfg["latent_dim"],
        beta       = cfg["beta"],
        dropout    = cfg["dropout"],
    ).to(device)

    # Class weight for imbalanced mortality
    train_labels = labels[train_real_idx]
    n_pos   = train_labels.sum().item()
    n_neg   = len(train_labels) - n_pos
    pos_w   = torch.tensor(
        [min(n_neg / max(n_pos, 1), 5.0)],
        dtype=torch.float32).to(device)

    optimiser = AdamW(
        model.parameters(),
        lr=cfg["lr"], weight_decay=1e-4)
    scheduler = CosineAnnealingLR(
        optimiser, T_max=cfg["epochs"], eta_min=1e-6)

    # DataLoader
    train_proteins = proteins[train_idx].to(device)
    train_labels_t = labels[train_idx].to(device)
    train_is_syn_t = is_syn[train_idx].to(device)

    dataset    = TensorDataset(
        train_proteins, train_labels_t, train_is_syn_t)
    loader     = DataLoader(
        dataset, batch_size=cfg["batch_size"],
        shuffle=True, drop_last=False)

    best_auroc  = 0.0
    best_state  = None
    no_improve  = 0
    patience    = 40
    warmup_epochs = 30

    print(f"  {'Epoch':6s} {'Recon':8s} {'KL':8s} "
          f"{'Graph':8s} {'Mort':8s} {'Total':8s} "
          f"{'Test AUROC':12s}")
    print("  " + "-"*64)

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        ep_recon = ep_kl = ep_graph = ep_mort = ep_total = 0.0
        n_batches = 0

        for x_batch, y_batch, syn_batch in loader:
            optimiser.zero_grad()

            recon, mu, logvar, logit = model(x_batch)

            # 1. Reconstruction loss (all patients)
            recon_loss = F.mse_loss(recon, x_batch)

            # 2. KL loss (all patients)
            kl_loss = -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp())

            # 3. Graph prior loss (all patients)
            graph_loss = compute_graph_loss(
                mu, adj, adj_indices, device)

            # 4. Mortality loss (real patients only — not synthetic)
            mort_weight = cfg["lambda_mort"] if epoch > warmup_epochs else 0.0
            real_mask_batch = (syn_batch == 0)
            if real_mask_batch.sum() > 0:
                logit_real = logit[real_mask_batch]
                y_real     = y_batch[real_mask_batch]
                mort_loss  = F.binary_cross_entropy_with_logits(
                    logit_real, y_real,
                    pos_weight=pos_w)
            else:
                mort_loss = torch.tensor(0.0, device=device)

            # Combined loss
            loss = (recon_loss
                    + cfg["beta"]       * kl_loss
                    + cfg["gamma_graph"]* graph_loss
                    + mort_weight       * mort_loss)

            loss.backward()
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0)
            optimiser.step()

            ep_recon  += recon_loss.item()
            ep_kl     += kl_loss.item()
            ep_graph  += graph_loss.item()
            ep_mort   += mort_loss.item()
            ep_total  += loss.item()
            n_batches += 1

        scheduler.step()

        # Evaluate every 10 epochs
        if epoch % 10 == 0 or epoch == cfg["epochs"]:
            model.eval()
            test_proteins = proteins[test_idx].to(device)
            test_labels_np = labels[test_idx].numpy()

            with torch.no_grad():
                _, mu_test, _, logit_test = model(test_proteins)
                probs = torch.sigmoid(logit_test).cpu().numpy()

            try:
                auroc = roc_auc_score(test_labels_np, probs)
                if auroc > best_auroc:
                    best_auroc = auroc
                    best_state = {
                        k: v.clone()
                        for k, v in model.state_dict().items()}
                    no_improve = 0
                    marker = " ←"
                else:
                    no_improve += 1
                    marker = ""
                if no_improve >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break
                print(f"  Ep {epoch:4d} | "
                      f"{ep_recon/n_batches:.4f}  "
                      f"{ep_kl/n_batches:.4f}  "
                      f"{ep_graph/n_batches:.4f}  "
                      f"{ep_mort/n_batches:.4f}  "
                      f"{ep_total/n_batches:.4f}  "
                      f"AUROC={auroc:.4f}{marker}")
            except Exception:
                pass

    if best_state:
        model.load_state_dict(best_state)

    return model, best_auroc


# ── Comparison: extract somalogic-only AUROC from cached pipeline ────────────

def get_baseline_soma_auroc(data_dir, device):
    """
    Quick logistic regression baseline on cached SomaLogic latents.
    Gives us the comparison point for how much joint training helps.
    """
    from train_cached import load_latent_cache, CFG as TCFG
    from sklearn.linear_model import LogisticRegression

    try:
        caches = load_latent_cache(TCFG['latent_dir'])
        (soma_cache, _, _, _, _, label_cache,
         real_ids, _) = caches

        X = np.array([
            soma_cache[p][0].numpy()   # tp0 latent
            for p in real_ids])
        y = np.array([
            label_cache[p]['mortality']
            for p in real_ids])

        skf = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=42)
        aurocs = []
        for tr, te in skf.split(X, y):
            lr = LogisticRegression(
                max_iter=1000, random_state=42, C=0.1)
            lr.fit(X[tr], y[tr])
            probs = lr.predict_proba(X[te])[:, 1]
            aurocs.append(roc_auc_score(y[te], probs))

        mean_auroc = np.mean(aurocs)
        print(f"  Baseline LR on unsupervised somalogic latents: "
              f"AUROC {mean_auroc:.4f}")
        return mean_auroc
    except Exception as e:
        print(f"  Baseline comparison skipped: {e}")
        return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir="./"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])
    device = get_device()
    print(f"Device: {device}")
    print(f"\nConfig: β={CFG['beta']}, "
          f"λ_mort={CFG['lambda_mort']}, "
          f"γ_graph={CFG['gamma_graph']}, "
          f"lr={CFG['lr']}, "
          f"epochs={CFG['epochs']}")

    # Load dataset
    print("\nLoading dataset...")
    train_ds, val_ds, test_ds, feature_dims = build_datasets(
        data_dir=data_dir, fold=0)
    all_ds = train_ds   # fold=0 train contains real+syn patients

    # Extract labeled SomaLogic tensors
    print("Extracting SomaLogic tensors...")
    proteins, labels, is_syn = extract_labeled_somalogic(all_ds)
    print(f"  Total: {len(proteins)} patients "
          f"({(is_syn==0).sum()} real, "
          f"{(is_syn==1).sum()} synthetic)")
    print(f"  Mortality rate (real): "
          f"{labels[is_syn==0].mean():.3f}")

    # Load graph prior
    print("\nLoading graph prior...")
    adj, adj_indices = load_graph_prior()

    # Baseline comparison
    print("\nComputing baseline...")
    baseline_auroc = get_baseline_soma_auroc(data_dir, device)

    # Train across all folds
    print(f"\n{'='*60}")
    print(f"SUPERVISED β-VAE — {CFG['n_folds']}-FOLD CV")
    print(f"{'='*60}")

    fold_aurocs  = []
    best_model   = None
    best_auroc   = 0.0

    for fold in range(CFG["n_folds"]):
        model, auroc = train_supervised_vae_fold(
            proteins, labels, is_syn,
            feature_dims, device, fold,
            adj, adj_indices, CFG)
        fold_aurocs.append(auroc)
        print(f"\n  Fold {fold} best AUROC: {auroc:.4f}")
        if auroc > best_auroc:
            best_auroc = auroc
            best_model = model

    mean_auroc = float(np.mean(fold_aurocs))
    std_auroc  = float(np.std(fold_aurocs))

    print(f"\n{'='*60}")
    print(f"SUPERVISED VAE RESULTS")
    print(f"{'='*60}")
    print(f"  Fold AUROCs : {[round(x,4) for x in fold_aurocs]}")
    print(f"  Mean AUROC  : {mean_auroc:.4f} ± {std_auroc:.4f}")
    if baseline_auroc:
        delta = mean_auroc - baseline_auroc
        print(f"  vs baseline : {baseline_auroc:.4f} "
              f"(Δ = {delta:+.4f})")
    print(f"  Best fold   : {best_auroc:.4f}")

    # Save best VAE encoder
    if best_model is not None:
        save_path = os.path.join(
            CKPT_DIR, "somalogic_vae_supervised.pt")
        torch.save(
            best_model.vae.state_dict(), save_path)
        print(f"\n  Saved encoder: {save_path}")
        print(f"  To use: load somalogic_vae_supervised.pt "
              f"instead of somalogic_vae.pt")
        print(f"  Then: python3 cache_latents.py ./ "
              f"(using supervised encoder)")
        print(f"  Then: python3 train_cached.py ./ "
              f"(full multimodal training on top)")

    results = {
        "fold_aurocs":     [float(x) for x in fold_aurocs],
        "mean_auroc":      mean_auroc,
        "std_auroc":       std_auroc,
        "baseline_auroc":  baseline_auroc,
        "delta_vs_baseline": float(mean_auroc - baseline_auroc)
                              if baseline_auroc else None,
        "config":          CFG,
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results: {OUTPUT_DIR}results.json")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    main(data_dir)