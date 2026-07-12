"""
plasma_effect_model.py
======================
Answers: "If this patient had received prehospital plasma,
what would their coagulation profile look like at admission?"

The PAMPer RCT gave plasma BEFORE the tp0 blood draw.
So the cross-sectional difference between arms at tp0 IS the plasma effect.
We learn to map control arm proteomes to plasma arm proteomes.
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR   = "coag_data/"
OUTPUT_DIR = "outputs/plasma_effect/"
CKPT_PATH  = "outputs/checkpoints/plasma_effect_model.pt"

CFG = {
    "protein_dim": 33,
    "hidden_dim":  128,
    "dropout":     0.3,
    "lr":          1e-3,
    "epochs":      500,
    "batch_size":  16,
    "patience":    60,
    "seed":        42,
    "n_folds":     5,
}

HIGH_SIGNAL_PROTEINS = [
    "coagulation factor xi",
    "coagulation factor v",
    "coagulation factor xa",
    "coagulation factor xiii",
    "coagulation factor x",
    "vitamin k-dependent protein s",
    "coagulation factor ixab",
    "coagulation factor vii",
    "activated protein c",
    "coagulation factor ix",
    "coagulation factor xiii b chain",
    "plasminogen",
    "thrombin",
]


# ── Model ─────────────────────────────────────────────────────────────────────

class PlasmaEffectModel(nn.Module):
    def __init__(self, protein_dim, hidden_dim, dropout=0.3):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(protein_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.delta_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ELU(),
            nn.Linear(hidden_dim // 2, protein_dim),
        )
        nn.init.uniform_(self.delta_head[-1].weight, -0.01, 0.01)
        nn.init.zeros_(self.delta_head[-1].bias)

    def forward(self, z_control):
        h     = self.encoder(z_control)
        delta = self.delta_head(h)
        return z_control + delta, delta


# ── Data ──────────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(os.path.join(DATA_DIR, "pamper_coag.csv"))
    meta = ["patient_id", "timepoint", "intervention", "mortality", "cohort"]
    protein_cols = [c for c in df.columns if c not in meta]

    tp0     = df[df["timepoint"] == 0].copy()
    plasma  = tp0[tp0["intervention"] == 1].reset_index(drop=True)
    control = tp0[tp0["intervention"] == 0].reset_index(drop=True)

    all_vals = tp0[protein_cols].values.astype(np.float32)
    col_meds = np.nanmedian(all_vals, axis=0)
    col_meds = np.where(np.isnan(col_meds), 0.0, col_meds)

    def impute(df_sub):
        v = df_sub[protein_cols].values.astype(np.float32)
        for j in range(v.shape[1]):
            v[np.isnan(v[:, j]), j] = col_meds[j]
        return v

    plasma_z  = impute(plasma)
    control_z = impute(control)
    pop_delta = plasma_z.mean(0) - control_z.mean(0)

    print(f"  Plasma:   {len(plasma_z)} patients")
    print(f"  Control:  {len(control_z)} patients")
    print(f"  Proteins: {len(protein_cols)}")
    return plasma_z, control_z, protein_cols, col_meds, pop_delta


# ── Training ──────────────────────────────────────────────────────────────────

def train_one(model, X_tr, Y_tr, X_val, Y_val, device, cfg):
    Xt = torch.tensor(X_tr).to(device)
    Yt = torch.tensor(Y_tr).to(device)
    Xv = torch.tensor(X_val).to(device)
    Yv = torch.tensor(Y_val).to(device)

    ds     = TensorDataset(Xt, Yt)
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=True)
    opt    = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    sched  = CosineAnnealingLR(opt, T_max=cfg["epochs"], eta_min=1e-6)

    best_val   = float("inf")
    best_state = None
    no_improve = 0

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred, delta = model(xb)
            loss = F.mse_loss(pred, yb) + 0.1 * (delta ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            vp, _ = model(Xv)
            vloss = F.mse_loss(vp, Yv).item()
        if vloss < best_val:
            best_val   = vloss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= cfg["patience"]:
            break

    if best_state:
        model.load_state_dict(best_state)
    return model, best_val


def build_pairs(control_z, plasma_z, rng):
    n_ctrl  = len(control_z)
    n_plasm = len(plasma_z)
    n_pair   = min(len(control_z), n_plasm)
    ctrl_use = control_z[:n_pair]
    Xs, Ys   = [], []
    for _ in range(3):
        perm = rng.permutation(n_plasm)[:n_pair]
        Xs.append(ctrl_use)
        Ys.append(plasma_z[perm])
    # Identity pairs — same size as one random pairing
    id_perm = rng.permutation(n_plasm)[:n_pair]
    Xs.append(plasma_z[id_perm])
    Ys.append(plasma_z[id_perm])
    return np.vstack(Xs).astype(np.float32), np.vstack(Ys).astype(np.float32)


def train_full(plasma_z, control_z, protein_cols, device, cfg):
    rng  = np.random.default_rng(cfg["seed"])
    kf   = KFold(n_splits=cfg["n_folds"], shuffle=True,
                 random_state=cfg["seed"])
    fold_losses = []
    mean_plasma = plasma_z.mean(0, keepdims=True)

    print("  Fold  Val Loss  MAE")
    print("  " + "-" * 25)

    for fold, (tr_idx, val_idx) in enumerate(kf.split(np.arange(len(control_z)))):
        ctrl_tr  = control_z[tr_idx]
        ctrl_val = control_z[val_idx]

        X_tr, Y_tr = build_pairs(ctrl_tr, plasma_z, rng)
        X_val = ctrl_val.astype(np.float32)
        Y_val = np.tile(mean_plasma, (len(ctrl_val), 1)).astype(np.float32)

        m = PlasmaEffectModel(cfg["protein_dim"],
                              cfg["hidden_dim"],
                              cfg["dropout"]).to(device)
        m, vloss = train_one(m, X_tr, Y_tr, X_val, Y_val, device, cfg)

        m.eval()
        with torch.no_grad():
            vp, _ = m(torch.tensor(X_val).to(device))
            mae   = (vp - torch.tensor(Y_val).to(device)).abs().mean().item()

        fold_losses.append(vloss)
        print(f"  {fold}     {vloss:.4f}    {mae:.4f}")

    print(f"\n  Mean val loss: {np.mean(fold_losses):.4f} +/- {np.std(fold_losses):.4f}")

    print("\n  Training final model on all data...")
    X_all, Y_all = build_pairs(control_z, plasma_z, rng)
    final = PlasmaEffectModel(cfg["protein_dim"],
                              cfg["hidden_dim"],
                              cfg["dropout"]).to(device)
    final, _ = train_one(final, X_all, Y_all, X_all, Y_all, device, cfg)
    return final, fold_losses


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model, plasma_z, control_z, protein_cols, device):
    print("\n" + "=" * 60)
    print("BIOLOGICAL VALIDATION")
    print("=" * 60)

    model.eval()
    ctrl_t = torch.tensor(control_z).to(device)
    with torch.no_grad():
        _, pred_delta = model(ctrl_t)
    pred_delta = pred_delta.cpu().numpy()

    true_delta      = plasma_z.mean(0) - control_z.mean(0)
    pred_mean_delta = pred_delta.mean(0)

    print(f"\n  {'':2s} {'Protein':35s} {'True':8s} {'Pred':8s}")
    print("  " + "-" * 55)

    correct = 0
    valid_proteins = [p for p in HIGH_SIGNAL_PROTEINS if p in protein_cols]
    for p in valid_proteins:
        i   = protein_cols.index(p)
        td  = true_delta[i]
        pd_ = pred_mean_delta[i]
        ok  = (td > 0 and pd_ > 0) or (td < 0 and pd_ < 0)
        if ok:
            correct += 1
        mark = "ok" if ok else "WRONG"
        print(f"  {'v' if ok else 'x'} {p:35s} {td:+.4f}   {pd_:+.4f}   {mark}")

    r, pval = stats.pearsonr(true_delta, pred_mean_delta)
    print(f"\n  Correct direction: {correct}/{len(valid_proteins)}")
    print(f"  Correlation: r={r:.3f}, p={pval:.4f}")
    return pred_delta, true_delta


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(pred_delta, true_delta, protein_cols, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    ax.scatter(true_delta, pred_delta.mean(0), alpha=0.7, s=60,
               color='#1565C0')
    for p in HIGH_SIGNAL_PROTEINS:
        if p not in protein_cols:
            continue
        i = protein_cols.index(p)
        short = (p.replace("coagulation factor ", "F")
                  .replace("vitamin k-dependent protein ", "VitK-")
                  .replace("activated protein c", "APC")
                  .title())
        ax.annotate(short, (true_delta[i], pred_delta.mean(0)[i]),
                    fontsize=6, alpha=0.8)
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axvline(0, color='gray', lw=0.8, ls='--')
    r, _ = stats.pearsonr(true_delta, pred_delta.mean(0))
    ax.set_xlabel("True plasma effect (log1p)")
    ax.set_ylabel("Predicted plasma effect (log1p)")
    ax.set_title(f"Plasma Effect: True vs Predicted  r={r:.3f}",
                 fontweight='bold')
    ax.grid(True, alpha=0.3)

    ax2 = axes[1]
    sort_idx = np.argsort(true_delta)
    names = [protein_cols[i]
             .replace("coagulation factor ", "F")
             .replace("antithrombin-iii", "AT-III")
             .replace("tissue factor pathway inhibitor", "TFPI")
             .replace("tissue factor pathway inhibitor 2", "TFPI-2")
             .replace("tissue-type plasminogen activator", "tPA")
             .replace("urokinase-type plasminogen activator", "uPA")
             .replace("urokinase plasminogen activator surface receptor", "uPAR")
             .replace("plasminogen activator inhibitor 1", "PAI-1")
             .replace("vitamin k-dependent protein ", "VitK-")
             .replace("activated protein c", "APC")
             .replace("mannose-binding protein c", "MBP-C")
             .replace("fibrinogen-like protein 1", "FGL1")
             .replace("fibrinogen c domain-containing protein 1", "FCDom1")
             .replace("soluble endothelial protein c receptor", "sEPCR")
             for i in sort_idx]
    tv = true_delta[sort_idx]
    pv = pred_delta.mean(0)[sort_idx]
    x  = np.arange(len(sort_idx))
    w  = 0.35
    ax2.barh(x - w/2, tv, w,
             color=['#1565C0' if v > 0 else '#C62828' for v in tv],
             alpha=0.8, label='True')
    ax2.barh(x + w/2, pv, w,
             color=['#42A5F5' if v > 0 else '#EF9A9A' for v in pv],
             alpha=0.8, label='Predicted')
    ax2.set_yticks(x)
    ax2.set_yticklabels(names, fontsize=7)
    ax2.axvline(0, color='black', lw=0.8)
    ax2.set_xlabel("Plasma effect (log1p)")
    ax2.set_title("Per-Protein Plasma Effect", fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='x', alpha=0.3)

    plt.suptitle("Plasma Effect Model Validation — PAMPer RCT",
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, "plasma_effect_validation.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure: {path}")
    plt.close()


# ── Patient Report ────────────────────────────────────────────────────────────

def patient_report(model, z0, protein_cols, device, patient_id="?"):
    model.eval()
    z_t = torch.tensor(z0, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        z_plasma, delta = model(z_t)
    z_plasma = z_plasma.squeeze(0).cpu().numpy()
    delta    = delta.squeeze(0).cpu().numpy()

    print(f"\n  Patient {patient_id}")
    print(f"  {'Protein':35s} {'Actual':8s} {'If Plasma':10s} {'Change':8s}")
    print("  " + "-" * 63)

    show = sorted([p for p in HIGH_SIGNAL_PROTEINS if p in protein_cols],
                  key=lambda p: -abs(delta[protein_cols.index(p)]))
    for p in show[:10]:
        i   = protein_cols.index(p)
        act = z0[i]
        pls = z_plasma[i]
        d   = delta[i]
        print(f"  {'up' if d>0 else 'dn'} {p:35s} {act:.3f}    {pls:.3f}"
              f"       {d:+.3f}")
    return z_plasma, delta


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir="./"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs("outputs/checkpoints", exist_ok=True)
    torch.manual_seed(CFG["seed"])
    np.random.seed(CFG["seed"])

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading data...")
    plasma_z, control_z, protein_cols, col_meds, pop_delta = load_data()
    CFG["protein_dim"] = len(protein_cols)

    print("\n" + "=" * 60)
    print("TRAINING PLASMA EFFECT MODEL")
    print("=" * 60)
    model, fold_losses = train_full(
        plasma_z, control_z, protein_cols, device, CFG)

    torch.save(model.state_dict(), CKPT_PATH)
    print(f"\nSaved: {CKPT_PATH}")

    pred_delta, true_delta = validate(
        model, plasma_z, control_z, protein_cols, device)

    plot_results(pred_delta, true_delta, protein_cols, OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("PATIENT REPORTS — Control arm counterfactuals")
    print("=" * 60)

    df   = pd.read_csv(os.path.join(DATA_DIR, "pamper_coag.csv"))
    tp0  = df[df["timepoint"] == 0]
    ctrl = tp0[tp0["intervention"] == 0]

    ctrl_surv = ctrl[ctrl["mortality"] == 0].iloc[0]
    ctrl_mort = ctrl[ctrl["mortality"] == 1].iloc[0]

    for row, label in [(ctrl_surv, "control_survivor"),
                       (ctrl_mort, "control_nonsurvivour")]:
        z0 = row[protein_cols].values.astype(np.float32)
        z0[np.isnan(z0)] = col_meds[np.isnan(z0)]
        patient_report(model, z0, protein_cols, device,
                       f"{row['patient_id']} ({label})")

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump({
            "protein_cols":  protein_cols,
            "fold_losses":   [float(x) for x in fold_losses],
            "mean_val_loss": float(np.mean(fold_losses)),
        }, f, indent=2)

    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"  Mean CV loss: {np.mean(fold_losses):.4f}")
    print(f"  Checkpoint:   {CKPT_PATH}")
    print(f"  Figure:       {OUTPUT_DIR}plasma_effect_validation.png")
    print()
    print("  This model answers:")
    print("  'If this patient had received prehospital plasma,")
    print("   what would their coagulation profile look like?'")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./")