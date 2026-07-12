"""
swat_permutation_test.py
========================
Formal permutation test for SWAT external validation.

Tests whether the dose-response correlations observed in SWAT
are significantly better than chance, given that:
1. PAMPer-trained model predicts protein deltas
2. SWAT patients have continuous plasma overlap scores
3. We observe correlations between predicted delta and actual protein level

Permutation strategy:
    Null hypothesis: plasma overlap scores are randomly assigned
    to patients — i.e., proteome and dose are independent.
    
    Under H0: shuffle Overlap_plasma labels across patients,
    recompute dose-response correlations.
    Repeat 1000 times to build null distribution.

    p-value = fraction of permutations where
    mean |r| >= observed mean |r|

Usage:
    python3 swat_permutation_test.py ./
"""

import os, sys, json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from plasma_effect_model_v2 import (
    PlasmaEffectModelV2, CFG, HIGH_SIGNAL_PROTEINS, EXCLUDE
)

N_PERMUTATIONS = 1000
SEED           = 42
OUTPUT_DIR     = "outputs/plasma_effect/"
CKPT_PATH      = "outputs/checkpoints/plasma_effect_model_v2.pt"
SUMMARY        = "outputs/plasma_effect/summary_v2.json"


def load_swat_data(protein_cols):
    swat = pd.read_csv("SWAT_proteins_clinical.csv")
    tp0  = swat[swat["Time point"] == 0].copy()

    avail = [p for p in protein_cols if p in tp0.columns]
    vals  = tp0[avail].values.astype(np.float32)
    vals  = np.log1p(np.clip(vals, 0, None))

    meds = np.nanmedian(vals, axis=0)
    meds = np.where(np.isnan(meds), 0.0, meds)
    for j in range(vals.shape[1]):
        vals[np.isnan(vals[:, j]), j] = meds[j]

    # Pad missing proteins
    if len(avail) < len(protein_cols):
        full      = np.zeros((len(vals), len(protein_cols)), dtype=np.float32)
        avail_idx = [protein_cols.index(p) for p in avail]
        full[:, avail_idx] = vals
        vals = full

    doses = tp0["Overlap_plasma"].values.astype(np.float32)
    return vals, doses


def compute_observed_stats(model, swat_z, swat_doses, protein_cols, device):
    """Compute observed dose-response correlations."""
    model.eval()
    swat_t  = torch.tensor(swat_z).to(device)
    doses_t = torch.tensor(swat_doses).unsqueeze(-1).to(device)

    with torch.no_grad():
        _, pred_delta = model(swat_t, doses_t)
    pred_delta = pred_delta.cpu().numpy()

    exclude = EXCLUDE
    valid   = [p for p in HIGH_SIGNAL_PROTEINS
               if p in protein_cols and p not in exclude]

    # Per-protein: correlation between dose and predicted delta
    r_pred_dose  = []  # predicted delta vs dose
    r_obs_dose   = []  # observed protein level vs dose

    for p in valid:
        i = protein_cols.index(p)
        r_pd, _ = stats.pearsonr(swat_doses, pred_delta[:, i])
        r_od, _ = stats.pearsonr(swat_doses, swat_z[:, i])
        r_pred_dose.append(r_pd)
        r_obs_dose.append(r_od)

    return (np.array(r_pred_dose), np.array(r_obs_dose),
            valid, pred_delta)


def permutation_test(model, swat_z, swat_doses, protein_cols,
                     device, n_perm=N_PERMUTATIONS):
    """
    Permutation test: shuffle dose labels, recompute correlations.
    """
    rng = np.random.default_rng(SEED)

    # Observed
    r_pred, r_obs, valid, pred_delta_obs = compute_observed_stats(
        model, swat_z, swat_doses, protein_cols, device)

    obs_mean_r     = np.mean(r_pred)
    obs_mean_abs_r = np.mean(np.abs(r_pred))
    obs_n_pos      = np.sum(r_pred > 0)
    obs_concordant = np.sum((r_pred > 0) == (r_obs > 0))

    print(f"\nObserved statistics:")
    print(f"  Mean r (pred delta vs dose):     {obs_mean_r:.4f}")
    print(f"  Mean |r|:                        {obs_mean_abs_r:.4f}")
    print(f"  Positive direction:              {obs_n_pos}/{len(valid)}")
    print(f"  Concordant with observed effect: {obs_concordant}/{len(valid)}")

    # Permutation null distribution
    print(f"\nRunning {n_perm} permutations...")
    null_mean_r     = []
    null_mean_abs_r = []
    null_n_pos      = []
    null_concordant = []

    model.eval()
    for perm in range(n_perm):
        # Shuffle dose labels
        shuffled_doses = rng.permutation(swat_doses)

        swat_t  = torch.tensor(swat_z).to(device)
        doses_t = torch.tensor(
            shuffled_doses).unsqueeze(-1).to(device)

        with torch.no_grad():
            _, pred_delta_perm = model(swat_t, doses_t)
        pred_delta_perm = pred_delta_perm.cpu().numpy()

        r_perm = []
        for p in valid:
            i = protein_cols.index(p)
            r, _ = stats.pearsonr(shuffled_doses,
                                   pred_delta_perm[:, i])
            r_perm.append(r)

        r_perm = np.array(r_perm)
        null_mean_r.append(np.mean(r_perm))
        null_mean_abs_r.append(np.mean(np.abs(r_perm)))
        null_n_pos.append(np.sum(r_perm > 0))
        null_concordant.append(
            np.sum((r_perm > 0) == (r_obs > 0)))

        if (perm + 1) % 100 == 0:
            print(f"  Permutation {perm+1}/{n_perm}...")

    null_mean_r     = np.array(null_mean_r)
    null_mean_abs_r = np.array(null_mean_abs_r)
    null_n_pos      = np.array(null_n_pos)
    null_concordant = np.array(null_concordant)

    # P-values
    p_mean_r     = (null_mean_r >= obs_mean_r).mean()
    p_abs_r      = (null_mean_abs_r >= obs_mean_abs_r).mean()
    p_n_pos      = (null_n_pos >= obs_n_pos).mean()
    p_concordant = (null_concordant >= obs_concordant).mean()

    print(f"\nPermutation test results (n={n_perm}):")
    print(f"  {'Statistic':35s} {'Observed':10s} {'Null mean':10s} {'p-value':8s}")
    print(f"  " + "-"*65)
    print(f"  {'Mean r (pred delta vs dose)':35s} "
          f"{obs_mean_r:.4f}     {null_mean_r.mean():.4f}     {p_mean_r:.4f}")
    print(f"  {'Mean |r|':35s} "
          f"{obs_mean_abs_r:.4f}     {null_mean_abs_r.mean():.4f}     {p_abs_r:.4f}")
    print(f"  {'N positive direction':35s} "
          f"{obs_n_pos:>8d}     {null_n_pos.mean():.1f}       {p_n_pos:.4f}")
    print(f"  {'N concordant with observed':35s} "
          f"{obs_concordant:>8d}     {null_concordant.mean():.1f}       {p_concordant:.4f}")

    return {
        "observed":     {
            "mean_r":     float(obs_mean_r),
            "mean_abs_r": float(obs_mean_abs_r),
            "n_pos":      int(obs_n_pos),
            "n_concordant": int(obs_concordant),
            "n_valid":    len(valid),
        },
        "null":         {
            "mean_r_dist":     null_mean_r.tolist(),
            "mean_abs_r_dist": null_mean_abs_r.tolist(),
        },
        "p_values":     {
            "mean_r":       float(p_mean_r),
            "mean_abs_r":   float(p_abs_r),
            "n_pos":        float(p_n_pos),
            "n_concordant": float(p_concordant),
        },
        "r_per_protein": dict(zip(valid, r_pred.tolist())),
    }


def plot_permutation(results, output_dir):
    """Visualize permutation test results."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    null_r     = np.array(results["null"]["mean_r_dist"])
    null_abs_r = np.array(results["null"]["mean_abs_r_dist"])
    obs_r      = results["observed"]["mean_r"]
    obs_abs_r  = results["observed"]["mean_abs_r"]
    p_r        = results["p_values"]["mean_r"]
    p_abs      = results["p_values"]["mean_abs_r"]

    # Panel 1: Mean r null distribution
    ax = axes[0]
    ax.hist(null_r, bins=50, color='#90CAF9', edgecolor='white',
            alpha=0.8, label='Null distribution')
    ax.axvline(obs_r, color='#C62828', lw=2.5,
               label=f'Observed r={obs_r:.3f}')
    ax.axvline(np.percentile(null_r, 95), color='#F57C00',
               lw=1.5, ls='--', label='95th percentile null')
    ax.set_xlabel("Mean Pearson r (predicted delta vs plasma dose)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Permutation Test — Mean Dose-Response r\n"
                 f"p={p_r:.4f} (n={len(null_r)} permutations)",
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 2: Per-protein r values
    ax2 = axes[1]
    valid     = list(results["r_per_protein"].keys())
    r_vals    = list(results["r_per_protein"].values())
    sort_idx  = np.argsort(r_vals)[::-1]
    short     = [valid[i].replace("coagulation factor ","F")
                          .replace("vitamin k-dependent protein ","VitK-")
                          .replace("activated protein c","APC")
                          .replace("coagulation factor xiii b chain","FXIII-B")
                          .title()
                 for i in sort_idx]
    r_sorted  = [r_vals[i] for i in sort_idx]
    bar_cols  = ['#1B5E20' if r>0 else '#B71C1C' for r in r_sorted]
    ax2.barh(short, r_sorted, color=bar_cols, alpha=0.8)
    ax2.axvline(0, color='black', lw=0.8)

    # Add null 95th percentile band
    p95 = np.percentile(np.abs(null_r), 95)
    ax2.axvspan(-p95, p95, alpha=0.1, color='gray',
                label='Null 95% band')
    ax2.set_xlabel("Pearson r (predicted delta vs dose)")
    ax2.set_title("Per-Protein Dose-Response Correlation\n"
                  "Gray band = null 95% CI",
                  fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, axis='x', alpha=0.3)

    plt.suptitle(
        "SWAT External Validation — Permutation Test\n"
        f"PAMPer-trained model, n={len(null_r)} permutations, "
        f"p={p_r:.4f}",
        fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, "swat_permutation_test.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure: {path}")
    plt.close()


def main(data_dir="./"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    np.random.seed(SEED)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    with open(SUMMARY) as f:
        summary = json.load(f)
    protein_cols = summary["protein_cols"]
    CFG["protein_dim"] = len(protein_cols)

    model = PlasmaEffectModelV2(
        CFG["protein_dim"], CFG["hidden_dim"],
        CFG["dose_dim"],    CFG["dropout"])
    model.load_state_dict(
        torch.load(CKPT_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded: {CKPT_PATH}")

    # Load SWAT
    print("Loading SWAT...")
    swat_z, swat_doses = load_swat_data(protein_cols)
    print(f"  {len(swat_z)} patients, dose range "
          f"{swat_doses.min():.3f}-{swat_doses.max():.3f}")

    # Run permutation test
    print(f"\n{'='*60}")
    print("SWAT PERMUTATION TEST")
    print(f"{'='*60}")
    results = permutation_test(
        model, swat_z, swat_doses, protein_cols, device)

    # Save results
    with open(os.path.join(OUTPUT_DIR, "swat_permutation_results.json"),
              "w") as f:
        json.dump(results, f, indent=2)

    # Plot
    plot_permutation(results, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print("PERMUTATION TEST COMPLETE")
    print(f"{'='*60}")
    print(f"  p(mean_r):       {results['p_values']['mean_r']:.4f}")
    print(f"  p(n_positive):   {results['p_values']['n_pos']:.4f}")
    print(f"  p(concordance):  {results['p_values']['n_concordant']:.4f}")
    print()
    p = results['p_values']['mean_r']
    if p < 0.001:
        print("  *** p < 0.001: Highly significant external validation")
    elif p < 0.01:
        print("  ** p < 0.01: Significant external validation")
    elif p < 0.05:
        print("  * p < 0.05: Significant external validation")
    else:
        print("  p >= 0.05: Not significant")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "./")
