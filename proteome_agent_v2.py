"""
proteome_agent_v2.py — Interventional Proteome Agent v2
=========================================================
Upgraded agent using plasma_effect_model_v2.
Now answers dose-response questions, not just binary predictions.

Clinical queries:
    1. What would this patient's proteome look like at any plasma dose?
    2. What is the minimum dose needed to normalize key coagulation factors?
    3. How does this patient's dose-response compare to similar patients?
    4. Which proteins are most sensitive to plasma dose for this patient?

Usage:
    python3 proteome_agent_v2.py ./ --demo
    python3 proteome_agent_v2.py ./ --patient PAMP1002
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

sys.path.insert(0, os.path.dirname(__file__))
from plasma_effect_model_v2 import (
    PlasmaEffectModelV2, CFG, HIGH_SIGNAL_PROTEINS, EXCLUDE
)

DATA_DIR   = "coag_data/"
OUTPUT_DIR = "outputs/proteome_agent_v2/"
CKPT_PATH  = "outputs/checkpoints/plasma_effect_model_v2.pt"
SUMMARY    = "outputs/plasma_effect/summary_v2.json"

PATHWAY_GROUPS = {
    "Extrinsic":          ["coagulation factor vii", "tissue factor",
                           "tissue factor pathway inhibitor",
                           "tissue factor pathway inhibitor 2"],
    "Intrinsic":          ["coagulation factor viii", "coagulation factor ix",
                           "coagulation factor xi", "coagulation factor ixab"],
    "Common pathway":     ["coagulation factor x", "coagulation factor xa",
                           "coagulation factor v", "prothrombin", "thrombin",
                           "coagulation factor xiii",
                           "coagulation factor xiii b chain"],
    "Fibrinogen":         ["fibrinogen", "fibrinogen beta chain",
                           "fibrinogen gamma chain",
                           "fibrinogen-like protein 1"],
    "Anticoagulants":     ["antithrombin-iii", "activated protein c",
                           "vitamin k-dependent protein c",
                           "vitamin k-dependent protein s",
                           "soluble endothelial protein c receptor"],
    "Fibrinolysis":       ["plasminogen", "plasminogen activator inhibitor 1",
                           "urokinase-type plasminogen activator"],
}

DOSE_LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


# ── Agent ─────────────────────────────────────────────────────────────────────

class ProteomeAgentV2:
    def __init__(self, model, protein_cols, pamper_df, device):
        self.model        = model
        self.protein_cols = protein_cols
        self.pamper_df    = pamper_df
        self.device       = device
        self.model.eval()
        self._build_reference_db()

    def _build_reference_db(self):
        self.ref_db   = {}
        self.col_meds = np.nanmedian(
            self.pamper_df[self.pamper_df["timepoint"]==0][self.protein_cols]
            .values.astype(np.float32), axis=0)
        self.col_meds = np.where(np.isnan(self.col_meds), 0.0, self.col_meds)

        tp0 = self.pamper_df[self.pamper_df["timepoint"]==0]
        for _, row in tp0.iterrows():
            prof = row[self.protein_cols].values.astype(np.float32)
            prof[np.isnan(prof)] = self.col_meds[np.isnan(prof)]
            self.ref_db[row["patient_id"]] = {
                "profile":      prof,
                "intervention": int(row["intervention"]),
                "mortality":    int(row["mortality"]),
            }
        print(f"  Reference DB: {len(self.ref_db)} PAMPer patients")

    def _impute(self, z0):
        z0 = z0.copy()
        z0[np.isnan(z0)] = self.col_meds[np.isnan(z0)]
        return z0

    def predict_at_dose(self, z0, dose):
        """Predict proteome at a specific plasma dose."""
        z0  = self._impute(z0)
        z_t = torch.tensor(z0, dtype=torch.float32).unsqueeze(0).to(self.device)
        d_t = torch.tensor([[dose]], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            z_pred, delta = self.model(z_t, d_t)
        return (z_pred.squeeze(0).cpu().numpy(),
                delta.squeeze(0).cpu().numpy())

    def dose_response_curve(self, z0, doses=None):
        """Predict proteome at multiple dose levels."""
        if doses is None:
            doses = DOSE_LEVELS
        results = {}
        for d in doses:
            z_pred, delta = self.predict_at_dose(z0, d)
            results[d] = {"z_pred": z_pred, "delta": delta}
        return results

    def minimum_effective_dose(self, z0, target_proteins=None,
                               target_percentile=75):
        """
        Find the minimum dose needed to bring key proteins
        above the population median for plasma arm patients.
        """
        if target_proteins is None:
            target_proteins = [
                p for p in HIGH_SIGNAL_PROTEINS
                if p in self.protein_cols and p not in EXCLUDE]

        # Compute plasma arm median from reference DB
        plasma_profiles = np.stack([
            v["profile"] for v in self.ref_db.values()
            if v["intervention"] == 1])
        plasma_targets = np.percentile(
            plasma_profiles, target_percentile, axis=0)

        # Sweep doses
        doses_sweep = np.linspace(0, 1, 51)
        z0 = self._impute(z0)

        med_results = {}
        for p in target_proteins:
            if p not in self.protein_cols:
                continue
            i      = self.protein_cols.index(p)
            target = plasma_targets[i]
            med    = None
            for d in doses_sweep:
                z_pred, _ = self.predict_at_dose(z0, d)
                if z_pred[i] >= target:
                    med = float(d)
                    break
            med_results[p] = {
                "min_dose":      med,
                "baseline":      float(z0[i]),
                "plasma_target": float(target),
                "achievable":    med is not None,
            }
        return med_results

    def pathway_dose_response(self, z0):
        """Pathway-level dose-response summary."""
        curve = self.dose_response_curve(z0)
        results = {}
        for pathway, proteins in PATHWAY_GROUPS.items():
            idxs = [self.protein_cols.index(p)
                    for p in proteins if p in self.protein_cols]
            if not idxs:
                continue
            dose_effects = []
            for d in DOSE_LEVELS:
                delta_mean = curve[d]["delta"][idxs].mean()
                dose_effects.append(float(delta_mean))
            results[pathway] = {
                "dose_effects": dict(zip(DOSE_LEVELS, dose_effects)),
                "max_effect":   max(dose_effects),
                "n_proteins":   len(idxs),
            }
        return results

    def find_similar(self, z0, n=5):
        """Find most similar PAMPer patients."""
        z0   = self._impute(z0)
        dists = {}
        for pid, info in self.ref_db.items():
            mask = np.isfinite(z0) & np.isfinite(info["profile"])
            if mask.sum() < 10:
                continue
            d = np.sqrt(((z0[mask] - info["profile"][mask])**2).mean())
            dists[pid] = d
        similar = sorted(dists.items(), key=lambda x: x[1])[:n]
        return [{"patient_id": pid, "distance": float(d),
                 "treatment": "plasma" if self.ref_db[pid]["intervention"]==1
                              else "control",
                 "mortality": int(self.ref_db[pid]["mortality"])}
                for pid, d in similar]

    def clinical_report(self, z0, patient_id="Unknown"):
        """Full clinical report."""
        print(f"\n{'='*68}")
        print(f"PROTEOME AGENT v2 — Patient: {patient_id}")
        print(f"{'='*68}")

        z0   = self._impute(z0)
        curve = self.dose_response_curve(z0)

        # 1. Dose-response table
        print(f"\n1. COAGULATION DOSE-RESPONSE (key proteins)")
        print(f"   {'Protein':35s} {'d=0':8s} {'d=0.5':8s} {'d=1.0':10s} "
              f"{'Max gain':8s}")
        print(f"   " + "-"*72)

        gains = []
        for p in HIGH_SIGNAL_PROTEINS:
            if p not in self.protein_cols or p in EXCLUDE:
                continue
            i    = self.protein_cols.index(p)
            z0v  = curve[0.0]["z_pred"][i]
            z05  = curve[0.5]["z_pred"][i]
            z1   = curve[1.0]["z_pred"][i]
            gain = z1 - z0v
            gains.append((p, gain))
            mark = "↑" if gain > 0 else "↓"
            print(f"   {mark} {p:35s} {z0v:.3f}    {z05:.3f}    "
                  f"{z1:.3f}      {gain:+.3f}")

        # 2. Minimum effective dose
        print(f"\n2. MINIMUM EFFECTIVE DOSE (to reach plasma-arm 75th percentile)")
        med = self.minimum_effective_dose(z0)
        print(f"   {'Protein':35s} {'Min dose':10s} {'Baseline':10s} "
              f"{'Achievable':10s}")
        print(f"   " + "-"*65)
        for p, info in sorted(med.items(),
                               key=lambda x: x[1]["min_dose"] or 2):
            d_str = f"{info['min_dose']:.2f}" if info["achievable"] else ">1.0"
            print(f"   {p:35s} {d_str:10s} {info['baseline']:.3f}     "
                  f"{'YES' if info['achievable'] else 'NO'}")

        # 3. Pathway response
        print(f"\n3. PATHWAY-LEVEL RESPONSE AT FULL DOSE")
        pw = self.pathway_dose_response(z0)
        print(f"   {'Pathway':25s} {'Effect at d=0.5':16s} "
              f"{'Effect at d=1.0':16s} {'N':4s}")
        print(f"   " + "-"*63)
        for pathway, info in sorted(pw.items(),
                                     key=lambda x: -x[1]["max_effect"]):
            e05 = info["dose_effects"][0.5]
            e10 = info["dose_effects"][1.0]
            mark = "↑" if e10 > 0 else "↓"
            print(f"   {pathway:25s} {mark}{abs(e05):.4f}          "
                  f"{mark}{abs(e10):.4f}          {info['n_proteins']}")

        # 4. Similar patients
        print(f"\n4. SIMILAR PAMPer PATIENTS")
        similar = self.find_similar(z0)
        print(f"   {'Patient':12s} {'Dist':8s} {'Treatment':12s} {'Outcome':8s}")
        print(f"   " + "-"*42)
        for s in similar:
            outcome = "Died" if s["mortality"]==1 else "Survived"
            print(f"   {s['patient_id']:12s} {s['distance']:.4f}   "
                  f"{s['treatment']:12s} {outcome}")

        print(f"\n{'='*68}")
        return curve, med, pw


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_agent_report(agent, z0, patient_id, output_dir):
    """4-panel dose-response report figure."""
    z0    = agent._impute(z0)
    curve = agent.dose_response_curve(z0)
    pw    = agent.pathway_dose_response(z0)
    med   = agent.minimum_effective_dose(z0)

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.42, wspace=0.35)

    # Panel 1: Dose-response curves for top proteins
    ax1 = fig.add_subplot(gs[0, 0])
    top_proteins = sorted(
        [p for p in HIGH_SIGNAL_PROTEINS
         if p in agent.protein_cols and p not in EXCLUDE],
        key=lambda p: -abs(
            curve[1.0]["delta"][agent.protein_cols.index(p)]))[:6]

    import matplotlib.cm as cm
    colors = cm.Set1(np.linspace(0, 0.8, len(top_proteins)))
    for k, (p, color) in enumerate(zip(top_proteins, colors)):
        i      = agent.protein_cols.index(p)
        vals   = [curve[d]["z_pred"][i] for d in DOSE_LEVELS]
        short  = p.replace("coagulation factor ", "F") \
                  .replace("antithrombin-iii", "AT-III") \
                  .replace("vitamin k-dependent protein ", "VitK-").title()
        ax1.plot(DOSE_LEVELS, vals, color=color, marker='o',
                 label=short, lw=2)

    ax1.set_xlabel("Plasma dose")
    ax1.set_ylabel("Predicted protein level (log1p)")
    ax1.set_title(f"Dose-Response Curves\nPatient {patient_id}",
                  fontweight='bold')
    ax1.legend(fontsize=7, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(DOSE_LEVELS)

    # Panel 2: Maximum gain per protein (waterfall)
    ax2 = fig.add_subplot(gs[0, 1])
    proteins_sorted = sorted(
        [p for p in HIGH_SIGNAL_PROTEINS
         if p in agent.protein_cols and p not in EXCLUDE],
        key=lambda p: curve[1.0]["delta"][agent.protein_cols.index(p)])
    gains = [curve[1.0]["delta"][agent.protein_cols.index(p)]
             for p in proteins_sorted]
    short_names = [p.replace("coagulation factor ", "F")
                    .replace("antithrombin-iii","AT-III")
                    .replace("vitamin k-dependent protein ","VitK-")
                    .replace("activated protein c","APC")
                    .replace("coagulation factor xiii b chain","FXIII-B")
                    .title() for p in proteins_sorted]
    bar_colors = ['#1565C0' if g>0 else '#C62828' for g in gains]
    ax2.barh(short_names, gains, color=bar_colors, alpha=0.8)
    ax2.axvline(0, color='black', lw=0.8)
    ax2.set_xlabel("Predicted gain at dose=1.0 (log1p)")
    ax2.set_title("Protein Response to Full Plasma\n(dose=0 → dose=1.0)",
                  fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)

    # Panel 3: Pathway dose-response
    ax3 = fig.add_subplot(gs[1, 0])
    pw_names   = list(pw.keys())
    pw_effects = [[pw[p]["dose_effects"][d] for d in DOSE_LEVELS]
                  for p in pw_names]
    pw_colors  = plt.cm.Set2(np.linspace(0, 1, len(pw_names)))
    for effects, name, color in zip(pw_effects, pw_names, pw_colors):
        ax3.plot(DOSE_LEVELS, effects, color=color,
                 marker='s', label=name, lw=2)
    ax3.axhline(0, color='gray', lw=0.8, ls='--')
    ax3.set_xlabel("Plasma dose")
    ax3.set_ylabel("Mean pathway delta (log1p)")
    ax3.set_title("Pathway-Level Dose-Response",
                  fontweight='bold')
    ax3.legend(fontsize=7)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(DOSE_LEVELS)

    # Panel 4: Minimum effective dose heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    med_proteins = [p for p in HIGH_SIGNAL_PROTEINS
                    if p in med and p not in EXCLUDE]
    med_doses    = [med[p]["min_dose"] if med[p]["achievable"] else 1.1
                    for p in med_proteins]
    short_med    = [p.replace("coagulation factor ","F")
                     .replace("vitamin k-dependent protein ","VitK-")
                     .replace("activated protein c","APC")
                     .replace("antithrombin-iii","AT-III")
                     .replace("coagulation factor xiii b chain","FXIII-B")
                     .title() for p in med_proteins]
    bar_col2 = [plt.cm.RdYlGn(1 - d) for d in
                [min(d, 1.0) for d in med_doses]]
    bars = ax4.barh(short_med, med_doses, color=bar_col2, alpha=0.85)
    ax4.axvline(1.0, color='gray', lw=1, ls='--', alpha=0.5)
    ax4.set_xlabel("Minimum plasma dose required")
    ax4.set_xlim(0, 1.15)
    ax4.set_title("Minimum Effective Dose\n(to reach plasma-arm 75th pctile)",
                  fontweight='bold')
    ax4.grid(True, axis='x', alpha=0.3)
    for bar, d in zip(bars, med_doses):
        label = f"{d:.2f}" if d <= 1.0 else ">1.0"
        ax4.text(min(d, 1.0) + 0.02, bar.get_y() + bar.get_height()/2,
                 label, va='center', fontsize=8)

    plt.suptitle(
        f"Proteome Agent v2 — Patient {patient_id}\n"
        f"Dose-Response Analysis | Plasma Effect Model v2",
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(output_dir, f"agent_v2_{patient_id}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"  Figure: {path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir="./", demo=False, patient_id=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"Loaded: {CKPT_PATH}")

    # Load PAMPer reference
    pamper = pd.read_csv(os.path.join(DATA_DIR, "pamper_coag.csv"))
    agent  = ProteomeAgentV2(model, protein_cols, pamper, device)

    if demo or patient_id is None:
        print(f"\n{'='*68}")
        print("DEMO — Control arm patients")
        print(f"{'='*68}")

        tp0  = pamper[pamper["timepoint"]==0]
        ctrl = tp0[tp0["intervention"]==0]

        mort_row = ctrl[ctrl["mortality"]==1].iloc[0]
        surv_row = ctrl[ctrl["mortality"]==0].iloc[0]

        for row, label in [(mort_row, "control_nonsurvivour"),
                            (surv_row, "control_survivor")]:
            z0  = row[protein_cols].values.astype(np.float32)
            pid = row["patient_id"]
            curve, med, pw = agent.clinical_report(z0, pid)
            plot_agent_report(agent, z0, pid, OUTPUT_DIR)

    else:
        tp0_row = pamper[(pamper["timepoint"]==0) &
                          (pamper["patient_id"]==patient_id)]
        if len(tp0_row) == 0:
            print(f"Patient {patient_id} not found")
            return
        z0 = tp0_row.iloc[0][protein_cols].values.astype(np.float32)
        agent.clinical_report(z0, patient_id)
        plot_agent_report(agent, z0, patient_id, OUTPUT_DIR)

    print(f"\nOutputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", default="./")
    parser.add_argument("--demo",    action="store_true")
    parser.add_argument("--patient", type=str, default=None)
    args = parser.parse_args()
    main(args.data_dir,
         demo=args.demo or args.patient is None,
         patient_id=args.patient)
