"""
proteome_agent.py — Interventional Proteome Agent
===================================================
A query interface over the intervention-conditioned neural ODE.
Given a patient's admission coagulation profile and a proposed
intervention, predicts the coagulation trajectory and answers
clinical questions about treatment timing and patient response.

Clinical queries supported:
    1. Trajectory prediction: what happens to each coagulation
       factor over 72 hours under plasma vs control?
    2. Intervention comparison: how does plasma vs saline change
       the coagulation state at 24h and 72h for this patient?
    3. Timing sensitivity: how does delaying plasma by 6h change
       the predicted trajectory?
    4. Patient similarity: which PAMPer patients are most similar
       to this new patient at admission?
    5. Pathway summary: which coagulation proteins are most
       responsive to plasma in this patient?

Usage:
    python3 proteome_agent.py ./
    python3 proteome_agent.py ./ --demo          # run demo queries
    python3 proteome_agent.py ./ --patient PAMP1001
"""

import os, sys, json, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Import ODE components
sys.path.insert(0, os.path.dirname(__file__))
from intervention_ode import (
    InterventionODE, integrate, CFG, SENTINEL_PROTEINS
)

DATA_DIR   = "coag_data/"
OUTPUT_DIR = "outputs/proteome_agent/"
CKPT_PATH  = "outputs/checkpoints/intervention_ode.pt"

# Coagulation pathway groupings for clinical interpretation
PATHWAY_GROUPS = {
    "Extrinsic pathway":   ["coagulation factor vii", "tissue factor",
                             "tissue factor pathway inhibitor",
                             "tissue factor pathway inhibitor 2"],
    "Intrinsic pathway":   ["coagulation factor viii", "coagulation factor ix",
                             "coagulation factor xi", "coagulation factor ixab"],
    "Common pathway":      ["coagulation factor x", "coagulation factor xa",
                             "coagulation factor v", "prothrombin", "thrombin",
                             "coagulation factor xiii",
                             "coagulation factor xiii b chain"],
    "Fibrinogen system":   ["fibrinogen", "fibrinogen beta chain",
                             "fibrinogen gamma chain",
                             "fibrinogen-like protein 1",
                             "fibrinogen c domain-containing protein 1"],
    "Natural anticoagulants": ["antithrombin-iii", "activated protein c",
                                "activated protein c.1",
                                "vitamin k-dependent protein c",
                                "vitamin k-dependent protein s",
                                "soluble endothelial protein c receptor"],
    "Fibrinolysis":        ["plasminogen", "plasminogen activator inhibitor 1",
                             "tissue-type plasminogen activator",
                             "urokinase-type plasminogen activator",
                             "urokinase plasminogen activator surface receptor"],
}


# ── Agent ─────────────────────────────────────────────────────────────────────

class ProteomeAgent:
    """
    Clinical query interface for the intervention-conditioned ODE.
    """

    def __init__(self, model, protein_cols, pamper_df, device):
        self.model       = model
        self.protein_cols = protein_cols
        self.pamper_df   = pamper_df
        self.device      = device
        self.model.eval()

        # Precompute PAMPer admission profiles for similarity search
        self._build_reference_db()

    def _build_reference_db(self):
        """Build reference database of PAMPer admission profiles."""
        self.ref_db = {}
        meta_cols = ["patient_id", "timepoint", "intervention",
                     "mortality", "cohort"]

        tp0      = self.pamper_df[self.pamper_df["timepoint"] == 0]
        all_profs = tp0[self.protein_cols].values.astype(np.float32)
        col_meds  = np.nanmedian(all_profs, axis=0)
        col_meds  = np.where(np.isnan(col_meds), 0.0, col_meds)

        for _, row in tp0.iterrows():
            pid  = row["patient_id"]
            prof = row[self.protein_cols].values.astype(np.float32)
            nan_mask      = np.isnan(prof)
            prof[nan_mask] = col_meds[nan_mask]
            self.ref_db[pid] = {
                "profile":      prof,
                "intervention": int(row["intervention"]),
                "mortality":    int(row["mortality"]),
            }
        print(f"  Reference database: {len(self.ref_db)} PAMPer patients")

    def predict_trajectory(self, z0, intervention_label, timepoints=None):
        """
        Core prediction: given admission proteome and intervention,
        return predicted coagulation state at each timepoint.

        Args:
            z0:                 np.array (33,) — admission coagulation profile
            intervention_label: 'plasma' | 'control' | 'unknown'
            timepoints:         list of hours, default [0, 24, 72]

        Returns:
            dict with predicted protein levels at each timepoint
        """
        if timepoints is None:
            timepoints = [0, 24, 72]

        intv_map = {'plasma': 1, 'control': 0, 'unknown': 2}
        intv_idx = intv_map.get(intervention_label.lower(), 2)

        # Impute NaNs with population median from reference DB
        z0 = z0.copy()
        if np.isnan(z0).any():
            ref_profiles = np.stack([v["profile"] for v in self.ref_db.values()])
            col_medians  = np.nanmedian(ref_profiles, axis=0)
            nan_mask     = np.isnan(z0)
            z0[nan_mask] = col_medians[nan_mask]

        z0_t   = torch.tensor(z0, dtype=torch.float32).unsqueeze(0).to(self.device)
        intv_t = torch.tensor([intv_idx], dtype=torch.long).to(self.device)

        with torch.no_grad():
            d24, d72 = self.model.func.predict_deltas(z0_t, intv_t)
            z24 = z0_t + d24
            z72 = z0_t + d72

        # Build result for requested timepoints
        result = {0: z0}
        tp_map = {24: z24.squeeze(0).cpu().numpy(),
                  72: z72.squeeze(0).cpu().numpy()}

        for tp in timepoints:
            if tp == 0:
                result[0] = z0
            elif tp <= 24:
                alpha = tp / 24.0
                result[tp] = z0 + alpha * (tp_map[24] - z0)
            elif tp in tp_map:
                result[tp] = tp_map[tp]
            else:
                alpha = (tp - 24) / 48.0
                result[tp] = tp_map[24] + alpha * (tp_map[72] - tp_map[24])

        return result

    def compare_interventions(self, z0):
        """
        For a given patient admission profile, predict the trajectory
        under plasma vs control and return the difference.

        This is the counterfactual: what would happen if we gave
        plasma vs if we didn't?
        """
        plasma_traj  = self.predict_trajectory(z0, 'plasma')
        control_traj = self.predict_trajectory(z0, 'control')

        diff = {}
        for tp in [24, 72]:
            diff[tp] = plasma_traj[tp] - control_traj[tp]

        return plasma_traj, control_traj, diff

    def timing_sensitivity(self, z0, delay_hours=6):
        """
        Compare immediate plasma vs delayed plasma.
        Models delay as: patient evolves under 'control' for delay_hours,
        then receives plasma.
        """
        # Immediate plasma from tp0
        immediate = self.predict_trajectory(z0, 'plasma', [0, 24, 72])

        # Delayed: evolve under control for delay_hours, then switch to plasma
        pre_delay  = self.predict_trajectory(z0, 'control',
                                              [0, delay_hours])
        z_delayed  = pre_delay[delay_hours]
        post_delay = self.predict_trajectory(
            z_delayed, 'plasma',
            [delay_hours, 24, 72])

        # Merge trajectories
        delayed = {0: pre_delay[0], delay_hours: pre_delay[delay_hours],
                   24: post_delay[24], 72: post_delay[72]}

        return immediate, delayed

    def find_similar_patients(self, z0, n=5):
        """Find the most similar PAMPer patients at admission."""
        dists = {}
        for pid, info in self.ref_db.items():
            ref = info["profile"]
            # Euclidean distance, ignoring NaNs
            mask = np.isfinite(z0) & np.isfinite(ref)
            if mask.sum() < 10:
                continue
            d = np.sqrt(((z0[mask] - ref[mask]) ** 2).mean())
            dists[pid] = d

        similar = sorted(dists.items(), key=lambda x: x[1])[:n]
        result  = []
        for pid, dist in similar:
            info = self.ref_db[pid]
            result.append({
                "patient_id":   pid,
                "distance":     float(dist),
                "intervention": "plasma" if info["intervention"] == 1
                                else "control",
                "mortality":    int(info["mortality"]),
            })
        return result

    def pathway_response(self, z0):
        """
        Compute pathway-level plasma response for this patient.
        Returns the mean predicted plasma effect per pathway at 24h.
        """
        plasma_traj, control_traj, diff = self.compare_interventions(z0)

        pathway_summary = {}
        for pathway, proteins in PATHWAY_GROUPS.items():
            idxs = [self.protein_cols.index(p)
                    for p in proteins if p in self.protein_cols]
            if not idxs:
                continue
            mean_effect_24h = diff[24][idxs].mean()
            mean_effect_72h = diff[72][idxs].mean()
            pathway_summary[pathway] = {
                "effect_24h": float(mean_effect_24h),
                "effect_72h": float(mean_effect_72h),
                "n_proteins": len(idxs),
            }

        return pathway_summary

    def clinical_report(self, z0, patient_id="Unknown"):
        """
        Generate a full clinical report for a patient.
        This is the demo output for the ASC presentation.
        """
        print(f"\n{'='*65}")
        print(f"PROTEOME AGENT REPORT — Patient: {patient_id}")
        print(f"{'='*65}")

        # 1. Trajectory comparison
        plasma_t, control_t, diff = self.compare_interventions(z0)

        print(f"\n1. PREDICTED COAGULATION TRAJECTORIES")
        print(f"   (plasma vs control, key proteins)")
        print(f"\n   {'Protein':30s} {'tp0':8s} {'Plasma 24h':12s} "
              f"{'Control 24h':12s} {'Plasma Δ':10s}")
        print(f"   " + "-"*72)

        for pname in SENTINEL_PROTEINS:
            if pname not in self.protein_cols:
                continue
            i     = self.protein_cols.index(pname)
            tp0_v = z0[i]
            p24   = plasma_t[24][i]
            c24   = control_t[24][i]
            delta = p24 - c24
            marker = "↑" if delta > 0 else "↓"
            print(f"   {pname:30s} {tp0_v:.3f}    {p24:.3f}        "
                  f"{c24:.3f}        {marker}{abs(delta):.3f}")

        # 2. Timing sensitivity
        print(f"\n2. TIMING SENSITIVITY — 6-HOUR DELAY")
        immediate, delayed = self.timing_sensitivity(z0, delay_hours=6)
        print(f"   Effect of delaying plasma by 6 hours:")
        print(f"\n   {'Protein':30s} {'Immediate 24h':14s} "
              f"{'Delayed 24h':12s} {'Cost of delay':12s}")
        print(f"   " + "-"*68)

        for pname in ["fibrinogen", "coagulation factor v",
                       "antithrombin-iii", "thrombin"]:
            if pname not in self.protein_cols:
                continue
            i    = self.protein_cols.index(pname)
            imm  = immediate[24][i]
            del_ = delayed[24][i]
            cost = del_ - imm  # negative = delayed is worse
            print(f"   {pname:30s} {imm:.3f}          {del_:.3f}        "
                  f"{cost:+.3f}")

        # 3. Pathway response
        print(f"\n3. PATHWAY-LEVEL PLASMA RESPONSE")
        pathway_resp = self.pathway_response(z0)
        print(f"\n   {'Pathway':30s} {'Effect at 24h':14s} "
              f"{'Effect at 72h':14s} {'N proteins':10s}")
        print(f"   " + "-"*68)

        for pw, info in sorted(pathway_resp.items(),
                                key=lambda x: -abs(x[1]["effect_24h"])):
            e24 = info["effect_24h"]
            e72 = info["effect_72h"]
            n   = info["n_proteins"]
            mark = "↑" if e24 > 0 else "↓"
            print(f"   {pw:30s} {mark}{abs(e24):.4f}         "
                  f"{mark}{abs(e72):.4f}         {n}")

        # 4. Similar patients
        print(f"\n4. MOST SIMILAR PAMPer PATIENTS")
        similar = self.find_similar_patients(z0, n=5)
        print(f"\n   {'Patient':12s} {'Distance':10s} "
              f"{'Treatment':12s} {'Outcome':8s}")
        print(f"   " + "-"*44)
        for s in similar:
            outcome = "Died" if s["mortality"] == 1 else "Survived"
            print(f"   {s['patient_id']:12s} {s['distance']:.4f}      "
                  f"{s['intervention']:12s} {outcome}")

        print(f"\n{'='*65}")
        return {
            "plasma_trajectory":   {k: v.tolist()
                                     for k, v in plasma_t.items()},
            "control_trajectory":  {k: v.tolist()
                                     for k, v in control_t.items()},
            "pathway_response":    pathway_resp,
            "similar_patients":    similar,
        }


# ── Visualization ─────────────────────────────────────────────────────────────

def plot_agent_report(agent, z0, patient_id, output_dir):
    """
    Generate the presentation-quality visualization.
    4-panel figure: trajectories, pathway response,
    timing sensitivity, patient comparison.
    """
    plasma_t, control_t, diff = agent.compare_interventions(z0)
    immediate, delayed        = agent.timing_sensitivity(z0, 6)
    pathway_resp              = agent.pathway_response(z0)

    fig = plt.figure(figsize=(16, 12))
    gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

    # ── Panel 1: Key coagulation trajectories ────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    tps = [0, 24, 72]
    colors = plt.cm.Set1(np.linspace(0, 1, 5))

    for k, pname in enumerate(["fibrinogen", "coagulation factor v",
                                "prothrombin", "antithrombin-iii",
                                "thrombin"]):
        if pname not in agent.protein_cols:
            continue
        i   = agent.protein_cols.index(pname)
        p_v = [z0[i], plasma_t[24][i], plasma_t[72][i]]
        c_v = [z0[i], control_t[24][i], control_t[72][i]]
        lbl = pname.replace("coagulation factor ", "F").replace(
            "antithrombin-iii", "AT-III").title()
        ax1.plot(tps, p_v, color=colors[k], ls="-",
                 marker='o', label=f"{lbl} (plasma)", lw=2)
        ax1.plot(tps, c_v, color=colors[k], ls="--",
                 marker='x', lw=1.5, alpha=0.6)

    ax1.set_title(f"Predicted Coagulation Trajectories\n"
                  f"Patient {patient_id}", fontweight='bold')
    ax1.set_xlabel("Hours post-injury")
    ax1.set_ylabel("log1p protein level")
    ax1.set_xticks([0, 24, 72])
    ax1.legend(fontsize=7, loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.axvline(x=0, color='gray', ls=':', alpha=0.5,
                label='Admission')

    # ── Panel 2: Pathway plasma response at 24h ───────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    pw_names  = list(pathway_resp.keys())
    effects24 = [pathway_resp[p]["effect_24h"] for p in pw_names]
    pw_short  = [p.replace("Natural anticoagulants", "Anticoag.")
                  .replace("Fibrinogen system", "Fibrinogen")
                  .replace("Common pathway", "Common") for p in pw_names]

    bar_colors = ['#1565C0' if e > 0 else '#C62828' for e in effects24]
    bars = ax2.barh(pw_short, effects24, color=bar_colors, alpha=0.8)
    ax2.axvline(x=0, color='black', lw=0.8)
    ax2.set_xlabel("Predicted plasma effect (log1p units)")
    ax2.set_title("Pathway-Level Plasma Response\nat 24 Hours",
                  fontweight='bold')
    ax2.grid(True, axis='x', alpha=0.3)
    for bar, val in zip(bars, effects24):
        ax2.text(val + (0.001 if val >= 0 else -0.001),
                 bar.get_y() + bar.get_height()/2,
                 f"{val:+.4f}", va='center',
                 ha='left' if val >= 0 else 'right', fontsize=8)

    # ── Panel 3: Timing sensitivity ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    timing_proteins = ["fibrinogen", "coagulation factor v",
                        "antithrombin-iii", "thrombin",
                        "coagulation factor vii"]
    imm_24  = []
    del_24  = []
    t_names = []
    for pname in timing_proteins:
        if pname not in agent.protein_cols:
            continue
        i = agent.protein_cols.index(pname)
        imm_24.append(immediate[24][i])
        del_24.append(delayed[24][i])
        t_names.append(pname.replace("coagulation factor ", "F")
                           .replace("antithrombin-iii", "AT-III").title())

    x = np.arange(len(t_names))
    w = 0.35
    ax3.bar(x - w/2, imm_24, w, label='Immediate plasma',
            color='#1565C0', alpha=0.8)
    ax3.bar(x + w/2, del_24, w, label='6h delayed plasma',
            color='#42A5F5', alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(t_names, rotation=30, ha='right', fontsize=8)
    ax3.set_ylabel("Predicted level at 24h")
    ax3.set_title("Cost of 6-Hour Plasma Delay\nfor This Patient",
                  fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, axis='y', alpha=0.3)

    # ── Panel 4: Plasma effect magnitude per protein ──────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    effect_24 = diff[24]  # plasma - control at 24h
    top_idx   = np.argsort(np.abs(effect_24))[-10:]
    top_names = [agent.protein_cols[i].replace("coagulation factor ", "F")
                     .replace("antithrombin-iii", "AT-III")
                     .replace("tissue factor", "TF")
                     .title() for i in top_idx]
    top_vals  = effect_24[top_idx]
    bar_col   = ['#1565C0' if v > 0 else '#C62828' for v in top_vals]

    ax4.barh(top_names, top_vals, color=bar_col, alpha=0.8)
    ax4.axvline(x=0, color='black', lw=0.8)
    ax4.set_xlabel("Plasma - Control at 24h (log1p units)")
    ax4.set_title("Top 10 Protein Responses\nto Plasma at 24 Hours",
                  fontweight='bold')
    ax4.grid(True, axis='x', alpha=0.3)

    plt.suptitle(
        f"Proteome Agent Clinical Report — Patient {patient_id}\n"
        f"Intervention-Conditioned Neural ODE | PAMPer + PRECISE",
        fontsize=13, fontweight='bold', y=1.01)

    path = os.path.join(output_dir, f"agent_report_{patient_id}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f"\n  Report figure: {path}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir="./", demo=False, patient_id=None):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device(
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    print("\nLoading intervention ODE...")
    with open(os.path.join("outputs/intervention_ode/", "summary.json")) as f:
        summary = json.load(f)
    protein_cols = summary["protein_cols"]
    CFG["protein_dim"] = len(protein_cols)

    model = InterventionODE(CFG)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    print(f"  Loaded: {CKPT_PATH}")

    # Load PAMPer data for reference DB
    pamper = pd.read_csv(os.path.join(DATA_DIR, "pamper_coag.csv"))

    # Build agent
    print("\nBuilding agent...")
    agent = ProteomeAgent(model, protein_cols, pamper, device)

    if demo or patient_id is None:
        # Demo: run report on a few PAMPer patients
        print("\n" + "="*65)
        print("DEMO MODE — Running reports on representative patients")
        print("="*65)

        tp0 = pamper[pamper["timepoint"] == 0]

        # Pick one plasma survivor, one control non-survivor
        plasma_surv = tp0[(tp0["intervention"]==1) &
                           (tp0["mortality"]==0)].iloc[0]
        ctrl_mort   = tp0[(tp0["intervention"]==0) &
                           (tp0["mortality"]==1)].iloc[0]

        for row, label in [
                (plasma_surv, "plasma_survivor"),
                (ctrl_mort,   "control_nonsurvivour")]:

            z0  = row[protein_cols].values.astype(np.float32)
            pid = row["patient_id"]
            intv = "plasma" if row["intervention"] == 1 else "control"
            mort = "died" if row["mortality"] == 1 else "survived"

            print(f"\nPatient: {pid} | Actual: {intv}, {mort}")
            report = agent.clinical_report(z0, patient_id=pid)
            plot_agent_report(agent, z0, pid, OUTPUT_DIR)

            with open(os.path.join(OUTPUT_DIR,
                                    f"report_{pid}.json"), "w") as f:
                json.dump(report, f, indent=2)

    else:
        # Specific patient
        tp0_row = pamper[(pamper["timepoint"]==0) &
                          (pamper["patient_id"]==patient_id)]
        if len(tp0_row) == 0:
            print(f"Patient {patient_id} not found in PAMPer data")
            return

        z0     = tp0_row.iloc[0][protein_cols].values.astype(np.float32)
        report = agent.clinical_report(z0, patient_id=patient_id)
        plot_agent_report(agent, z0, patient_id, OUTPUT_DIR)

        with open(os.path.join(OUTPUT_DIR,
                                f"report_{patient_id}.json"), "w") as f:
            json.dump(report, f, indent=2)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", nargs="?", default="./")
    parser.add_argument("--demo",    action="store_true")
    parser.add_argument("--patient", type=str, default=None)
    args = parser.parse_args()
    main(args.data_dir, demo=args.demo or args.patient is None,
         patient_id=args.patient)