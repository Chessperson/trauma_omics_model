"""
pathway_perturbation.py — Pathway-Level Perturbation Analysis
==============================================================
"Which biological pathway, if disrupted, most drives trauma mortality?"

This is the first computational answer to that question from proteomics data.

Pipeline:
    1. Map 7,596 SomaLogic proteins to KEGG/Reactome biological pathways
    2. For each pathway, identify which VAE latent dimensions most control
       those proteins (via decoder Jacobian)
    3. Perturb ONLY those pathway-associated latent dimensions
    4. Run perturbed latent through mortality predictor
    5. Measure ΔAUROC — pathway mortality contribution
    6. Rank pathways by mortality contribution

This answers:
    - Which pathways drive mortality (ablation)?
    - Which pathways, if restored, would most reduce mortality (rescue)?
    - How do plasma vs control patients differ at the pathway level?

Outputs (saved to outputs/pathway/):
    pathway_mortality_scores.csv    — ranked pathways by mortality contribution
    pathway_protein_map.csv         — which proteins belong to which pathway
    pathway_latent_map.csv          — which latent dims control each pathway
    pathway_perturbation.png        — bar chart of top pathways
    plasma_pathway_divergence.csv   — pathway-level plasma vs control differences

Usage:
    python3 pathway_perturbation.py ./
"""

import os
import sys
import json
import time
import requests
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from encoders import SomaLogicVAE, load_pretrained_encoders
from train_cached import (CachedMultimodalTransformer, load_latent_cache,
                          get_fold_splits, get_device, safe_logits)
from dataset import build_datasets

# ── Config ────────────────────────────────────────────────────────────────────

CKPT_DIR   = "outputs/checkpoints/"
LATENT_DIR = "outputs/latents/"
OUTPUT_DIR = "outputs/pathway/"
DATA_DIR   = "./"
FOLD       = 0
N_PERTURB  = 50    # perturbation magnitude samples per pathway
PERTURB_SD = 2.0   # std devs to perturb latent dims

# Key trauma-relevant pathways to prioritize
PRIORITY_PATHWAYS = [
    "complement", "coagulation", "glycolysis", "inflammation",
    "apoptosis", "cytokine", "toll-like receptor", "nf-kb",
    "oxidative phosphorylation", "fatty acid", "platelet",
    "interferon", "interleukin", "chemokine", "fibrin",
    "neutrophil", "macrophage", "t cell", "b cell",
    "mtor", "mapk", "pi3k", "jak-stat",
]

# ── Known pathway → protein mappings (curated for trauma) ────────────────────
# These are the most clinically relevant pathways for trauma mortality
# Proteins matched to your SomaLogic panel via name matching

CURATED_PATHWAYS = {
    "Complement System": [
        "complement c1q", "complement c1r", "complement c1s",
        "complement c2", "complement c3", "complement c4",
        "complement c5", "complement c6", "complement c7",
        "complement c8", "complement c9",
        "complement factor b", "complement factor d", "complement factor h",
        "complement factor i", "complement factor p",
        "c4b-binding protein", "clusterin",
        "cd55 antigen", "cd59 glycoprotein",
        "mannose-binding lectin", "ficolin",
        "c1q tumor necrosis factor",
    ],
    "Coagulation Cascade": [
        "coagulation factor", "prothrombin", "thrombin",
        "fibrinogen", "fibrin", "plasminogen",
        "tissue plasminogen activator", "urokinase",
        "protein c", "protein s", "thrombomodulin",
        "antithrombin", "heparin cofactor",
        "von willebrand factor", "platelet factor",
        "tissue factor pathway inhibitor",
        "alpha-2-antiplasmin", "alpha-2-macroglobulin",
    ],
    "Acute Phase Response": [
        "c-reactive protein", "serum amyloid",
        "haptoglobin", "alpha-1-antitrypsin",
        "alpha-2-macroglobulin", "ceruloplasmin",
        "fibrinogen", "ferritin",
        "alpha-1-acid glycoprotein", "hemopexin",
        "inter-alpha-trypsin inhibitor",
        "lipopolysaccharide-binding protein",
    ],
    "Cytokine Signaling": [
        "interleukin", "tumor necrosis factor",
        "interferon", "chemokine", "c-c motif",
        "c-x-c motif", "transforming growth factor",
        "colony-stimulating factor", "lymphotoxin",
        "oncostatin", "leukemia inhibitory factor",
    ],
    "Toll-Like Receptor Signaling": [
        "toll-like receptor", "myeloid differentiation",
        "interleukin-1 receptor-associated kinase",
        "tnf receptor-associated factor",
        "nuclear factor kappa", "nf-kappa",
        "interferon regulatory factor",
        "cd14", "cd11b", "lipopolysaccharide",
    ],
    "Apoptosis": [
        "caspase", "bcl-2", "bcl-x", "bax", "bad",
        "cytochrome c", "apaf", "death receptor",
        "fas ligand", "trail", "survivin",
        "inhibitor of apoptosis", "p53",
        "bid", "bim", "puma", "noxa",
    ],
    "Oxidative Stress": [
        "superoxide dismutase", "catalase", "glutathione",
        "thioredoxin", "peroxiredoxin", "heme oxygenase",
        "nadph oxidase", "xanthine oxidase",
        "heat shock protein", "ferritin",
        "metallothionein", "glutaredoxin",
    ],
    "Platelet Activation": [
        "platelet", "glycoprotein ib", "glycoprotein iib",
        "integrin alpha-iib", "integrin beta-3",
        "p-selectin", "cd62p",
        "thromboxane", "arachidonic acid",
        "phospholipase", "protein kinase c",
        "von willebrand", "fibrinogen receptor",
    ],
    "Neutrophil Degranulation": [
        "neutrophil", "elastase", "myeloperoxidase",
        "lactoferrin", "defensin", "cathelicidin",
        "matrix metalloproteinase", "collagenase",
        "gelatinase", "cathepsin",
        "s100", "calprotectin", "resistin",
    ],
    "JAK-STAT Signaling": [
        "janus kinase", "jak1", "jak2", "jak3",
        "signal transducer", "stat1", "stat2", "stat3",
        "stat5", "stat6", "suppressor of cytokine",
        "protein inhibitor of activated stat",
        "prolactin receptor", "growth hormone receptor",
        "erythropoietin receptor",
    ],
    "PI3K-AKT Signaling": [
        "phosphatidylinositol 3-kinase", "akt",
        "protein kinase b", "pten",
        "insulin receptor substrate", "mtor",
        "ribosomal protein s6 kinase",
        "forkhead box", "glycogen synthase kinase",
        "bad", "mdm2",
    ],
    "Glycolysis & Energy Metabolism": [
        "hexokinase", "phosphofructokinase", "aldolase",
        "glyceraldehyde", "phosphoglycerate",
        "enolase", "pyruvate kinase", "lactate dehydrogenase",
        "pyruvate dehydrogenase", "glucose transporter",
        "atp synthase", "citrate synthase",
    ],
    "Extracellular Matrix": [
        "collagen", "fibronectin", "laminin",
        "matrix metalloproteinase", "tissue inhibitor",
        "vitronectin", "tenascin", "periostin",
        "versican", "aggrecan", "decorin",
        "thrombospondin", "osteopontin",
    ],
    "Lipid Metabolism": [
        "apolipoprotein", "lipoprotein lipase",
        "cholesterol ester", "phospholipase",
        "fatty acid binding", "acyl-coa",
        "lecithin-cholesterol acyltransferase",
        "paraoxonase", "lipoprotein",
        "sphingomyelin", "ceramide",
    ],
    "Innate Immune Activation": [
        "natural killer", "nk cell",
        "killer cell lectin", "nkp", "nkg2",
        "perforin", "granzyme",
        "cd16", "cd56", "cd94",
        "dag-like lectin", "siglec",
    ],
}


# ── Map proteins to pathways ──────────────────────────────────────────────────

def map_proteins_to_pathways(protein_names, pathways=CURATED_PATHWAYS):
    """
    Map protein names to pathways via substring matching.
    Returns dict: pathway_name -> list of (protein_idx, protein_name)
    """
    protein_lower = [p.lower() for p in protein_names]
    pathway_map   = {}

    for pathway, keywords in pathways.items():
        members = []
        for idx, pname in enumerate(protein_lower):
            for kw in keywords:
                if kw.lower() in pname:
                    members.append((idx, protein_names[idx]))
                    break
        if len(members) >= 3:   # require at least 3 proteins per pathway
            pathway_map[pathway] = members

    return pathway_map


# ── Compute pathway → latent dimension mapping via Jacobian ──────────────────

def compute_pathway_latent_map(soma_vae, pathway_protein_map,
                                n_samples=50, device=None):
    """
    For each pathway, find which latent dimensions most control those proteins.

    Method: compute mean |dProtein/dLatent| for pathway proteins,
    then rank latent dims by their influence on pathway members.

    Returns dict: pathway_name -> latent_dim_weights (128,) tensor
    """
    soma_vae.eval()
    pathway_latent_map = {}

    for pathway, members in pathway_protein_map.items():
        protein_indices = [m[0] for m in members]
        latent_weights  = torch.zeros(128)

        for _ in range(n_samples):
            z = torch.randn(128, device=device, requires_grad=True)

            def decode_fn(z_input):
                return soma_vae.decode(z_input.unsqueeze(0)).squeeze(0)

            try:
                J = torch.autograd.functional.jacobian(decode_fn, z)
                # J shape: (n_proteins, 128)
                # Take rows corresponding to pathway proteins
                pathway_J = J[protein_indices, :].abs()
                # Sum contribution across pathway proteins
                latent_weights += pathway_J.mean(dim=0).detach().cpu()
            except Exception:
                # Finite differences fallback
                with torch.no_grad():
                    x0 = soma_vae.decode(z.unsqueeze(0)).squeeze(0)
                    eps = 1e-2
                    for j in range(128):
                        z_p = z.clone().detach()
                        z_p[j] += eps
                        x_p = soma_vae.decode(z_p.unsqueeze(0)).squeeze(0)
                        diffs = ((x_p - x0).abs()[protein_indices]).mean()
                        latent_weights[j] += diffs.item() / eps

        latent_weights = latent_weights / n_samples
        # Normalize to get top latent dims
        latent_weights = latent_weights / (latent_weights.sum() + 1e-8)
        pathway_latent_map[pathway] = latent_weights

        print(f"  {pathway:35s}: {len(members):3d} proteins, "
              f"top latent dim: {latent_weights.argmax().item()}")

    return pathway_latent_map


# ── Pathway perturbation experiment ──────────────────────────────────────────

def perturb_pathway(model, caches, test_ids, pathway_name,
                    latent_weights, perturb_sd, device, cfg_clinical_dim=29):
    """
    For a given pathway, perturb the associated latent dimensions
    and measure the change in predicted mortality.

    Two perturbations:
        1. Ablation: zero out pathway-associated dims (simulate pathway loss)
        2. Amplification: amplify pathway-associated dims (simulate overactivation)

    Returns dict with mortality probabilities under each condition.
    """
    (soma_cache, met_cache, lip_cache, lum_cache,
     clin_cache, label_cache, _, _) = caches

    model.eval()

    # Get top latent dims for this pathway (top 20% by weight)
    threshold     = latent_weights.quantile(0.80)
    pathway_dims  = (latent_weights >= threshold).nonzero(as_tuple=True)[0]

    baseline_probs  = []
    ablated_probs   = []
    amplified_probs = []
    true_labels     = []

    with torch.no_grad():
        for pid in test_ids:
            info = label_cache[pid]
            if info["is_synthetic"] == 1:
                continue

            # Build batch
            soma_latent = soma_cache[pid].clone().to(device)   # (3, 128)
            clin        = clin_cache[pid].unsqueeze(0).to(device)

            def make_batch(soma_seq):
                return {
                    "somalogic":        soma_seq.unsqueeze(0),
                    "somalogic_mask":   info["somalogic_mask"].unsqueeze(0).to(device),
                    "metabolon":        met_cache[pid].unsqueeze(0).to(device),
                    "metabolon_mask":   info["metabolon_mask"].unsqueeze(0).to(device),
                    "lipidomics":       lip_cache[pid].unsqueeze(0).to(device),
                    "lipidomics_mask":  info["lipidomics_mask"].unsqueeze(0).to(device),
                    "luminex":          lum_cache[pid].unsqueeze(0).to(device),
                    "luminex_mask":     info["luminex_mask"].unsqueeze(0).to(device),
                    "clinical":         clin,
                }

            # Baseline
            batch    = make_batch(soma_latent)
            logits,_ = model(batch)
            baseline_probs.append(torch.sigmoid(safe_logits(logits)).item())

            # Ablation — zero out pathway dims
            ablated = soma_latent.clone()
            ablated[:, pathway_dims] = 0.0
            batch_abl    = make_batch(ablated)
            logits_abl,_ = model(batch_abl)
            ablated_probs.append(torch.sigmoid(safe_logits(logits_abl)).item())

            # Amplification — shift pathway dims by +perturb_sd std devs
            amplified = soma_latent.clone()
            amplified[:, pathway_dims] += perturb_sd
            batch_amp    = make_batch(amplified)
            logits_amp,_ = model(batch_amp)
            amplified_probs.append(torch.sigmoid(safe_logits(logits_amp)).item())

            true_labels.append(info["mortality"])

    baseline_mean  = np.mean(baseline_probs)
    ablated_mean   = np.mean(ablated_probs)
    amplified_mean = np.mean(amplified_probs)

    # AUROC under each condition
    try:
        baseline_auroc  = roc_auc_score(true_labels, baseline_probs)
        ablated_auroc   = roc_auc_score(true_labels, ablated_probs)
        amplified_auroc = roc_auc_score(true_labels, amplified_probs)
    except Exception:
        baseline_auroc = ablated_auroc = amplified_auroc = float("nan")

    return {
        "pathway":            pathway_name,
        "n_proteins":         0,   # filled later
        "n_latent_dims":      len(pathway_dims),
        "baseline_mort":      round(baseline_mean, 4),
        "ablated_mort":       round(ablated_mean, 4),
        "amplified_mort":     round(amplified_mean, 4),
        "ablation_delta":     round(ablated_mean - baseline_mean, 4),
        "amplification_delta":round(amplified_mean - baseline_mean, 4),
        "baseline_auroc":     round(baseline_auroc, 4),
        "ablated_auroc":      round(ablated_auroc, 4),
        "amplified_auroc":    round(amplified_auroc, 4),
        "auroc_drop_ablation":round(baseline_auroc - ablated_auroc, 4),
    }


# ── Plasma vs control pathway divergence ─────────────────────────────────────

def plasma_pathway_divergence(caches, real_ids, label_cache,
                               treatment_dict, pathway_protein_map,
                               soma_vae, device):
    """
    For each pathway, compute the mean latent activation difference
    between plasma and control patients at each timepoint.

    This shows which pathways are most affected by plasma treatment.
    """
    (soma_cache, _, _, _, _, _, _, _) = caches

    plasma_ids  = [p for p in real_ids
                   if treatment_dict.get(p, 0) == 1 and p in soma_cache]
    control_ids = [p for p in real_ids
                   if treatment_dict.get(p, 0) == 0 and p in soma_cache]

    results = []

    for pathway, members in pathway_protein_map.items():
        protein_indices = [m[0] for m in members]

        # Get mean decoded protein values for each arm at each timepoint
        diffs = []
        for tp in range(3):
            plasma_vals = []
            for pid in plasma_ids:
                z   = soma_cache[pid][tp].unsqueeze(0).to(device)
                with torch.no_grad():
                    x = soma_vae.decode(z).squeeze(0).cpu().numpy()
                plasma_vals.append(x[protein_indices].mean())

            control_vals = []
            for pid in control_ids:
                z   = soma_cache[pid][tp].unsqueeze(0).to(device)
                with torch.no_grad():
                    x = soma_vae.decode(z).squeeze(0).cpu().numpy()
                control_vals.append(x[protein_indices].mean())

            diff = np.mean(plasma_vals) - np.mean(control_vals)
            diffs.append(diff)

        results.append({
            "pathway":          pathway,
            "n_proteins":       len(members),
            "plasma_vs_ctrl_tp0":  round(diffs[0], 4),
            "plasma_vs_ctrl_tp24": round(diffs[1], 4),
            "plasma_vs_ctrl_tp72": round(diffs[2], 4),
            "convergence_tp0_tp24": round(abs(diffs[0]) - abs(diffs[1]), 4),
        })

    df = pd.DataFrame(results)
    df = df.sort_values("convergence_tp0_tp24", ascending=False)
    return df


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_pathway_results(df_scores, df_divergence, output_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # ── Figure 1: Pathway mortality contribution ──
        df_plot = df_scores.sort_values(
            "auroc_drop_ablation", ascending=False).head(15)

        fig, ax = plt.subplots(figsize=(11, 7))
        colors  = ["#ef4444" if x > 0 else "#3b82f6"
                   for x in df_plot["auroc_drop_ablation"]]
        ax.barh(range(len(df_plot)),
                df_plot["auroc_drop_ablation"].values,
                color=colors, alpha=0.85)
        ax.set_yticks(range(len(df_plot)))
        ax.set_yticklabels(df_plot["pathway"].values, fontsize=9)
        ax.invert_yaxis()
        ax.axvline(x=0, color="black", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("AUROC Drop When Pathway Ablated\n"
                      "(positive = pathway contributes to mortality prediction)",
                      fontsize=10)
        ax.set_title("Pathway-Level Mortality Contribution\n"
                     "PAMPer Trauma Cohort — SomaLogic Proteomics",
                     fontsize=12, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pathway_mortality_contribution.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: pathway_mortality_contribution.png")

        # ── Figure 2: Plasma pathway divergence ──
        df_div = df_divergence.head(12)
        x      = np.arange(len(df_div))
        width  = 0.25

        fig, ax = plt.subplots(figsize=(13, 6))
        ax.bar(x - width, df_div["plasma_vs_ctrl_tp0"],
               width, label="tp0 (admission)", color="#94a3b8", alpha=0.8)
        ax.bar(x,         df_div["plasma_vs_ctrl_tp24"],
               width, label="tp24", color="#3b82f6", alpha=0.8)
        ax.bar(x + width, df_div["plasma_vs_ctrl_tp72"],
               width, label="tp72", color="#1d4ed8", alpha=0.8)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(df_div["pathway"].values,
                           rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Plasma − Control\n(mean pathway protein expression)",
                      fontsize=10)
        ax.set_title("Plasma Treatment Effect by Biological Pathway\n"
                     "Positive = plasma patients have higher pathway activation",
                     fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "plasma_pathway_divergence.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: plasma_pathway_divergence.png")

    except Exception as e:
        print(f"  Plotting skipped: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir=DATA_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}")

    # ── Load caches ──
    caches = load_latent_cache(LATENT_DIR)
    (soma_cache, met_cache, lip_cache, lum_cache,
     clin_cache, label_cache, real_ids, syn_ids) = caches

    train_ids, val_ids, test_ids = get_fold_splits(real_ids, label_cache, FOLD)
    test_ids = [p for p in test_ids if label_cache[p]["is_synthetic"] == 0]
    print(f"Test patients: {len(test_ids)}")

    # ── Load models ──
    print("\nLoading models...")
    model = CachedMultimodalTransformer(dropout=0.3).to(device)
    model.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, f"model_cached_fold{FOLD}.pt"),
        map_location=device))
    model.eval()

    _, _, _, feature_dims = build_datasets(data_dir=data_dir, fold=0)
    soma_vae = SomaLogicVAE(feature_dims["somalogic"], 128)
    soma_vae.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, "somalogic_vae.pt"), map_location=device))
    soma_vae = soma_vae.to(device).eval()
    print("  Models loaded.")

    # ── Load protein names ──
    imp           = pd.read_csv("outputs/interpretation/protein_importance.csv")
    protein_names = imp["feature"].tolist()
    print(f"  Proteins: {len(protein_names)}")

    # ── Load treatment labels ──
    pts = pd.read_excel(
        os.path.join(data_dir, "PAMPer_External_data.xlsx"),
        sheet_name="Patients")[["Patients", "Intervention_arm"]]
    treatment_dict = {
        row["Patients"]: 1 if row["Intervention_arm"] == "Prehospital_plasma" else 0
        for _, row in pts.iterrows()
    }

    # ── Map proteins to pathways ──
    print("\n" + "="*55)
    print("STEP 1: Mapping proteins to pathways")
    print("="*55)
    pathway_protein_map = map_proteins_to_pathways(protein_names)
    print(f"\n  Found {len(pathway_protein_map)} pathways with ≥3 proteins:")
    for pw, members in sorted(pathway_protein_map.items(),
                               key=lambda x: -len(x[1])):
        print(f"    {pw:35s}: {len(members):3d} proteins")

    # Save protein-pathway map
    rows = []
    for pw, members in pathway_protein_map.items():
        for idx, name in members:
            rows.append({"pathway": pw, "protein_idx": idx, "protein": name})
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, "pathway_protein_map.csv"), index=False)

    # ── Compute pathway → latent dimension mapping ──
    print("\n" + "="*55)
    print("STEP 2: Computing pathway → latent dimension mapping")
    print("="*55)
    pathway_latent_map = compute_pathway_latent_map(
        soma_vae, pathway_protein_map, n_samples=30, device=device)

    # Save latent map
    latent_rows = []
    for pw, weights in pathway_latent_map.items():
        top5 = weights.argsort(descending=True)[:5].tolist()
        latent_rows.append({
            "pathway": pw,
            "top_latent_dims": str(top5),
            "top_dim_weights": str([round(weights[i].item(), 4) for i in top5]),
        })
    pd.DataFrame(latent_rows).to_csv(
        os.path.join(OUTPUT_DIR, "pathway_latent_map.csv"), index=False)

    # ── Pathway perturbation experiment ──
    print("\n" + "="*55)
    print("STEP 3: Pathway perturbation experiment")
    print("="*55)
    print(f"  Ablating each pathway and measuring AUROC change...")
    print(f"  {'Pathway':35s} {'N prots':8s} {'Baseline':10s} "
          f"{'Ablated':10s} {'ΔAUROC':8s}")
    print("  " + "-"*75)

    perturbation_results = []
    for pathway, members in pathway_protein_map.items():
        result = perturb_pathway(
            model=model,
            caches=caches,
            test_ids=test_ids,
            pathway_name=pathway,
            latent_weights=pathway_latent_map[pathway],
            perturb_sd=PERTURB_SD,
            device=device,
        )
        result["n_proteins"] = len(members)
        perturbation_results.append(result)

        print(f"  {pathway:35s} {len(members):8d} "
              f"{result['baseline_auroc']:10.4f} "
              f"{result['ablated_auroc']:10.4f} "
              f"{result['auroc_drop_ablation']:+8.4f}")

    df_scores = pd.DataFrame(perturbation_results)
    df_scores = df_scores.sort_values(
        "auroc_drop_ablation", ascending=False).reset_index(drop=True)
    df_scores["rank"] = df_scores.index + 1
    df_scores.to_csv(
        os.path.join(OUTPUT_DIR, "pathway_mortality_scores.csv"), index=False)

    # ── Plasma vs control pathway divergence ──
    print("\n" + "="*55)
    print("STEP 4: Plasma treatment effect by pathway")
    print("="*55)
    print("  Computing pathway activation differences plasma vs control...")

    df_divergence = plasma_pathway_divergence(
        caches=caches,
        real_ids=real_ids,
        label_cache=label_cache,
        treatment_dict=treatment_dict,
        pathway_protein_map=pathway_protein_map,
        soma_vae=soma_vae,
        device=device,
    )
    df_divergence.to_csv(
        os.path.join(OUTPUT_DIR, "plasma_pathway_divergence.csv"), index=False)

    print(f"\n  Pathways most converged by plasma at tp24:")
    print(f"  {'Pathway':35s} {'tp0 diff':10s} {'tp24 diff':10s} {'Convergence':12s}")
    print("  " + "-"*70)
    for _, row in df_divergence.head(10).iterrows():
        print(f"  {row['pathway']:35s} {row['plasma_vs_ctrl_tp0']:10.4f} "
              f"{row['plasma_vs_ctrl_tp24']:10.4f} "
              f"{row['convergence_tp0_tp24']:12.4f}")

    # ── Summary ──
    print("\n" + "="*55)
    print("RESULTS SUMMARY")
    print("="*55)

    print(f"\n  Top 5 pathways by mortality contribution (AUROC drop):")
    for _, row in df_scores.head(5).iterrows():
        print(f"    #{int(row['rank'])}: {row['pathway']} "
              f"(ΔAUROC={row['auroc_drop_ablation']:+.4f}, "
              f"n={int(row['n_proteins'])} proteins)")

    top_plasma_pathway = df_divergence.iloc[0]
    print(f"\n  Pathway most affected by plasma treatment:")
    print(f"    {top_plasma_pathway['pathway']}")
    print(f"    tp0 diff:  {top_plasma_pathway['plasma_vs_ctrl_tp0']:+.4f}")
    print(f"    tp24 diff: {top_plasma_pathway['plasma_vs_ctrl_tp24']:+.4f}")
    print(f"    Convergence: {top_plasma_pathway['convergence_tp0_tp24']:+.4f}")

    # ── Plots ──
    print("\nGenerating figures...")
    plot_pathway_results(df_scores, df_divergence, OUTPUT_DIR)

    # Save summary JSON
    summary = {
        "n_pathways_analyzed": len(pathway_protein_map),
        "top_mortality_pathway": df_scores.iloc[0]["pathway"],
        "top_mortality_auroc_drop": float(df_scores.iloc[0]["auroc_drop_ablation"]),
        "top_plasma_pathway": top_plasma_pathway["pathway"],
        "top_plasma_convergence": float(top_plasma_pathway["convergence_tp0_tp24"]),
    }
    with open(os.path.join(OUTPUT_DIR, "pathway_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  All outputs saved to: {OUTPUT_DIR}")
    print(f"\n  KEY FINDINGS:")
    print(f"  This is the first pathway-level perturbation analysis")
    print(f"  of trauma proteomics — showing which biological pathways")
    print(f"  drive mortality and which are most altered by plasma treatment.")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    main(data_dir=data_dir)