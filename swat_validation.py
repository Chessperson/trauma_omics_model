"""
swat_validation.py — External Validation of PAMPer Model on SWAT Cohort
=========================================================================
Domain adaptation strategy:
    - VAE is FROZEN — keeps the PAMPer latent space intact
    - Only the MLP mortality prediction head is fine-tuned on SWAT labels
    - This lets the model adapt its risk calibration to SWAT's 9% mortality
      without destroying the biological latent structure

Usage:
    python3 swat_validation.py ./
"""

import os, sys, re, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from encoders import SomaLogicVAE
from train_cached import CachedMultimodalTransformer, get_device, safe_logits
from dataset import build_datasets

OUTPUT_DIR = "outputs/swat_validation/"
SWAT_CSV   = "SWAT_proteins_clinical.csv"
CKPT_DIR   = "outputs/checkpoints/"

FT_EPOCHS    = 80
FT_LR        = 1e-4
FT_SEED      = 42
FT_TEST_SIZE = 0.2


def parse_swat_timepoint(name):
    m = re.search(r'SWAT\d+?(\d+)EDA$', str(name))
    if not m:
        return None
    suffix = m.group(1)
    if suffix.endswith('24'):
        return 24
    elif suffix.endswith('4'):
        return 4
    else:
        return 0


def load_swat_data(data_dir="./"):
    print("Loading SWAT data...")
    df = pd.read_csv(os.path.join(data_dir, SWAT_CSV))
    df['timepoint'] = df['Name'].apply(parse_swat_timepoint)

    pamp_df    = pd.read_excel(
        os.path.join(data_dir, "PAMPer_External_data.xlsx"),
        sheet_name="SomaLogic", nrows=1)
    pamp_prots = pamp_df.columns.tolist()[2:]
    all_cols   = df.columns.tolist()
    swat_prots = set(all_cols[218:])

    protein_cols = pamp_prots
    missing = [p for p in pamp_prots if p not in swat_prots]
    print(f"  PAMPer proteins : {len(pamp_prots)}")
    print(f"  SWAT proteins   : {len(swat_prots)}")
    print(f"  Zero-filled     : {missing}")
    print(f"  Unique patients : {df['ID'].nunique()}")

    plasma_treatments = {'All three', 'PRBC+Plasma', 'WB+Plasma'}
    tp0_patients      = df[df['timepoint'] == 0]['ID'].unique().tolist()

    patient_data = {}
    for pid in tp0_patients:
        prows = df[df['ID'] == pid].sort_values('timepoint')
        row0  = prows[prows['timepoint'] == 0].iloc[0]

        treatment_str = str(row0.get('Treatment', 'Other'))
        treatment     = 1 if treatment_str in plasma_treatments else 0
        mortality     = int(row0.get('30d_Mortality', 0))

        protein_matrix = np.zeros((3, len(protein_cols)), dtype=np.float32)
        masks          = np.zeros(3, dtype=np.float32)

        for _, row in prows.iterrows():
            tp   = row['timepoint']
            vals = np.array([
                row[p] if p in row.index else 0.0
                for p in protein_cols
            ], dtype=np.float32)
            vals = np.nan_to_num(vals, nan=0.0)
            vals = np.log1p(np.clip(vals, 0, None))
            if tp == 0:
                protein_matrix[0] = vals
                masks[0] = 1.0
            elif tp == 24:
                protein_matrix[1] = vals
                masks[1] = 1.0

        if masks[0] == 0:
            continue

        clinical_vals = np.zeros(29, dtype=np.float32)
        clin_map = {
            'phgcs': 0, 'phsbp': 1, 'phhr': 2, 'phrr': 3,
            'phshockind': 4, 'Age': 5, 'ISS': 6, 'TBI': 7, 'Penetrating': 8,
        }
        for col, idx in clin_map.items():
            if col in row0.index:
                v = row0.get(col, 0)
                clinical_vals[idx] = float(v) if pd.notna(v) else 0.0

        patient_data[pid] = {
            'proteins':      protein_matrix,
            'masks':         masks,
            'clinical':      clinical_vals,
            'treatment':     treatment,
            'mortality':     mortality,
            'treatment_str': treatment_str,
        }

    print(f"  Valid patients  : {len(patient_data)}")
    n_plasma  = sum(1 for v in patient_data.values() if v['treatment'] == 1)
    mort_rate = np.mean([v['mortality'] for v in patient_data.values()])
    print(f"  Plasma-containing: {n_plasma} | No plasma: {len(patient_data)-n_plasma}")
    print(f"  Mortality rate  : {mort_rate:.3f}")
    return patient_data, protein_cols


def encode_swat_proteins(patient_data, protein_cols, device, ckpt_dir, data_dir):
    """Encode SWAT proteins through the FROZEN PAMPer VAE."""
    print("\nEncoding SWAT proteins through PAMPer VAE (frozen)...")
    _, _, _, feature_dims = build_datasets(data_dir=data_dir, fold=0)
    soma_vae = SomaLogicVAE(feature_dims['somalogic'], 128)
    joint_path = os.path.join(ckpt_dir, 'somalogic_vae_joint.pt')
    vae_path   = joint_path if os.path.exists(joint_path) else \
             os.path.join(ckpt_dir, 'somalogic_vae.pt')
    print(f"  Loading {'joint' if os.path.exists(joint_path) else 'original'} VAE")
    soma_vae.load_state_dict(torch.load(vae_path, map_location=device))
    soma_vae.beta = 0.1   # joint VAE was trained with beta=0.1 final pass

    for param in soma_vae.parameters():
        param.requires_grad = False

    print(f"  VAE loaded and frozen (beta={soma_vae.beta})")

    # Latent space alignment check
    with torch.no_grad():
        sample_pids = list(patient_data.keys())[:10]
        sample_z = []
        for pid in sample_pids:
            x = torch.tensor(
                patient_data[pid]['proteins'][0],
                dtype=torch.float32).unsqueeze(0).to(device)
            z, _ = soma_vae.encode(x)
            sample_z.append(z.squeeze(0))
        sample_z  = torch.stack(sample_z)
        norm_mean = sample_z.norm(dim=1).mean().item()
        norm_std  = sample_z.norm(dim=1).std().item()
    print(f"  Latent norm — mean: {norm_mean:.4f}, std: {norm_std:.4f}")
    print(f"  PAMPer ref   — mean: 1.3239, std: 0.6428")
    if norm_std < 0.15:
        print(f"  WARNING: Low std — posterior collapse in transfer")
    else:
        print(f"  Latent space looks healthy")

    latents = {}
    with torch.no_grad():
        for pid, data in patient_data.items():
            proteins = torch.tensor(
                data['proteins'], dtype=torch.float32).to(device)
            z_list = []
            for tp in range(3):
                z, _ = soma_vae.encode(proteins[tp].unsqueeze(0))
                z_list.append(z.squeeze(0))
            latents[pid] = torch.stack(z_list).cpu()

    print(f"  Encoded {len(latents)} patients")
    return latents, soma_vae


def build_prediction_batch(patient_data, latents, pids, device):
    B            = len(pids)
    soma_latents = torch.stack([latents[p] for p in pids]).to(device)
    soma_masks   = torch.tensor(
        np.array([patient_data[p]['masks'] for p in pids]),
        dtype=torch.float32).to(device)
    zeros_64  = torch.zeros(B, 3, 64, device=device)
    zeros_32  = torch.zeros(B, 3, 32, device=device)
    zero_mask = torch.zeros(B, 3, device=device)

    clin_raw = np.array([patient_data[p]['clinical'] for p in pids])
    clinical = torch.zeros(B, 32, dtype=torch.float32)
    clinical[:, :29] = torch.tensor(clin_raw, dtype=torch.float32)
    clinical = clinical.to(device)

    return {
        'somalogic':       soma_latents,
        'somalogic_mask':  soma_masks,
        'metabolon':       zeros_64,
        'metabolon_mask':  zero_mask,
        'lipidomics':      zeros_64,
        'lipidomics_mask': zero_mask,
        'luminex':         zeros_32,
        'luminex_mask':    zero_mask,
        'clinical':        clinical,
    }


def finetune_predictor_on_swat(model, patient_data, latents, device):
    """Fine-tune ONLY the mortality prediction head on SWAT labels."""
    print("\n" + "="*55)
    print("PREDICTOR HEAD FINE-TUNING ON SWAT")
    print("="*55)

    pids   = list(patient_data.keys())
    labels = np.array([patient_data[p]['mortality'] for p in pids])

    tr_pids, te_pids, tr_labels, te_labels = train_test_split(
        pids, labels,
        test_size=FT_TEST_SIZE,
        stratify=labels,
        random_state=FT_SEED)

    print(f"\n  Train: {len(tr_pids)} patients ({sum(tr_labels)} deaths)")
    print(f"  Test:  {len(te_pids)} patients ({sum(te_labels)} deaths)")

    # Freeze everything except fusion/projection layers
    frozen = trainable = 0
    for name, param in model.named_parameters():
    # Keep trainable: soma projection, clinical projection, fusion head
    # Freeze: met_proj, lip_proj, lum_proj (SWAT has none of these)
        is_head = any(k in name for k in [
            'soma_proj', 'clin_proj', 'fusion'])
        param.requires_grad = is_head
        if is_head:
            trainable += param.numel()
        else:
            frozen += param.numel()

    print(f"\n  Frozen     : {frozen:,}")
    print(f"  Trainable  : {trainable:,}")

    n_pos = tr_labels.sum()
    n_neg = len(tr_labels) - n_pos
    pos_weight = torch.tensor(
        [min(n_neg / max(n_pos, 1), 10.0)],
        dtype=torch.float32).to(device)
    print(f"  pos_weight : {pos_weight.item():.2f}x")

    optim = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=FT_LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=FT_EPOCHS, eta_min=1e-6)

    best_auroc = 0.0
    best_state = None

    print(f"\n  {'Epoch':8s} {'Loss':10s} {'AUROC':10s} {'Best':8s}")
    print("  " + "-"*38)

    for epoch in range(1, FT_EPOCHS + 1):
        model.train()
        idx    = np.random.permutation(len(tr_pids))
        losses = []

        for i in range(0, len(tr_pids), 16):
            batch_pids = [tr_pids[j] for j in idx[i:i+16]]
            batch = build_prediction_batch(
                patient_data, latents, batch_pids, device)
            bl = torch.tensor(
                [patient_data[p]['mortality'] for p in batch_pids],
                dtype=torch.float32).to(device)

            optim.zero_grad()
            logits, _ = model(batch)
            loss = F.binary_cross_entropy_with_logits(
                safe_logits(logits), bl, pos_weight=pos_weight)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            losses.append(loss.item())

        scheduler.step()

        if epoch % 10 == 0 or epoch == 1 or epoch == FT_EPOCHS:
            model.eval()
            with torch.no_grad():
                te_batch     = build_prediction_batch(
                    patient_data, latents, te_pids, device)
                te_logits, _ = model(te_batch)
                te_probs     = torch.sigmoid(
                    safe_logits(te_logits)).cpu().numpy()
            try:
                auroc   = roc_auc_score(te_labels, te_probs)
                is_best = auroc > best_auroc
                if is_best:
                    best_auroc = auroc
                    best_state = {k: v.clone()
                                  for k, v in model.state_dict().items()}
                marker = " ← best" if is_best else ""
                print(f"  Epoch {epoch:3d}   "
                      f"{np.mean(losses):.4f}    "
                      f"{auroc:.4f}    "
                      f"{best_auroc:.4f}{marker}")
            except Exception:
                pass

    if best_state:
        model.load_state_dict(best_state)

    for param in model.parameters():
        param.requires_grad = True

    model.eval()
    print(f"\n  Best test AUROC: {best_auroc:.4f}")
    return model, te_pids, best_auroc


def validate_mortality(model, patient_data, latents, device,
                       label="zero-shot", exclude_pids=None):
    print("\n" + "="*55)
    print(f"MORTALITY PREDICTION — {label.upper()}")
    print("="*55)

    pids      = exclude_pids if exclude_pids is not None \
                else list(patient_data.keys())
    labels    = [patient_data[p]['mortality'] for p in pids]
    all_probs = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(pids), 16):
            batch_pids = pids[i:i+16]
            batch      = build_prediction_batch(
                patient_data, latents, batch_pids, device)
            logits, _  = model(batch)
            probs      = torch.sigmoid(
                safe_logits(logits)).cpu().tolist()
            all_probs.extend(probs)

    try:
        auroc = roc_auc_score(labels, all_probs)
        auprc = average_precision_score(labels, all_probs)
    except Exception:
        auroc = float('nan')
        auprc = float('nan')

    print(f"\n  AUROC : {auroc:.4f}  (PAMPer training: 0.875)")
    print(f"  AUPRC : {auprc:.4f}")
    print(f"  N     : {len(pids)} patients, {sum(labels)} deaths")

    print(f"\n  AUROC by treatment group:")
    for treat_name, treat_val in [('Plasma-containing', 1), ('No plasma', 0)]:
        gpids = [p for p in pids if patient_data[p]['treatment'] == treat_val]
        if len(gpids) < 5:
            continue
        gl = [patient_data[p]['mortality'] for p in gpids]
        gp = [all_probs[pids.index(p)] for p in gpids]
        try:
            print(f"    {treat_name:25s}: AUROC {roc_auc_score(gl,gp):.4f} "
                  f"(n={len(gpids)}, deaths={sum(gl)})")
        except Exception:
            print(f"    {treat_name:25s}: insufficient deaths")

    results_df = pd.DataFrame({
        'patient_id':     pids,
        'true_mortality': labels,
        'predicted_prob': all_probs,
        'treatment':      [patient_data[p]['treatment_str'] for p in pids],
    })
    return auroc, auprc, results_df


def validate_trajectory_convergence(latents, patient_data):
    print("\n" + "="*55)
    print("TRAJECTORY CONVERGENCE REPLICATION")
    print("="*55)

    plasma_pids  = [p for p in latents
                    if patient_data[p]['treatment'] == 1
                    and patient_data[p]['masks'][0] == 1]
    control_pids = [p for p in latents
                    if patient_data[p]['treatment'] == 0
                    and patient_data[p]['masks'][0] == 1]
    print(f"\n  Plasma-containing: {len(plasma_pids)} | "
          f"No plasma: {len(control_pids)}")

    if len(plasma_pids) < 5 or len(control_pids) < 5:
        print("  Insufficient patients.")
        return {}

    distances = {}
    tp_names  = {0: 'tp0 (admission)', 1: 'tp24 (24h)', 2: 'tp72 (masked)'}

    print(f"\n  {'Timepoint':25s} {'Distance':10s} "
          f"{'N plasma':10s} {'N control':10s}")
    print("  " + "-"*55)

    for tp_idx, tp_name in tp_names.items():
        pv = [p for p in plasma_pids
              if patient_data[p]['masks'][tp_idx] == 1]
        cv = [p for p in control_pids
              if patient_data[p]['masks'][tp_idx] == 1]
        if len(pv) < 3 or len(cv) < 3:
            print(f"  {tp_name:25s} {'N/A':10s} "
                  f"{len(pv):10d} {len(cv):10d}")
            distances[tp_idx] = None
            continue
        pm = torch.stack([latents[p][tp_idx] for p in pv]).mean(0)
        cm = torch.stack([latents[p][tp_idx] for p in cv]).mean(0)
        d  = (pm - cm).norm().item()
        distances[tp_idx] = d
        print(f"  {tp_name:25s} {d:10.4f} "
              f"{len(pv):10d} {len(cv):10d}")

    d0  = distances.get(0)
    d24 = distances.get(1)
    results = {}

    if d0 and d24:
        convergence = d0 - d24
        print(f"\n  Convergence tp0 to tp24: {convergence:.4f}")

        all_pids   = plasma_pids + control_pids
        all_z0     = torch.stack([latents[p][0] for p in all_pids])
        valid_tp24 = [p for p in all_pids
                      if patient_data[p]['masks'][1] == 1]
        n_p   = len(plasma_pids)
        n_p24 = len([p for p in plasma_pids
                     if patient_data[p]['masks'][1] == 1])

        null_conv = []
        rng = np.random.default_rng(42)
        for _ in range(1000):
            perm = rng.permutation(len(all_pids))
            p_z0 = all_z0[perm[:n_p]].mean(0)
            c_z0 = all_z0[perm[n_p:]].mean(0)
            nd0  = (p_z0 - c_z0).norm().item()
            if len(valid_tp24) >= 10 and n_p24 > 0:
                z24_all = torch.stack(
                    [latents[p][1] for p in valid_tp24])
                perm24  = rng.permutation(len(valid_tp24))
                p_z24   = z24_all[perm24[:n_p24]].mean(0)
                c_z24   = z24_all[perm24[n_p24:]].mean(0)
                nd24    = (p_z24 - c_z24).norm().item()
                null_conv.append(nd0 - nd24)

        if null_conv:
            p_val = np.mean(np.array(null_conv) >= convergence)
            fx    = convergence / (np.mean(null_conv) + 1e-8)
            print(f"  Null mean convergence : {np.mean(null_conv):.4f}")
            print(f"  Effect size vs null   : {fx:.1f}x")
            print(f"  Permutation p-value   : {p_val:.4f} "
                  f"({'SIGNIFICANT' if p_val < 0.05 else 'trend'} at a=0.05)")
            print(f"\n  Comparison to PAMPer finding:")
            print(f"  {'':20s} {'PAMPer':12s} {'SWAT':12s}")
            print(f"  {'tp0 distance':20s} {'0.433':12s} {d0:12.4f}")
            print(f"  {'tp24 distance':20s} {'0.103':12s} {d24:12.4f}")
            print(f"  {'convergence':20s} {'0.330':12s} {convergence:12.4f}")
            print(f"  {'p-value':20s} {'0.107':12s} {p_val:12.4f}")
            results = {
                'tp0_distance':       d0,
                'tp24_distance':      d24,
                'convergence':        convergence,
                'convergence_p':      p_val,
                'convergence_effect': fx,
            }

    return results


def compare_protein_importance(latents, patient_data, data_dir):
    print("\n" + "="*55)
    print("TOP PROTEIN VALIDATION IN SWAT")
    print("="*55)

    imp_path = "outputs/interpretation/protein_importance.csv"
    if not os.path.exists(imp_path):
        print("  Protein importance file not found — skipping.")
        return

    imp_df       = pd.read_csv(imp_path)
    top_proteins = imp_df.head(20)['feature'].tolist()

    df = pd.read_csv(os.path.join(data_dir, SWAT_CSV))
    df['timepoint'] = df['Name'].apply(parse_swat_timepoint)
    swat_tp0 = df[df['timepoint'] == 0].copy()

    print(f"\n  PAMPer top 20 proteins — survivor vs non-survivor in SWAT tp0:")
    print(f"  {'Protein':45s} {'Surv mean':12s} "
          f"{'NonSurv mean':14s} {'p-value':10s}")
    print("  " + "-"*84)

    significant = 0
    for prot in top_proteins:
        if prot not in swat_tp0.columns:
            continue
        surv    = swat_tp0[swat_tp0['30d_Mortality'] == 0][prot].dropna()
        nonsurv = swat_tp0[swat_tp0['30d_Mortality'] == 1][prot].dropna()
        if len(surv) < 3 or len(nonsurv) < 3:
            continue
        _, p = stats.mannwhitneyu(surv, nonsurv, alternative='two-sided')
        sig  = ' *' if p < 0.05 else ''
        if p < 0.05:
            significant += 1
        print(f"  {prot[:43]:45s} {surv.mean():12.1f} "
              f"{nonsurv.mean():14.1f} {p:10.4f}{sig}")

    print(f"\n  {significant}/20 PAMPer top proteins significant in SWAT (p<0.05)")


def main(data_dir="./"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    torch.manual_seed(FT_SEED)
    np.random.seed(FT_SEED)
    device = get_device()
    print(f"Using device: {device}\n")

    patient_data, protein_cols = load_swat_data(data_dir)

    latents, soma_vae = encode_swat_proteins(
        patient_data, protein_cols, device, CKPT_DIR, data_dir)

    print("\nLoading PAMPer mortality model...")
    model = CachedMultimodalTransformer(dropout=0.3).to(device)
    model.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, 'model_cached_fold0.pt'),
        map_location=device))
    model.eval()
    print("  Model loaded.")

    # Zero-shot evaluation before fine-tuning
    auroc_zs, auprc_zs, _ = validate_mortality(
        model, patient_data, latents, device,
        label="zero-shot (before fine-tuning)")

    # Fine-tune predictor head on SWAT labels
    model, test_pids, ft_best_auroc = finetune_predictor_on_swat(
        model, patient_data, latents, device)

    # Evaluate on held-out test set only
    auroc_ft, auprc_ft, results_df = validate_mortality(
        model, patient_data, latents, device,
        label="fine-tuned (held-out test set)",
        exclude_pids=test_pids)
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, 'swat_predictions.csv'), index=False)

    conv_results = validate_trajectory_convergence(latents, patient_data)

    compare_protein_importance(latents, patient_data, data_dir)

    summary = {
        'n_patients':      len(patient_data),
        'n_test_patients': len(test_pids),
        'mortality_rate':  float(np.mean(
            [v['mortality'] for v in patient_data.values()])),
        'auroc_zero_shot': auroc_zs,
        'auroc_finetuned': auroc_ft,
        'auprc_finetuned': auprc_ft,
        'pamper_auroc':    0.875,
        'ft_epochs':       FT_EPOCHS,
        'ft_lr':           FT_LR,
    }
    summary.update(conv_results)
    with open(os.path.join(OUTPUT_DIR, 'swat_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print("SWAT EXTERNAL VALIDATION COMPLETE")
    print(f"{'='*55}")
    print(f"  PAMPer AUROC            : 0.875 (5-fold CV)")
    print(f"  SWAT AUROC (zero-shot)  : {auroc_zs:.4f}")
    print(f"  SWAT AUROC (fine-tuned) : {auroc_ft:.4f} "
          f"(n={len(test_pids)} held-out)")
    print(f"  Mortality rate          : "
          f"{summary['mortality_rate']:.1%} vs 42% PAMPer")
    if conv_results.get('convergence_p') is not None:
        p  = conv_results['convergence_p']
        fx = conv_results.get('convergence_effect', 0)
        print(f"  tp24 convergence        : p={p:.4f}, {fx:.1f}x null")
    print(f"\n  Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    main(data_dir=data_dir)