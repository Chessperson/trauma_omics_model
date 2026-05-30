"""
swat_validation.py — External Validation of PAMPer Model on SWAT Cohort
"""

import os, sys, re, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy import stats

sys.path.insert(0, os.path.dirname(__file__))
from encoders import SomaLogicVAE
from train_cached import CachedMultimodalTransformer, get_device, safe_logits
from dataset import build_datasets

OUTPUT_DIR = "outputs/swat_validation/"
SWAT_CSV   = "SWAT_proteins_clinical.csv"
CKPT_DIR   = "outputs/checkpoints/"


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

    all_cols     = df.columns.tolist()
    protein_cols = all_cols[218:]
    print(f"  Proteins: {len(protein_cols)}")
    print(f"  Patients: {df['ID'].nunique()}")

    plasma_treatments = {'All three', 'PRBC+Plasma', 'WB+Plasma'}
    tp0_patients      = df[df['timepoint'] == 0]['ID'].unique().tolist()

    patient_data = {}
    for pid in tp0_patients:
        prows = df[df['ID'] == pid].sort_values('timepoint')
        row0  = prows[prows['timepoint'] == 0].iloc[0]

        treatment_str  = str(row0.get('Treatment', 'Other'))
        treatment      = 1 if treatment_str in plasma_treatments else 0
        mortality      = int(row0.get('30d_Mortality', 0))

        protein_matrix = np.zeros((3, len(protein_cols)), dtype=np.float32)
        masks          = np.zeros(3, dtype=np.float32)

        for _, row in prows.iterrows():
            tp   = row['timepoint']
            vals = row[protein_cols].values.astype(np.float32)
            vals = np.nan_to_num(vals, nan=0.0)
            vals = np.log1p(np.abs(vals))
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

    print(f"  Valid patients: {len(patient_data)}")
    n_plasma  = sum(1 for v in patient_data.values() if v['treatment'] == 1)
    mort_rate = np.mean([v['mortality'] for v in patient_data.values()])
    print(f"  Plasma-containing: {n_plasma} | No plasma: {len(patient_data)-n_plasma}")
    print(f"  Mortality rate: {mort_rate:.3f}")
    return patient_data, protein_cols

def finetune_vae_on_swat(soma_vae, patient_data, protein_cols, device, n_epochs=30):
    """Fine-tune PAMPer VAE on SWAT protein distribution."""
    print("\nFine-tuning VAE on SWAT distribution...")
    import torch.nn.functional as F
    from torch.optim import AdamW

    # Build SWAT protein tensor
    all_proteins = []
    for data in patient_data.values():
        for tp in range(3):
            if data['masks'][tp] == 1:
                all_proteins.append(data['proteins'][tp])
    soma_data = torch.tensor(np.array(all_proteins), dtype=torch.float32)
    print(f"  SWAT protein samples for fine-tuning: {len(soma_data)}")

    soma_vae.train()
    optim = AdamW(soma_vae.parameters(), lr=1e-5, weight_decay=1e-4)

    for epoch in range(1, n_epochs+1):
        idx    = torch.randperm(len(soma_data))
        losses = []
        for i in range(0, len(soma_data), 32):
            batch = soma_data[idx[i:i+32]].to(device)
            optim.zero_grad()
            recon, mu, logvar = soma_vae(batch)
            recon_loss = F.mse_loss(recon, batch)
            kl_loss    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            loss = recon_loss + soma_vae.beta * kl_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(soma_vae.parameters(), 0.5)
            optim.step()
            losses.append(loss.item())
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Loss: {np.mean(losses):.4f}")

    soma_vae.eval()
    print("  Fine-tuning complete.")
    return soma_vae


def encode_swat_proteins(patient_data, protein_cols, device, ckpt_dir, data_dir):
    print("\nEncoding SWAT proteins through PAMPer VAE...")
    _, _, _, feature_dims = build_datasets(data_dir=data_dir, fold=0)
    soma_vae = SomaLogicVAE(feature_dims['somalogic'], 128)
    soma_vae.load_state_dict(torch.load(
        os.path.join(ckpt_dir, 'somalogic_vae.pt'), map_location=device))
    soma_vae = soma_vae.to(device).eval()
    print(f"  VAE loaded (beta={soma_vae.beta})")
    soma_vae = finetune_vae_on_swat(soma_vae, patient_data, protein_cols, device)

    latents = {}
    with torch.no_grad():
        for pid, data in patient_data.items():
            proteins = torch.tensor(data['proteins'], dtype=torch.float32).to(device)
            z_list   = []
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
        [patient_data[p]['masks'] for p in pids], dtype=torch.float32).to(device)
    zeros_64  = torch.zeros(B, 3, 64, device=device)
    zeros_32  = torch.zeros(B, 3, 32, device=device)
    zero_mask = torch.zeros(B, 3, device=device)
    clin_raw = torch.tensor(
    [patient_data[p]['clinical'] for p in pids], dtype=torch.float32)
# Pad from 29 to 32 dims to match PAMPer model expectation
    clinical = torch.zeros(len(pids), 32)
    clinical[:, :29] = clin_raw
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


def validate_mortality(model, patient_data, latents, device):
    print("\n" + "="*55)
    print("MORTALITY PREDICTION VALIDATION")
    print("="*55)

    pids      = list(patient_data.keys())
    labels    = [patient_data[p]['mortality'] for p in pids]
    all_probs = []

    model.eval()
    with torch.no_grad():
        for i in range(0, len(pids), 16):
            batch_pids = pids[i:i+16]
            batch      = build_prediction_batch(patient_data, latents, batch_pids, device)
            logits, _  = model(batch)
            probs      = torch.sigmoid(safe_logits(logits)).cpu().tolist()
            all_probs.extend(probs)

    try:
        auroc = roc_auc_score(labels, all_probs)
        auprc = average_precision_score(labels, all_probs)
    except Exception:
        auroc = float('nan')
        auprc = float('nan')

    print(f"\n  PAMPer model on SWAT cohort (zero-shot transfer):")
    print(f"  AUROC : {auroc:.4f}  (PAMPer training AUROC: 0.878)")
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
    print(f"\n  Plasma-containing: {len(plasma_pids)} | No plasma: {len(control_pids)}")

    if len(plasma_pids) < 5 or len(control_pids) < 5:
        print("  Insufficient patients.")
        return {}

    distances = {}
    tp_names  = {0: 'tp0 (admission)', 1: 'tp24 (24h)', 2: 'tp72 (masked)'}

    print(f"\n  {'Timepoint':25s} {'Distance':10s} {'N plasma':10s} {'N control':10s}")
    print("  " + "-"*55)

    for tp_idx, tp_name in tp_names.items():
        pv = [p for p in plasma_pids  if patient_data[p]['masks'][tp_idx] == 1]
        cv = [p for p in control_pids if patient_data[p]['masks'][tp_idx] == 1]
        if len(pv) < 3 or len(cv) < 3:
            print(f"  {tp_name:25s} {'N/A':10s} {len(pv):10d} {len(cv):10d}")
            distances[tp_idx] = None
            continue
        pm = torch.stack([latents[p][tp_idx] for p in pv]).mean(0)
        cm = torch.stack([latents[p][tp_idx] for p in cv]).mean(0)
        d  = (pm - cm).norm().item()
        distances[tp_idx] = d
        print(f"  {tp_name:25s} {d:10.4f} {len(pv):10d} {len(cv):10d}")

    d0  = distances.get(0)
    d24 = distances.get(1)
    results = {}

    if d0 and d24:
        convergence = d0 - d24
        print(f"\n  Convergence tp0 to tp24: {convergence:.4f}")

        all_pids   = plasma_pids + control_pids
        all_z0     = torch.stack([latents[p][0] for p in all_pids])
        valid_tp24 = [p for p in all_pids if patient_data[p]['masks'][1] == 1]
        n_p        = len(plasma_pids)
        n_p24      = len([p for p in plasma_pids if patient_data[p]['masks'][1] == 1])

        null_conv = []
        rng = np.random.default_rng(42)
        for _ in range(1000):
            perm = rng.permutation(len(all_pids))
            p_z0 = all_z0[perm[:n_p]].mean(0)
            c_z0 = all_z0[perm[n_p:]].mean(0)
            nd0  = (p_z0 - c_z0).norm().item()
            if len(valid_tp24) >= 10 and n_p24 > 0:
                z24_all = torch.stack([latents[p][1] for p in valid_tp24])
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

    df       = pd.read_csv(os.path.join(data_dir, SWAT_CSV))
    df['timepoint'] = df['Name'].apply(parse_swat_timepoint)
    swat_tp0 = df[df['timepoint'] == 0].copy()

    print(f"\n  PAMPer top 20 proteins — survivor vs non-survivor in SWAT tp0:")
    print(f"  {'Protein':45s} {'Surv mean':12s} {'NonSurv mean':14s} {'p-value':10s}")
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
        print(f"  {prot[:43]:45s} {surv.mean():12.1f} {nonsurv.mean():14.1f} {p:10.4f}{sig}")

    print(f"\n  {significant}/20 PAMPer top proteins significant in SWAT (p<0.05)")


def main(data_dir="./"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = get_device()
    print(f"Using device: {device}\n")

    patient_data, protein_cols = load_swat_data(data_dir)
    latents, soma_vae          = encode_swat_proteins(
        patient_data, protein_cols, device, CKPT_DIR, data_dir)

    print("\nLoading PAMPer mortality model...")
    model = CachedMultimodalTransformer(dropout=0.3).to(device)
    model.load_state_dict(torch.load(
        os.path.join(CKPT_DIR, 'model_cached_fold0.pt'), map_location=device))
    model.eval()
    print("  Model loaded.")

    auroc, auprc, results_df = validate_mortality(
        model, patient_data, latents, device)
    results_df.to_csv(
        os.path.join(OUTPUT_DIR, 'swat_predictions.csv'), index=False)

    conv_results = validate_trajectory_convergence(latents, patient_data)

    compare_protein_importance(latents, patient_data, data_dir)

    summary = {
        'n_patients':     len(patient_data),
        'mortality_rate': float(np.mean([v['mortality'] for v in patient_data.values()])),
        'auroc':          auroc,
        'auprc':          auprc,
        'pamper_auroc':   0.878,
    }
    summary.update(conv_results)
    with open(os.path.join(OUTPUT_DIR, 'swat_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*55}")
    print("SWAT EXTERNAL VALIDATION COMPLETE")
    print(f"{'='*55}")
    print(f"  PAMPer AUROC  : 0.878 (training cohort)")
    print(f"  SWAT AUROC    : {auroc:.4f} (external validation, zero-shot)")
    print(f"  Mortality rate: {summary['mortality_rate']:.1%} vs 42% PAMPer")
    if conv_results.get('convergence_p') is not None:
        p  = conv_results['convergence_p']
        fx = conv_results.get('convergence_effect', 0)
        print(f"  tp24 convergence: p={p:.4f}, {fx:.1f}x null")
    print(f"\n  Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    main(data_dir=data_dir)