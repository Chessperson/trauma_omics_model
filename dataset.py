"""
dataset.py — Multimodal trauma proteomics dataloader
======================================================
Loads and merges real + synthetic data across four files:
  - PAMPer_External_data.xlsx         (real proteomics)
  - synthetic_PAMPer_External_full.xlsx (synthetic proteomics)
  - PAMPer_Metabolon_metabolomics.xlsx  (real metabolomics)
  - synthetic_metabolomics.xlsx         (synthetic metabolomics)

Each patient is represented as a dict of tensors, one per modality,
with a missingness mask for absent modalities.

Usage:
    from dataset import build_datasets
    train_ds, val_ds, test_ds = build_datasets(data_dir='data/', fold=0)
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

# ── Constants ────────────────────────────────────────────────────────────────

TIMEPOINTS     = [0, 24, 72]
N_FOLDS        = 5
RANDOM_SEED    = 42

# Latent dimensions after VAE encoding (used for mask token sizing)
LATENT_DIMS = {
    "somalogic":  128,
    "metabolon":  64,
    "luminex":    32,
    "lipidomics": 64,
    "clinical":   32,
}

# ── File loading helpers ──────────────────────────────────────────────────────

def load_patients(xl, is_synthetic=False):
    """Load patient-level clinical data + mortality label."""
    df = pd.read_excel(xl, sheet_name="Patients")
    df["is_synthetic"] = int(is_synthetic)

    # Mortality label — use 30_Mortality (binary 0/1)
    if "30_Mortality" in df.columns:
        df["mortality"] = pd.to_numeric(df["30_Mortality"], errors="coerce").fillna(0).astype(int)
    else:
        raise ValueError("Could not find 30_Mortality column in Patients sheet")

    # Standardise the patient ID column to 'patient_id'
    # Real data uses 'Patients' (e.g. PAMP1001), synthetic uses 'Patients' too
    df = df.rename(columns={"Patients": "patient_id", "GenID": "gen_id"})
    return df


def load_somalogic(xl, patient_ids):
    """Load SomaLogic proteomics — shape (n_patients, 3_timepoints, 7596_proteins)."""
    df = pd.read_excel(xl, sheet_name="SomaLogic")
    df = df.rename(columns={"Patient": "patient_id", "Time point": "timepoint"})
    protein_cols = [c for c in df.columns if c not in ["patient_id", "timepoint"]]

    # Keep only patients we care about
    df = df[df["patient_id"].isin(patient_ids)]

    # Log-transform (proteins are right-skewed)
    df[protein_cols] = np.log1p(df[protein_cols].apply(pd.to_numeric, errors="coerce").clip(lower=0))
    return df, protein_cols


def load_luminex(xl, patient_ids):
    """Load Luminex cytokines — shape (n_patients, 3_timepoints, n_cytokines)."""
    df = pd.read_excel(xl, sheet_name="Luminex")
    df = df.rename(columns={"GenID": "gen_id", "Patients": "patient_id", "Time point": "timepoint"})

    # Luminex has censored values like '<117' — coerce to NaN then fill with col min
    cytokine_cols = [c for c in df.columns if c not in ["gen_id", "patient_id", "timepoint"]]
    for col in cytokine_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[cytokine_cols] = df[cytokine_cols].apply(lambda c: c.fillna(c.min()))
    df[cytokine_cols] = np.log1p(df[cytokine_cols].clip(lower=0))

    df = df[df["patient_id"].isin(patient_ids)]
    return df, cytokine_cols


def load_metabolon(xl, patient_ids):
    """Load Metabolon metabolomics — shape (n_patients, 3_timepoints, n_metabolites)."""
    if not hasattr(xl, 'sheet_names'):
        xl = pd.ExcelFile(xl)
    if "Metabolon_U" in xl.sheet_names:
        met_sheet = "Metabolon_U"
    elif "PAMPer_Metabolon_metabolomics" in xl.sheet_names:
        met_sheet = "PAMPer_Metabolon_metabolomics"
    else:
        met_sheet = xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=met_sheet)
    df = df.rename(columns={"GenID": "patient_id", "Time": "timepoint"})

    met_cols = [c for c in df.columns if c not in ["patient_id", "ID", "timepoint"]]
    df[met_cols] = np.log1p(df[met_cols].apply(pd.to_numeric, errors="coerce").clip(lower=0))

    # Drop healthy controls (HC*) — keep patients only
    df = df[~df["patient_id"].astype(str).str.startswith("HC")]
    df["patient_id"] = df["patient_id"].astype(str)
    df["patient_id"] = df["patient_id"].astype(str)
    df = df[df["patient_id"].isin([str(p) for p in patient_ids])]
    return df, met_cols


def load_lipidomics(xl, patient_ids):
    """Load Species concentrations (lipidomics) — temporal, IDs like PAMP1001-0HR-CT-C."""
    # Try real data sheet name first, fall back to first sheet for synthetic
    if not hasattr(xl, "sheet_names"):
        xl = pd.ExcelFile(xl)
    sheet = "Species_concentrations_unscaled" if "Species_concentrations_unscaled" in xl.sheet_names else xl.sheet_names[0]
    df = pd.read_excel(xl, sheet_name=sheet)
    id_col = df.columns[0]
    df = df.rename(columns={id_col: "raw_id"})
    lipid_cols = df.columns[1:].tolist()
    df[lipid_cols] = np.log1p(df[lipid_cols].apply(pd.to_numeric, errors="coerce").clip(lower=0))

    # Drop healthy controls
    df = df[~df["raw_id"].astype(str).str.contains("HC", na=False)]

    # Parse patient_id and timepoint from raw_id (e.g. PAMP1001-0HR-CT-C)
    def parse_id(raw):
        raw = str(raw)
        parts = raw.split("-")
        pat = parts[0]  # e.g. PAMP1001
        tp = 0
        for p in parts[1:]:
            if "0HR" in p or p == "0HR": tp = 0
            elif "24HR" in p: tp = 24
            elif "72HR" in p: tp = 72
        return pat, tp

    df[["patient_id", "timepoint"]] = pd.DataFrame(
        df["raw_id"].apply(parse_id).tolist(), index=df.index)
    df = df[df["patient_id"].isin([str(p) for p in patient_ids])]
    return df, lipid_cols


# ── Per-patient tensor builder ────────────────────────────────────────────────

def build_temporal_tensor(df, value_cols, patient_id, scaler=None, fit_scaler=False):
    """
    For a given patient, return a (3, n_features) tensor across timepoints.
    Missing timepoints are filled with zeros (mask token handles it in model).
    Returns (tensor, mask) where mask[t]=1 means timepoint t is present.
    """
    rows = df[df["patient_id"].astype(str) == str(patient_id)]
    mat  = np.zeros((3, len(value_cols)), dtype=np.float32)
    mask = np.zeros(3, dtype=np.float32)

    for i, tp in enumerate(TIMEPOINTS):
        row = rows[rows["timepoint"] == tp]
        if len(row) > 0:
            vals = row[value_cols].values[0].astype(np.float32)
            vals = np.nan_to_num(vals, nan=0.0)
            mat[i] = vals
            mask[i] = 1.0

    return torch.tensor(mat), torch.tensor(mask)


def build_clinical_tensor(pat_row, clinical_cols, scaler=None):
    """Convert a single patient's clinical row to a tensor."""
    vals = pat_row[clinical_cols].values.astype(np.float32)
    vals = np.nan_to_num(vals, nan=0.0)
    if scaler is not None:
        vals = scaler.transform(vals.reshape(1, -1))[0]
    return torch.tensor(vals)


# ── Dataset class ─────────────────────────────────────────────────────────────

class TraumaDataset(Dataset):
    """
    PyTorch Dataset for multimodal trauma mortality prediction.

    Each item is a dict:
    {
        "somalogic":       (3, n_soma_proteins) float tensor,
        "somalogic_mask":  (3,) float tensor  [1=present, 0=missing],
        "luminex":         (3, n_cytokines)   float tensor,
        "luminex_mask":    (3,) float tensor,
        "metabolon":       (3, n_metabolites) float tensor,
        "metabolon_mask":  (3,) float tensor,
        "lipidomics":      (1, n_lipids)      float tensor,
        "lipidomics_mask": (1,) float tensor,
        "clinical":        (n_clinical_feats,) float tensor,
        "mortality":       scalar int tensor  [0 or 1],
        "patient_id":      str,
        "is_synthetic":    int [0=real, 1=synthetic],
    }
    """

    def __init__(self, patient_df, soma_df, soma_cols,
                 lum_df, lum_cols, met_df, met_cols,
                 lip_df, lip_cols, clinical_cols,
                 clinical_scaler=None):

        self.patients       = patient_df.reset_index(drop=True)
        self.soma_df        = soma_df
        self.soma_cols      = soma_cols
        self.lum_df         = lum_df
        self.lum_cols       = lum_cols
        self.met_df         = met_df
        self.met_cols       = met_cols
        self.lip_df         = lip_df
        self.lip_cols       = lip_cols
        self.clinical_cols  = clinical_cols
        self.clinical_scaler = clinical_scaler

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        row        = self.patients.iloc[idx]
        patient_id = str(row["patient_id"])
        mortality  = int(row["mortality"])
        is_syn     = int(row["is_synthetic"])

        # ── SomaLogic ──
        soma_tensor, soma_mask = build_temporal_tensor(
            self.soma_df, self.soma_cols, patient_id)

        # ── Luminex ──
        lum_tensor, lum_mask = build_temporal_tensor(
            self.lum_df, self.lum_cols, patient_id)

        # ── Metabolon ──
        met_tensor, met_mask = build_temporal_tensor(
            self.met_df, self.met_cols, str(row["gen_id"]))

        # ── Lipidomics (temporal — same structure as other modalities) ──
        lip_tensor, lip_mask = build_temporal_tensor(
            self.lip_df, self.lip_cols, patient_id)

        # ── Clinical ──
        clin_vals = pd.to_numeric(row[self.clinical_cols], errors='coerce').fillna(0).values.astype(np.float32)
        clin_vals = np.nan_to_num(clin_vals, nan=0.0)
        if self.clinical_scaler is not None:
            clin_vals = self.clinical_scaler.transform(
                clin_vals.reshape(1, -1))[0]
        clin_tensor = torch.tensor(clin_vals)

        return {
            "somalogic":       soma_tensor,
            "somalogic_mask":  soma_mask,
            "luminex":         lum_tensor,
            "luminex_mask":    lum_mask,
            "metabolon":       met_tensor,
            "metabolon_mask":  met_mask,
            "lipidomics":      lip_tensor,
            "lipidomics_mask": lip_mask,
            "clinical":        clin_tensor,
            "mortality":       torch.tensor(mortality, dtype=torch.long),
            "patient_id":      patient_id,
            "is_synthetic":    is_syn,
        }


# ── Main builder function ─────────────────────────────────────────────────────

def build_datasets(data_dir="data/", fold=0):
    """
    Load all data, merge real + synthetic, split into train/val/test
    using nested stratified 5-fold CV.

    Args:
        data_dir: path to folder containing all four data files
        fold:     which outer fold to use as test set (0-4)

    Returns:
        train_ds, val_ds, test_ds  — TraumaDataset instances
        feature_dims               — dict of feature dimensions per modality
    """

    print("Loading data files...")

    # ── Load Excel files ──
    real_proto_path  = os.path.join(data_dir, "PAMPer_External_data.xlsx")
    syn_proto_path   = os.path.join(data_dir, "synthetic_PAMPer_External_full.xlsx")
    real_met_path    = os.path.join(data_dir, "PAMPer_Metabolon_metabolomics.xlsx")
    syn_met_path     = os.path.join(data_dir, "synthetic_metabolomics.xlsx")

    for path in [real_proto_path, syn_proto_path, real_met_path, syn_met_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}\nMake sure all four files are in {data_dir}")

    real_proto_xl = pd.ExcelFile(real_proto_path)
    syn_proto_xl  = pd.ExcelFile(syn_proto_path)
    real_met_xl   = pd.ExcelFile(real_met_path)
    # synthetic metabolomics may be a single-sheet CSV or Excel
    if syn_met_path.endswith(".xlsx"):
        syn_met_xl = pd.ExcelFile(syn_met_path)
    else:
        syn_met_xl = None

    # ── Patients ──
    print("  Loading patient demographics...")
    real_pat = load_patients(real_proto_xl, is_synthetic=False)
    syn_pat  = load_patients(syn_proto_xl,  is_synthetic=True)
    all_pat  = pd.concat([real_pat, syn_pat], ignore_index=True)

    # Real patients only for CV splitting
    real_ids      = real_pat["patient_id"].tolist()
    real_labels   = real_pat["mortality"].values

    # ── Clinical feature columns ──
    clinical_cols = [
        c for c in real_pat.columns
        if c not in ["patient_id", "gen_id", "is_synthetic", "mortality",
                     "30_Mortality", "Outcome", "Date_of_death",
                     "ED_hosp_arrival_date", "Discharge_date", "ID",
                     "Biological_sex", "Race", "Ethnicity", "Burn",
                     "Injury_type", "TBI", "Intervention_arm"]
        and real_pat[c].dtype in [np.float64, np.int64, float, int]
    ]
    print(f"  Clinical features: {len(clinical_cols)}")

    # ── SomaLogic ──
    print("  Loading SomaLogic proteomics (7596 proteins)...")
    all_soma_ids = all_pat["patient_id"].tolist()
    real_soma, soma_cols = load_somalogic(real_proto_xl, all_soma_ids)
    syn_soma,  _         = load_somalogic(syn_proto_xl,  all_soma_ids)
    all_soma = pd.concat([real_soma, syn_soma], ignore_index=True)
    print(f"  SomaLogic proteins: {len(soma_cols)}")

    # ── Luminex ──
    print("  Loading Luminex cytokines...")
    real_lum, lum_cols = load_luminex(real_proto_xl, all_soma_ids)
    syn_lum,  _        = load_luminex(syn_proto_xl,  all_soma_ids)
    all_lum = pd.concat([real_lum, syn_lum], ignore_index=True)
    print(f"  Luminex cytokines: {len(lum_cols)}")

    # ── Metabolon ──
    print("  Loading Metabolon metabolomics...")
    real_gen_ids = real_pat["gen_id"].astype(str).tolist()
    syn_gen_ids  = syn_pat["gen_id"].astype(str).tolist()
    real_met, met_cols = load_metabolon(real_met_xl, real_gen_ids)
    if syn_met_xl:
        syn_met, _ = load_metabolon(syn_met_xl, syn_gen_ids)
        all_met = pd.concat([real_met, syn_met], ignore_index=True)
    else:
        all_met = real_met
    print(f"  Metabolon metabolites: {len(met_cols)}")

    # ── Lipidomics ──
    print("  Loading lipidomics...")
    real_lip, lip_cols = load_lipidomics(real_proto_xl, all_soma_ids)
    syn_lip,  _        = load_lipidomics(syn_proto_xl,  all_soma_ids)
    all_lip = pd.concat([real_lip, syn_lip], ignore_index=True)
    print(f"  Lipidomic species: {len(lip_cols)}")

    # ── Stratified 5-fold CV split on real patients only ──
    print(f"\nSplitting into {N_FOLDS}-fold CV (fold {fold} = test set)...")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    folds = list(skf.split(real_ids, real_labels))

    train_val_idx, test_idx = folds[fold]

    # Inner fold: split train_val into train and val (use fold+1 mod 5)
    inner_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED + 1)
    train_val_ids     = [real_ids[i] for i in train_val_idx]
    train_val_labels  = real_labels[train_val_idx]
    inner_folds       = list(inner_skf.split(train_val_ids, train_val_labels))
    train_inner_idx, val_inner_idx = inner_folds[fold % N_FOLDS]

    train_real_ids = [train_val_ids[i] for i in train_inner_idx]
    val_ids        = [train_val_ids[i] for i in val_inner_idx]
    test_ids       = [real_ids[i] for i in test_idx]
    syn_ids        = syn_pat["patient_id"].tolist()

    print(f"  Train real patients : {len(train_real_ids)}")
    print(f"  Train synthetic     : {len(syn_ids)}  (added to training only)")
    print(f"  Val patients        : {len(val_ids)}  (real only)")
    print(f"  Test patients       : {len(test_ids)} (real only)")

    # ── Build patient subsets ──
    def subset(ids, include_synthetic=False):
        mask = all_pat["patient_id"].isin(ids)
        if include_synthetic:
            mask = mask | (all_pat["is_synthetic"] == 1)
        return all_pat[mask].copy()

    train_pat = subset(train_real_ids, include_synthetic=True)
    val_pat   = subset(val_ids,        include_synthetic=False)
    test_pat  = subset(test_ids,       include_synthetic=False)

    # ── Fit clinical scaler on training data only ──
    scaler = StandardScaler()
    train_clin = train_pat[clinical_cols].apply(
        pd.to_numeric, errors="coerce").fillna(0).values
    scaler.fit(train_clin)

    # ── Build Dataset objects ──
    def make_ds(pat_subset):
        return TraumaDataset(
            patient_df=pat_subset,
            soma_df=all_soma, soma_cols=soma_cols,
            lum_df=all_lum,   lum_cols=lum_cols,
            met_df=all_met,   met_cols=met_cols,
            lip_df=all_lip,   lip_cols=lip_cols,
            clinical_cols=clinical_cols,
            clinical_scaler=scaler,
        )

    train_ds = make_ds(train_pat)
    val_ds   = make_ds(val_pat)
    test_ds  = make_ds(test_pat)

    feature_dims = {
        "somalogic":  len(soma_cols),
        "luminex":    len(lum_cols),
        "metabolon":  len(met_cols),
        "lipidomics": len(lip_cols),
        "clinical":   len(clinical_cols),
    }

    print(f"\nDatasets ready.")
    print(f"  Feature dims: {feature_dims}")
    return train_ds, val_ds, test_ds, feature_dims


# ── Quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "data/"

    print("=" * 55)
    print("DATASET SANITY CHECK")
    print("=" * 55)

    train_ds, val_ds, test_ds, dims = build_datasets(data_dir=data_dir, fold=0)

    print("\nChecking first training sample...")
    sample = train_ds[0]
    for key, val in sample.items():
        if isinstance(val, torch.Tensor):
            print(f"  {key:20s}: shape={tuple(val.shape)}, dtype={val.dtype}")
        else:
            print(f"  {key:20s}: {val}")

    # Check class balance
    train_labels = [int(train_ds[i]["mortality"]) for i in range(len(train_ds))]
    val_labels   = [int(val_ds[i]["mortality"])   for i in range(len(val_ds))]
    test_labels  = [int(test_ds[i]["mortality"])  for i in range(len(test_ds))]

    print(f"\nClass balance:")
    print(f"  Train: {sum(train_labels)} positive / {len(train_labels)} total "
          f"({100*sum(train_labels)/len(train_labels):.1f}%)")
    print(f"  Val:   {sum(val_labels)} positive / {len(val_labels)} total "
          f"({100*sum(val_labels)/len(val_labels):.1f}%)")
    print(f"  Test:  {sum(test_labels)} positive / {len(test_labels)} total "
          f"({100*sum(test_labels)/len(test_labels):.1f}%)")

    # Check synthetic data is only in training
    train_syn = sum(int(train_ds[i]["is_synthetic"]) for i in range(len(train_ds)))
    val_syn   = sum(int(val_ds[i]["is_synthetic"])   for i in range(len(val_ds)))
    test_syn  = sum(int(test_ds[i]["is_synthetic"])  for i in range(len(test_ds)))
    print(f"\nSynthetic data leakage check:")
    print(f"  Train synthetic: {train_syn}  (expected > 0)")
    print(f"  Val synthetic:   {val_syn}    (expected 0)")
    print(f"  Test synthetic:  {test_syn}   (expected 0)")

    print("\nAll checks passed!" if val_syn == 0 and test_syn == 0
          else "\nWARNING: synthetic data found in val/test!")