"""
coag_trajectory.py — Coagulation Cascade Trajectory Extractor
==============================================================
Extracts coagulation protein time series from PAMPer and PRECISE,
aligns them into a unified dataset for the intervention-conditioned
neural ODE.

PAMPer:  195 patients, tp0/tp24/tp72, plasma vs saline RCT
PRECISE: ~42 patients, tp0/tp24, observational multi-site

Output:
    coag_data/pamper_coag.csv     — PAMPer coagulation trajectories
    coag_data/precise_coag.csv    — PRECISE coagulation trajectories
    coag_data/protein_map.json    — PAMPer <-> PRECISE protein mapping
    coag_data/summary.json        — dataset summary statistics

Usage:
    python3 coag_trajectory.py ./
"""

import os, sys, json, re
import numpy as np
import pandas as pd

DATA_DIR   = "./"
OUTPUT_DIR = "coag_data/"

# ── Core coagulation proteins (exact match against PAMPer names) ─────────────
COAG_PROTEINS = [
    "coagulation factor v",
    "coagulation factor vii",
    "coagulation factor viii",
    "coagulation factor ix",
    "coagulation factor ixab",
    "coagulation factor x",
    "coagulation factor xa",
    "coagulation factor xi",
    "coagulation factor xiii",
    "coagulation factor xiii b chain",
    "fibrinogen",
    "fibrinogen beta chain",
    "fibrinogen gamma chain",
    "fibrinogen-like protein 1",
    "fibrinogen c domain-containing protein 1",
    "plasminogen",
    "plasminogen activator inhibitor 1",
    "tissue-type plasminogen activator",
    "urokinase-type plasminogen activator",
    "urokinase plasminogen activator surface receptor",
    "thrombin",
    "prothrombin",
    "antithrombin-iii",
    "activated protein c",
    "activated protein c.1",
    "vitamin k-dependent protein c",
    "vitamin k-dependent protein s",
    "soluble endothelial protein c receptor",
    "von willebrand factor",
    "tissue factor",
    "tissue factor pathway inhibitor",
    "tissue factor pathway inhibitor 2",
    "mannose-binding protein c",
]

# PRECISE equivalents by UniProt ID (confirmed from panel inspection)
# PAMPer name -> PRECISE gene symbol / target
PAMPER_TO_PRECISE = {
    "coagulation factor v":               "F5",
    "coagulation factor vii":             "F7",
    "coagulation factor viii":            "Coagulation Factor VIII",
    "coagulation factor ix":              "F9",
    "coagulation factor x":               "F10",
    "coagulation factor xi":              "F11",
    "coagulation factor xiii":            "Coagulation factor XIII",
    "coagulation factor xiii b chain":    "coagulation factor XIII B",
    "fibrinogen":                         "Fibrinogen",
    "fibrinogen beta chain":              "Fibrinogen B",
    "fibrinogen gamma chain":             "Fibrinogen g-chain dimer",
    "plasminogen":                        "Plasminogen",
    "plasminogen activator inhibitor 1":  "PAI-1",
    "thrombin":                           "Thrombin",
    "prothrombin":                        "Prothrombin",
    "antithrombin-iii":                   "Antithrombin III",
    "activated protein c":                "Activated Protein C",
    "vitamin k-dependent protein c":      "Protein C",
    "vitamin k-dependent protein s":      "Protein S",
    "von willebrand factor":              "vWF",
    "tissue factor":                      "TF",
    "tissue factor pathway inhibitor":    "TFPI",
}


# ── Preprocessing ─────────────────────────────────────────────────────────────

def log1p_clip(x):
    """Same preprocessing as dataset.py — clip(0) then log1p."""
    return np.log1p(np.clip(x, 0, None))


# ── PAMPer extraction ─────────────────────────────────────────────────────────

def extract_pamper(data_dir):
    print("\n" + "="*60)
    print("EXTRACTING PAMPer COAGULATION TRAJECTORIES")
    print("="*60)

    soma = pd.read_excel(
        os.path.join(data_dir, "PAMPer_External_data.xlsx"),
        sheet_name="SomaLogic")
    patients = pd.read_excel(
        os.path.join(data_dir, "PAMPer_External_data.xlsx"),
        sheet_name="Patients")

    # Build patient metadata
    pat_meta = patients.set_index("Patients")[
        ["30_Mortality", "Intervention_arm"]].to_dict(orient="index")

    # Find coagulation columns (exact match)
    all_proteins = soma.columns[2:].tolist()
    protein_lower = {p.lower().strip(): p for p in all_proteins}

    found_proteins = {}
    for coag_name in COAG_PROTEINS:
        if coag_name in protein_lower:
            found_proteins[coag_name] = protein_lower[coag_name]

    print(f"  Coagulation proteins found: {len(found_proteins)}/{len(COAG_PROTEINS)}")
    for k, v in found_proteins.items():
        print(f"    {k}")

    # Extract trajectories
    records = []
    for _, row in soma.iterrows():
        pid = row["Patient"]
        tp  = int(row["Time point"])
        if pid not in pat_meta:
            continue
        meta    = pat_meta[pid]
        arm     = 1 if "plasma" in str(meta["Intervention_arm"]).lower() else 0
        outcome = int(meta["30_Mortality"])

        rec = {
            "patient_id":       pid,
            "timepoint":        tp,
            "intervention":     arm,   # 1=plasma, 0=saline/control
            "mortality":        outcome,
            "cohort":           "PAMPer",
        }
        for coag_name, col in found_proteins.items():
            val = row[col]
            rec[coag_name] = log1p_clip(float(val)) if pd.notna(val) else np.nan

        records.append(rec)

    df = pd.DataFrame(records)
    print(f"\n  Total records: {len(df)}")
    print(f"  Patients: {df['patient_id'].nunique()}")
    print(f"  Timepoints: {sorted(df['timepoint'].unique())}")
    print(f"  Plasma arm: {(df['intervention']==1).sum()//3} patients")
    print(f"  Control arm: {(df['intervention']==0).sum()//3} patients")
    print(f"  Deaths: {df[df['timepoint']==0]['mortality'].sum()}")

    return df, list(found_proteins.keys())


# ── PRECISE extraction ────────────────────────────────────────────────────────

def extract_precise(data_dir, pamper_coag_proteins):
    print("\n" + "="*60)
    print("EXTRACTING PRECISE COAGULATION TRAJECTORIES")
    print("="*60)

    # Find all PRECISE run files
    run_files = [f for f in os.listdir(data_dir)
                 if "dragen-protein-quant" in f and f.endswith(".csv")]
    run_files.sort()
    print(f"  Run files found: {run_files}")

    all_records = []
    protein_map = {}  # PAMPer name -> PRECISE column name

    for run_file in run_files:
        print(f"\n  Processing {run_file}...")
        df_raw = pd.read_csv(
            os.path.join(data_dir, run_file),
            header=None, low_memory=False)

        # Extract metadata rows
        targets  = df_raw.iloc[3,  1:].tolist()   # protein target names
        uniprot  = df_raw.iloc[7,  1:].tolist()   # UniProt IDs
        genes    = df_raw.iloc[9,  1:].tolist()   # gene symbols
        col_indices = list(range(1, df_raw.shape[1]))

        # Build PRECISE protein lookup: target_lower -> col_index
        precise_target_lower = {}
        for idx, (t, u, g) in enumerate(zip(targets, uniprot, genes)):
            if str(t) != 'nan':
                precise_target_lower[str(t).lower().strip()] = idx
                precise_target_lower[str(g).lower().strip()] = idx

        # Map PAMPer coag proteins to PRECISE columns
        if not protein_map:
            for pamper_name in pamper_coag_proteins:
                # Try direct name match
                if pamper_name in precise_target_lower:
                    protein_map[pamper_name] = precise_target_lower[pamper_name]
                    continue
                # Try PRECISE equivalent name
                precise_equiv = PAMPER_TO_PRECISE.get(pamper_name, "")
                if precise_equiv.lower() in precise_target_lower:
                    protein_map[pamper_name] = precise_target_lower[
                        precise_equiv.lower()]
                    continue
                # Try partial match on key terms
                key_terms = pamper_name.replace("coagulation ", "").split()
                for term in key_terms:
                    if len(term) > 4:
                        matches = [k for k in precise_target_lower
                                  if term in k and len(k) < 40]
                        if matches:
                            protein_map[pamper_name] = precise_target_lower[
                                matches[0]]
                            break

            print(f"  Mapped {len(protein_map)}/{len(pamper_coag_proteins)} "
                  f"coagulation proteins to PRECISE")

        # Extract patient data (rows 24 onwards)
        patient_rows = df_raw.iloc[24:, :]

        for _, row in patient_rows.iterrows():
            sample_id = str(row.iloc[0])
            m = re.match(r'([A-Z]+)_(\d+)_(0HR|24HR)', sample_id)
            if not m:
                continue
            site, pid, tp_str = m.groups()
            tp = 0 if tp_str == "0HR" else 24
            patient_key = f"{site}_{pid}"

            rec = {
                "patient_id":   patient_key,
                "timepoint":    tp,
                "intervention": -1,  # unknown — observational
                "mortality":    -1,  # unknown — need REDCap data
                "cohort":       "PRECISE",
                "site":         site,
            }
            for pamper_name, col_idx in protein_map.items():
                try:
                    val = float(row.iloc[col_idx + 1])  # +1 for row label col
                    rec[pamper_name] = log1p_clip(val)
                except (ValueError, IndexError):
                    rec[pamper_name] = np.nan

            all_records.append(rec)

    df = pd.DataFrame(all_records)

    # Deduplicate — keep first occurrence per patient/timepoint
    df = df.drop_duplicates(subset=["patient_id", "timepoint"])

    print(f"\n  Total PRECISE records: {len(df)}")
    print(f"  Unique patients: {df['patient_id'].nunique()}")
    print(f"  Timepoints: {sorted(df['timepoint'].unique())}")
    print(f"  Sites: {df['site'].value_counts().to_dict()}")

    # Count paired patients
    paired = df.groupby("patient_id")["timepoint"].count()
    print(f"  Patients with both 0HR+24HR: {(paired==2).sum()}")

    return df, protein_map


# ── Main ──────────────────────────────────────────────────────────────────────

def main(data_dir="./"):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract PAMPer
    pamper_df, found_coag = extract_pamper(data_dir)
    pamper_df.to_csv(os.path.join(OUTPUT_DIR, "pamper_coag.csv"), index=False)
    print(f"\n  Saved: {OUTPUT_DIR}pamper_coag.csv")

    # Extract PRECISE
    precise_df, protein_map = extract_precise(data_dir, found_coag)
    precise_df.to_csv(os.path.join(OUTPUT_DIR, "precise_coag.csv"), index=False)
    print(f"  Saved: {OUTPUT_DIR}precise_coag.csv")

    # Save protein mapping
    with open(os.path.join(OUTPUT_DIR, "protein_map.json"), "w") as f:
        json.dump({
            "pamper_proteins":  found_coag,
            "protein_map":      {k: int(v) for k, v in protein_map.items()},
            "pamper_to_precise": PAMPER_TO_PRECISE,
        }, f, indent=2)

    # Summary statistics
    summary = {
        "pamper_patients":   int(pamper_df["patient_id"].nunique()),
        "pamper_proteins":   len(found_coag),
        "pamper_timepoints": sorted(pamper_df["timepoint"].unique().tolist()),
        "pamper_plasma":     int((pamper_df[pamper_df["timepoint"]==0]
                                 ["intervention"]==1).sum()),
        "pamper_control":    int((pamper_df[pamper_df["timepoint"]==0]
                                 ["intervention"]==0).sum()),
        "pamper_deaths":     int(pamper_df[pamper_df["timepoint"]==0]
                                ["mortality"].sum()),
        "precise_patients":  int(precise_df["patient_id"].nunique()),
        "precise_proteins":  len(protein_map),
        "precise_timepoints": sorted(precise_df["timepoint"].unique().tolist()),
        "precise_sites":     precise_df["site"].value_counts().to_dict(),
        "shared_proteins":   found_coag,
    }

    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("COAGULATION TRAJECTORY EXTRACTION COMPLETE")
    print("="*60)
    print(f"  PAMPer:  {summary['pamper_patients']} patients, "
          f"{summary['pamper_proteins']} proteins, "
          f"3 timepoints")
    print(f"  PRECISE: {summary['precise_patients']} patients, "
          f"{summary['precise_proteins']} proteins, "
          f"2 timepoints")
    print(f"  Shared proteins: {len(found_coag)}")
    print(f"\n  Next: python3 intervention_ode.py ./")
    print(f"  This trains the intervention-conditioned neural ODE")
    print(f"  on coagulation cascade trajectories")


if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    main(data_dir)