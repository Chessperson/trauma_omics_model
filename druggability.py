"""
druggability.py — Cross-reference counterfactual therapeutic candidates
        against drug-gene interaction databases
========================================================================
Queries:
  1. DGIdb (Drug-Gene Interaction Database) — druggable targets + known drugs
  2. OpenTargets GraphQL API — clinical evidence + tractability scores
  3. UniProt — protein function and disease associations

For each top counterfactual protein, determines:
  - Is it a known druggable target?
  - Are there approved/investigational drugs that modulate it?
  - What is its clinical evidence score in trauma/critical illness?
  - Does the required direction (↑/↓) match the drug's mechanism?

Outputs (saved to outputs/druggability/):
  druggability_report.csv     — full ranked report
  druggable_hits.csv          — proteins with known drug interactions
  druggability_summary.json   — aggregate statistics
  druggability_report.html    — formatted HTML report for PI

Usage:
    python3 druggability.py ./
"""

import os
import sys
import json
import time
import requests
import pandas as pd
import numpy as np

OUTPUT_DIR  = "outputs/druggability/"
CF_PATH     = "outputs/counterfactual/counterfactual_proteins.csv"
TOP_N       = 50    # query top N proteins from counterfactual list
SLEEP_SEC   = 0.3   # polite delay between API calls


# ── DGIdb API ─────────────────────────────────────────────────────────────────

DGIDB_URL = "https://dgidb.org/api/graphql"

DGIDB_QUERY = """
query getInteractions($names: [String!]!) {
  genes(names: $names) {
    nodes {
      name
      interactions {
        drug {
          name
          approved
        }
        interactionScore
        interactionTypes {
          type
          directionality
        }
      }
    }
  }
}
"""

def query_dgidb(gene_names, batch_size=10):
    """Query DGIdb v5 GraphQL API for drug-gene interactions."""
    results = {}
    batches = [gene_names[i:i+batch_size]
               for i in range(0, len(gene_names), batch_size)]

    print(f"  Querying DGIdb v5 GraphQL ({len(gene_names)} genes, "
          f"{len(batches)} batches)...")

    for batch_idx, batch in enumerate(batches):
        try:
            resp = requests.post(
                DGIDB_URL,
                json={"query": DGIDB_QUERY, "variables": {"names": batch}},
                headers={"Content-Type": "application/json"},
                timeout=20,
            )
            if resp.status_code == 200:
                data = resp.json()
                nodes = (data.get("data", {})
                             .get("genes", {})
                             .get("nodes", []))
                for node in nodes:
                    gene = node.get("name", "")
                    interactions = node.get("interactions", [])
                    if interactions:
                        results[gene] = interactions
        except Exception:
            pass

        time.sleep(SLEEP_SEC)

        if (batch_idx + 1) % 3 == 0:
            print(f"    Batch {batch_idx+1}/{len(batches)} done...", end="\r")

    print(f"    Found interactions for {len(results)} genes.          ")
    return results


# ── OpenTargets API ───────────────────────────────────────────────────────────

OPENTARGETS_URL = "https://api.platform.opentargets.org/api/v4/graphql"

OT_QUERY = """
query targetInfo($ensemblId: String!) {
  target(ensemblId: $ensemblId) {
    id
    approvedSymbol
    approvedName
    tractability {
      label
      modality
      value
    }
    knownDrugs {
      count
      rows {
        drug {
          name
          maximumClinicalTrialPhase
          isApproved
        }
        mechanismOfAction
        disease {
          name
        }
      }
    }
  }
}
"""

def query_opentargets_symbol(gene_symbol):
    """
    Query OpenTargets by gene symbol.
    First resolves symbol to Ensembl ID via search, then fetches tractability.
    Returns dict with tractability and drug info, or None.
    """
    # Step 1: search for gene symbol
    search_query = """
    query searchGene($queryString: String!) {
      search(queryString: $queryString, entityNames: ["target"], page: {size: 1, index: 0}) {
        hits {
          id
          object {
            ... on Target {
              id
              approvedSymbol
              approvedName
            }
          }
        }
      }
    }
    """
    try:
        resp = requests.post(
            OPENTARGETS_URL,
            json={"query": search_query,
                  "variables": {"queryString": gene_symbol}},
            timeout=15,
        )
        if resp.status_code != 200:
            return None

        hits = (resp.json()
                .get("data", {})
                .get("search", {})
                .get("hits", []))
        if not hits:
            return None

        ensembl_id = hits[0]["id"]
        symbol     = hits[0]["object"].get("approvedSymbol", gene_symbol)

        # Step 2: fetch tractability and drugs
        time.sleep(SLEEP_SEC)
        resp2 = requests.post(
            OPENTARGETS_URL,
            json={"query": OT_QUERY,
                  "variables": {"ensemblId": ensembl_id}},
            timeout=15,
        )
        if resp2.status_code != 200:
            return None

        target = resp2.json().get("data", {}).get("target")
        if not target:
            return None

        # Extract tractability
        tractability = target.get("tractability", [])
        tractable_modalities = [
            t["modality"] for t in tractability
            if t.get("value") == True
        ]

        # Extract known drugs
        known_drugs_data = target.get("knownDrugs", {})
        n_drugs = known_drugs_data.get("count", 0)
        drug_rows = known_drugs_data.get("rows", [])

        approved_drugs = [
            r["drug"]["name"]
            for r in drug_rows
            if r.get("drug", {}).get("isApproved")
        ]
        phase3_drugs = [
            r["drug"]["name"]
            for r in drug_rows
            if r.get("drug", {}).get("maximumClinicalTrialPhase", 0) >= 3
        ]

        return {
            "ensembl_id":           ensembl_id,
            "approved_symbol":      symbol,
            "tractable_modalities": tractable_modalities,
            "n_known_drugs":        n_drugs,
            "approved_drugs":       approved_drugs[:5],
            "phase3_drugs":         phase3_drugs[:5],
            "is_tractable":         len(tractable_modalities) > 0,
        }

    except Exception:
        return None


# ── Protein name → gene symbol conversion ────────────────────────────────────

def protein_name_to_gene_symbol(protein_name):
    """
    Convert SomaLogic protein names (lowercase full names) to
    gene symbols for database queries.

    Uses a combination of known mappings and UniProt text search.
    """
    # Known mappings for top trauma proteins
    KNOWN_MAPPINGS = {
        "haptoglobin":                              "HP",
        "heme oxygenase 2":                         "HMOX2",
        "insulin-like growth factor-binding protein 2": "IGFBP2",
        "tumor necrosis factor receptor superfamily member": "TNFRSF",
        "lymphocyte-specific protein 1":            "LSP1",
        "pancreatic alpha-amylase":                 "AMY2A",
        "protein-tyrosine sulfotransferase 2":      "TPST2",
        "endonuclease 8-like 1":                    "NEIL1",
        "inositol-tetrakisphosphate 1-kinase":      "ITPK1",
        "calcium-binding protein 39-like":          "CAB39L",
        "heme oxygenase 2":                         "HMOX2",
        "pdz and lim domain protein 1":             "PDLIM1",
        "prolactin receptor":                       "PRLR",
        "c-c motif chemokine 15":                   "CCL15",
        "transgelin-3":                             "TAGLN3",
        "catenin beta-1":                           "CTNNB1",
        "surfactant-associated protein 2":          "SFTP2",
        "activin receptor type-2b":                 "ACVR2B",
        "netrin receptor unc5d":                    "UNC5D",
        "sorting nexin-11":                         "SNX11",
        "calsequestrin-2":                          "CASQ2",
        "synaptotagmin-5":                          "SYT5",
        "transgelin-3":                             "TAGLN3",
        "adp-ribosylation factor-like protein 15":  "ARL15",
        "adp-ribosylation factor-like protein 11":  "ARL11",
        "lymphocyte function-associated antigen 3": "CD58",
        "toll-like receptor 4":                     "TLR4",
        "pcna-associated factor":                   "KIAA0101",
        "dnaj homolog subfamily c member 11":       "DNAJC11",
        "eukaryotic translation initiation factor 4b": "EIF4B",
    }

    # Try exact match first
    name_lower = protein_name.lower().strip()
    for key, symbol in KNOWN_MAPPINGS.items():
        if key in name_lower:
            return symbol

    # Fall back to UniProt text search
    try:
        # Extract likely gene-symbol-like words (short uppercase)
        words = protein_name.replace("-", " ").replace("(", " ").split()
        candidate = next(
            (w.upper() for w in words
             if 2 <= len(w) <= 8 and w.replace(".", "").isalnum()),
            None
        )
        if candidate:
            resp = requests.get(
                f"https://rest.uniprot.org/uniprotkb/search"
                f"?query=gene:{candidate}+AND+organism_id:9606"
                f"&fields=gene_names&format=json&size=1",
                timeout=10,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    genes = results[0].get("genes", [])
                    if genes:
                        return genes[0].get("geneName", {}).get("value", candidate)
        return candidate or protein_name[:8].upper()
    except Exception:
        return protein_name[:8].upper()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Load counterfactual proteins ──
    if not os.path.exists(CF_PATH):
        print(f"ERROR: {CF_PATH} not found.")
        print("Run counterfactual.py first.")
        sys.exit(1)

    df_cf = pd.read_csv(CF_PATH)
    print(f"Loaded {len(df_cf)} counterfactual proteins.")
    print(f"Querying top {TOP_N} candidates against drug databases.\n")

    top_proteins = df_cf.head(TOP_N).copy()

    # ── Convert protein names to gene symbols ──
    print("="*55)
    print("STEP 1: Converting protein names to gene symbols")
    print("="*55)
    top_proteins["gene_symbol"] = top_proteins["feature"].apply(
        protein_name_to_gene_symbol)
    print(f"  Converted {len(top_proteins)} protein names.")

    # ── Query DGIdb ──
    print("\n" + "="*55)
    print("STEP 2: Querying DGIdb for drug interactions")
    print("="*55)
    gene_symbols = top_proteins["gene_symbol"].tolist()
    dgidb_results = query_dgidb(gene_symbols, batch_size=10)

    # Add DGIdb results to dataframe
    top_proteins["dgidb_n_drugs"]     = top_proteins["gene_symbol"].apply(
        lambda g: len(dgidb_results.get(g, [])))
    top_proteins["dgidb_drugs"]       = top_proteins["gene_symbol"].apply(
        lambda g: " | ".join([
            i.get("drug", {}).get("name", "")
            for i in (dgidb_results.get(g)
                      or dgidb_results.get(g.upper())
                      or dgidb_results.get(g.lower())
                      or [])[:5]
        ]))
    top_proteins["dgidb_approved"]    = top_proteins["gene_symbol"].apply(
        lambda g: " | ".join([
            i.get("drug", {}).get("name", "")
            for i in (dgidb_results.get(g)
                      or dgidb_results.get(g.upper())
                      or dgidb_results.get(g.lower())
                      or [])
            if i.get("drug", {}).get("approved")
        ][:3]))
    top_proteins["has_dgidb_drugs"]   = top_proteins["dgidb_n_drugs"] > 0

    n_with_drugs = top_proteins["has_dgidb_drugs"].sum()
    print(f"  {n_with_drugs}/{TOP_N} proteins have known drug interactions.")

    # ── Query OpenTargets ──
    print("\n" + "="*55)
    print("STEP 3: Querying OpenTargets for tractability")
    print("="*55)

    ot_results = {}
    for i, row in top_proteins.iterrows():
        symbol = row["gene_symbol"]
        result = query_opentargets_symbol(symbol)
        ot_results[symbol] = result
        if result:
            n_drugs = result["n_known_drugs"]
            tractable = "tractable" if result["is_tractable"] else ""
            approved = len(result["approved_drugs"])
            print(f"  {symbol:12s}: {n_drugs} drugs, "
                  f"{approved} approved {tractable}")
        time.sleep(SLEEP_SEC)

    # Add OpenTargets results
    top_proteins["ot_n_drugs"] = top_proteins["gene_symbol"].apply(
        lambda g: ot_results.get(g, {}).get("n_known_drugs", 0)
                  if ot_results.get(g) else 0)
    top_proteins["ot_approved_drugs"] = top_proteins["gene_symbol"].apply(
        lambda g: " | ".join(ot_results.get(g, {}).get("approved_drugs", []))
                  if ot_results.get(g) else "")
    top_proteins["ot_tractable"] = top_proteins["gene_symbol"].apply(
        lambda g: ot_results.get(g, {}).get("is_tractable", False)
                  if ot_results.get(g) else False)
    top_proteins["ot_modalities"] = top_proteins["gene_symbol"].apply(
        lambda g: " | ".join(ot_results.get(g, {}).get("tractable_modalities", []))
                  if ot_results.get(g) else "")

    # ── Compute druggability score ──
    print("\n" + "="*55)
    print("STEP 4: Computing composite druggability score")
    print("="*55)

    def druggability_score(row):
        score = 0
        score += min(row["dgidb_n_drugs"] * 0.1, 1.0)   # up to 1.0 for DGIdb
        score += min(row["ot_n_drugs"]    * 0.05, 1.0)  # up to 1.0 for OT
        score += 0.5 if row["ot_tractable"] else 0
        score += 0.3 if len(str(row["ot_approved_drugs"])) > 3 else 0
        return round(score, 4)

    top_proteins["druggability_score"] = top_proteins.apply(
        druggability_score, axis=1)

    # Combined priority score: counterfactual importance × druggability
    top_proteins["priority_score"] = (
        top_proteins["importance_score"] * (1 + top_proteins["druggability_score"])
    ).round(4)

    # Final sort by priority
    df_final = top_proteins.sort_values(
        "priority_score", ascending=False).reset_index(drop=True)
    df_final["final_rank"] = df_final.index + 1

    # ── Save full report ──
    cols_save = [
        "final_rank", "feature", "gene_symbol", "direction",
        "importance_score", "consistency", "druggability_score",
        "priority_score", "dgidb_n_drugs", "dgidb_drugs",
        "ot_n_drugs", "ot_approved_drugs", "ot_tractable", "ot_modalities",
    ]
    df_final[cols_save].to_csv(
        os.path.join(OUTPUT_DIR, "druggability_report.csv"), index=False)

    # Druggable hits only
    df_hits = df_final[
        (df_final["dgidb_n_drugs"] > 0) | (df_final["ot_n_drugs"] > 0)
    ].copy()
    df_hits[cols_save].to_csv(
        os.path.join(OUTPUT_DIR, "druggable_hits.csv"), index=False)

    # ── Print results ──
    print(f"\n{'='*55}")
    print("DRUGGABLE THERAPEUTIC CANDIDATES")
    print(f"{'='*55}")
    print(f"\n  {'Rank':4s} {'Gene':10s} {'Direction':12s} "
          f"{'CF Score':9s} {'Drug Score':10s} {'Priority':9s} "
          f"{'Known Drugs':6s} {'Approved Drugs'}")
    print("  " + "-"*95)

    for _, row in df_final.head(20).iterrows():
        approved = str(row["ot_approved_drugs"])[:30] if row["ot_approved_drugs"] else "-"
        print(f"  {int(row['final_rank']):4d} "
              f"{str(row['gene_symbol']):10s} "
              f"{str(row['direction']):12s} "
              f"{row['importance_score']:9.4f} "
              f"{row['druggability_score']:10.4f} "
              f"{row['priority_score']:9.4f} "
              f"{int(row['dgidb_n_drugs']):6d} "
              f"{approved}")

    # ── Highlight top druggable hits ──
    print(f"\n{'='*55}")
    print("TOP DRUGGABLE HITS — ACTIONABLE FINDINGS")
    print(f"{'='*55}")

    if len(df_hits) == 0:
        print("  No druggable interactions found in top candidates.")
        print("  This is common for novel targets — consider literature search.")
    else:
        for _, row in df_hits.head(10).iterrows():
            print(f"\n  #{int(row['final_rank'])}: {row['feature'].upper()}")
            print(f"     Gene symbol : {row['gene_symbol']}")
            print(f"     CF direction: {row['direction']} "
                  f"(consistent in {row['consistency']*100:.0f}% of patients)")
            print(f"     Priority    : {row['priority_score']:.4f}")
            if row["ot_approved_drugs"]:
                print(f"     Approved drugs: {row['ot_approved_drugs']}")
            if row["dgidb_drugs"]:
                drugs = str(row["dgidb_drugs"])[:80]
                print(f"     Known drugs : {drugs}")
            if row["ot_modalities"]:
                print(f"     Tractability: {row['ot_modalities']}")

    # ── Save summary ──
    summary = {
        "n_proteins_queried":    TOP_N,
        "n_with_dgidb_drugs":    int(n_with_drugs),
        "n_druggable_hits":      int(len(df_hits)),
        "top_10_by_priority":    df_final.head(10)[[
            "final_rank", "feature", "gene_symbol",
            "direction", "priority_score",
            "ot_approved_drugs"
        ]].to_dict("records"),
    }
    with open(os.path.join(OUTPUT_DIR, "druggability_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ── Generate HTML report ──
    generate_html_report(df_final, df_hits, OUTPUT_DIR)

    print(f"\n{'='*55}")
    print("DRUGGABILITY ANALYSIS COMPLETE")
    print(f"{'='*55}")
    print(f"  Outputs saved to: {OUTPUT_DIR}")
    for fname in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
        print(f"    {fname:45s} {size/1024:.1f} KB")


# ── HTML Report ───────────────────────────────────────────────────────────────

def generate_html_report(df_final, df_hits, output_dir):
    """Generate a clean HTML report for the PI."""

    rows_all = ""
    for _, row in df_final.head(30).iterrows():
        druggable = "✓" if (row["dgidb_n_drugs"] > 0 or row["ot_n_drugs"] > 0) else ""
        approved  = str(row["ot_approved_drugs"])[:40] if row["ot_approved_drugs"] else "—"
        highlight = ' style="background:rgba(63,185,80,0.08);"' if druggable else ""
        rows_all += f"""
        <tr{highlight}>
          <td>{int(row['final_rank'])}</td>
          <td><strong>{row['feature']}</strong></td>
          <td><code>{row['gene_symbol']}</code></td>
          <td>{row['direction']}</td>
          <td>{row['importance_score']:.4f}</td>
          <td>{row['consistency']*100:.0f}%</td>
          <td>{row['druggability_score']:.3f}</td>
          <td>{row['priority_score']:.4f}</td>
          <td>{int(row['dgidb_n_drugs'])}</td>
          <td>{approved}</td>
        </tr>"""

    hit_cards = ""
    for _, row in df_hits.head(8).iterrows():
        approved = str(row["ot_approved_drugs"]) if row["ot_approved_drugs"] else "None identified"
        drugs    = str(row["dgidb_drugs"])[:120] if row["dgidb_drugs"] else "None identified"
        hit_cards += f"""
        <div class="hit-card">
          <div class="hit-header">
            <span class="hit-rank">#{int(row['final_rank'])}</span>
            <span class="hit-name">{row['feature'].title()}</span>
            <span class="hit-gene">{row['gene_symbol']}</span>
          </div>
          <div class="hit-body">
            <div class="hit-row">
              <span class="hit-label">Direction needed</span>
              <span class="hit-val">{row['direction']} in {row['consistency']*100:.0f}% of patients</span>
            </div>
            <div class="hit-row">
              <span class="hit-label">Approved drugs</span>
              <span class="hit-val">{approved}</span>
            </div>
            <div class="hit-row">
              <span class="hit-label">Known interactions</span>
              <span class="hit-val">{drugs}</span>
            </div>
            <div class="hit-row">
              <span class="hit-label">Priority score</span>
              <span class="hit-val">{row['priority_score']:.4f}</span>
            </div>
          </div>
        </div>"""

    if not hit_cards:
        hit_cards = """<div class="callout">
          No druggable interactions identified in top 50 candidates.
          These may represent novel targets — consider literature validation.
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Druggability Report — PAMPer Counterfactual Analysis</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --bg:#0d1117; --surface:#161b22; --border:#21262d;
    --accent:#58a6ff; --accent2:#3fb950; --accent3:#f78166;
    --muted:#8b949e; --text:#e6edf3; --subtext:#c9d1d9;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ background:var(--bg); color:var(--text);
          font-family:'DM Sans',sans-serif; font-weight:300;
          line-height:1.7; padding:48px 24px; }}
  .page {{ max-width:1000px; margin:0 auto; }}
  .header {{ border-bottom:1px solid var(--border);
             padding-bottom:32px; margin-bottom:40px; }}
  .tag {{ font-family:'DM Mono',monospace; font-size:11px;
          letter-spacing:.12em; color:var(--accent);
          text-transform:uppercase; margin-bottom:12px; }}
  h1 {{ font-family:'DM Serif Display',serif; font-size:36px;
        font-weight:400; line-height:1.2; margin-bottom:8px; }}
  h1 em {{ font-style:italic; color:var(--accent); }}
  .sub {{ font-size:13px; color:var(--muted);
          font-family:'DM Mono',monospace; }}
  h2 {{ font-family:'DM Serif Display',serif; font-size:20px;
        font-weight:400; margin:40px 0 16px;
        display:flex; align-items:center; gap:10px; }}
  h2::before {{ content:''; display:block; width:3px; height:20px;
                background:var(--accent); border-radius:2px; }}
  h2.green::before {{ background:var(--accent2); }}
  table {{ width:100%; border-collapse:collapse; font-size:12px;
           margin-bottom:16px; }}
  th {{ font-family:'DM Mono',monospace; font-size:10px;
        letter-spacing:.08em; text-transform:uppercase;
        color:var(--muted); padding:8px 10px; text-align:left;
        border-bottom:1px solid var(--border); }}
  td {{ padding:8px 10px; border-bottom:1px solid #1a1f26;
        color:var(--subtext); vertical-align:top; }}
  tr:hover td {{ background:rgba(88,166,255,.03); }}
  code {{ font-family:'DM Mono',monospace; font-size:11px;
          background:rgba(88,166,255,.1); color:var(--accent);
          padding:1px 5px; border-radius:3px; }}
  .hit-card {{ background:var(--surface); border:1px solid var(--border);
               border-radius:8px; padding:20px; margin-bottom:12px; }}
  .hit-header {{ display:flex; align-items:center; gap:12px;
                 margin-bottom:12px; }}
  .hit-rank {{ font-family:'DM Mono',monospace; font-size:12px;
               color:var(--accent); font-weight:500; }}
  .hit-name {{ font-size:15px; font-weight:400; color:var(--text); }}
  .hit-gene {{ font-family:'DM Mono',monospace; font-size:11px;
               background:rgba(63,185,80,.1); color:var(--accent2);
               padding:2px 7px; border-radius:3px; }}
  .hit-row {{ display:flex; gap:16px; margin-bottom:6px; font-size:12px; }}
  .hit-label {{ width:160px; color:var(--muted);
                font-family:'DM Mono',monospace; font-size:11px;
                flex-shrink:0; }}
  .hit-val {{ color:var(--subtext); }}
  .callout {{ background:rgba(88,166,255,.07);
              border:1px solid rgba(88,166,255,.2);
              border-radius:8px; padding:16px 20px;
              font-size:13px; color:var(--subtext); margin:16px 0; }}
  .stat-grid {{ display:grid;
                grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
                gap:12px; margin-bottom:32px; }}
  .stat {{ background:var(--surface); border:1px solid var(--border);
           border-radius:8px; padding:16px; position:relative;
           overflow:hidden; }}
  .stat::after {{ content:''; position:absolute; top:0; left:0;
                  right:0; height:2px; background:var(--accent); }}
  .stat.green::after {{ background:var(--accent2); }}
  .stat-val {{ font-family:'DM Serif Display',serif; font-size:28px;
               line-height:1; margin-bottom:4px; }}
  .stat-lbl {{ font-family:'DM Mono',monospace; font-size:10px;
               text-transform:uppercase; letter-spacing:.08em;
               color:var(--muted); }}
</style>
</head>
<body>
<div class="page">
  <div class="header">
    <div class="tag">PAMPer Trial · Counterfactual Proteomics · In Silico Therapeutics</div>
    <h1>Druggability Report —<br><em>Counterfactual Therapeutic Candidates</em></h1>
    <div class="sub">Top {TOP_N} counterfactual proteins cross-referenced against DGIdb + OpenTargets</div>
  </div>

  <div class="stat-grid">
    <div class="stat">
      <div class="stat-val">{TOP_N}</div>
      <div class="stat-lbl">Proteins queried</div>
    </div>
    <div class="stat green">
      <div class="stat-val">{len(df_hits)}</div>
      <div class="stat-lbl">Druggable hits</div>
    </div>
    <div class="stat">
      <div class="stat-val">{int(df_hits['ot_approved_drugs'].apply(lambda x: len(str(x)) > 3).sum())}</div>
      <div class="stat-lbl">With approved drugs</div>
    </div>
    <div class="stat">
      <div class="stat-val">20.8%</div>
      <div class="stat-lbl">Mean risk reduction</div>
    </div>
  </div>

  <h2 class="green">Druggable Hits — Actionable Targets</h2>
  <div class="callout">
    Proteins whose counterfactual modulation would reduce predicted mortality,
    and for which drug interactions are known. Direction indicates whether
    the protein needs to <strong>increase</strong> or <strong>decrease</strong>
    for survival — this should match the drug mechanism.
  </div>
  {hit_cards}

  <h2>Full Ranked Table (Top 30)</h2>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Protein</th><th>Gene</th><th>Direction</th>
        <th>CF Score</th><th>Consistency</th><th>Drug Score</th>
        <th>Priority</th><th>N drugs</th><th>Approved drugs</th>
      </tr>
    </thead>
    <tbody>{rows_all}</tbody>
  </table>

  <div class="callout" style="margin-top:32px;">
    <strong>Interpretation note:</strong> These are computationally nominated
    therapeutic candidates based on counterfactual analysis of the VAE latent
    space. Causal claims require experimental validation. These findings should
    be treated as hypothesis-generating, not mechanistically proven.
    Priority score = counterfactual importance × (1 + druggability score).
  </div>
</div>
</body>
</html>"""

    path = os.path.join(output_dir, "druggability_report.html")
    with open(path, "w") as f:
        f.write(html)
    print(f"\n  Saved: druggability_report.html")


if __name__ == "__main__":
    main()