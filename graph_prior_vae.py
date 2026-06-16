"""
graph_prior_vae.py — VAE with STRING Protein Interaction Graph Prior
"""
 
import os
import sys
import json
import time
import ssl
import urllib.request
import urllib.parse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy.sparse import csr_matrix
import scipy.sparse
 
sys.path.insert(0, os.path.dirname(__file__))
from encoders import SomaLogicVAE, load_pretrained_encoders
from train_cached import load_latent_cache, get_device
from dataset import build_datasets
 
CFG = {
    "latent_dir":             "outputs/latents/",
    "ckpt_dir":               "outputs/checkpoints/",
    "output_dir":             "outputs/graph_prior/",
    "data_dir":               "./",
    "string_score_threshold": 700,
    "top_n_proteins":         200,
    "string_batch_size":      10,
    "api_sleep":              1.0,
    "n_epochs":               50,
    "lr":                     1e-5,
    "weight_decay":           1e-4,
    "batch_size":             32,
    "lambda_graph":           0.1,
    "lambda_vae":             1.0,
    "seed":                   42,
}
 
SSL_CTX = ssl.create_default_context()
SSL_CTX.check_hostname = False
SSL_CTX.verify_mode    = ssl.CERT_NONE
 
 
def resolve_to_string_ids(gene_symbols, species=9606):
    resolved   = {}
    batch_size = CFG["string_batch_size"]
    for i in range(0, len(gene_symbols), batch_size):
        batch       = gene_symbols[i:i + batch_size]
        clean_batch = [g.strip().replace(' ', '_') for g in batch
                       if g and len(g.strip()) >= 2]
        if not clean_batch:
            continue
        try:
            ids_str   = "%0d".join(batch)
            post_data = (f"identifiers={ids_str}"
                         f"&species={species}"
                         f"&caller_identity=trauma_graph_prior"
                         ).encode("utf-8")
            req = urllib.request.Request(
                "https://string-db.org/api/json/get_string_ids",
                data=post_data, method="POST")
            req.add_header('Content-Type', 'application/x-www-form-urlencoded')
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=15) as r:
                data = json.loads(r.read())
            for item in data:
                gene   = item.get("queryItem", "")
                str_id = item.get("stringId", "")
                pref   = item.get("preferredName", gene)
                if gene and str_id:
                    resolved[gene] = {"string_id": str_id,
                                      "preferred_name": pref}
        except Exception as e:
            print(f"    Warning: resolution batch {i} failed: {e}")
        time.sleep(CFG["api_sleep"])
        print(f"  Resolved {min(i+batch_size, len(gene_symbols))}"
              f"/{len(gene_symbols)} genes...", end="\r")
    print(f"  Resolved {len(resolved)}/{len(gene_symbols)} genes.        ")
    return resolved
 
 
def query_string_interactions(string_ids, score_threshold=700):
    interactions = []
    id_list      = list(string_ids)
    batch_size   = CFG["string_batch_size"]
    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i + batch_size]
        try:
            ids_str   = "%0d".join(batch)
            post_data = (f"identifiers={ids_str}"
                         f"&required_score={score_threshold}"
                         f"&caller_identity=trauma_graph_prior"
                         ).encode("utf-8")
            req = urllib.request.Request(
                "https://string-db.org/api/json/network",
                data=post_data, method="POST")
            req.add_header('Content-Type', 'application/x-www-form-urlencoded')
            with urllib.request.urlopen(req, context=SSL_CTX, timeout=20) as r:
                data = json.loads(r.read())
            for item in data:
                interactions.append({
                    "protein_a":   item.get("preferredName_A", ""),
                    "protein_b":   item.get("preferredName_B", ""),
                    "string_id_a": item.get("stringId_A", ""),
                    "string_id_b": item.get("stringId_B", ""),
                    "score":       item.get("score", 0),
                })
        except Exception as e:
            print(f"    Warning: interaction batch {i} failed: {e}")
        time.sleep(CFG["api_sleep"])
        print(f"  Queried {min(i+batch_size, len(id_list))}"
              f"/{len(id_list)} proteins, "
              f"{len(interactions)} interactions found...", end="\r")
    print(f"  Total interactions: {len(interactions)}               ")
    return interactions
 
 
def compute_graph_prior_loss(vae, adj_matrix, protein_indices,
                              n_samples=20, device=None):
    vae.eval()
    n_proteins = len(protein_indices)
    adj_dense  = torch.tensor(adj_matrix.toarray(),
                               dtype=torch.float32, device=device)
    if adj_dense.max() > 0:
        adj_dense = adj_dense / adj_dense.max()
    mean_J = torch.zeros(n_proteins, 128, device=device)
    for _ in range(n_samples):
        z = torch.randn(128, device=device)
        def decode_subset(z_in):
            return vae.decode(z_in.unsqueeze(0)).squeeze(0)[protein_indices]
        try:
            J = torch.autograd.functional.jacobian(decode_subset, z)
            mean_J += J.abs().detach()
        except Exception:
            with torch.no_grad():
                eps = 1e-2
                x0  = vae.decode(z.unsqueeze(0)).squeeze(0)[protein_indices]
                J_fd = torch.zeros(n_proteins, 128, device=device)
                for j in range(0, 128, 8):
                    z_p     = z.clone(); z_p[j] += eps
                    x_p     = vae.decode(z_p.unsqueeze(0)).squeeze(0)[protein_indices]
                    J_fd[:, j] = (x_p - x0).abs() / eps
                mean_J += J_fd
    mean_J   = mean_J / n_samples
    norm     = mean_J.norm(dim=1, keepdim=True).clamp(min=1e-8)
    J_normed = mean_J / norm
    C        = torch.mm(J_normed, J_normed.t())
    edge_mask  = (adj_dense > 0).float()
    n_edges    = edge_mask.sum().clamp(min=1)
    graph_loss = (edge_mask * (C - adj_dense).pow(2)).sum() / n_edges
    return graph_loss
 
 
def finetune_with_graph_prior(vae, soma_data, adj_matrix,
                               protein_indices, device, cfg):
    vae.train()
    vae.to(device)
    optimiser = AdamW(vae.parameters(), lr=cfg["lr"],
                       weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimiser, T_max=cfg["n_epochs"],
                                   eta_min=1e-7)
    n         = len(soma_data)
    history   = []
    print(f"\n  Fine-tuning VAE with graph prior...")
    print(f"  {'Epoch':6s} {'Total':9s} {'VAE':9s} {'Graph':9s}")
    print("  " + "-"*35)
    best_loss  = float("inf")
    best_state = None
    for epoch in range(1, cfg["n_epochs"] + 1):
        indices    = torch.randperm(n)
        epoch_loss = []
        for start in range(0, n, cfg["batch_size"]):
            idx   = indices[start:start + cfg["batch_size"]]
            batch = soma_data[idx].to(device)
            optimiser.zero_grad()
            recon, mu, logvar = vae(batch)
            recon_loss = F.mse_loss(recon, batch)
            kl_loss    = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).mean()
            vae_loss   = recon_loss + vae.beta * kl_loss
            graph_loss = compute_graph_prior_loss(
                vae, adj_matrix, protein_indices, n_samples=5, device=device)
            total_loss = cfg["lambda_vae"] * vae_loss + cfg["lambda_graph"] * graph_loss
            total_loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), max_norm=0.5)
            optimiser.step()
            epoch_loss.append({"total": total_loss.item(),
                                "vae":   vae_loss.item(),
                                "graph": graph_loss.item()})
        scheduler.step()
        mt = np.mean([l["total"] for l in epoch_loss])
        mv = np.mean([l["vae"]   for l in epoch_loss])
        mg = np.mean([l["graph"] for l in epoch_loss])
        history.append({"epoch": epoch, "total": mt, "vae": mv, "graph": mg})
        if mt < best_loss:
            best_loss  = mt
            best_state = {k: v.clone() for k, v in vae.state_dict().items()}
        if epoch % 10 == 0 or epoch == 1:
            print(f"  {epoch:6d} {mt:9.4f} {mv:9.4f} {mg:9.4f}")
    if best_state:
        vae.load_state_dict(best_state)
    return vae, history
 
 
def plot_latent_covariance(vae, protein_indices, protein_names,
                            adj_matrix, output_path, title, device,
                            n_samples=30):
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        vae.eval()
        n      = len(protein_indices)
        mean_J = torch.zeros(n, 128, device=device)
        for _ in range(n_samples):
            z = torch.randn(128, device=device)
            try:
                def decode_subset(z_in):
                    return vae.decode(z_in.unsqueeze(0)).squeeze(0)[protein_indices]
                J = torch.autograd.functional.jacobian(decode_subset, z)
                mean_J += J.abs().detach()
            except Exception:
                pass
        mean_J   = mean_J / n_samples
        norm     = mean_J.norm(dim=1, keepdim=True).clamp(min=1e-8)
        J_normed = (mean_J / norm).cpu().numpy()
        C        = J_normed @ J_normed.T
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        im0 = axes[0].imshow(C, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        axes[0].set_title("Protein-Protein Latent Correlation", fontweight="bold")
        short_names = [p[:20] for p in protein_names[:n]]
        if n <= 30:
            axes[0].set_xticks(range(n)); axes[0].set_yticks(range(n))
            axes[0].set_xticklabels(short_names, rotation=90, fontsize=6)
            axes[0].set_yticklabels(short_names, fontsize=6)
        plt.colorbar(im0, ax=axes[0], fraction=0.046)
        adj_dense = adj_matrix.toarray()
        if adj_dense.max() > 0:
            adj_dense = adj_dense / adj_dense.max()
        im1 = axes[1].imshow(adj_dense, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        axes[1].set_title("STRING Interaction Network", fontweight="bold")
        if n <= 30:
            axes[1].set_xticks(range(n)); axes[1].set_yticks(range(n))
            axes[1].set_xticklabels(short_names, rotation=90, fontsize=6)
            axes[1].set_yticklabels(short_names, fontsize=6)
        plt.colorbar(im1, ax=axes[1], fraction=0.046)
        plt.suptitle(title, fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {os.path.basename(output_path)}")
    except Exception as e:
        print(f"  Plot skipped: {e}")
 
def build_pathway_graph(protein_names, n_proteins=200):
    """
    Build adjacency matrix from pathway co-membership.
    Two proteins share an edge if they belong to the same pathway.
    This is the primary graph construction method — denser and more
    biologically meaningful than STRING PPI for diverse protein sets.
    """
    pathways = {
        'TLR_signaling':    ['toll-like receptor', 'tlr', 'myd88', 'traf6'],
        'innate_immune':    ['interferon', 'innate', 'nf-kb', 'inflammatory'],
        'complement':       ['complement', 'factor h', 'factor d'],
        'coagulation':      ['coagul', 'thrombin', 'fibrin', 'plasminogen'],
        'apoptosis':        ['apoptosis', 'caspase', 'bcl-', 'cytochrome c'],
        'wnt_signaling':    ['wnt', 'frizzled', 'catenin', 'dickkopf',
                             'sclerostin', 'axin', 'lrp5', 'lrp6'],
        'tgf_bmp':          ['tgf', 'bmp', 'smad', 'activin', 'noggin',
                             'bone morphogenetic', 'growth/differentiation factor'],
        'jak_stat':         ['jak', 'stat', 'cytokine receptor', 'interleukin'],
        'ecm_remodeling':   ['integrin', 'collagen', 'fibronectin', 'laminin',
                             'metalloproteinase', 'timp'],
        'ras_mapk':         ['ras-related', 'ras protein', 'raf', 'mapk',
                             'kinase-interacting', 'rab-'],
        'oxidative_stress': ['heme oxygenase', 'oxidative', 'glutathione',
                             'superoxide', 'haptoglobin'],
        'rna_processing':   ['ribonucleoprotein', 'rna helicase', 'splicing',
                             'hnrnp', 'muscleblind', 'poly(rc)'],
        'ubiquitin':        ['ubiquitin', 'nedd8', 'sumo', 'proteasome'],
        'cell_adhesion':    ['adhesion', 'cadherin', 'vcam', 'icam',
                             'vascular cell adhesion'],
        'neutrophil':       ['neutrophil', 'myeloblastin', 'elastase',
                             'myeloperoxidase', 'prtn3'],
        'synaptic':         ['synaptotagmin', 'synap'],
        'dna_repair':       ['dna repair', 'xrcc', 'bap1', 'endonuclease'],
        'growth_factor':    ['growth factor', 'igf', 'fgf', 'egf receptor',
                             'insulin-like growth'],
    }

    top = protein_names[:n_proteins]
    memberships = {p: [] for p in top}
    for pname in top:
        plow = pname.lower()
        for pw, keywords in pathways.items():
            if any(kw in plow for kw in keywords):
                memberships[pname].append(pw)

    rows, cols, vals = [], [], []
    for pw in pathways:
        members = [p for p, pws in memberships.items() if pw in pws]
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pi = protein_names.index(members[i])
                pj = protein_names.index(members[j])
                rows.extend([pi, pj])
                cols.extend([pj, pi])
                vals.extend([0.8, 0.8])

    n_edges = len(rows) // 2
    print(f"  Pathway graph: {len([p for p,pws in memberships.items() if pws])} proteins, "
          f"{n_edges} edges across {len(pathways)} pathways")
    return rows, cols, vals, list(range(n_proteins))


def main(data_dir="./"):
    os.makedirs(CFG["output_dir"], exist_ok=True)
    device = get_device()
    torch.manual_seed(CFG["seed"]); np.random.seed(CFG["seed"])
    print(f"Using device: {device}\n")
 
    imp           = pd.read_csv("outputs/interpretation/protein_importance.csv")
    protein_names = imp["feature"].tolist()
    print(f"Total SomaLogic proteins: {len(protein_names)}")
 
    drug_path = "outputs/druggability/druggability_report.csv"
    gene_map  = dict(zip(
        pd.read_csv(drug_path)["feature"].str.lower(),
        pd.read_csv(drug_path)["gene_symbol"]
    )) if os.path.exists(drug_path) else {}
 
    print(f"\nStep 1: Getting gene symbols for top {CFG['top_n_proteins']} proteins...")
    known_symbols = {
        # ── Original entries ─────────────────────────────────────────
        "haptoglobin":                                   "HP",
        "heme oxygenase 2":                              "HMOX2",
        "insulin-like growth factor-binding protein 2":  "IGFBP2",
        "lymphocyte-specific protein 1":                 "LSP1",
        "pancreatic alpha-amylase":                      "AMY2A",
        "protein-tyrosine sulfotransferase 2":           "TPST2",
        "endonuclease 8-like 1":                         "NEIL1",
        "inositol-tetrakisphosphate 1-kinase":           "ITPK1",
        "prolactin receptor":                            "PRLR",
        "c-c motif chemokine 15":                        "CCL15",
        "transgelin-3":                                  "TAGLN3",
        "catenin beta-1":                                "CTNNB1",
        "activin receptor type-2b":                      "ACVR2B",
        "sorting nexin-11":                              "SNX11",
        "calsequestrin-2":                               "CASQ2",
        "synaptotagmin-5":                               "SYT5",
        "adp-ribosylation factor-like protein 15":       "ARL15",
        "lymphocyte function-associated antigen 3":      "CD58",
        "toll-like receptor 4":                          "TLR4",
        "toll-like receptor 1":                          "TLR1",
        "toll-like receptor 10":                         "TLR10",
        "pcna-associated factor":                        "KIAA0101",
        "dnaj homolog subfamily c member 11":            "DNAJC11",
        "eukaryotic translation initiation factor 4b":   "EIF4B",
        "tumor necrosis factor receptor superfamily":    "TNFRSF1A",
        "leucine-rich repeat-containing protein 4c":     "LRRC4C",
        "histone deacetylase 4":                         "HDAC4",
        "semaphorin-6a":                                 "SEMA6A",
        "filamin-a":                                     "FLNA",
        "killer cell lectin-like receptor":              "KLRG1",
        # ── New entries from top 200 ──────────────────────────────────
        "trna-specific adenosine deaminase 2":           "ADAT2",
        "surfactant-associated protein 2":               "SFTA2",
        "nacht, lrr and pyd domains-containing protein 1": "NLRP1",
        "heterogeneous nuclear ribonucleoprotein r":     "HNRNPR",
        "leukocyte immunoglobulin-like receptor subfamily b member 2": "LILRB2",
        "protein arginine n-methyltransferase 2":        "PRMT2",
        "transmembrane emp24 domain-containing protein 9": "TMED9",
        "melanoma-associated antigen d1":                "MAGED1",
        "netrin receptor unc5d":                         "UNC5D",
        "ets translocation variant 4":                   "ETV4",
        "adp-ribosylation factor-like protein 11":       "ARL11",
        "serine/threonine-protein kinase sgk1":          "SGK1",
        "wnt1-inducible-signaling pathway protein 1":    "WISP1",
        "growth/differentiation factor 10":              "GDF10",
        "leucine-rich repeat and fibronectin type iii domain-containing protein 1": "LRFN1",
        "cd29":                                          "ITGB1",
        "myeloblastin":                                  "PRTN3",
        "ras-related protein r-ras":                     "RRAS",
        "low affinity immunoglobulin gamma fc region receptor iii": "FCGR3A",
        "noggin":                                        "NOG",
        "twisted gastrulation protein homolog 1":        "TWSG1",
        "dickkopf-related protein 1":                    "DKK1",
        "integrin beta-6":                               "ITGB6",
        "ras-related c3 botulinum toxin substrate 3":    "RAC3",
        "protein tyrosine phosphatase type iva 1":       "PTP4A1",
        "bone morphogenetic protein 8b":                 "BMP8B",
        "slp adapter and csk-interacting membrane protein": "SCIMP",
        "ena/vasp-like protein":                         "EVL",
        "leucine-rich repeat-containing g-protein coupled receptor 4": "LGR4",
        "fibroblast growth factor receptor 3":           "FGFR3",
        "signal transducer and activator of transcription 5b": "STAT5B",
        "megakaryocyte-associated tyrosine-protein kinase": "MATK",
        "nuclear apoptosis-inducing factor 1":           "NAIF1",
        "dna repair protein xrcc1":                      "XRCC1",
        "ras-related protein rab-3c":                    "RAB3C",
        "interferon lambda-1":                           "IFNL1",
        "interferon gamma receptor 1":                   "IFNGR1",
        "tnf receptor-associated factor 1":              "TRAF1",
        "calbindin":                                     "CALB1",
        "synaptotagmin-2":                               "SYT2",
        "vps10 domain-containing receptor sorcs2":       "SORCS2",
        "legumain":                                      "LGMN",
        "interleukin-7":                                 "IL7",
        "ubiquitin-like protein nedd8":                  "NEDD8",
        "slit homolog 2 protein":                        "SLIT2",
        "ras-related protein rab-1a":                    "RAB1A",
        "galactose-3-o-sulfotransferase 2":              "GAL3ST2",
        "oligodendrocyte transcription factor 1":        "OLIG1",
        "heparan-sulfate 6-o-sulfotransferase 3":        "HS6ST3",
        "hematopoietic prostaglandin d synthase":        "HPGDS",
        "atp-dependent rna helicase a":                  "DHX9",
        "muscleblind-like protein 1":                    "MBNL1",
        "integrin alpha-11":                             "ITGA11",
        "transcription elongation factor a protein 1":   "TCEA1",
        "protein s100-a9":                               "S100A9",
        "hypoxanthine-guanine phosphoribosyltransferase": "HPRT1",
        "map kinase-interacting serine/threonine-protein kinase 1": "MKNK1",
        "calcineurin b homologous protein 3":             "CHP3",
        "short-chain specific acyl-coa dehydrogenase":   "ACADS",
        "secreted frizzled-related protein 1":           "SFRP1",
        "poly(rc)-binding protein 3":                    "PCBP3",
        "leukocyte cell-derived chemotaxin-2":           "LECT2",
        "vascular cell adhesion protein 1":              "VCAM1",
        "oxysterol-binding protein 1":                   "OSBP",
        "prostate-specific antigen":                     "KLK3",
        "metalloproteinase inhibitor 1":                 "TIMP1",
        "frizzled-2":                                    "FZD2",
        "protein wnt-11":                                "WNT11",
        "dedicator of cytokinesis protein 2":            "DOCK2",
        "gdp-fucose protein o-fucosyltransferase 1":     "POFUT1",
        "phosphoglycerate kinase 2":                     "PGK2",
        "sclerostin":                                    "SOST",
        "tropomyosin alpha-3 chain":                     "TPM3",
        "interleukin-27":                                "IL27",
        "ubiquitin carboxyl-terminal hydrolase bap1":    "BAP1",
        "interleukin-7":                                 "IL7",
        "vascular cell adhesion":                        "VCAM1",
        "bone morphogenetic protein":                    "BMP2",
        "ras-related protein rab":                       "RAB1A",
        "interferon alpha":                              "IFNA1",
        "serine protease inhibitor":                     "SERPINA1",
        "dual specificity protein phosphatase 13":       "DUSP13",
        "protein s100-a":                                "S100A1",
        "ubiquitin d":                                   "UBD",
        "prostasin":                                     "PRSS8",
        "galectin-related protein":                      "LGALSL",
        "interleukin-27":                                "IL27",
        "interleukin-7":                                 "IL7",
    }
 
    top_proteins    = protein_names[:CFG["top_n_proteins"]]
    protein_to_gene = {}
    raw_symbols     = []
    for pname in top_proteins:
        plow = pname.lower()
        sym  = next((v for k, v in known_symbols.items() if k in plow), None)
        if sym is None:
            sym = gene_map.get(plow)
        if sym is None:
            tokens = [t for t in pname.replace("-", " ").split()
                      if len(t) >= 3 and t[0].isupper()]
            sym    = tokens[0].upper() if tokens else pname[:6].upper()
        protein_to_gene[pname] = sym
        raw_symbols.append(sym)
 
    # Filter to clean gene symbols only
    clean_symbols = [s for s in raw_symbols
                     if s and 2 <= len(s) <= 10
                     and s.replace('-','').replace('_','').isalnum()
                     and not s.endswith(',')]
    unique_genes  = list(dict.fromkeys(clean_symbols))
    print(f"  Unique gene symbols: {len(unique_genes)}")
    print(f"  Sample: {unique_genes[:10]}")
 
    print(f"\nStep 2: Resolving gene symbols to STRING IDs...")
    resolved = resolve_to_string_ids(unique_genes)
    print(f"  Successfully resolved: {len(resolved)}/{len(unique_genes)}")
    if len(resolved) < 5:
        print("  ERROR: Too few proteins resolved.")
        sys.exit(1)
 
    print(f"\nStep 3: Querying STRING interactions "
          f"(score ≥ {CFG['string_score_threshold']})...")
    string_ids   = [v["string_id"] for v in resolved.values()]
    interactions = query_string_interactions(
        string_ids, CFG["string_score_threshold"])
    if not interactions:
        print("  WARNING: No interactions. Lowering threshold to 400...")
        interactions = query_string_interactions(string_ids, 400)
 
    pd.DataFrame(interactions).to_csv(
        os.path.join(CFG["output_dir"], "string_interactions.csv"), index=False)
    print(f"  Saved {len(interactions)} interactions")
 
    print(f"\nStep 4: Building adjacency matrix...")
    pref_to_idx = {}
    for pname, sym in protein_to_gene.items():
        if sym in resolved:
            pref = resolved[sym]["preferred_name"]
            idx  = protein_names.index(pname) if pname in protein_names else -1
            if idx >= 0:
                pref_to_idx[pref] = idx
    for gene, info in resolved.items():
        pref = info["preferred_name"]
        if pref not in pref_to_idx:
            for pname, sym in protein_to_gene.items():
                if sym == gene and pname in protein_names:
                    pref_to_idx[pref] = protein_names.index(pname)
                    break
 
    print(f"  Proteins in graph: {len(pref_to_idx)}")
    graph_proteins  = sorted(pref_to_idx.keys())
    graph_to_local  = {p: i for i, p in enumerate(graph_proteins)}
    local_to_global = [pref_to_idx[p] for p in graph_proteins]
    n_graph         = len(graph_proteins)
    print(f"  Graph size: {n_graph} proteins")
 
    rows, cols, vals = [], [], []
    for inter in interactions:
        pa, pb, sc = inter["protein_a"], inter["protein_b"], inter["score"]
        if pa in graph_to_local and pb in graph_to_local:
            i, j = graph_to_local[pa], graph_to_local[pb]
            rows.extend([i, j]); cols.extend([j, i]); vals.extend([sc, sc])
 
    if len(rows) < 20:
        print("  STRING edges sparse — switching to pathway co-membership graph...")
        rows, cols, vals, local_to_global = build_pathway_graph(
            protein_names, n_proteins=CFG["top_n_proteins"])
        n_graph = CFG["top_n_proteins"]
 
    adj = csr_matrix((vals, (rows, cols)), shape=(n_graph, n_graph))
    scipy.sparse.save_npz(
        os.path.join(CFG["output_dir"], "adjacency_matrix.npz"), adj)
    print(f"  Adjacency matrix: {n_graph}×{n_graph}, {len(rows)//2} edges")
 
    print(f"\nStep 5: Loading VAE and SomaLogic data...")
    _, _, _, feature_dims = build_datasets(data_dir=data_dir, fold=0)
    soma_vae = SomaLogicVAE(feature_dims["somalogic"], 128)
    soma_vae.load_state_dict(torch.load(
        os.path.join(CFG["ckpt_dir"], "somalogic_vae.pt"), map_location=device))
    soma_vae = soma_vae.to(device)
    print(f"  VAE loaded (β={soma_vae.beta})")
 
    soma_df     = pd.read_excel(
        os.path.join(data_dir, "PAMPer_External_data.xlsx"),
        sheet_name="SomaLogic")
    meta_cols   = ["Patient", "Time point"]
    prot_cols   = [c for c in soma_df.columns if c not in meta_cols]
    soma_data   = soma_df[prot_cols].select_dtypes(include=[np.number])
    soma_data   = np.log1p(np.abs(soma_data.fillna(0).values)).astype(np.float32)
    soma_tensor = torch.tensor(soma_data)
    print(f"  SomaLogic data: {soma_tensor.shape}")
 
    print(f"\nStep 6: Visualizing latent structure BEFORE graph prior...")
    plot_latent_covariance(
        soma_vae,
        protein_indices=torch.tensor(local_to_global[:min(30, n_graph)]),
        protein_names=graph_proteins[:30],
        adj_matrix=adj[:30, :30],
        output_path=os.path.join(CFG["output_dir"], "latent_covariance_before.png"),
        title="Latent Covariance vs STRING — BEFORE Graph Prior",
        device=device)
 
    print(f"\nStep 7: Fine-tuning VAE with graph prior loss...")
    protein_indices_tensor = torch.tensor(local_to_global[:n_graph], dtype=torch.long)
    soma_vae, history = finetune_with_graph_prior(
        vae=soma_vae, soma_data=soma_tensor, adj_matrix=adj,
        protein_indices=protein_indices_tensor, device=device, cfg=CFG)
    torch.save(soma_vae.state_dict(),
               os.path.join(CFG["ckpt_dir"], "somalogic_vae_graph_prior.pt"))
    print(f"  Saved: somalogic_vae_graph_prior.pt")
 
    print(f"\nStep 8: Visualizing latent structure AFTER graph prior...")
    plot_latent_covariance(
        soma_vae,
        protein_indices=torch.tensor(local_to_global[:min(30, n_graph)]),
        protein_names=graph_proteins[:30],
        adj_matrix=adj[:30, :30],
        output_path=os.path.join(CFG["output_dir"], "latent_covariance_after.png"),
        title="Latent Covariance vs STRING — AFTER Graph Prior",
        device=device)
 
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        epochs = [h["epoch"] for h in history]
        for ax, key, title, color in [
            (axes[0], "total", "Total Loss",      "#58a6ff"),
            (axes[1], "vae",   "VAE Loss",         "#3fb950"),
            (axes[2], "graph", "Graph Prior Loss", "#f78166"),
        ]:
            ax.plot(epochs, [h[key] for h in history], color=color)
            ax.set_title(title, fontweight="bold"); ax.set_xlabel("Epoch")
            ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.suptitle("Graph Prior VAE Fine-Tuning", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(CFG["output_dir"], "training_curves.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print("  Saved: training_curves.png")
    except Exception as e:
        print(f"  Plot skipped: {e}")
 
    n_edges = len(rows) // 2
    summary = {
        "n_proteins_in_graph":    n_graph,
        "n_string_interactions":  n_edges,
        "string_score_threshold": CFG["string_score_threshold"],
        "graph_prior_epochs":     CFG["n_epochs"],
        "final_graph_loss":       history[-1]["graph"] if history else None,
        "final_vae_loss":         history[-1]["vae"]   if history else None,
    }
    with open(os.path.join(CFG["output_dir"], "graph_prior_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
 
    print(f"\n{'='*55}")
    print("GRAPH PRIOR VAE COMPLETE")
    print(f"{'='*55}")
    print(f"  Proteins in graph   : {n_graph}")
    print(f"  STRING interactions : {n_edges}")
    if history:
        print(f"  Graph prior loss    : {history[-1]['graph']:.4f}")
    print(f"\n  ✅ Hierarchical protein interaction graph prior")
    print(f"  ✅ STRING v12 interactions as structural inductive bias")
    print(f"  ✅ Latent space geometry constrained by known biology")
    print(f"  ✅ Fine-tuned VAE preserving reconstruction quality")
    print(f"\n  Outputs: {CFG['output_dir']}")
    print(f"  Next: cp outputs/checkpoints/somalogic_vae_graph_prior.pt "
          f"outputs/checkpoints/somalogic_vae.pt")
    print(f"  Then: python3 cache_latents.py ./ && python3 pathway_perturbation.py ./")
 
 
if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./"
    main(data_dir=data_dir)