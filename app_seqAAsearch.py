# app.py
# Peptide-oriented clustering app for cyanobacterial metabolite databases
# Author: Ricardo M. Borges workflow draft

import io
import re
import itertools
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

try:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem, Descriptors
    from rdkit.Chem.Draw import rdMolDraw2D
    from rdkit.Chem import rdDepictor
    RDKIT_AVAILABLE = True
except Exception:
    RDKIT_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except Exception:
    NETWORKX_AVAILABLE = False


# =============================================================================
# Streamlit setup
# =============================================================================

st.set_page_config(
    page_title="NPSAA Scaffold Architecture",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 NPSAA Biosynthetic Class → Scaffold → Motif Architecture Analysis")
st.caption(
    "Detector hierárquico de peptídeos naturais: BIOSYNTHETIC_CLASS → SCAFFOLD_CLASS → CANONICAL_MOTIF → ARCHITECTURE, com features de scaffold, decoradores químicos e comparação com Morgan/Tanimoto."
)


# =============================================================================
# Helper functions
# =============================================================================

@st.cache_data
def load_table(uploaded_file):
    """Load CSV, TSV or TXT table."""
    filename = uploaded_file.name.lower()
    raw = uploaded_file.read()

    if filename.endswith(".csv"):
        return pd.read_csv(io.BytesIO(raw))

    if filename.endswith(".tsv") or filename.endswith(".txt"):
        # Try tab first, then comma fallback
        try:
            return pd.read_csv(io.BytesIO(raw), sep="\t")
        except Exception:
            return pd.read_csv(io.BytesIO(raw))

    # Generic fallback
    try:
        return pd.read_csv(io.BytesIO(raw), sep="\t")
    except Exception:
        return pd.read_csv(io.BytesIO(raw))


def normalize_columns(df):
    """Normalize likely column names without destroying original columns."""
    rename_map = {}
    for col in df.columns:
        clean = col.strip().lower().replace(" ", "_").replace("-", "_")
        if clean in ["compound_name", "name", "compound", "metabolite", "compoundname"]:
            rename_map[col] = "compound_name"
        elif clean in ["smiles", "canonical_smiles", "structure_smiles"]:
            rename_map[col] = "SMILES"
        elif clean in ["inchi", "in_chi"]:
            rename_map[col] = "Inchi"
        elif clean in ["inchi_key", "inchikey", "inchi_key_"]:
            rename_map[col] = "Inchi_key"

    df = df.rename(columns=rename_map)
    return df


def csv_bytes(df):
    """Return semicolon-separated CSV bytes for Brazilian/European Excel compatibility."""
    return df.to_csv(sep=";", index=False, encoding="utf-8-sig").encode("utf-8-sig")






def zip_results_bytes(result_tables):
    """Create a ZIP file containing all available result tables as semicolon-separated CSV files."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        manifest_rows = []
        for file_name, table in result_tables.items():
            if table is None or not isinstance(table, pd.DataFrame):
                continue
            export_table = table.drop(columns=["mol"], errors="ignore")
            csv_text = export_table.to_csv(sep=";", index=False, encoding="utf-8-sig")
            zf.writestr(file_name, csv_text)
            manifest_rows.append({
                "file_name": file_name,
                "rows": len(export_table),
                "columns": len(export_table.columns),
            })
        manifest = pd.DataFrame(manifest_rows)
        zf.writestr(
            "README_manifest.csv",
            manifest.to_csv(sep=";", index=False, encoding="utf-8-sig")
        )
    buffer.seek(0)
    return buffer.getvalue()

def safe_mol_from_smiles(smiles):
    if not RDKIT_AVAILABLE:
        return None
    if pd.isna(smiles):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        return mol
    except Exception:
        return None


def count_peptide_bonds_from_mol(mol):
    """Approximate count of amide/peptide bonds."""
    if mol is None:
        return 0
    peptide_pattern = Chem.MolFromSmarts("C(=O)N")
    return len(mol.GetSubstructMatches(peptide_pattern))


def classify_family_from_name(name):
    """Simple family extraction from compound name."""
    if pd.isna(name):
        return "Unknown"
    s = str(name).lower()

    families = [
        "microcystin", "nodularin", "aeruginosin", "cyanopeptolin", "micropeptin",
        "anabaenopeptin", "hassallidin", "jizanpeptin", "nostocyclopeptide",
        "jamaicamide", "cryptophycin", "largazole", "lyngbyabellin", "lyngbyastatin",
        "apratoxin", "symplostatin", "dolastatin", "portoamide", "oscillapeptin",
        "microginin", "microviridin", "aerucyclamide", "patellamide",
        "anacyclamide", "piricyclamide", "oscillacyclamide", "kamptornamide",
        "hassallidin", "portoamide", "tenuecyclamide", "trichamide",
        "comoramide", "nostocyclamide", "cyanobactin", "venturamide",
        "arthrospiramide", "puwainaphycin", "minutissamide", "hectochlorin",
        "kempopeptin", "tutuilamide", "microcyclamide", "laxaphycin",
        "lyngbyacyclamide", "kawaguchipeptin", "venturamide", "microspinosamide",
        "heinamide", "scytocyclamide", "tychonamide", "pahayokolide",
        "lyngbyaureidamide", "lyngbyaureidamides",
        # v21 expanded ontology families
        "dragonamide", "malevamide", "janadolide", "almiramide", "palmyramide",
        "hantupeptin", "grenadamide", "lobocyclamide", "tolytoxin", "trichormamide",
        "lyngbyazothrin", "schizotrin", "aeruginazole", "tolybyssidin",
        "microviridin", "microviridin-like"
    ]

    for fam in families:
        if fam in s:
            return fam.capitalize()
    return "Other/Unknown"


# Canonical amino acid side-chain SMARTS are intentionally approximate.
# Goal: generate a comparable residue signature, not a perfect chemical sequence.
RESIDUE_SMARTS = {
    "Gly": "NCC(=O)",
    "Ala": "N[C@H](C)C(=O)",
    "Val": "N[C@H](C(C)C)C(=O)",
    "Leu/Ile": "N[C@H](CC(C)C)C(=O)",
    "Ser": "N[C@H](CO)C(=O)",
    "Thr": "N[C@H](C(O)C)C(=O)",
    "Phe": "N[C@H](Cc1ccccc1)C(=O)",
    "Tyr": "N[C@H](Cc1ccc(O)cc1)C(=O)",
    "Trp": "N[C@H](Cc1c[nH]c2ccccc12)C(=O)",
    "Asp": "N[C@H](CC(=O)O)C(=O)",
    "Glu": "N[C@H](CCC(=O)O)C(=O)",
    "Asn": "N[C@H](CC(=O)N)C(=O)",
    "Gln": "N[C@H](CCC(=O)N)C(=O)",
    "Lys": "N[C@H](CCCCN)C(=O)",
    "Arg": "N[C@H](CCCNC(=N)N)C(=O)",
    "His": "N[C@H](Cc1c[nH]cn1)C(=O)",
    "Pro": "N1CCCC1C(=O)",
}

# Non-proteinogenic / cyanobacterial peptide motifs, approximate SMARTS.
SPECIAL_MOTIFS = {
    # Calibrated motifs. These SMARTS are still heuristic, but less brittle than
    # the first draft and better adapted to the SMILES observed in the database.
    # Ahp-like: 3-amino-6-hydroxy-piperidone motif used in micropeptins/cyanopeptolins.
    "Ahp_like": "N1C(=O)C(N)CCC(O)1",
    # Choi-like: bicyclic proline-like core observed in aeruginosins.
    "Choi_like": "N1C(C(=O))CC2C1CCCC2",
    # Adda-like: deliberately restrictive proxy for the methoxy-diene/isoprenoid region of Adda.
    # The previous broad diene pattern generated many false positives.
    # This pattern requires a methoxy substituent and an extended substituted diene chain.
    "Adda_like": "CO[C;!R][C;!R]([CH3])[C;!R]=[C;!R][C;!R]=[C;!R][C;!R]([CH3])",
    # N-methyl amide; useful as a modifier, but excluded from automatic motif mining.
    "NMe_amide": "C(=O)N(C)",
    "Guanidino": "NC(=N)N",
    "Sulfate": "OS(=O)(=O)O",
    "Halogenated": "[F,Cl,Br,I]",
    "Sugar_like": "O[C@H]1O[C@H](CO)[C@H](O)[C@@H](O)[C@H]1O",
    # Scaffold-level motifs used only as broad architecture features. These are intentionally
    # conservative proxies and should be interpreted together with family/name support.
    # RiPP/cyclamide heterocycles: multiple SMARTS variants are intentionally
    # used because aromatic/cyclized forms appear differently across SMILES exports.
    "Thiazole_like": "c1nccs1",
    "Thiazole_alt_like": "c1scnc1",
    "Thiazoline_like": "C1=NCCS1",
    "Thiazoline_alt_like": "N1CCS=C1",
    "Oxazole_like": "c1ncco1",
    "Oxazole_alt_like": "c1ocnc1",
    "Oxazoline_like": "C1=NCCO1",
    "Prenyl_like": "CC(C)=CC",
    "Polyketide_like": "C(=O)CC(=O)",
    # Broad microginin hydroxy-amino-acid / amino-hydroxy fatty acid proxy.
    # Used only as scaffold feature context; family/name evidence remains primary.
    "Ahda_Ahdo_like": "[C;!R](O)[C;!R](N)[C;!R][C;!R][C;!R][C;!R]",
    # Broad lipid tail proxy for lipopeptide spaces.
    "Lipid_tail_like": "CCCCCCCC",

    # v21 expanded NPSAA diagnostic feature proxies.
    # These are intentionally broad and should be interpreted as feature evidence,
    # not as stand-alone proof of scaffold identity. Name/ontology rules remain primary.
    "Dhb_like": "C(=O)N[C;!R]=[C;!R]",
    "Mdha_like": "C(=O)N(C)[C;!R]=[C;!R]",
    "Dhoya_like": "[C;!R](O)[C;!R]=[C;!R][C;!R][C;!R][C;!R]",
    "Dhmoya_like": "[C;!R](O)[C;!R](C)[C;!R]=[C;!R][C;!R][C;!R]",
    "Hmoya_like": "[C;!R](O)[C;!R][C;!R][C;!R][C;!R](C)",
    "Athmu_like": "N[C;!R](C)[C;!R](O)[C;!R][C;!R]",
    "Amha_like": "N[C;!R][C;!R](O)[C;!R][C;!R][C;!R]",
    "Ahoa_like": "N[C;!R][C;!R](O)[C;!R][C;!R][C;!R][C;!R]",
    "Amoa_like": "N[C;!R][C;!R](O)[C;!R][C;!R][C;!R][C;!R][C;!R]",
    "Aba_like": "N[C;!R](C)C(=O)",
    "Ada_like": "N[C;!R](C)CC(=O)",
    "Aound_like": "N[C;!R][C;!R][C;!R][C;!R][C;!R][C;!R][C;!R][C;!R]",
    "Htya_like": "N[C;!R](Cc1ccc(O)cc1)[C;!R](O)",
    "Aeap_like": "NCCCN",
    "Aeca_like": "NCCCC(=O)",
}

# =============================================================================
# Calibrated cyanopeptide structural signatures
# =============================================================================
# These rules intentionally separate high-confidence structural motifs from broad,
# family-supported rules. Broad signatures use a family/name guard to reduce false positives.

MICROPEPTIN_SIGNATURES = {
    "Ahp_core_like": {
        "tier": "high-confidence structural",
        "required": {"motif_Ahp_like": 1},
        "diagnostic_msms": "Ahp-containing micropeptin/cyanopeptolin core",
        "interpretation": "Contains an Ahp-like piperidone motif. Strong indicator of micropeptin/cyanopeptolin-type chemistry."
    },
    "Ahp-Phe_core_like": {
        "tier": "high-confidence structural",
        "required": {"motif_Ahp_like": 1, "res_Phe": 1},
        "diagnostic_msms": "[Ahp-Phe+H-H2O]+, approximately m/z 243 in reported micropeptin/cyanopeptolin workflows",
        "interpretation": "Ahp plus phenylalanine-like residue. Compatible with Phe-containing micropeptins/cyanopeptolins."
    },
    "Ahp-Phe-NMePhe_core_like": {
        "tier": "high-confidence structural",
        "required": {"motif_Ahp_like": 1, "res_Phe": 2, "motif_NMe_amide": 1},
        "diagnostic_msms": "[Ahp-Phe-N-MePhe+H-H2O]+, approximately m/z 404 in reported workflows",
        "interpretation": "Ahp + two Phe-like aromatic residues + N-methyl amide. Compatible with Ahp-Phe-NMePhe micropeptin/cyanopeptolin cores."
    },
    "BTA-Gln-Thr_like": {
        "tier": "exploratory",
        "required": {"res_Gln": 1, "res_Thr": 1},
        "any_of": {"BTA_or_short_acyl": ["CCCC(=O)", "CCCC(=O)N"]},
        "diagnostic_msms": "BTA-Gln-Thr-related ions",
        "interpretation": "Putative butyric-acid/acyl starter plus Gln-Thr motif."
    },
}

MICROCYSTIN_SIGNATURES = {
    "Adda_core_like": {
        "tier": "high-confidence structural",
        "required": {"motif_Adda_like": 1},
        "diagnostic_msms": "Adda-related diagnostic ions, e.g. m/z 135 and related fragments",
        "interpretation": "Contains an Adda-like conjugated diene motif, characteristic of microcystins/nodularins."
    },
    "Adda_Glu_like": {
        "tier": "high-confidence structural",
        "required": {"motif_Adda_like": 1, "res_Glu": 1},
        "diagnostic_msms": "Adda + Glu-compatible microcystin/nodularin evidence",
        "interpretation": "Adda-like motif plus Glu-like residue. Stronger support for microcystin/nodularin-type chemistry."
    },
    "Microcystin_or_Nodularin_name_supported": {
        "tier": "name-supported",
        "required": {"motif_Adda_like": 1, "amide_bond_count": 4},
        "family_contains": ["microcystin", "nodularin"],
        "diagnostic_msms": "Name/family-supported Adda peptide",
        "interpretation": "Adda-like peptide in a compound already classified by name as microcystin/nodularin."
    },
}

AERUGINOSIN_SIGNATURES = {
    "Choi_core_like": {
        "tier": "high-confidence structural",
        "required": {"motif_Choi_like": 1},
        "diagnostic_msms": "Choi-containing aeruginosin-type fragment",
        "interpretation": "Contains a Choi-like bicyclic residue. Strong indicator of aeruginosin-type chemistry."
    },
    "Choi_Guanidino_like": {
        "tier": "high-confidence structural",
        "required": {"motif_Choi_like": 1, "motif_Guanidino": 1},
        "diagnostic_msms": "Choi + guanidino/argininol-compatible aeruginosin motif",
        "interpretation": "Choi-like motif plus guanidino group. Strong support for aeruginosin-type chemistry."
    },
    "Aeruginosin_name_supported": {
        "tier": "name-supported",
        "required": {"motif_Choi_like": 1, "amide_bond_count": 2},
        "family_contains": ["aeruginosin"],
        "diagnostic_msms": "Name/family-supported Choi peptide",
        "interpretation": "Choi-like peptide in a compound already classified by name as aeruginosin."
    },
}

ANABAENOPEPTIN_SIGNATURES = {
    "Lys_ureido_name_supported": {
        "tier": "name-supported",
        "required": {"res_Lys": 1, "amide_bond_count": 4},
        "family_contains": ["anabaenopeptin", "ferintoic"],
        "diagnostic_msms": "Lys/ureido-compatible anabaenopeptin evidence",
        "interpretation": "Lys-containing peptide in a compound already classified as anabaenopeptin/ferintoic acid."
    },
}

MICROGININ_SIGNATURES = {
    "Microginin_name_supported": {
        "tier": "name-supported",
        "required": {"amide_bond_count": 2},
        "family_contains": ["microginin"],
        "diagnostic_msms": "Microginin-compatible peptide evidence",
        "interpretation": "Name-supported microginin-like peptide. Tyr/Phe residue SMARTS are not required because aromatic residue calls can be noisy."
    },
}

MICROVIRIDIN_SIGNATURES = {
    "Microviridin_name_supported": {
        "tier": "name-supported",
        "required": {"amide_bond_count": 6},
        "family_contains": ["microviridin"],
        "diagnostic_msms": "Microviridin-compatible highly amidated peptide evidence",
        "interpretation": "Highly amidated peptide in a compound already classified by name as microviridin."
    },
}

CYANOPEPTIDE_SIGNATURE_GROUPS = {
    "Micropeptin / Cyanopeptolin": MICROPEPTIN_SIGNATURES,
    "Microcystin / Nodularin": MICROCYSTIN_SIGNATURES,
    "Aeruginosin": AERUGINOSIN_SIGNATURES,
    "Anabaenopeptin / Ferintoic acid": ANABAENOPEPTIN_SIGNATURES,
    "Microginin": MICROGININ_SIGNATURES,
    "Microviridin": MICROVIRIDIN_SIGNATURES,
}


@st.cache_data
def compile_smarts_dict(smarts_dict):
    if not RDKIT_AVAILABLE:
        return {}
    compiled = {}
    for name, smarts in smarts_dict.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None:
            compiled[name] = patt
    return compiled


def count_substructures(mol, compiled_patterns):
    counts = {}
    if mol is None:
        return {k: 0 for k in compiled_patterns}
    for name, patt in compiled_patterns.items():
        try:
            counts[name] = len(mol.GetSubstructMatches(patt))
        except Exception:
            counts[name] = 0
    return counts


def has_smarts(mol, smarts):
    """Return True when the molecule matches a SMARTS pattern."""
    if mol is None or not smarts:
        return False
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return False
    return mol.HasSubstructMatch(patt)


def detect_cyanopeptide_signatures(row, signatures):
    """
    Detect micropeptin/cyanopeptolin signatures based on required residue/motif
    columns plus optional SMARTS checks.

    This is not MS/MS ion detection. It is a structural proxy using SMILES/RDKit.
    """
    mol = row.get("mol", None)
    hits = []

    for signature_name, spec in signatures.items():
        required = spec.get("required", {})
        ok = True

        for col, min_count in required.items():
            if int(row.get(col, 0) or 0) < int(min_count):
                ok = False
                break

        if not ok:
            continue

        # Optional name/family guard. This is used only for chemically broad
        # signatures that would otherwise create many false positives.
        family_contains = spec.get("family_contains", [])
        if family_contains:
            fam = str(row.get("family", "")).lower()
            name = str(row.get("compound_name", "")).lower()
            if not any(term.lower() in fam or term.lower() in name for term in family_contains):
                continue

        # any_of groups: at least one SMARTS inside each group must match.
        any_of = spec.get("any_of", {})
        for group_name, smarts_list in any_of.items():
            if not any(has_smarts(mol, smarts) for smarts in smarts_list):
                ok = False
                break

        if ok:
            hits.append(signature_name)

    return hits


def collect_highlight_atoms_and_bonds(mol, compiled_patterns, selected_pattern_names):
    """Collect atoms and bonds from selected SMARTS matches for RDKit highlighting."""
    atoms = set()
    bonds = set()

    if mol is None:
        return [], []

    for name in selected_pattern_names:
        patt = compiled_patterns.get(name)
        if patt is None:
            continue

        for match in mol.GetSubstructMatches(patt):
            atoms.update(match)
            for i in range(len(match)):
                for j in range(i + 1, len(match)):
                    bond = mol.GetBondBetweenAtoms(int(match[i]), int(match[j]))
                    if bond is not None:
                        bonds.add(bond.GetIdx())

    return sorted(atoms), sorted(bonds)


def draw_molecule_png(smiles, compiled_patterns, selected_pattern_names, width=900, height=650):
    """Return PNG bytes of molecule with selected substructures highlighted."""
    mol = safe_mol_from_smiles(smiles)
    if mol is None:
        return None

    rdDepictor.Compute2DCoords(mol)
    highlight_atoms, highlight_bonds = collect_highlight_atoms_and_bonds(
        mol, compiled_patterns, selected_pattern_names
    )

    drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
    options = drawer.drawOptions()
    options.addAtomIndices = False
    options.bondLineWidth = 2

    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()




def make_simplified_sequence(row, residue_cols, motif_cols):
    """
    Build a simplified residue signature.
    This is not a true ordered peptide sequence. It is a comparable peptide-like token string.
    """
    tokens = []
    for col in residue_cols:
        n = int(row.get(col, 0) or 0)
        residue = col.replace("res_", "")
        tokens.extend([residue] * n)

    special = []
    for col in motif_cols:
        n = int(row.get(col, 0) or 0)
        motif = col.replace("motif_", "")
        if n > 0:
            special.append(f"{motif}:{n}")

    if not tokens and not special:
        return ""

    base = "-".join(tokens) if tokens else "Unresolved_peptide"
    if special:
        base += " | " + "; ".join(special)
    return base


def token_counter(sequence):
    if pd.isna(sequence) or not str(sequence).strip():
        return Counter()

    text = str(sequence).replace("|", ";")
    raw_tokens = [t.strip() for t in re.split(r"[-; ,]+", text) if t.strip()]

    tokens = []
    for token in raw_tokens:
        # Convert motif annotations such as Ahp_like:1 into Ahp_like.
        if ":" in token:
            token = token.split(":", 1)[0]
        if token and token != "Unresolved_peptide":
            tokens.append(token)

    return Counter(tokens)


def sequence_jaccard(seq_a, seq_b):
    """Multiset Jaccard similarity between simplified residue-token sequences."""
    ca, cb = token_counter(seq_a), token_counter(seq_b)
    if not ca and not cb:
        return 0.0
    keys = set(ca) | set(cb)
    inter = sum(min(ca[k], cb[k]) for k in keys)
    union = sum(max(ca[k], cb[k]) for k in keys)
    return inter / union if union else 0.0


def compute_sequence_similarity(sequences):
    n = len(sequences)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0
        for j in range(i + 1, n):
            value = sequence_jaccard(sequences[i], sequences[j])
            sim[i, j] = value
            sim[j, i] = value
    return sim


def compute_morgan_similarity(smiles_list, radius=2, n_bits=2048):
    if not RDKIT_AVAILABLE:
        return None

    fps = []
    valid = []
    for smiles in smiles_list:
        mol = safe_mol_from_smiles(smiles)
        if mol is None:
            fps.append(None)
            valid.append(False)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            fps.append(fp)
            valid.append(True)

    n = len(smiles_list)
    sim = np.zeros((n, n), dtype=float)
    for i in range(n):
        sim[i, i] = 1.0 if valid[i] else 0.0
        for j in range(i + 1, n):
            if fps[i] is None or fps[j] is None:
                value = 0.0
            else:
                value = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            sim[i, j] = value
            sim[j, i] = value
    return sim


def make_heatmap(sim_matrix, labels, title):
    fig = px.imshow(
        sim_matrix,
        x=labels,
        y=labels,
        color_continuous_scale="Viridis",
        zmin=0,
        zmax=1,
        title=title,
        aspect="auto"
    )
    fig.update_layout(height=700)
    return fig


def make_dendrogram_figure(sim_matrix, labels, title):
    # Convert similarity to distance. Clip to avoid negative values.
    dist = 1 - np.clip(sim_matrix, 0, 1)
    np.fill_diagonal(dist, 0)
    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    dendro = dendrogram(Z, labels=labels, no_plot=True)

    icoord = dendro["icoord"]
    dcoord = dendro["dcoord"]
    ordered_labels = dendro["ivl"]

    fig = go.Figure()
    for xs, ys in zip(icoord, dcoord):
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False))

    tickvals = [5 + 10 * i for i in range(len(ordered_labels))]
    fig.update_layout(
        title=title,
        xaxis=dict(tickmode="array", tickvals=tickvals, ticktext=ordered_labels, tickangle=90),
        yaxis_title="Distance",
        height=700,
        margin=dict(l=40, r=20, t=60, b=220)
    )
    return fig, Z


def make_network(sim_matrix, labels, metadata_df, threshold=0.55):
    if not NETWORKX_AVAILABLE:
        return None, None

    G = nx.Graph()
    for i, label in enumerate(labels):
        family = metadata_df.iloc[i].get("family", "Unknown")
        G.add_node(label, family=family)

    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            if sim_matrix[i, j] >= threshold:
                G.add_edge(labels[i], labels[j], weight=float(sim_matrix[i, j]))

    if G.number_of_edges() == 0:
        return G, go.Figure().update_layout(title="No edges at selected threshold")

    pos = nx.spring_layout(G, seed=42, weight="weight")

    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, node_text, node_size = [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        degree = G.degree(node)
        family = G.nodes[node].get("family", "Unknown")
        node_text.append(f"{node}<br>Family: {family}<br>Degree: {degree}")
        node_size.append(8 + degree * 3)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.7),
        hoverinfo="none",
        showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=[str(x)[:20] for x in labels],
        textposition="top center",
        hovertext=node_text,
        hoverinfo="text",
        marker=dict(size=node_size, line=dict(width=1)),
        showlegend=False
    ))
    fig.update_layout(
        title=f"Similarity network, threshold ≥ {threshold:.2f}",
        height=750,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return G, fig


def matrix_to_long_table(sim_matrix, labels, metric_name):
    rows = []
    for i, j in itertools.combinations(range(len(labels)), 2):
        rows.append({
            "source": labels[i],
            "target": labels[j],
            metric_name: sim_matrix[i, j]
        })
    return pd.DataFrame(rows).sort_values(metric_name, ascending=False)


# =============================================================================
# Sidebar
# =============================================================================

with st.sidebar:
    LOGO_PATH = Path(__file__).parent / "static" / "LAABio.png"
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)
    else:
        st.caption("Logo not found: static/LAABio.png")

    st.info("by Ricardo Moreira Borges (IPPN-UFRJ; 06-2026)")
    
    st.link_button(
        "📖 Documentation / Tutorial",
        "https://github.com/RicardoMBorges/app_seqAAsearch/blob/main/README.md",
        use_container_width=True,
    )

    st.header("Input")
    uploaded_file = st.file_uploader(
        "Upload database file",
        type=["csv", "tsv", "txt"]
    )

    st.header("Peptide detection")
    min_amide_bonds = st.slider(
        "Minimum amide bonds to classify as peptide-like",
        min_value=1,
        max_value=10,
        value=3,
        step=1
    )

    use_family_name_filter = st.checkbox(
        "Also keep known peptide families by name",
        value=True
    )

    st.header("Clustering")
    max_items = st.slider(
        "Maximum compounds for plots",
        min_value=20,
        max_value=500,
        value=120,
        step=20
    )

    sequence_network_threshold = st.slider(
        "Sequence network threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.50,
        step=0.05
    )

    structural_network_threshold = st.slider(
        "Morgan/Tanimoto network threshold",
        min_value=0.1,
        max_value=1.0,
        value=0.55,
        step=0.05
    )

    morgan_radius = st.slider("Morgan radius", 1, 4, 2, 1)
    morgan_bits = st.selectbox("Morgan bits", [1024, 2048, 4096], index=1)


# =============================================================================
# Main app
# =============================================================================

if not uploaded_file:
    st.info("Upload a table containing at least compound name and SMILES columns.")
    st.stop()

if not RDKIT_AVAILABLE:
    st.error(
        "RDKit is not available in this environment. Install it with: `conda install -c conda-forge rdkit`."
    )
    st.stop()

# Load data
df = load_table(uploaded_file)
df = normalize_columns(df)

required_cols = ["SMILES"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"Missing required column(s): {missing}. The app needs a SMILES column.")
    st.stop()

if "compound_name" not in df.columns:
    df["compound_name"] = [f"compound_{i+1}" for i in range(len(df))]

st.subheader("1. Database overview")
col1, col2, col3 = st.columns(3)
col1.metric("Total rows", len(df))
col2.metric("Rows with SMILES", int(df["SMILES"].notna().sum()))
col3.metric("Unique names", int(df["compound_name"].nunique()))

with st.expander("Preview input table", expanded=False):
    st.dataframe(df.head(50), use_container_width=True)

# Molecule processing
residue_patterns = compile_smarts_dict(RESIDUE_SMARTS)
motif_patterns = compile_smarts_dict(SPECIAL_MOTIFS)

processed = df.copy()
processed["mol"] = processed["SMILES"].apply(safe_mol_from_smiles)
processed["valid_smiles"] = processed["mol"].notna()
processed["amide_bond_count"] = processed["mol"].apply(count_peptide_bonds_from_mol)
processed["family"] = processed["compound_name"].apply(classify_family_from_name)

# Basic descriptors
processed["MolWt"] = processed["mol"].apply(lambda m: Descriptors.MolWt(m) if m is not None else np.nan)
processed["NumAtoms"] = processed["mol"].apply(lambda m: m.GetNumAtoms() if m is not None else np.nan)

# Residue and motif counts
for name, patt in residue_patterns.items():
    processed[f"res_{name}"] = processed["mol"].apply(lambda m, p=patt: len(m.GetSubstructMatches(p)) if m is not None else 0)

for name, patt in motif_patterns.items():
    processed[f"motif_{name}"] = processed["mol"].apply(lambda m, p=patt: len(m.GetSubstructMatches(p)) if m is not None else 0)

ALL_CYANOPEPTIDE_SIGNATURES = {
    signature_name: {**spec, "group": group_name}
    for group_name, group_dict in CYANOPEPTIDE_SIGNATURE_GROUPS.items()
    for signature_name, spec in group_dict.items()
}

processed["cyanopeptide_signature_hits_list"] = processed.apply(
    lambda row: detect_cyanopeptide_signatures(row, ALL_CYANOPEPTIDE_SIGNATURES),
    axis=1
)

processed["cyanopeptide_signature_hits"] = processed["cyanopeptide_signature_hits_list"].apply(
    lambda hits: "; ".join(hits) if hits else ""
)

def build_signature_summary_from_list(hits):
    if not hits:
        return ""
    return "; ".join(hits)


def cyanopeptide_signature_table(row, signatures):
    hits = set(row.get("cyanopeptide_signature_hits_list", []))
    data = {}
    for signature_name in signatures:
        data[f"sig_{signature_name}"] = int(signature_name in hits)
    return pd.Series(data)


processed["cyanopeptide_signature_hits"] = processed[
    "cyanopeptide_signature_hits_list"
].apply(build_signature_summary_from_list)

signature_hit_table = processed.apply(
    lambda row: cyanopeptide_signature_table(row, ALL_CYANOPEPTIDE_SIGNATURES),
    axis=1
)

processed = pd.concat([processed, signature_hit_table], axis=1)


residue_cols = [c for c in processed.columns if c.startswith("res_")]
motif_cols = [c for c in processed.columns if c.startswith("motif_")]
signature_cols = [c for c in processed.columns if c.startswith("sig_")]

processed["detected_residue_count"] = processed[residue_cols].sum(axis=1)
processed["detected_special_motif_count"] = processed[motif_cols].sum(axis=1)

known_peptide_families = {
    "Microcystin", "Nodularin", "Aeruginosin", "Cyanopeptolin", "Micropeptin",
    "Anabaenopeptin", "Hassallidin", "Jizanpeptin", "Nostocyclopeptide", "Jamaicamide",
    "Cryptophycin", "Largazole", "Lyngbyabellin", "Lyngbyastatin", "Apratoxin",
    "Symplostatin", "Dolastatin", "Portoamide", "Oscillapeptin", "Microginin",
    "Microviridin", "Aerucyclamide", "Patellamide", "Anacyclamide", "Piricyclamide",
    "Oscillacyclamide", "Kamptornamide", "Tenuecyclamide", "Trichamide",
    "Comoramide", "Nostocyclamide", "Cyanobactin", "Venturamide",
    "Arthrospiramide", "Puwainaphycin", "Minutissamide", "Hectochlorin",
    "Kempopeptin", "Tutuilamide", "Microcyclamide", "Laxaphycin",
    "Lyngbyacyclamide", "Kawaguchipeptin", "Venturamide", "Microspinosamide",
    "Heinamide", "Scytocyclamide", "Tychonamide", "Pahayokolide",
    "Lyngbyaureidamide", "Lyngbyaureidamides",
    "Dragonamide", "Malevamide", "Janadolide", "Almiramide", "Palmyramide",
    "Hantupeptin", "Grenadamide", "Lobocyclamide", "Tolytoxin", "Trichormamide",
    "Lyngbyazothrin", "Schizotrin", "Aeruginazole", "Tolybyssidin"
}

processed["peptide_by_amide"] = processed["amide_bond_count"] >= min_amide_bonds
processed["peptide_by_name"] = processed["family"].isin(known_peptide_families)
processed["is_peptide_like"] = processed["peptide_by_amide"] | (
    processed["peptide_by_name"] if use_family_name_filter else False
)

processed["simplified_sequence"] = processed.apply(
    lambda row: make_simplified_sequence(row, residue_cols, motif_cols),
    axis=1
)
processed["AA_signature"] = processed.apply(
    lambda row: (
        row["simplified_sequence"]
        + (" | Cyanopeptide_signatures: " + row["cyanopeptide_signature_hits"] if row["cyanopeptide_signature_hits"] else "")
    ).strip(),
    axis=1
)

peptides = processed[processed["is_peptide_like"] & processed["valid_smiles"]].copy()
peptides = peptides.sort_values(["family", "amide_bond_count", "MolWt"], ascending=[True, False, False])

st.subheader("2. Automatic peptide-like detection")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Valid SMILES", int(processed["valid_smiles"].sum()))
col2.metric("Peptide-like compounds", len(peptides))
col3.metric("Known peptide families", int(peptides["family"].nunique()))
col4.metric("Median amide bonds", float(peptides["amide_bond_count"].median()) if len(peptides) else 0)

family_counts = peptides["family"].value_counts().reset_index()
family_counts.columns = ["family", "count"]
if len(family_counts):
    fig_family = px.bar(family_counts, x="family", y="count", title="Detected peptide-like compounds by family")
    fig_family.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig_family, use_container_width=True)

show_cols = [
    "compound_name", "family", "amide_bond_count", "detected_residue_count",
    "detected_special_motif_count", "cyanopeptide_signature_hits", "MolWt", "AA_signature", "SMILES"
]
st.dataframe(peptides[show_cols].head(500), use_container_width=True)

csv_processed = csv_bytes(peptides.drop(columns=["mol"], errors="ignore"))
st.download_button(
    "Download peptide-like table with simplified sequences",
    data=csv_processed,
    file_name="cyano_peptide_like_sequences.csv",
    mime="text/csv"
)

st.subheader("2b. Cyanopeptide structural signature detection")
st.caption(
    "These signatures are structural proxies inspired by diagnostic MS/MS fragments. "
    "They use SMILES/RDKit substructure matching, not direct MS/MS ion detection."
)

signature_summary_cols = [
    "compound_name", "family", "cyanopeptide_signature_hits", "AA_signature", "SMILES"
] + signature_cols

micro_hits_df = peptides.loc[
    peptides["cyanopeptide_signature_hits"].astype(str).str.len() > 0,
    signature_summary_cols
].copy()

col_sig1, col_sig2 = st.columns(2)
col_sig1.metric("Compounds with cyanopeptide signatures", len(micro_hits_df))
col_sig2.metric("Signature types searched", len(ALL_CYANOPEPTIDE_SIGNATURES))

if len(micro_hits_df):
    st.dataframe(micro_hits_df, use_container_width=True)

    # Summarize detected signatures by confidence tier.
    signature_tier_map = {
        name: spec.get("tier", "exploratory")
        for name, spec in ALL_CYANOPEPTIDE_SIGNATURES.items()
    }
    tier_rows = []
    for sig_name, tier in signature_tier_map.items():
        col = f"sig_{sig_name}"
        if col in peptides.columns:
            n_hits = int(peptides[col].sum())
            if n_hits > 0:
                tier_rows.append({
                    "signature": sig_name,
                    "tier": tier,
                    "count": n_hits,
                })
    tier_summary_df = pd.DataFrame(tier_rows)
    if len(tier_summary_df):
        st.markdown("**Detected signature confidence tiers**")
        st.dataframe(
            tier_summary_df.sort_values(["tier", "count"], ascending=[True, False]),
            use_container_width=True
        )
else:
    st.info("No cyanopeptide signatures were detected with the current SMARTS heuristics.")

st.download_button(
    "Download cyanopeptide signature hits",
    data=csv_bytes(micro_hits_df),
    file_name="cyanopeptide_signature_hits.csv",
    mime="text/csv"
)

with st.expander("Signature dictionary used for detection"):
    signature_dictionary_df = pd.DataFrame([
        {
            "signature": name,
            "tier": spec.get("tier", "exploratory"),
            "required_columns": "; ".join(f"{k}>={v}" for k, v in spec.get("required", {}).items()),
            "optional_smarts_groups": "; ".join(spec.get("any_of", {}).keys()),
            "diagnostic_msms_reference": spec.get("diagnostic_msms", ""),
            "interpretation": spec.get("interpretation", "")
        }
        for name, spec in ALL_CYANOPEPTIDE_SIGNATURES.items()
    ])
    st.dataframe(signature_dictionary_df, use_container_width=True)
    st.download_button(
        "Download signature dictionary",
        data=csv_bytes(signature_dictionary_df),
        file_name="cyanopeptide_signature_dictionary.csv",
        mime="text/csv"
    )

st.subheader("2c. Structure inspector with RDKit highlights")
if len(peptides):
    selected_compound = st.selectbox(
        "Select a compound to inspect",
        options=peptides["compound_name"].astype(str).tolist(),
        index=0
    )

    selected_row = peptides[peptides["compound_name"].astype(str) == selected_compound].iloc[0]
    selected_mol = selected_row.get("mol", None)

    all_highlight_patterns = {}
    all_highlight_patterns.update({f"res_{k}": v for k, v in residue_patterns.items()})
    all_highlight_patterns.update({f"motif_{k}": v for k, v in motif_patterns.items()})

    detected_pattern_names = [
        name for name in all_highlight_patterns
        if int(selected_row.get(name, 0) or 0) > 0
    ]

    selected_patterns = st.multiselect(
        "Substructures to highlight",
        options=detected_pattern_names,
        default=detected_pattern_names[:8]
    )

    png_bytes = draw_molecule_png(
        selected_row["SMILES"],
        all_highlight_patterns,
        selected_patterns
    )

    col_struct1, col_struct2 = st.columns([2, 1])
    with col_struct1:
        if png_bytes is not None:
            st.image(png_bytes, caption=selected_compound, use_container_width=True)
        else:
            st.warning("Could not render structure.")
    with col_struct2:
        st.markdown("**Detected AA/signature**")
        st.write(selected_row.get("AA_signature", ""))
        st.markdown("**Cyanopeptide signatures**")
        st.write(selected_row.get("cyanopeptide_signature_hits", "None"))
        st.download_button(
            "Download highlighted structure PNG",
            data=png_bytes if png_bytes is not None else b"",
            file_name=f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', selected_compound)}_highlighted.png",
            mime="image/png",
            disabled=png_bytes is None
        )

if peptides.empty:
    st.warning("No peptide-like compounds were detected with the current settings.")
    st.stop()

# Limit for clustering plots
plot_df = peptides.head(max_items).copy()
labels = plot_df["compound_name"].astype(str).tolist()


# =============================================================================
# CyanoPeptide Signature Builder
# =============================================================================

TOKEN_LABEL_MAP = {
    "Ahp_like": "Ahp",
    "Choi_like": "Choi",
    "Adda_like": "Adda",
    "NMe_amide": "NMe",
    "Sugar_like": "Sugar",
    "Halogenated": "Halogen",
    "Sulfate": "Sulfate",
    "Guanidino": "Guanidino",
    "Thiazole_like": "Thiazole",
    "Thiazole_alt_like": "Thiazole",
    "Thiazoline_like": "Thiazoline",
    "Thiazoline_alt_like": "Thiazoline",
    "Oxazole_like": "Oxazole",
    "Oxazole_alt_like": "Oxazole",
    "Oxazoline_like": "Oxazoline",
    "Prenyl_like": "Prenyl",
    "Polyketide_like": "Polyketide",
    "Ahda_Ahdo_like": "Ahda/Ahdo",
    "Lipid_tail_like": "Lipid_tail",
    "Dhb_like": "Dhb",
    "Mdha_like": "Mdha",
    "Dhoya_like": "Dhoya",
    "Dhmoya_like": "Dhmoya",
    "Hmoya_like": "Hmoya",
    "Athmu_like": "Athmu",
    "Amha_like": "Amha",
    "Ahoa_like": "Ahoa",
    "Amoa_like": "Amoa",
    "Aba_like": "Aba",
    "Ada_like": "Ada",
    "Aound_like": "Aound",
    "Htya_like": "Htya",
    "Aeap_like": "Aeap",
    "Aeca_like": "Aeca",
}


# Automatic mining should prioritize motifs that are informative for cyanopeptide chemistry.
# The full residue table is still exported, but the unsupervised Signature Builder is deliberately stricter.
INFORMATIVE_RESIDUE_TOKENS = {
    "Phe", "Tyr", "Trp", "Arg", "Lys", "Glu", "Gln"
}

# NMe is kept as a contextual modifier, but it is not an anchor by itself.
INFORMATIVE_MOTIF_TOKENS = {
    "Ahp", "Choi", "Adda", "Sugar", "Halogen", "Sulfate", "Guanidino",
    "Thiazole", "Thiazoline", "Oxazole", "Oxazoline", "Prenyl", "Polyketide",
    "Ahda/Ahdo", "Lipid_tail",
    "Dhb", "Mdha", "Dhoya", "Dhmoya", "Hmoya", "Athmu", "Amha",
    "Ahoa", "Amoa", "Aba", "Ada", "Aound", "Htya", "Aeap", "Aeca"
}

# Core anchors are the most family-informative cyanopeptide motifs.
# Secondary anchors can be useful, but are less specific when used alone.
PRIMARY_SIGNATURE_BUILDER_ANCHORS = {"Ahp", "Choi", "Adda"}
SECONDARY_SIGNATURE_BUILDER_ANCHORS = {
    "Guanidino", "Thiazole", "Thiazoline", "Oxazole", "Oxazoline", "Prenyl",
    "Polyketide", "Ahda/Ahdo", "Lipid_tail",
    "Dhb", "Mdha", "Dhoya", "Dhmoya", "Hmoya", "Athmu", "Amha",
    "Ahoa", "Amoa", "Aba", "Ada", "Aound", "Htya", "Aeap", "Aeca"
}
ALL_SIGNATURE_BUILDER_ANCHORS = (
    PRIMARY_SIGNATURE_BUILDER_ANCHORS | SECONDARY_SIGNATURE_BUILDER_ANCHORS
)

# Tokens protected from automatic prevalence-based exclusion.
# These may be common in a peptide-rich database but remain chemically informative.
PROTECTED_SIGNATURE_BUILDER_TOKENS = {
    "Ahp", "Choi", "Adda",
    "Phe", "Tyr", "Trp", "Arg", "Lys", "Glu", "Gln"
}

GENERIC_TOKEN_PREFIXES = ("Gly", "Ala")


def is_generic_signature_builder_token(token):
    """Exclude generic backbone/proxy tokens and their counted variants."""
    return any(str(token).startswith(prefix) for prefix in GENERIC_TOKEN_PREFIXES)


def strip_count_suffix(token):
    """Convert Phex2 -> Phe or Ahpx3 -> Ahp for anchor testing."""
    return re.sub(r"x\d+$", "", str(token))


def has_signature_builder_anchor(tokens, anchor_tokens=None):
    """Return True when a token combination contains at least one selected anchor."""
    anchor_tokens = set(anchor_tokens or ALL_SIGNATURE_BUILDER_ANCHORS)
    base_tokens = {strip_count_suffix(t) for t in tokens}
    return bool(base_tokens & anchor_tokens)


def extract_detected_tokens_from_row(row, residue_cols, motif_cols):
    """
    Extract compact, chemically informative tokens from residue/motif columns.

    Important:
    - This does not infer true residue order.
    - Gly/Ala and their counted variants are excluded from automatic signature mining.
    - NMe is retained only as a contextual modifier; counted NMe variants are excluded.
    - Automatic signatures are built from special cyanopeptide motifs plus selected informative residues.
    """
    tokens = []

    for col in residue_cols:
        count = int(row.get(col, 0) or 0)
        if count <= 0:
            continue
        token = col.replace("res_", "")
        if token not in INFORMATIVE_RESIDUE_TOKENS:
            continue
        if is_generic_signature_builder_token(token):
            continue
        tokens.append(token)
        if count > 1:
            counted = f"{token}x{count}"
            if not is_generic_signature_builder_token(counted):
                tokens.append(counted)

    for col in motif_cols:
        count = int(row.get(col, 0) or 0)
        if count <= 0:
            continue
        motif = col.replace("motif_", "")
        token = TOKEN_LABEL_MAP.get(motif, motif)
        if token not in INFORMATIVE_MOTIF_TOKENS:
            continue
        if is_generic_signature_builder_token(token):
            continue
        tokens.append(token)
        # NMe count is usually generic; keep only the presence/absence modifier.
        if count > 1 and token != "NMe":
            counted = f"{token}x{count}"
            if not is_generic_signature_builder_token(counted):
                tokens.append(counted)

    return sorted(set(tokens))


def build_recurring_signature_table(
    peptides_df,
    residue_cols,
    motif_cols,
    motif_sizes=(2, 3),
    min_support=3,
    max_token_prevalence=0.60,
    exclude_tokens=None,
    require_anchor=True,
    anchor_tokens=None,
    protected_tokens=None,
):
    """
    Build recurring compositional signatures from the loaded database.

    Accepted examples: Ahp-Phe, Ahp-Phe-NMe, Adda-Glu, Choi-Arg.
    Rejected when require_anchor=True: Ser-Thr, Asn-Gln, Leu/Ile-Pro.
    """
    rows = []
    motif_to_indices = {}

    token_series = peptides_df.apply(
        lambda row: extract_detected_tokens_from_row(row, residue_cols, motif_cols),
        axis=1
    )

    exclude_tokens = set(exclude_tokens or [])
    n_compounds = max(len(peptides_df), 1)
    token_counts = Counter()
    for tokens in token_series:
        token_counts.update(set(tokens))
    ubiquitous_tokens = {
        token for token, count in token_counts.items()
        if (count / n_compounds) > float(max_token_prevalence)
    }
    protected_tokens = set(protected_tokens or PROTECTED_SIGNATURE_BUILDER_TOKENS)
    protected_base_tokens = {strip_count_suffix(t) for t in protected_tokens}
    ubiquitous_tokens = {
        token for token in ubiquitous_tokens
        if strip_count_suffix(token) not in protected_base_tokens
    }
    excluded = exclude_tokens | ubiquitous_tokens

    for idx, tokens in token_series.items():
        tokens = [
            t for t in tokens
            if t and t not in excluded and not is_generic_signature_builder_token(t)
        ]
        for size in motif_sizes:
            if len(tokens) < size:
                continue
            for combo in itertools.combinations(tokens, size):
                if require_anchor and not has_signature_builder_anchor(combo, anchor_tokens=anchor_tokens):
                    continue
                motif_to_indices.setdefault(combo, []).append(idx)

    for combo, indices in motif_to_indices.items():
        unique_indices = list(dict.fromkeys(indices))
        if len(unique_indices) < min_support:
            continue

        sub = peptides_df.loc[unique_indices]
        families = (
            sub["family"]
            .dropna()
            .astype(str)
            .value_counts()
            .head(5)
            .to_dict()
        )

        rows.append({
            "auto_signature": "-".join(combo),
            "size": len(combo),
            "count": len(unique_indices),
            "families": "; ".join([f"{k}:{v}" for k, v in families.items()]),
            "compounds_preview": "; ".join(sub["compound_name"].astype(str).head(8).tolist()),
            "tokens": list(combo),
            "anchor_required": require_anchor,
            "excluded_tokens": "; ".join(sorted(excluded)),
        })

    if not rows:
        return pd.DataFrame(columns=[
            "auto_signature", "size", "count", "families", "compounds_preview",
            "tokens", "anchor_required", "excluded_tokens"
        ])

    out = pd.DataFrame(rows)
    out = out.sort_values(["count", "size", "auto_signature"], ascending=[False, True, True])
    return out.reset_index(drop=True)


def filter_by_auto_signature(peptides_df, selected_tokens, residue_cols, motif_cols):
    """Return rows containing all selected auto-signature tokens."""
    if not selected_tokens:
        return peptides_df.iloc[0:0].copy()

    row_tokens = peptides_df.apply(
        lambda row: set(extract_detected_tokens_from_row(row, residue_cols, motif_cols)),
        axis=1
    )

    selected_tokens = set(selected_tokens)
    mask = row_tokens.apply(lambda tokens: selected_tokens.issubset(tokens))
    return peptides_df.loc[mask].copy()


st.subheader("2d. CyanoPeptide Signature Builder")
st.caption(
    "Automatically discovers recurring compositional signatures from the uploaded database. "
    "The builder prioritizes special cyanopeptide anchors such as Ahp, Choi and Adda, "
    "plus selected informative residues, reducing residue-only artifacts."
)

builder_col1, builder_col2, builder_col3, builder_col4 = st.columns(4)

with builder_col1:
    auto_motif_sizes = st.multiselect(
        "Motif size",
        options=[2, 3, 4],
        default=[2, 3]
    )

with builder_col2:
    auto_min_support = st.slider(
        "Minimum number of compounds",
        min_value=2,
        max_value=50,
        value=3,
        step=1
    )

with builder_col3:
    auto_top_n = st.slider(
        "Maximum signatures to show",
        min_value=20,
        max_value=500,
        value=100,
        step=20
    )

with builder_col4:
    auto_max_prevalence = st.slider(
        "Max token prevalence",
        min_value=0.10,
        max_value=1.00,
        value=0.60,
        step=0.05,
        help="Tokens present in more than this fraction of compounds are ignored during automatic motif mining."
    )

auto_require_anchor = st.checkbox(
    "Require at least one special cyanopeptide motif in auto-built signatures",
    value=True,
    help="When enabled, automatic signatures must contain at least one selected anchor motif. This suppresses generic residue-only pairs such as Ser-Thr or Asn-Gln."
)

auto_strict_primary_anchors = st.checkbox(
    "Use strict family anchors only: Ahp, Choi and Adda",
    value=True,
    help="Recommended. When enabled, auto-built signatures must contain Ahp, Choi or Adda. Disable only for exploratory searches involving Sugar, Halogen, Sulfate or Guanidino anchors."
)

active_anchor_tokens = (
    PRIMARY_SIGNATURE_BUILDER_ANCHORS
    if auto_strict_primary_anchors
    else ALL_SIGNATURE_BUILDER_ANCHORS
)

with st.expander("Signature Builder token policy", expanded=False):
    st.markdown("""
**Primary family anchors:** Ahp, Choi, Adda  
**Secondary exploratory anchors:** Sugar, Halogen, Sulfate, Guanidino  
**Decorators excluded from module mining:** NMe, Sugar, Halogen, Sulfate  
**Protected tokens:** Ahp, Choi, Adda, Phe, Tyr, Trp, Arg, Lys, Glu, Gln  
**Informative residues used by the builder:** Phe, Tyr, Trp, Arg, Lys, Glu, Gln  
**Excluded from automatic mining:** Gly, Ala, Ser, Thr, Val, Leu/Ile, Pro, Asn, Asp, His and generic counted variants.

By default, the builder uses strict anchors: Ahp, Choi or Adda. This favors motifs such as Ahp-Phe, Ahp-Phe-NMe, Adda-Glu and Choi-Arg over less specific combinations such as Arg-Guanidino or Halogen-Tyr. The full residue table is still kept in the main peptide table. These filters affect only the automatic Signature Builder.
""")

GENERIC_SIGNATURE_BUILDER_TOKENS = set()  # prefix filtering is handled by is_generic_signature_builder_token().

auto_signature_df = build_recurring_signature_table(
    peptides,
    residue_cols,
    motif_cols,
    motif_sizes=tuple(auto_motif_sizes) if auto_motif_sizes else (2, 3),
    min_support=auto_min_support,
    max_token_prevalence=auto_max_prevalence,
    exclude_tokens=GENERIC_SIGNATURE_BUILDER_TOKENS,
    require_anchor=auto_require_anchor,
    anchor_tokens=active_anchor_tokens,
    protected_tokens=PROTECTED_SIGNATURE_BUILDER_TOKENS
)

st.caption("Active Signature Builder anchors: " + ", ".join(sorted(active_anchor_tokens)))

if len(auto_signature_df):
    st.dataframe(
        auto_signature_df.drop(columns=["tokens"]).head(auto_top_n),
        use_container_width=True
    )

    st.download_button(
        "Download auto-built recurring signatures",
        data=csv_bytes(auto_signature_df.drop(columns=["tokens"])),
        file_name="auto_built_cyanopeptide_signatures.csv",
        mime="text/csv"
    )
else:
    st.info("No recurring signatures were found with the current settings.")



# =============================================================================
# CyanoPeptide Module Discovery
# =============================================================================

MODULE_PARTNER_TOKENS = {
    # Decorators such as NMe, Sugar, Halogen and Sulfate are deliberately excluded
    # from local module discovery. They are handled later as decorators in the
    # scaffold architecture model, not as independent modules.
    "Phe", "Tyr", "Trp", "Arg", "Lys", "Glu", "Gln", "Guanidino"
}


def build_token_pattern_map(residue_patterns, motif_patterns):
    """Map compact NPPSA tokens to compiled RDKit SMARTS patterns."""
    token_map = {}

    for residue, patt in residue_patterns.items():
        if residue in INFORMATIVE_RESIDUE_TOKENS:
            token_map[residue] = patt

    motif_to_token = {
        "Ahp_like": "Ahp",
        "Choi_like": "Choi",
        "Adda_like": "Adda",
        "NMe_amide": "NMe",
        "Sugar_like": "Sugar",
        "Halogenated": "Halogen",
        "Sulfate": "Sulfate",
        "Guanidino": "Guanidino",
        "Thiazole_like": "Thiazole",
        "Oxazole_like": "Oxazole",
        "Prenyl_like": "Prenyl",
        "Polyketide_like": "Polyketide",
        "Ahda_Ahdo_like": "Ahda/Ahdo",
        "Lipid_tail_like": "Lipid_tail",
        "Dhb_like": "Dhb",
        "Mdha_like": "Mdha",
        "Dhoya_like": "Dhoya",
        "Dhmoya_like": "Dhmoya",
        "Hmoya_like": "Hmoya",
        "Athmu_like": "Athmu",
        "Amha_like": "Amha",
        "Ahoa_like": "Ahoa",
        "Amoa_like": "Amoa",
        "Aba_like": "Aba",
        "Ada_like": "Ada",
        "Aound_like": "Aound",
        "Htya_like": "Htya",
        "Aeap_like": "Aeap",
        "Aeca_like": "Aeca",
    }

    for motif, token in motif_to_token.items():
        patt = motif_patterns.get(motif)
        if patt is not None:
            token_map[token] = patt

    return token_map


def min_match_distance(mol, patt_a, patt_b):
    """
    Minimum topological atom distance between two SMARTS matches.

    This is an approximation of local structural proximity, not proof of a direct
    peptide bond between two residues.
    """
    if mol is None or patt_a is None or patt_b is None:
        return None

    matches_a = mol.GetSubstructMatches(patt_a)
    matches_b = mol.GetSubstructMatches(patt_b)

    if not matches_a or not matches_b:
        return None

    dist_matrix = Chem.GetDistanceMatrix(mol)
    best = None

    for ma in matches_a:
        atoms_a = set(ma)
        for mb in matches_b:
            atoms_b = set(mb)
            if atoms_a & atoms_b:
                d = 0
            else:
                d = min(dist_matrix[i, j] for i in atoms_a for j in atoms_b)
            if best is None or d < best:
                best = d

    return int(best) if best is not None else None


def detect_local_modules_for_row(row, token_pattern_map, anchor_tokens, partner_tokens, max_distance=6, max_module_size=3):
    """
    Detect anchor-centered local modules.

    A module is called when an anchor token such as Ahp, Choi or Adda is close to
    another informative token within a user-defined topological atom distance.
    """
    mol = row.get("mol", None)
    if mol is None:
        return []

    present_tokens = set(extract_detected_tokens_from_row(row, residue_cols, motif_cols))
    present_base_tokens = {strip_count_suffix(t) for t in present_tokens}

    anchors = sorted(set(anchor_tokens) & present_base_tokens & set(token_pattern_map))
    partners = sorted(set(partner_tokens) & present_base_tokens & set(token_pattern_map))

    modules = set()
    anchor_to_local_partners = {}

    for anchor in anchors:
        patt_anchor = token_pattern_map.get(anchor)
        local_partners = []

        for partner in partners:
            if partner == anchor:
                continue
            patt_partner = token_pattern_map.get(partner)
            distance = min_match_distance(mol, patt_anchor, patt_partner)
            if distance is not None and distance <= int(max_distance):
                local_partners.append(partner)
                modules.add(f"{anchor}-{partner}")

        anchor_to_local_partners[anchor] = sorted(set(local_partners))

    if max_module_size >= 3:
        for anchor, local_partners in anchor_to_local_partners.items():
            for combo in itertools.combinations(local_partners, 2):
                # Put NMe before amino-acid partner for readability.
                ordered = [anchor] + sorted(combo, key=lambda x: (x != "NMe", x))
                modules.add("-".join(ordered))

    return sorted(modules)


def build_module_tables(peptides_df, token_pattern_map, anchor_tokens, partner_tokens, max_distance=6, max_module_size=3, min_support=3):
    """Create module summary, compound-module and module co-occurrence edge tables."""
    compound_rows = []
    module_to_compounds = {}

    for idx, row in peptides_df.iterrows():
        modules = detect_local_modules_for_row(
            row,
            token_pattern_map=token_pattern_map,
            anchor_tokens=anchor_tokens,
            partner_tokens=partner_tokens,
            max_distance=max_distance,
            max_module_size=max_module_size,
        )

        if not modules:
            continue

        compound = str(row.get("compound_name", idx))
        family = str(row.get("family", "Unknown"))

        for module in modules:
            module_to_compounds.setdefault(module, []).append(idx)
            compound_rows.append({
                "compound_index": idx,
                "compound_name": compound,
                "family": family,
                "module": module,
                "SMILES": row.get("SMILES", ""),
            })

    summary_rows = []
    for module, indices in module_to_compounds.items():
        unique_indices = list(dict.fromkeys(indices))
        if len(unique_indices) < int(min_support):
            continue
        sub = peptides_df.loc[unique_indices]
        families = sub["family"].astype(str).value_counts().head(5).to_dict()
        summary_rows.append({
            "module": module,
            "size": len(module.split("-")),
            "count": len(unique_indices),
            "families": "; ".join(f"{k}:{v}" for k, v in families.items()),
            "compounds_preview": "; ".join(sub["compound_name"].astype(str).head(8).tolist()),
        })

    module_summary_df = pd.DataFrame(summary_rows)
    if len(module_summary_df):
        module_summary_df = module_summary_df.sort_values(["count", "size", "module"], ascending=[False, True, True]).reset_index(drop=True)
    else:
        module_summary_df = pd.DataFrame(columns=["module", "size", "count", "families", "compounds_preview"])

    compound_module_df = pd.DataFrame(compound_rows)
    if len(compound_module_df) and len(module_summary_df):
        valid_modules = set(module_summary_df["module"])
        compound_module_df = compound_module_df[compound_module_df["module"].isin(valid_modules)].reset_index(drop=True)
    else:
        compound_module_df = pd.DataFrame(columns=["compound_index", "compound_name", "family", "module", "SMILES"])

    edge_counter = Counter()
    if len(compound_module_df):
        for compound, sub in compound_module_df.groupby("compound_name"):
            mods = sorted(set(sub["module"]))
            for a, b in itertools.combinations(mods, 2):
                edge_counter[(a, b)] += 1

    edge_rows = [
        {"source": a, "target": b, "shared_compounds": c}
        for (a, b), c in edge_counter.items()
    ]
    module_edges_df = pd.DataFrame(edge_rows)
    if len(module_edges_df):
        module_edges_df = module_edges_df.sort_values("shared_compounds", ascending=False).reset_index(drop=True)
    else:
        module_edges_df = pd.DataFrame(columns=["source", "target", "shared_compounds"])

    return module_summary_df, compound_module_df, module_edges_df



def compact_module_tokens(modules):
    """Return sorted unique module tokens from a list of module strings."""
    tokens = set()
    for module in modules:
        for token in str(module).split("-"):
            token = token.strip()
            if token:
                tokens.add(token)
    return sorted(tokens)


def build_module_architecture_table(peptides_df, compound_module_df):
    """
    Build a compound-level MODULE_SIGNATURE table.

    MODULE_SIGNATURE is intentionally module-first and uses only the curated NPPSA
    module vocabulary detected in the module discovery step. It avoids using the
    full noisy residue SMARTS table as the primary representation.
    """
    rows = []

    module_groups = {}
    if compound_module_df is not None and len(compound_module_df):
        for idx, sub in compound_module_df.groupby("compound_index"):
            modules = sorted(set(sub["module"].astype(str)))
            module_groups[idx] = modules

    for idx, row in peptides_df.iterrows():
        modules = module_groups.get(idx, [])
        core_modules = [m for m in modules if "NMe" not in m]
        modifier_modules = [m for m in modules if "NMe" in m]
        module_tokens = compact_module_tokens(modules)

        module_signature = " | ".join(modules) if modules else ""
        core_signature = " | ".join(core_modules) if core_modules else ""
        modifier_signature = " | ".join(modifier_modules) if modifier_modules else ""

        rows.append({
            "compound_index": idx,
            "compound_name": row.get("compound_name", ""),
            "family": row.get("family", ""),
            "MODULE_signature": module_signature,
            "core_modules": core_signature,
            "modifier_modules": modifier_signature,
            "module_tokens": "-".join(module_tokens),
            "module_count": len(modules),
            "cyanopeptide_signature_hits": row.get("cyanopeptide_signature_hits", ""),
            "AA_signature_low_weight": row.get("AA_signature", ""),
            "SMILES": row.get("SMILES", ""),
        })

    return pd.DataFrame(rows)


def choose_nppsa_signature(row):
    """
    Module-first signature used for downstream similarity.

    Priority:
    1. MODULE_signature: module architecture derived from local module discovery.
    2. cyanopeptide_signature_hits: curated structural signatures.
    3. AA_signature: low-weight fallback only when no module/signature is available.
    """
    module_sig = str(row.get("MODULE_signature", "") or "").strip()
    if module_sig:
        return module_sig

    cyano_sig = str(row.get("cyanopeptide_signature_hits", "") or "").strip()
    if cyano_sig:
        return "Cyanopeptide_signatures: " + cyano_sig

    aa_sig = str(row.get("AA_signature", "") or "").strip()
    if aa_sig:
        return "LOW_WEIGHT_AA: " + aa_sig

    return "Unresolved_module_signature"

def make_module_network(module_summary_df, module_edges_df, min_edge_support=2):
    """Plot module co-occurrence network."""
    if not NETWORKX_AVAILABLE or module_summary_df.empty:
        return None

    G = nx.Graph()
    counts = dict(zip(module_summary_df["module"], module_summary_df["count"]))

    for module, count in counts.items():
        G.add_node(module, count=int(count))

    for _, row in module_edges_df.iterrows():
        if int(row["shared_compounds"]) >= int(min_edge_support):
            if row["source"] in G.nodes and row["target"] in G.nodes:
                G.add_edge(row["source"], row["target"], weight=int(row["shared_compounds"]))

    if G.number_of_edges() == 0:
        return go.Figure().update_layout(title="No module co-occurrence edges at selected support")

    pos = nx.spring_layout(G, seed=42, weight="weight")

    edge_x, edge_y = [], []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, text, hover, size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        count = G.nodes[node].get("count", 1)
        text.append(node)
        hover.append(f"{node}<br>Compounds: {count}<br>Degree: {G.degree(node)}")
        size.append(10 + min(count, 100) * 0.25 + G.degree(node) * 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=edge_x, y=edge_y,
        mode="lines",
        line=dict(width=0.7),
        hoverinfo="none",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y,
        mode="markers+text",
        text=text,
        textposition="top center",
        hovertext=hover,
        hoverinfo="text",
        marker=dict(size=size, line=dict(width=1)),
        showlegend=False,
    ))
    fig.update_layout(
        title=f"CyanoPeptide module co-occurrence network, edge support ≥ {min_edge_support}",
        height=750,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


st.subheader("2f. CyanoPeptide Module Discovery")
st.caption(
    "Infers local, anchor-centered modules by combining detected NPPSA tokens with RDKit topological proximity. "
    "This moves beyond simple co-presence and asks whether anchors such as Ahp, Choi or Adda are structurally close "
    "to informative residues or modifiers."
)

module_token_pattern_map = build_token_pattern_map(residue_patterns, motif_patterns)

module_col1, module_col2, module_col3, module_col4 = st.columns(4)

with module_col1:
    module_anchor_mode = st.radio(
        "Module anchors",
        ["Strict: Ahp/Choi/Adda", "Exploratory: include secondary anchors"],
        index=0,
        horizontal=False,
    )

with module_col2:
    module_max_distance = st.slider(
        "Maximum atom distance",
        min_value=2,
        max_value=14,
        value=6,
        step=1,
        help="Maximum topological atom distance between two matched substructures. Larger values are more permissive."
    )

with module_col3:
    module_min_support = st.slider(
        "Minimum module support",
        min_value=2,
        max_value=50,
        value=3,
        step=1,
    )

with module_col4:
    module_max_size = st.selectbox(
        "Maximum module size",
        [2, 3],
        index=1,
    )

module_anchor_tokens = (
    PRIMARY_SIGNATURE_BUILDER_ANCHORS
    if module_anchor_mode.startswith("Strict")
    else ALL_SIGNATURE_BUILDER_ANCHORS
)

module_partner_tokens = MODULE_PARTNER_TOKENS | module_anchor_tokens

module_summary_df, compound_module_df, module_edges_df = build_module_tables(
    peptides,
    token_pattern_map=module_token_pattern_map,
    anchor_tokens=module_anchor_tokens,
    partner_tokens=module_partner_tokens,
    max_distance=module_max_distance,
    max_module_size=module_max_size,
    min_support=module_min_support,
)

module_architecture_df = build_module_architecture_table(peptides, compound_module_df)

# Add module-first signatures back to peptides for searching and clustering.
module_cols_to_merge = module_architecture_df[[
    "compound_index", "MODULE_signature", "core_modules", "modifier_modules",
    "module_tokens", "module_count"
]].copy()
peptides = peptides.drop(
    columns=["MODULE_signature", "core_modules", "modifier_modules", "module_tokens", "module_count", "NPPSA_signature"],
    errors="ignore"
).merge(
    module_cols_to_merge,
    left_index=True,
    right_on="compound_index",
    how="left"
).set_index("compound_index", drop=True)

for col in ["MODULE_signature", "core_modules", "modifier_modules", "module_tokens"]:
    peptides[col] = peptides[col].fillna("")
peptides["module_count"] = peptides["module_count"].fillna(0).astype(int)
peptides["NPPSA_signature"] = peptides.apply(choose_nppsa_signature, axis=1)

st.caption("Active module anchors: " + ", ".join(sorted(module_anchor_tokens)))

if len(module_summary_df):
    st.markdown("**Module summary**")
    st.dataframe(module_summary_df.head(200), use_container_width=True)

    st.markdown("**Module architecture table**")
    architecture_show_cols = [
        "compound_name", "family", "MODULE_signature", "core_modules",
        "modifier_modules", "module_tokens", "module_count",
        "cyanopeptide_signature_hits", "SMILES"
    ]
    st.dataframe(
        module_architecture_df[architecture_show_cols].head(500),
        use_container_width=True
    )

    mod_dl1, mod_dl2, mod_dl3, mod_dl4 = st.columns(4)
    with mod_dl1:
        st.download_button(
            "Download module summary",
            data=csv_bytes(module_summary_df),
            file_name="cyanopeptide_module_summary.csv",
            mime="text/csv",
        )
    with mod_dl2:
        st.download_button(
            "Download module architecture table",
            data=csv_bytes(module_architecture_df.drop(columns=["compound_index"], errors="ignore")),
            file_name="cyanopeptide_module_architecture_table.csv",
            mime="text/csv",
        )
    with mod_dl3:
        st.download_button(
            "Download compound-module table",
            data=csv_bytes(compound_module_df),
            file_name="cyanopeptide_compound_modules.csv",
            mime="text/csv",
        )
    with mod_dl4:
        st.download_button(
            "Download module network edges",
            data=csv_bytes(module_edges_df),
            file_name="cyanopeptide_module_network_edges.csv",
            mime="text/csv",
        )

    if NETWORKX_AVAILABLE and len(module_edges_df):
        module_edge_support = st.slider(
            "Module network minimum shared compounds",
            min_value=1,
            max_value=max(2, int(module_edges_df["shared_compounds"].max())),
            value=2,
            step=1,
        )
        fig_module_network = make_module_network(
            module_summary_df,
            module_edges_df,
            min_edge_support=module_edge_support,
        )
        if fig_module_network is not None:
            st.plotly_chart(fig_module_network, use_container_width=True)
else:
    st.info("No local modules were detected with the current module settings.")



# =============================================================================
# Biosynthetic Architecture Analysis
# =============================================================================

BIOSYNTHETIC_CORE_TOKENS = {"Ahp", "Adda", "Choi"}
BIOSYNTHETIC_PARTNER_TOKENS = {"Phe", "Tyr", "Trp", "Arg", "Lys", "Glu", "Gln"}
BIOSYNTHETIC_DECORATOR_TOKENS = {"NMe", "Sugar", "Halogen", "Sulfate"}
SCAFFOLD_FEATURE_TOKENS = {
    "Thiazole", "Thiazoline", "Oxazole", "Oxazoline", "Prenyl", "Polyketide",
    "Ahda/Ahdo", "Lipid_tail",
    "Dhb", "Mdha", "Dhoya", "Dhmoya", "Hmoya", "Athmu", "Amha",
    "Ahoa", "Amoa", "Aba", "Ada", "Aound", "Htya", "Aeap", "Aeca"
}

BIOSYNTHETIC_CLASS_BY_SCAFFOLD = {
    "Ahp_peptidase_inhibitor": "NRPS",
    "Aeruginosin": "NRPS",
    "Microcystin": "NRPS_PKS",
    "Microginin": "Linear_peptide",
    "RiPP": "RiPP",
    "PKS_NRPS_cyclodepsipeptide": "NRPS_PKS",
    "PKS_NRPS_macrolactam": "NRPS_PKS",
    "PKS_NRPS_alkynyl": "NRPS_PKS",
    "Linear_PKS_NRPS": "NRPS_PKS",
    "PKS_NRPS_hybrid": "NRPS_PKS",
    "Lipopeptide": "Lipopeptide",
    "Glycolipopeptide": "Glycolipopeptide",
    "Macrocyclic_peptide": "Macrocyclic_peptide",
    "Linear_peptide": "Linear_peptide",
    "Lyngbyacyclamide_like": "RiPP",
    "Kawaguchipeptin_like": "Macrocyclic_peptide",
    "Venturamide_like": "RiPP",
    "Microspinosamide_like": "RiPP",
    "Heinamide_like": "Lipopeptide",
    "Scytocyclamide_like": "Macrocyclic_peptide",
    "Tychonamide_Pahayokolide_like": "Lipopeptide",
    "Lyngbyaureidamide_like": "Lipopeptide",
    "Unknown": "Unknown",
}


# High-confidence structural signatures are the primary evidence for scaffolds/cores.
CORE_SIGNATURE_RULES = [
    ("sig_Ahp_core_like", "Ahp"),
    ("sig_Choi_core_like", "Choi"),
    ("sig_Adda_core_like", "Adda"),
]

PARTNER_SIGNATURE_RULES = [
    ("sig_Ahp-Phe_core_like", "Phe"),
    ("sig_Ahp-Phe-NMePhe_core_like", "Phe"),
    ("sig_Choi_Guanidino_like", "Arg"),
    ("sig_Adda_Glu_like", "Glu"),
]

# Scaffold-specific partner vocabulary.
# -------------------------------------------------------------------------
# NPSAA scaffold ontology
# -------------------------------------------------------------------------
# The ontology is evaluated before detailed architecture assignment.
# Each scaffold class defines a biosynthetic space, allowed partners, features
# and a canonical interpretation vocabulary.
SCAFFOLD_ONTOLOGY = {
    "Ahp_peptidase_inhibitor": {
        "label": "Ahp-containing peptidase inhibitor",
        "level": "NRPS/non-ribosomal peptide",
        "core": ["Ahp"],
        "families": ["micropeptin", "cyanopeptolin", "oscillapeptin", "jizanpeptin"],
        "allowed_partners": {"Phe", "Tyr", "Trp", "Gln", "Lys", "Arg"},
        "feature_tokens": set(),
        "canonical_motifs": ["Ahp-Phe", "Ahp-Tyr", "Ahp-Trp", "Ahp-Gln", "Ahp-Lys", "Ahp-Arg"],
        "description": "Ahp-centered micropeptin/cyanopeptolin-like peptidase inhibitors."
    },
    "Aeruginosin": {
        "label": "Choi-containing aeruginosin",
        "level": "NRPS/non-ribosomal peptide",
        "core": ["Choi"],
        "families": ["aeruginosin"],
        "allowed_partners": {"Arg"},
        "feature_tokens": set(),
        "canonical_motifs": ["Choi-Arg", "Choi"],
        "description": "Choi + guanidino/argininol aeruginosin-like architecture."
    },
    "Microcystin": {
        "label": "Adda-containing microcystin/nodularin",
        "level": "NRPS/PKS hybrid cyclic peptide",
        "core": ["Adda"],
        "families": ["microcystin", "nodularin"],
        "allowed_partners": {"Glu"},
        "feature_tokens": set(),
        "canonical_motifs": ["Adda-Glu", "Adda"],
        "description": "Adda-containing microcystin/nodularin-like architecture."
    },
    "Microginin": {
        "label": "Microginin-like linear peptide",
        "level": "linear peptide",
        "core": [],
        "core_features": ["Ahda", "Ahdo"],
        "families": ["microginin"],
        "allowed_partners": {"Tyr", "Phe", "Leu/Ile", "Val"},
        "feature_tokens": {"Ahda/Ahdo"},
        "canonical_motifs": ["Microginin-Ahda/Ahdo", "Microginin"],
        "description": "Name-supported microginin-like peptide; the true diagnostic core is Ahda/Ahdo-like hydroxy amino acid chemistry, not Phe/Tyr alone."
    },
    "RiPP": {
        "label": "RiPP/cyclamide-like peptide",
        "level": "ribosomally synthesized and post-translationally modified peptide",
        "core": [],
        "families": ["aerucyclamide", "patellamide", "microviridin", "anacyclamide", "piricyclamide", "tenuecyclamide", "trichamide", "comoramide", "nostocyclamide", "cyanobactin", "microcyclamide"],
        "allowed_partners": set(),
        "feature_tokens": {"Thiazole", "Thiazoline", "Oxazole", "Oxazoline", "Prenyl"},
        "canonical_motifs": ["RiPP-Thiazole", "RiPP-Thiazoline", "RiPP-Oxazole", "RiPP-Oxazoline", "RiPP-Prenyl", "RiPP"],
        "description": "RiPP/cyclamide-like scaffold described by thiazole/thiazoline/oxazole/oxazoline and tailoring features."
    },
    "PKS_NRPS_cyclodepsipeptide": {
        "label": "Apratoxin-like PKS-NRPS cyclodepsipeptide",
        "level": "hybrid biosynthetic assembly",
        "core": [],
        "families": ["apratoxin"],
        "allowed_partners": set(),
        "feature_tokens": {"Polyketide"},
        "canonical_motifs": ["Apratoxin-like PKS-NRPS", "PKS-NRPS"],
        "description": "Apratoxin-like cyclic depsipeptide PKS-NRPS hybrid space."
    },
    "PKS_NRPS_macrolactam": {
        "label": "Cryptophycin-like PKS-NRPS macrolactam",
        "level": "hybrid biosynthetic assembly",
        "core": [],
        "families": ["cryptophycin"],
        "allowed_partners": set(),
        "feature_tokens": {"Polyketide"},
        "canonical_motifs": ["Cryptophycin-like PKS-NRPS", "PKS-NRPS"],
        "description": "Cryptophycin-like macrolactam/macrolide PKS-NRPS hybrid space."
    },
    "PKS_NRPS_alkynyl": {
        "label": "Jamaicamide-like PKS-NRPS alkynyl lipopeptide",
        "level": "hybrid biosynthetic assembly",
        "core": [],
        "families": ["jamaicamide"],
        "allowed_partners": set(),
        "feature_tokens": {"Polyketide"},
        "canonical_motifs": ["Jamaicamide-like PKS-NRPS", "PKS-NRPS"],
        "description": "Jamaicamide-like alkynyl/lipopeptide PKS-NRPS hybrid space."
    },
    "Linear_PKS_NRPS": {
        "label": "Dolastatin/symplostatin-like linear PKS-NRPS",
        "level": "linear hybrid peptide",
        "core": [],
        "families": ["dolastatin", "symplostatin"],
        "allowed_partners": set(),
        "feature_tokens": {"Polyketide"},
        "canonical_motifs": ["Dolastatin-like linear PKS-NRPS", "PKS-NRPS"],
        "description": "Dolastatin/symplostatin-like linear PKS-NRPS peptide space."
    },
    "PKS_NRPS_hybrid": {
        "label": "generic PKS-NRPS hybrid",
        "level": "hybrid biosynthetic assembly",
        "core": [],
        "families": ["largazole"],
        "allowed_partners": set(),
        "feature_tokens": {"Polyketide"},
        "canonical_motifs": ["PKS-NRPS"],
        "description": "Generic hybrid polyketide-peptide natural product space when a more specific subclass is unavailable."
    },
    "Lipopeptide": {
        "label": "Lipopeptide/depsipeptide",
        "level": "lipidated peptide",
        "core": [],
        "families": ["lyngbyabellin", "lyngbyastatin", "kamptornamide", "puwainaphycin", "minutissamide", "hectochlorin", "kempopeptin", "tutuilamide", "laxaphycin"],
        "allowed_partners": set(),
        "feature_tokens": {"Polyketide", "Lipid_tail"},
        "canonical_motifs": ["Lipopeptide"],
        "description": "Lipidated/depsipeptide-like cyanobacterial peptide space."
    },
    "Glycolipopeptide": {
        "label": "Glycolipopeptide",
        "level": "glycosylated lipidated peptide",
        "core": [],
        "families": ["hassallidin"],
        "allowed_partners": set(),
        "feature_tokens": {"Sugar"},
        "canonical_motifs": ["Glycolipopeptide"],
        "description": "Glycosylated lipopeptide/glycolipopeptide scaffold."
    },
    "Macrocyclic_peptide": {
        "label": "Macrocyclic peptide",
        "level": "cyclic peptide",
        "core": [],
        "families": ["nostocyclopeptide", "anabaenopeptin", "oscillacyclamide", "arthrospiramide"],
        "allowed_partners": {"Lys"},
        "feature_tokens": set(),
        "canonical_motifs": ["Macrocyclic_peptide"],
        "description": "Name-supported macrocyclic peptide scaffold."
    },
    "Linear_peptide": {
        "label": "Linear peptide",
        "level": "linear peptide",
        "core": [],
        "families": ["portoamide"],
        "allowed_partners": set(),
        "feature_tokens": set(),
        "canonical_motifs": ["Linear_peptide"],
        "description": "Name-supported linear peptide scaffold."
    },
    "Lyngbyacyclamide_like": {
        "label": "Lyngbyacyclamide-like cyclamide scaffold",
        "level": "RiPP/cyclamide-like peptide",
        "core": [],
        "families": ["lyngbyacyclamide"],
        "allowed_partners": set(),
        "feature_tokens": {"Thiazole", "Thiazoline", "Oxazole", "Oxazoline"},
        "canonical_motifs": ["Lyngbyacyclamide-like", "RiPP-Thiazole", "RiPP-Oxazole"],
        "description": "Name-supported lyngbyacyclamide/cyclamide-like scaffold; interpreted as a RiPP/cyanobactin-like heterocycle-rich peptide when heterocycles are present."
    },
    "Kawaguchipeptin_like": {
        "label": "Kawaguchipeptin-like cyclic peptide",
        "level": "macrocyclic peptide",
        "core": [],
        "families": ["kawaguchipeptin"],
        "allowed_partners": set(),
        "feature_tokens": set(),
        "canonical_motifs": ["Kawaguchipeptin-like"],
        "description": "Name-supported kawaguchipeptin-like macrocyclic peptide subclass."
    },
    "Venturamide_like": {
        "label": "Venturamide-like cyanobactin/RiPP scaffold",
        "level": "RiPP/cyanobactin-like peptide",
        "core": [],
        "families": ["venturamide"],
        "allowed_partners": set(),
        "feature_tokens": {"Thiazole", "Thiazoline", "Oxazole", "Oxazoline", "Prenyl"},
        "canonical_motifs": ["Venturamide-like", "RiPP-Thiazole", "RiPP-Oxazole", "RiPP-Prenyl"],
        "description": "Name-supported venturamide-like cyanobactin/RiPP subclass, separated from the generic RiPP bin to reduce Unknown assignments."
    },
    "Microspinosamide_like": {
        "label": "Microspinosamide-like cyanobactin/RiPP scaffold",
        "level": "RiPP/cyanobactin-like peptide",
        "core": [],
        "families": ["microspinosamide"],
        "allowed_partners": set(),
        "feature_tokens": {"Thiazole", "Thiazoline", "Oxazole", "Oxazoline", "Prenyl"},
        "canonical_motifs": ["Microspinosamide-like", "RiPP-Thiazole", "RiPP-Oxazole"],
        "description": "Name-supported microspinosamide-like cyanobactin/RiPP subclass."
    },
    "Heinamide_like": {
        "label": "Heinamide-like lipopeptide/depsipeptide scaffold",
        "level": "lipopeptide / depsipeptide",
        "core": [],
        "families": ["heinamide"],
        "allowed_partners": set(),
        "feature_tokens": {"Lipid_tail", "Polyketide"},
        "canonical_motifs": ["Heinamide-like", "Lipopeptide"],
        "description": "Name-supported heinamide-like peptide/lipopeptide subclass, separated from generic lipopeptides for ontology-level interpretation."
    },
    "Scytocyclamide_like": {
        "label": "Scytocyclamide-like cyclic peptide",
        "level": "macrocyclic peptide",
        "core": [],
        "families": ["scytocyclamide"],
        "allowed_partners": set(),
        "feature_tokens": set(),
        "canonical_motifs": ["Scytocyclamide-like"],
        "description": "Name-supported scytocyclamide-like macrocyclic peptide subclass."
    },
    "Tychonamide_Pahayokolide_like": {
        "label": "Tychonamide/Pahayokolide-like lipopeptide",
        "level": "lipopeptide / large cyclic peptide",
        "core": [],
        "families": ["tychonamide", "pahayokolide"],
        "allowed_partners": set(),
        "feature_tokens": {"Lipid_tail"},
        "canonical_motifs": ["Tychonamide/Pahayokolide-like"],
        "description": "Name-supported tychonamide/pahayokolide-like subclass, separated from generic lipopeptides for ontology-level interpretation."
    },
    "Lyngbyaureidamide_like": {
        "label": "Lyngbyaureidamide-like ureido/lipopeptide",
        "level": "lipopeptide / ureido peptide",
        "core": [],
        "families": ["lyngbyaureidamide", "lyngbyaureidamides"],
        "allowed_partners": set(),
        "feature_tokens": {"Lipid_tail"},
        "canonical_motifs": ["Lyngbyaureidamide-like"],
        "description": "Name-supported lyngbyaureidamide-like ureido/lipopeptide subclass."
    },
    "Unknown": {
        "label": "Unknown scaffold",
        "level": "unresolved",
        "core": [],
        "families": [],
        "allowed_partners": set(),
        "feature_tokens": set(),
        "canonical_motifs": [],
        "description": "No scaffold-level evidence detected."
    },
}


# -------------------------------------------------------------------------
# v21 expanded scaffold ontology additions
# -------------------------------------------------------------------------
NPSAA_DIAGNOSTIC_FEATURES = {
    "Dhb", "Mdha", "Dhoya", "Dhmoya", "Hmoya", "Athmu", "Amha",
    "Ahoa", "Amoa", "Aba", "Ada", "Aound", "Htya", "Aeap", "Aeca"
}

SCAFFOLD_ONTOLOGY.update({
    "Microviridin_superfamily": {
        "label": "Microviridin-like RiPP/lactone-cage peptide",
        "level": "RiPP / tricyclic depsipeptide",
        "core": [],
        "families": ["microviridin"],
        "allowed_partners": set(),
        "feature_tokens": {"Dhb", "Mdha"},
        "canonical_motifs": ["Microviridin-like", "Microviridin-Dhb/Mdha"],
        "description": "Microviridin-like RiPPs, separated from generic RiPP because their defining logic is a highly constrained lactone/lactam cage rather than simple thiazole/oxazole cyclamide heterocycles."
    },
    "Marine_PKS_NRPS_lipopeptide": {
        "label": "Dragonamide/Malevamide/Janadolide-type marine PKS-NRPS",
        "level": "hybrid PKS-NRPS lipopeptide",
        "core": [],
        "families": ["dragonamide", "malevamide", "janadolide", "almiramide", "palmyramide", "hantupeptin", "grenadamide"],
        "allowed_partners": set(),
        "feature_tokens": {"Polyketide", "Lipid_tail", "Dhoya", "Dhmoya", "Hmoya", "Ahoa", "Amoa", "Aound", "Htya", "Dhb", "Mdha"},
        "canonical_motifs": ["Marine-PKS-NRPS", "Dragonamide/Malevamide-like", "Janadolide/Almiramide-like"],
        "description": "Marine cyanobacterial PKS-NRPS lipopeptide space enriched in long lipid/polyketide chains and diagnostic non-proteinogenic amino acids."
    },
    "Cyclamide_RiPP_superfamily": {
        "label": "Cyclamide/cyanobactin RiPP superfamily",
        "level": "RiPP / cyanobactin / cyclamide",
        "core": [],
        "families": ["aerucyclamide", "tenuecyclamide", "lobocyclamide", "trichamide", "nostocyclamide", "lyngbyacyclamide", "kawaguchipeptin", "anacyclamide", "piricyclamide", "patellamide", "microcyclamide", "cyanobactin"],
        "allowed_partners": set(),
        "feature_tokens": {"Thiazole", "Thiazoline", "Oxazole", "Oxazoline", "Prenyl"},
        "canonical_motifs": ["Cyclamide-RiPP", "RiPP-Thiazole", "RiPP-Oxazole", "RiPP-Prenyl"],
        "description": "Unified cyanobactin/cyclamide scaffold class for small cyclic RiPPs with thiazole/oxazole-derived heterocycles and related tailoring."
    },
    "Laxaphycin_Hassallidin_superfamily": {
        "label": "Laxaphycin–Scytocyclamide–Hassallidin superfamily",
        "level": "large cyclic/lipo/glycopeptide",
        "core": [],
        "families": ["laxaphycin", "scytocyclamide", "hassallidin", "tolytoxin", "hectochlorin"],
        "allowed_partners": set(),
        "feature_tokens": {"Lipid_tail", "Sugar", "Dhb", "Mdha", "Dhoya", "Dhmoya", "Hmoya", "Athmu", "Amha", "Ahoa", "Amoa", "Aba", "Ada", "Aound", "Htya"},
        "canonical_motifs": ["Laxaphycin/Hassallidin-like", "Large-cyclic-lipopeptide", "Glyco-lipopeptide"],
        "description": "Large cyanobacterial cyclic/lipo/glycopeptide superfamily where family-name support and NPSAA-rich feature profiles are more reliable than generic residue SMARTS."
    },
    "NPSAA_rich_depsipeptide": {
        "label": "NPSAA-rich depsipeptide / modified peptide",
        "level": "modified NRPS peptide",
        "core": [],
        "families": ["trichormamide", "lyngbyazothrin", "schizotrin", "aeruginazole", "tolybyssidin"],
        "allowed_partners": set(),
        "feature_tokens": NPSAA_DIAGNOSTIC_FEATURES | {"Lipid_tail", "Polyketide"},
        "canonical_motifs": ["NPSAA-rich-depsipeptide"],
        "description": "Catch-all name-supported class for families dominated by diagnostic non-proteinogenic amino acids and depsipeptide-like architectures."
    },
})

BIOSYNTHETIC_CLASS_BY_SCAFFOLD.update({
    "Microviridin_superfamily": "RiPP",
    "Marine_PKS_NRPS_lipopeptide": "NRPS_PKS",
    "Cyclamide_RiPP_superfamily": "RiPP",
    "Laxaphycin_Hassallidin_superfamily": "Lipopeptide",
    "NPSAA_rich_depsipeptide": "NRPS",
})

CANONICAL_MOTIF_LIBRARY = {
    "Ahp-Phe": {"scaffold": "Ahp_peptidase_inhibitor", "families": ["micropeptin", "cyanopeptolin"]},
    "Ahp-Tyr": {"scaffold": "Ahp_peptidase_inhibitor", "families": ["cyanopeptolin", "micropeptin"]},
    "Ahp-Trp": {"scaffold": "Ahp_peptidase_inhibitor", "families": ["micropeptin", "cyanopeptolin"]},
    "Ahp-Gln": {"scaffold": "Ahp_peptidase_inhibitor", "families": ["micropeptin", "cyanopeptolin"]},
    "Choi-Arg": {"scaffold": "Aeruginosin", "families": ["aeruginosin"]},
    "Adda-Glu": {"scaffold": "Microcystin", "families": ["microcystin", "nodularin"]},
    "RiPP-Thiazole": {"scaffold": "RiPP", "families": ["aerucyclamide", "patellamide", "anacyclamide", "piricyclamide"]},
    "RiPP-Thiazoline": {"scaffold": "RiPP", "families": ["aerucyclamide", "patellamide", "anacyclamide", "piricyclamide"]},
    "RiPP-Oxazole": {"scaffold": "RiPP", "families": ["aerucyclamide", "patellamide"]},
    "Microginin-Ahda/Ahdo": {"scaffold": "Microginin", "families": ["microginin"]},
    "Lyngbyacyclamide-like": {"scaffold": "Lyngbyacyclamide_like", "families": ["lyngbyacyclamide"]},
    "Kawaguchipeptin-like": {"scaffold": "Kawaguchipeptin_like", "families": ["kawaguchipeptin"]},
    "Venturamide-like": {"scaffold": "Venturamide_like", "families": ["venturamide"]},
    "Microspinosamide-like": {"scaffold": "Microspinosamide_like", "families": ["microspinosamide"]},
    "Heinamide-like": {"scaffold": "Heinamide_like", "families": ["heinamide"]},
    "Scytocyclamide-like": {"scaffold": "Scytocyclamide_like", "families": ["scytocyclamide"]},
    "Tychonamide/Pahayokolide-like": {"scaffold": "Tychonamide_Pahayokolide_like", "families": ["tychonamide", "pahayokolide"]},
    "Lyngbyaureidamide-like": {"scaffold": "Lyngbyaureidamide_like", "families": ["lyngbyaureidamide", "lyngbyaureidamides"]},
    "Apratoxin-like PKS-NRPS": {"scaffold": "PKS_NRPS_cyclodepsipeptide", "families": ["apratoxin"]},
    "Cryptophycin-like PKS-NRPS": {"scaffold": "PKS_NRPS_macrolactam", "families": ["cryptophycin"]},
    "Jamaicamide-like PKS-NRPS": {"scaffold": "PKS_NRPS_alkynyl", "families": ["jamaicamide"]},
    "Dolastatin-like linear PKS-NRPS": {"scaffold": "Linear_PKS_NRPS", "families": ["dolastatin", "symplostatin"]},
    "Microviridin-like": {"scaffold": "Microviridin_superfamily", "families": ["microviridin"]},
    "Microviridin-Dhb/Mdha": {"scaffold": "Microviridin_superfamily", "families": ["microviridin"]},
    "Marine-PKS-NRPS": {"scaffold": "Marine_PKS_NRPS_lipopeptide", "families": ["dragonamide", "malevamide", "janadolide", "almiramide", "palmyramide", "hantupeptin", "grenadamide"]},
    "Dragonamide/Malevamide-like": {"scaffold": "Marine_PKS_NRPS_lipopeptide", "families": ["dragonamide", "malevamide"]},
    "Janadolide/Almiramide-like": {"scaffold": "Marine_PKS_NRPS_lipopeptide", "families": ["janadolide", "almiramide", "palmyramide", "hantupeptin", "grenadamide"]},
    "Cyclamide-RiPP": {"scaffold": "Cyclamide_RiPP_superfamily", "families": ["aerucyclamide", "tenuecyclamide", "lobocyclamide", "trichamide", "nostocyclamide", "lyngbyacyclamide", "kawaguchipeptin"]},
    "Laxaphycin/Hassallidin-like": {"scaffold": "Laxaphycin_Hassallidin_superfamily", "families": ["laxaphycin", "scytocyclamide", "hassallidin", "tolytoxin", "hectochlorin"]},
    "NPSAA-rich-depsipeptide": {"scaffold": "NPSAA_rich_depsipeptide", "families": ["trichormamide", "lyngbyazothrin", "schizotrin", "aeruginazole", "tolybyssidin"]},
}

SCAFFOLD_ALLOWED_PARTNERS = {
    scaffold: set(spec.get("allowed_partners", set()))
    for scaffold, spec in SCAFFOLD_ONTOLOGY.items()
}

SCAFFOLD_TO_CORE = {
    scaffold: list(spec.get("core", []))
    for scaffold, spec in SCAFFOLD_ONTOLOGY.items()
}

FAMILY_SCAFFOLD_RULES = {}
for scaffold, spec in SCAFFOLD_ONTOLOGY.items():
    for family_term in spec.get("families", []):
        FAMILY_SCAFFOLD_RULES[family_term] = scaffold


def _positive(row, col):
    """Safe helper for binary signature columns."""
    return int(row.get(col, 0) or 0) > 0


def infer_core_from_high_confidence_signatures(row):
    """Infer biosynthetic core from high-confidence structural signature columns."""
    core = set()
    evidence = []
    for col, token in CORE_SIGNATURE_RULES:
        if _positive(row, col):
            core.add(token)
            evidence.append(col)
    return sorted(core), "; ".join(evidence)


def infer_scaffold_class(row, base_tokens):
    """
    Detect the first-level natural product scaffold class.

    v12 logic:
    1. High-confidence structural signatures define Ahp, Choi and Adda scaffolds.
    2. Family/name rules classify RiPP, PKS-NRPS hybrid, lipopeptide and microginin spaces.
    3. Token fallback is used only after signature/name evidence.
    """
    core, core_evidence = infer_core_from_high_confidence_signatures(row)
    if len(core) > 1:
        hybrid_name = "Hybrid_" + "_".join(core)
        return hybrid_name, core, core_evidence, "high-confidence structural signature; hybrid scaffold"
    if "Ahp" in core:
        return "Ahp_peptidase_inhibitor", core, core_evidence, "high-confidence structural signature"
    if "Choi" in core:
        return "Aeruginosin", core, core_evidence, "high-confidence structural signature"
    if "Adda" in core:
        return "Microcystin", core, core_evidence, "high-confidence structural signature"

    family_text = f"{row.get('family', '')} {row.get('compound_name', '')}".lower()
    for term, scaffold in FAMILY_SCAFFOLD_RULES.items():
        if term in family_text:
            return scaffold, SCAFFOLD_TO_CORE.get(scaffold, []), f"family/name:{term}", "name-supported scaffold"

    # Conservative token fallback for the three currently supported structural cores.
    token_core = sorted(base_tokens & BIOSYNTHETIC_CORE_TOKENS)
    if "Ahp" in token_core:
        return "Ahp_peptidase_inhibitor", ["Ahp"], "token_fallback:Ahp", "token-supported scaffold"
    if "Choi" in token_core:
        return "Aeruginosin", ["Choi"], "token_fallback:Choi", "token-supported scaffold"
    if "Adda" in token_core:
        return "Microcystin", ["Adda"], "token_fallback:Adda", "token-supported scaffold"

    return "Unknown", [], "", "unresolved"


def infer_scaffold_features(base_tokens, scaffold_class):
    """Infer scaffold-level features according to the scaffold ontology."""
    features = set(base_tokens & SCAFFOLD_FEATURE_TOKENS)
    scaffold_spec = SCAFFOLD_ONTOLOGY.get(scaffold_class, SCAFFOLD_ONTOLOGY["Unknown"])
    allowed_features = set(scaffold_spec.get("feature_tokens", set()))

    if scaffold_class in {"RiPP", "Cyclamide_RiPP_superfamily", "Lyngbyacyclamide_like", "Venturamide_like", "Microspinosamide_like"}:
        return sorted(features & {"Thiazole", "Thiazoline", "Oxazole", "Oxazoline", "Prenyl"})

    if scaffold_class in {"PKS_NRPS_hybrid", "PKS_NRPS_cyclodepsipeptide", "PKS_NRPS_macrolactam", "PKS_NRPS_alkynyl", "Linear_PKS_NRPS", "Marine_PKS_NRPS_lipopeptide"}:
        out = set(features & allowed_features)
        out.add("Peptide")
        out.add("Polyketide")
        return sorted(out)

    if scaffold_class in {"Lipopeptide", "Heinamide_like", "Tychonamide_Pahayokolide_like", "Lyngbyaureidamide_like", "Laxaphycin_Hassallidin_superfamily", "NPSAA_rich_depsipeptide"}:
        out = set(features & allowed_features)
        out.add("Lipid_tail")
        out.add("Peptide")
        return sorted(out)

    if scaffold_class == "Glycolipopeptide":
        out = set(features & allowed_features)
        out.add("Lipid_tail")
        out.add("Peptide")
        out.add("Sugar")
        return sorted(out)

    if scaffold_class == "Microginin":
        return ["Ahda/Ahdo"]

    return sorted(features & allowed_features)

def infer_partners_from_signatures(row):
    """Infer partner residues/groups from curated structural signature columns."""
    partners = set()
    evidence = []
    for col, token in PARTNER_SIGNATURE_RULES:
        if _positive(row, col):
            partners.add(token)
            evidence.append(col)
    return partners, evidence


def infer_partners_from_tokens(base_tokens):
    """Conservative partner fallback using only the curated partner vocabulary."""
    return set(base_tokens & BIOSYNTHETIC_PARTNER_TOKENS)


def filter_partners_by_scaffold(partners, scaffold_class):
    """Keep only partners that make sense for the detected scaffold class."""
    partners = set(partners)
    allowed = SCAFFOLD_ALLOWED_PARTNERS.get(scaffold_class, BIOSYNTHETIC_PARTNER_TOKENS)
    if not allowed:
        return []
    return sorted(partners & allowed)


def infer_decorators_from_tokens(base_tokens):
    """Infer tailoring/decorator features independently from scaffold/core and partners."""
    return sorted(base_tokens & BIOSYNTHETIC_DECORATOR_TOKENS)


def architecture_key_from_parts(scaffold_class, core_tokens, partner_tokens, decorator_tokens, scaffold_features):
    """Compact architecture key for searching, grouping and clustering."""
    core = "+".join(core_tokens) if core_tokens else "none"
    partners = "+".join(partner_tokens) if partner_tokens else "none"
    decorators = "+".join(decorator_tokens) if decorator_tokens else "none"
    features = "+".join(scaffold_features) if scaffold_features else "none"
    return f"{scaffold_class}|core:{core}|partners:{partners}|decorators:{decorators}|features:{features}"


def classify_biosynthetic_architecture(scaffold_class, core_tokens, partner_tokens, decorator_tokens, scaffold_features, family=""):
    """Assign an interpretable architecture family from scaffold/partner/decorator tokens."""
    core = set(core_tokens)
    partners = set(partner_tokens)
    decorators = set(decorator_tokens)
    features = set(scaffold_features)

    if scaffold_class == "Ahp_peptidase_inhibitor":
        if "Phe" in partners and "NMe" in decorators:
            return "Micropeptin-like Ahp-Phe N-methylated architecture"
        if "Phe" in partners:
            return "Micropeptin-like Ahp-Phe architecture"
        if "Tyr" in partners and "NMe" in decorators:
            return "Cyanopeptolin-like Ahp-Tyr N-methylated architecture"
        if "Tyr" in partners:
            return "Cyanopeptolin-like Ahp-Tyr architecture"
        if "Gln" in partners:
            return "Ahp-Gln peptidase-inhibitor architecture"
        return "Ahp-centered peptidase-inhibitor architecture"

    if scaffold_class == "Aeruginosin":
        if "Arg" in partners and "Sugar" in decorators:
            return "Aeruginosin-like Choi-Arg glycosylated architecture"
        if "Arg" in partners:
            return "Aeruginosin-like Choi-Arg architecture"
        return "Choi-centered aeruginosin-like architecture"

    if scaffold_class == "Microcystin":
        if "Glu" in partners:
            return "Microcystin/Nodularin-like Adda-Glu architecture"
        return "Adda-centered microcystin/nodularin-like architecture"

    if scaffold_class == "RiPP":
        if {"Thiazole", "Thiazoline", "Oxazole", "Oxazoline"} & features:
            return "RiPP heterocycle-rich cyclamide architecture"
        if "Prenyl" in features:
            return "RiPP prenylated architecture"
        return "RiPP name-supported architecture"

    if scaffold_class == "PKS_NRPS_cyclodepsipeptide":
        return "Apratoxin-like PKS-NRPS cyclodepsipeptide architecture"

    if scaffold_class == "PKS_NRPS_macrolactam":
        return "Cryptophycin-like PKS-NRPS macrolactam architecture"

    if scaffold_class == "PKS_NRPS_alkynyl":
        return "Jamaicamide-like PKS-NRPS alkynyl architecture"

    if scaffold_class == "Linear_PKS_NRPS":
        return "Dolastatin/symplostatin-like linear PKS-NRPS architecture"

    if scaffold_class == "PKS_NRPS_hybrid":
        return "generic PKS-NRPS hybrid architecture"

    if scaffold_class == "Lyngbyacyclamide_like":
        if {"Thiazole", "Thiazoline", "Oxazole", "Oxazoline"} & features:
            return "Lyngbyacyclamide-like heterocycle-rich cyclamide architecture"
        return "Lyngbyacyclamide-like name-supported cyclamide architecture"

    if scaffold_class == "Kawaguchipeptin_like":
        return "Kawaguchipeptin-like macrocyclic peptide architecture"

    if scaffold_class == "Venturamide_like":
        if {"Thiazole", "Thiazoline", "Oxazole", "Oxazoline"} & features:
            return "Venturamide-like heterocycle-rich cyanobactin/RiPP architecture"
        if "Prenyl" in features:
            return "Venturamide-like prenylated cyanobactin/RiPP architecture"
        return "Venturamide-like name-supported cyanobactin/RiPP architecture"

    if scaffold_class == "Microspinosamide_like":
        if {"Thiazole", "Thiazoline", "Oxazole", "Oxazoline"} & features:
            return "Microspinosamide-like heterocycle-rich cyanobactin/RiPP architecture"
        return "Microspinosamide-like name-supported cyanobactin/RiPP architecture"

    if scaffold_class == "Heinamide_like":
        return "Heinamide-like lipopeptide/depsipeptide architecture"

    if scaffold_class == "Scytocyclamide_like":
        return "Scytocyclamide-like macrocyclic peptide architecture"

    if scaffold_class == "Tychonamide_Pahayokolide_like":
        return "Tychonamide/Pahayokolide-like lipopeptide architecture"

    if scaffold_class == "Lyngbyaureidamide_like":
        return "Lyngbyaureidamide-like ureido/lipopeptide architecture"

    if scaffold_class == "Microviridin_superfamily":
        return "Microviridin-like RiPP/lactone-cage architecture"

    if scaffold_class == "Marine_PKS_NRPS_lipopeptide":
        return "Dragonamide/Malevamide/Janadolide-type marine PKS-NRPS lipopeptide architecture"

    if scaffold_class == "Cyclamide_RiPP_superfamily":
        if {"Thiazole", "Thiazoline", "Oxazole", "Oxazoline"} & features:
            return "Cyclamide/cyanobactin RiPP heterocycle-rich architecture"
        if "Prenyl" in features:
            return "Cyclamide/cyanobactin RiPP prenylated architecture"
        return "Cyclamide/cyanobactin RiPP name-supported architecture"

    if scaffold_class == "Laxaphycin_Hassallidin_superfamily":
        if "Sugar" in features:
            return "Laxaphycin/Hassallidin-like glyco-lipopeptide architecture"
        return "Laxaphycin/Scytocyclamide/Hassallidin-like large cyclic lipopeptide architecture"

    if scaffold_class == "NPSAA_rich_depsipeptide":
        return "NPSAA-rich depsipeptide / modified NRPS peptide architecture"

    if scaffold_class == "Lipopeptide":
        return "Lipopeptide architecture"

    if scaffold_class == "Microginin":
        return "name-supported microginin-like architecture"

    return "unresolved peptide architecture"


def canonical_motif_from_parts(scaffold_class, core_tokens, partner_tokens, scaffold_features=None):
    """Return the main canonical biosynthetic motif for scaffold-first interpretation."""
    partners = set(partner_tokens)
    features = set(scaffold_features or [])

    if scaffold_class == "Ahp_peptidase_inhibitor":
        if "Phe" in partners:
            return "Ahp-Phe"
        if "Tyr" in partners:
            return "Ahp-Tyr"
        if "Trp" in partners:
            return "Ahp-Trp"
        if "Gln" in partners:
            return "Ahp-Gln"
        if "Lys" in partners:
            return "Ahp-Lys"
        if "Arg" in partners:
            return "Ahp-Arg"
        return "Ahp"

    if scaffold_class == "Aeruginosin":
        if "Arg" in partners:
            return "Choi-Arg"
        return "Choi"

    if scaffold_class == "Microcystin":
        if "Glu" in partners:
            return "Adda-Glu"
        return "Adda"

    if scaffold_class == "Microginin":
        return "Microginin-Ahda/Ahdo"

    if scaffold_class == "RiPP":
        heterocycles = sorted({"Thiazole", "Thiazoline", "Oxazole", "Oxazoline"} & features)
        if heterocycles:
            return "RiPP-" + "+".join(heterocycles)
        if "Prenyl" in features:
            return "RiPP-Prenyl"
        return "RiPP"

    if scaffold_class == "PKS_NRPS_cyclodepsipeptide":
        return "Apratoxin-like PKS-NRPS"
    if scaffold_class == "PKS_NRPS_macrolactam":
        return "Cryptophycin-like PKS-NRPS"
    if scaffold_class == "PKS_NRPS_alkynyl":
        return "Jamaicamide-like PKS-NRPS"
    if scaffold_class == "Linear_PKS_NRPS":
        return "Dolastatin-like linear PKS-NRPS"
    if scaffold_class == "PKS_NRPS_hybrid":
        return "PKS-NRPS"

    if scaffold_class == "Lyngbyacyclamide_like":
        heterocycles = sorted({"Thiazole", "Thiazoline", "Oxazole", "Oxazoline"} & features)
        if heterocycles:
            return "Lyngbyacyclamide-like-" + "+".join(heterocycles)
        return "Lyngbyacyclamide-like"
    if scaffold_class == "Kawaguchipeptin_like":
        return "Kawaguchipeptin-like"
    if scaffold_class == "Venturamide_like":
        heterocycles = sorted({"Thiazole", "Thiazoline", "Oxazole", "Oxazoline"} & features)
        if heterocycles:
            return "Venturamide-like-" + "+".join(heterocycles)
        if "Prenyl" in features:
            return "Venturamide-like-Prenyl"
        return "Venturamide-like"
    if scaffold_class == "Microspinosamide_like":
        heterocycles = sorted({"Thiazole", "Thiazoline", "Oxazole", "Oxazoline"} & features)
        if heterocycles:
            return "Microspinosamide-like-" + "+".join(heterocycles)
        return "Microspinosamide-like"
    if scaffold_class == "Heinamide_like":
        return "Heinamide-like"
    if scaffold_class == "Scytocyclamide_like":
        return "Scytocyclamide-like"
    if scaffold_class == "Tychonamide_Pahayokolide_like":
        return "Tychonamide/Pahayokolide-like"
    if scaffold_class == "Lyngbyaureidamide_like":
        return "Lyngbyaureidamide-like"

    if scaffold_class == "Microviridin_superfamily":
        if {"Dhb", "Mdha"} & features:
            return "Microviridin-Dhb/Mdha"
        return "Microviridin-like"
    if scaffold_class == "Marine_PKS_NRPS_lipopeptide":
        return "Marine-PKS-NRPS"
    if scaffold_class == "Cyclamide_RiPP_superfamily":
        heterocycles = sorted({"Thiazole", "Thiazoline", "Oxazole", "Oxazoline"} & features)
        if heterocycles:
            return "Cyclamide-RiPP-" + "+".join(heterocycles)
        if "Prenyl" in features:
            return "Cyclamide-RiPP-Prenyl"
        return "Cyclamide-RiPP"
    if scaffold_class == "Laxaphycin_Hassallidin_superfamily":
        return "Laxaphycin/Hassallidin-like"
    if scaffold_class == "NPSAA_rich_depsipeptide":
        return "NPSAA-rich-depsipeptide"

    if scaffold_class == "Lipopeptide":
        return "Lipopeptide"
    if scaffold_class == "Glycolipopeptide":
        return "Glycolipopeptide"
    if scaffold_class == "Macrocyclic_peptide":
        return "Macrocyclic_peptide"
    if scaffold_class == "Linear_peptide":
        return "Linear_peptide"

    return "Unresolved"

def architecture_confidence(scaffold_evidence, partner_evidence, scaffold_class, partner_tokens):
    """Simple evidence tier for scaffold-first architecture calls."""
    if scaffold_class != "Unknown" and "high-confidence" in str(scaffold_evidence) and partner_evidence:
        return "high-confidence scaffold+partner"
    if scaffold_class != "Unknown" and "high-confidence" in str(scaffold_evidence):
        return "high-confidence scaffold only"
    if scaffold_class != "Unknown" and partner_tokens:
        return "name/token-supported scaffold+partner"
    if scaffold_class != "Unknown":
        return "name/token-supported scaffold"
    return "unresolved"


def scaffold_requires_structural_core(scaffold_class):
    """Return True for scaffold classes that should ideally have a structural anchor."""
    return scaffold_class in {
        "Ahp_peptidase_inhibitor",
        "Aeruginosin",
        "Microcystin",
    } or str(scaffold_class).startswith("Hybrid_")


def confidence_level_from_evidence(scaffold_class, scaffold_evidence, core_tokens, partner_tokens, canonical_motif):
    """
    Formal NPSAA confidence level.

    A = structural scaffold/core evidence plus motif/partner evidence
    B = structural scaffold/core evidence only
    C = scaffold assigned by family/name or token fallback
    D = unresolved scaffold
    """
    evidence_text = str(scaffold_evidence or "").lower()
    has_core = bool(core_tokens)
    has_partner_or_motif = bool(partner_tokens) or str(canonical_motif or "") not in {"", "Unresolved"}

    if scaffold_class == "Unknown":
        return "D_unresolved"
    if "high-confidence" in evidence_text and has_core and has_partner_or_motif:
        return "A_structural_scaffold_plus_motif"
    if "high-confidence" in evidence_text and has_core:
        return "B_structural_scaffold"
    return "C_name_or_token_supported"


def scaffold_core_status(scaffold_class, core_tokens, scaffold_evidence):
    """Audit whether an assigned scaffold is supported by its expected structural core."""
    if scaffold_class == "Unknown":
        return "unresolved"
    if str(scaffold_class).startswith("Hybrid_"):
        return "hybrid_multiple_structural_cores"
    if scaffold_requires_structural_core(scaffold_class):
        return "core_present" if core_tokens else "possible_like_name_supported_missing_core"
    if "family/name" in str(scaffold_evidence):
        return "family_supported_non_core_scaffold"
    return "not_core_required"


def biosynthetic_score_from_parts(scaffold_class, core_tokens, partner_tokens, decorator_tokens, scaffold_features, confidence_level):
    """Interpretable weighted biosynthetic score used for audit and sorting."""
    score = 0
    if scaffold_class != "Unknown":
        score += 5
    score += 5 * len(core_tokens)
    score += 3 * len(partner_tokens)
    score += 2 * len(scaffold_features)
    score += 1 * len(decorator_tokens)
    if str(confidence_level).startswith("A_"):
        score += 4
    elif str(confidence_level).startswith("B_"):
        score += 2
    elif str(confidence_level).startswith("C_"):
        score += 1
    return int(score)


def detect_biosynthetic_architecture_for_row(row, residue_cols, motif_cols):
    """
    Detect scaffold-first biosynthetic architecture.

    v12 hierarchy:
    Step 1: SCAFFOLD_CLASS
    Step 2: CORE -> PARTNERS -> DECORATORS within that scaffold.
    """
    tokens = set(extract_detected_tokens_from_row(row, residue_cols, motif_cols))
    base_tokens = {strip_count_suffix(t) for t in tokens}

    scaffold_class, core, core_evidence, scaffold_evidence = infer_scaffold_class(row, base_tokens)

    signature_partners, partner_evidence_list = infer_partners_from_signatures(row)
    token_partners = infer_partners_from_tokens(base_tokens)
    partners = filter_partners_by_scaffold(signature_partners | token_partners, scaffold_class)

    partner_evidence = "; ".join([
        ev for ev in partner_evidence_list
        if any(token in partners for col, token in PARTNER_SIGNATURE_RULES if col == ev)
    ])

    decorators = infer_decorators_from_tokens(base_tokens)
    scaffold_features = infer_scaffold_features(base_tokens, scaffold_class)

    # Critical v13 safeguard: partners/decorators/features are only interpreted
    # inside a resolved scaffold. Without scaffold context, residue/motif hits are
    # treated as raw evidence, not as a biosynthetic architecture.
    if scaffold_class == "Unknown":
        partners = []
        decorators = []
        scaffold_features = []
        partner_evidence = ""

    canonical_motif = canonical_motif_from_parts(scaffold_class, core, partners, scaffold_features)
    feature_set = "; ".join(scaffold_features) if scaffold_features else ""

    architecture_class = classify_biosynthetic_architecture(
        scaffold_class, core, partners, decorators, scaffold_features, family=row.get("family", "")
    )
    confidence = architecture_confidence(scaffold_evidence, partner_evidence, scaffold_class, partners)

    core_text = "+".join(core) if core else "No_core_token"
    partners_text = "+".join(partners) if partners else "No_informative_partners"
    decorators_text = "+".join(decorators) if decorators else "No_decorators"
    features_text = "+".join(scaffold_features) if scaffold_features else "No_scaffold_features"

    architecture = (
        f"Scaffold={scaffold_class} | Motif={canonical_motif} | Core={core_text} | Partners={partners_text} | "
        f"Decorators={decorators_text} | Features={features_text}"
    )
    architecture_core_partners = f"Scaffold={scaffold_class} | Motif={canonical_motif} | Core={core_text} | Partners={partners_text}"
    architecture_key = architecture_key_from_parts(scaffold_class, core, partners, decorators, scaffold_features)

    weighted_tokens = []
    biosynthetic_class_for_weight = "Hybrid" if str(scaffold_class).startswith("Hybrid_") else BIOSYNTHETIC_CLASS_BY_SCAFFOLD.get(scaffold_class, "Unknown")
    if biosynthetic_class_for_weight != "Unknown":
        weighted_tokens.extend([f"BIOSYNTHETICCLASS_{biosynthetic_class_for_weight}"] * 10)
    if scaffold_class != "Unknown":
        weighted_tokens.extend([f"SCAFFOLDCLASS_{scaffold_class}"] * 8)
    if canonical_motif and canonical_motif != "Unresolved":
        weighted_tokens.extend([f"CANONICAL_{canonical_motif}"] * 4)
    for token in core:
        weighted_tokens.extend([f"CORE_{token}"] * 5)
    for token in partners:
        weighted_tokens.extend([f"PARTNER_{token}"] * 2)
    for token in scaffold_features:
        weighted_tokens.extend([f"FEATURE_{token}"] * 2)
    for token in decorators:
        weighted_tokens.append(f"DECORATOR_{token}")

    weighted_signature = "-".join(weighted_tokens) if weighted_tokens else "Unresolved_biosynthetic_architecture"
    biosynthetic_class = "Hybrid" if str(scaffold_class).startswith("Hybrid_") else BIOSYNTHETIC_CLASS_BY_SCAFFOLD.get(scaffold_class, "Unknown")
    confidence_level = confidence_level_from_evidence(scaffold_class, scaffold_evidence, core, partners, canonical_motif)
    core_status = scaffold_core_status(scaffold_class, core, scaffold_evidence)
    biosynthetic_score = biosynthetic_score_from_parts(scaffold_class, core, partners, decorators, scaffold_features, confidence_level)

    return {
        "BIOSYNTHETIC_CLASS": biosynthetic_class,
        "NPSAA_CONFIDENCE_LEVEL": confidence_level,
        "SCAFFOLD_CORE_STATUS": core_status,
        "BIOSYNTHETIC_SCORE": biosynthetic_score,
        "SCAFFOLD_CLASS": scaffold_class,
        "SCAFFOLD_EVIDENCE": scaffold_evidence,
        "SCAFFOLD_FEATURES": "; ".join(scaffold_features),
        "FEATURE_SET": feature_set,
        "CANONICAL_MOTIF": canonical_motif,
        "BIOSYNTHETIC_SCAFFOLD": scaffold_class,
        "BIOSYNTHETIC_CORE": "; ".join(core),
        "BIOSYNTHETIC_PARTNERS": "; ".join(partners),
        "BIOSYNTHETIC_DECORATORS": "; ".join(decorators),
        "BIOSYNTHETIC_ARCHITECTURE": architecture,
        "BIOSYNTHETIC_CORE_PARTNERS": architecture_core_partners,
        "BIOSYNTHETIC_ARCHITECTURE_KEY": architecture_key,
        "BIOSYNTHETIC_SCAFFOLD_FAMILY": scaffold_class,
        "BIOSYNTHETIC_ARCHITECTURE_CLASS": architecture_class,
        "BIOSYNTHETIC_ARCHITECTURE_CONFIDENCE": confidence,
        "BIOSYNTHETIC_CORE_EVIDENCE": core_evidence,
        "BIOSYNTHETIC_PARTNER_EVIDENCE": partner_evidence,
        "BIOSYNTHETIC_WEIGHTED_SIGNATURE": weighted_signature,
        "biosynthetic_token_count": len(core) + len(partners) + len(decorators) + len(scaffold_features),
        "biosynthetic_core_count": len(core),
        "biosynthetic_partner_count": len(partners),
        "biosynthetic_decorator_count": len(decorators),
    }

def build_biosynthetic_architecture_table(peptides_df, residue_cols, motif_cols):
    """Create compound-level Biosynthetic Architecture Analysis table."""
    rows = []
    for idx, row in peptides_df.iterrows():
        arch = detect_biosynthetic_architecture_for_row(row, residue_cols, motif_cols)
        out = {
            "compound_index": idx,
            "compound_name": row.get("compound_name", ""),
            "family": row.get("family", ""),
            **arch,
            "MODULE_signature": row.get("MODULE_signature", ""),
            "MODULE_core": row.get("MODULE_core", ""),
            "MODULE_modifiers": row.get("MODULE_modifiers", ""),
            "cyanopeptide_signature_hits": row.get("cyanopeptide_signature_hits", ""),
            "AA_signature_low_weight": row.get("AA_signature", ""),
            "SMILES": row.get("SMILES", ""),
        }
        rows.append(out)
    return pd.DataFrame(rows)


def choose_biosynthetic_nppsa_signature(row):
    """
    Architecture-first NPPSA signature.

    Priority:
    1. BIOSYNTHETIC_WEIGHTED_SIGNATURE;
    2. MODULE_core/MODULE_modifiers;
    3. curated cyanopeptide signatures;
    4. AA_signature as low-weight fallback.
    """
    arch = str(row.get("BIOSYNTHETIC_WEIGHTED_SIGNATURE", "") or "").strip()
    if arch and arch != "Unresolved_biosynthetic_architecture":
        return arch

    module_core = str(row.get("MODULE_core", "") or "").strip()
    module_mod = str(row.get("MODULE_modifiers", "") or "").strip()
    if module_core:
        parts = [module_core, module_core, module_core]
        if module_mod:
            parts.append(module_mod)
        return " | ".join(parts)

    cyano_sig = str(row.get("cyanopeptide_signature_hits", "") or "").strip()
    if cyano_sig:
        return "Cyanopeptide_signatures: " + cyano_sig

    aa_sig = str(row.get("AA_signature", "") or "").strip()
    if aa_sig:
        return "LOW_WEIGHT_AA: " + aa_sig

    return "Unresolved_biosynthetic_architecture"


def architecture_components(text, include_decorators=True):
    """Parse BIOSYNTHETIC_ARCHITECTURE text into comparable component tokens."""
    text = str(text or "")
    components = set()
    for part in text.split("|"):
        part = part.strip()
        if not part:
            continue
        if part.startswith("Decorators=") and not include_decorators:
            continue
        value = part.split("=", 1)[-1]
        for token in re.split(r"[+; ,]+", value):
            token = token.strip()
            if token and token not in {"No_informative_partners", "No_decorators", "Unresolved_core"}:
                components.add(token)
    return components


def build_architecture_summary_and_edges(architecture_df, min_count=2, include_decorators=False, min_jaccard=0.34):
    """Build architecture-level summary and similarity network edge tables."""
    if architecture_df.empty:
        return (
            pd.DataFrame(columns=["architecture", "architecture_class", "count", "families", "compounds_preview"]),
            pd.DataFrame(columns=["source", "target", "architecture_jaccard"]),
        )

    group_col = "BIOSYNTHETIC_CORE_PARTNERS" if not include_decorators else "BIOSYNTHETIC_ARCHITECTURE"
    rows = []
    for architecture, sub in architecture_df.groupby(group_col):
        if len(sub) < int(min_count):
            continue
        families = sub["family"].astype(str).value_counts().head(5).to_dict()
        classes = sub["BIOSYNTHETIC_ARCHITECTURE_CLASS"].astype(str).value_counts().head(3).to_dict()
        rows.append({
            "architecture": architecture,
            "architecture_class": "; ".join(f"{k}:{v}" for k, v in classes.items()),
            "count": len(sub),
            "families": "; ".join(f"{k}:{v}" for k, v in families.items()),
            "compounds_preview": "; ".join(sub["compound_name"].astype(str).head(8).tolist()),
        })
    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values(["count", "architecture"], ascending=[False, True]).reset_index(drop=True)
    else:
        summary = pd.DataFrame(columns=["architecture", "architecture_class", "count", "families", "compounds_preview"])

    edge_rows = []
    architectures = summary["architecture"].tolist()
    component_map = {
        arch: architecture_components(arch, include_decorators=include_decorators)
        for arch in architectures
    }
    for a, b in itertools.combinations(architectures, 2):
        ca, cb = component_map[a], component_map[b]
        if not ca or not cb:
            continue
        j = len(ca & cb) / len(ca | cb)
        if j >= float(min_jaccard):
            edge_rows.append({"source": a, "target": b, "architecture_jaccard": j})
    edges = pd.DataFrame(edge_rows)
    if len(edges):
        edges = edges.sort_values("architecture_jaccard", ascending=False).reset_index(drop=True)
    else:
        edges = pd.DataFrame(columns=["source", "target", "architecture_jaccard"])
    return summary, edges


def make_architecture_network(architecture_summary_df, architecture_edges_df, min_edge_similarity=0.34):
    """Plot biosynthetic architecture similarity network."""
    if not NETWORKX_AVAILABLE or architecture_summary_df.empty:
        return None

    G = nx.Graph()
    counts = dict(zip(architecture_summary_df["architecture"], architecture_summary_df["count"]))
    classes = dict(zip(architecture_summary_df["architecture"], architecture_summary_df["architecture_class"]))

    for architecture, count in counts.items():
        G.add_node(architecture, count=int(count), architecture_class=classes.get(architecture, ""))

    for _, row in architecture_edges_df.iterrows():
        sim = float(row.get("architecture_jaccard", 0))
        if sim >= float(min_edge_similarity):
            if row["source"] in G.nodes and row["target"] in G.nodes:
                G.add_edge(row["source"], row["target"], weight=sim)

    if G.number_of_edges() == 0:
        return go.Figure().update_layout(title="No architecture similarity edges at selected threshold")

    pos = nx.spring_layout(G, seed=42, weight="weight")
    edge_x, edge_y = [], []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, text, hover, size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        count = G.nodes[node].get("count", 1)
        label = str(node).replace("Core=", "").replace(" | Partners=", " + ")
        text.append(label[:45])
        hover.append(f"{node}<br>Compounds: {count}<br>{G.nodes[node].get('architecture_class','')}<br>Degree: {G.degree(node)}")
        size.append(12 + min(count, 120) * 0.20 + G.degree(node) * 2)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.7), hoverinfo="none", showlegend=False))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=text, textposition="top center",
        hovertext=hover, hoverinfo="text", marker=dict(size=size, line=dict(width=1)), showlegend=False
    ))
    fig.update_layout(
        title=f"Biosynthetic architecture network, Jaccard ≥ {min_edge_similarity:.2f}",
        height=760,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig



def build_scaffold_summary_and_edges(architecture_df, min_count=1):
    """Build scaffold-level summary and scaffold co-occurrence/similarity edges.

    The scaffold layer is the first-order NPSAA representation. It intentionally
    summarizes compounds by SCAFFOLD_CLASS before inspecting motif-level variants.
    """
    if architecture_df.empty or "SCAFFOLD_CLASS" not in architecture_df.columns:
        return (
            pd.DataFrame(columns=[
                "BIOSYNTHETIC_CLASS", "SCAFFOLD_CLASS", "count", "canonical_motifs", "families",
                "architecture_classes", "feature_sets", "representative_compounds"
            ]),
            pd.DataFrame(columns=["source", "target", "shared_feature_jaccard", "shared_motifs"]),
        )

    rows = []
    for scaffold, sub in architecture_df.groupby("SCAFFOLD_CLASS"):
        if len(sub) < int(min_count):
            continue
        ontology = SCAFFOLD_ONTOLOGY.get(scaffold, SCAFFOLD_ONTOLOGY.get("Unknown", {}))
        motifs = sorted(set(sub.get("CANONICAL_MOTIF", pd.Series(dtype=str)).dropna().astype(str)))
        motifs = [m for m in motifs if m and m != "Unresolved"]
        families = sub["family"].astype(str).value_counts().head(8).to_dict() if "family" in sub else {}
        classes = sub["BIOSYNTHETIC_ARCHITECTURE_CLASS"].astype(str).value_counts().head(5).to_dict() if "BIOSYNTHETIC_ARCHITECTURE_CLASS" in sub else {}
        features = sorted(set(
            token.strip()
            for value in sub.get("FEATURE_SET", pd.Series(dtype=str)).fillna("").astype(str)
            for token in re.split(r"[;+]", value)
            if token.strip()
        ))
        rows.append({
            "BIOSYNTHETIC_CLASS": "Hybrid" if str(scaffold).startswith("Hybrid_") else BIOSYNTHETIC_CLASS_BY_SCAFFOLD.get(scaffold, "Unknown"),
            "SCAFFOLD_CLASS": scaffold,
            "ontology_label": ontology.get("label", scaffold),
            "biosynthetic_level": ontology.get("level", ""),
            "count": len(sub),
            "canonical_motifs": "; ".join(motifs),
            "families": "; ".join(f"{k}:{v}" for k, v in families.items()),
            "architecture_classes": "; ".join(f"{k}:{v}" for k, v in classes.items()),
            "feature_sets": "; ".join(features),
            "representative_compounds": "; ".join(sub["compound_name"].astype(str).head(10).tolist()),
        })

    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values(["count", "SCAFFOLD_CLASS"], ascending=[False, True]).reset_index(drop=True)
    else:
        summary = pd.DataFrame(columns=[
            "BIOSYNTHETIC_CLASS", "SCAFFOLD_CLASS", "ontology_label", "biosynthetic_level", "count",
            "canonical_motifs", "families", "architecture_classes", "feature_sets", "representative_compounds"
        ])

    # Edges are based on motif/feature overlap between scaffold classes.
    edge_rows = []
    scaffold_features = {}
    for _, row in summary.iterrows():
        scaffold = row["SCAFFOLD_CLASS"]
        tokens = set()
        for field in ["canonical_motifs", "feature_sets"]:
            tokens.update(t.strip() for t in str(row.get(field, "")).split(";") if t.strip())
        ontology = SCAFFOLD_ONTOLOGY.get(scaffold, {})
        tokens.update(ontology.get("core", []))
        tokens.update(ontology.get("feature_tokens", set()))
        scaffold_features[scaffold] = tokens

    for a, b in itertools.combinations(summary["SCAFFOLD_CLASS"].tolist(), 2):
        ta, tb = scaffold_features.get(a, set()), scaffold_features.get(b, set())
        if not ta or not tb:
            continue
        j = len(ta & tb) / len(ta | tb) if (ta | tb) else 0
        shared = sorted(ta & tb)
        if j > 0:
            edge_rows.append({
                "source": a,
                "target": b,
                "shared_feature_jaccard": j,
                "shared_motifs": "; ".join(shared),
            })
    edges = pd.DataFrame(edge_rows)
    if len(edges):
        edges = edges.sort_values("shared_feature_jaccard", ascending=False).reset_index(drop=True)
    else:
        edges = pd.DataFrame(columns=["source", "target", "shared_feature_jaccard", "shared_motifs"])
    return summary, edges




def build_canonical_motif_summary_and_edges(architecture_df, min_count=1):
    """Build canonical motif summary and motif similarity/co-occurrence network edges."""
    if architecture_df.empty or "CANONICAL_MOTIF" not in architecture_df.columns:
        return (
            pd.DataFrame(columns=["CANONICAL_MOTIF", "SCAFFOLD_CLASS", "BIOSYNTHETIC_CLASS", "count", "families", "feature_sets", "representative_compounds"]),
            pd.DataFrame(columns=["source", "target", "shared_context_jaccard", "shared_context"]),
        )

    rows = []
    df_m = architecture_df.copy()
    df_m = df_m[df_m["CANONICAL_MOTIF"].fillna("").astype(str).ne("")]
    df_m = df_m[df_m["CANONICAL_MOTIF"].astype(str).ne("Unresolved")]
    for motif, sub in df_m.groupby("CANONICAL_MOTIF"):
        if len(sub) < int(min_count):
            continue
        scaffolds = sub["SCAFFOLD_CLASS"].astype(str).value_counts().head(3).to_dict() if "SCAFFOLD_CLASS" in sub else {}
        bio_classes = sub["BIOSYNTHETIC_CLASS"].astype(str).value_counts().head(3).to_dict() if "BIOSYNTHETIC_CLASS" in sub else {}
        families = sub["family"].astype(str).value_counts().head(8).to_dict() if "family" in sub else {}
        features = sorted(set(
            token.strip()
            for value in sub.get("FEATURE_SET", pd.Series(dtype=str)).fillna("").astype(str)
            for token in re.split(r"[;+]", value)
            if token.strip()
        ))
        rows.append({
            "CANONICAL_MOTIF": motif,
            "SCAFFOLD_CLASS": "; ".join(f"{k}:{v}" for k, v in scaffolds.items()),
            "BIOSYNTHETIC_CLASS": "; ".join(f"{k}:{v}" for k, v in bio_classes.items()),
            "count": len(sub),
            "families": "; ".join(f"{k}:{v}" for k, v in families.items()),
            "feature_sets": "; ".join(features),
            "representative_compounds": "; ".join(sub["compound_name"].astype(str).head(10).tolist()),
        })
    summary = pd.DataFrame(rows)
    if len(summary):
        summary = summary.sort_values(["count", "CANONICAL_MOTIF"], ascending=[False, True]).reset_index(drop=True)
    else:
        summary = pd.DataFrame(columns=["CANONICAL_MOTIF", "SCAFFOLD_CLASS", "BIOSYNTHETIC_CLASS", "count", "families", "feature_sets", "representative_compounds"])

    context = {}
    for _, row in summary.iterrows():
        motif = row["CANONICAL_MOTIF"]
        tokens = {motif}
        for field in ["SCAFFOLD_CLASS", "BIOSYNTHETIC_CLASS", "feature_sets"]:
            tokens.update(t.split(":", 1)[0].strip() for t in str(row.get(field, "")).split(";") if t.strip())
        context[motif] = tokens
    edge_rows = []
    for a, b in itertools.combinations(summary["CANONICAL_MOTIF"].tolist(), 2):
        ta, tb = context.get(a, set()), context.get(b, set())
        j = len(ta & tb) / len(ta | tb) if (ta | tb) else 0
        if j > 0:
            edge_rows.append({"source": a, "target": b, "shared_context_jaccard": j, "shared_context": "; ".join(sorted(ta & tb))})
    edges = pd.DataFrame(edge_rows)
    if len(edges):
        edges = edges.sort_values("shared_context_jaccard", ascending=False).reset_index(drop=True)
    else:
        edges = pd.DataFrame(columns=["source", "target", "shared_context_jaccard", "shared_context"])
    return summary, edges


def make_canonical_motif_network(motif_summary_df, motif_edges_df, min_edge_similarity=0.01):
    """Plot canonical motif network."""
    if not NETWORKX_AVAILABLE or motif_summary_df.empty:
        return None
    G = nx.Graph()
    for _, row in motif_summary_df.iterrows():
        motif = row["CANONICAL_MOTIF"]
        G.add_node(motif, count=int(row.get("count", 1)), scaffold=row.get("SCAFFOLD_CLASS", ""), biosynthetic_class=row.get("BIOSYNTHETIC_CLASS", ""))
    for _, row in motif_edges_df.iterrows():
        sim = float(row.get("shared_context_jaccard", 0))
        if sim >= float(min_edge_similarity) and row["source"] in G.nodes and row["target"] in G.nodes:
            G.add_edge(row["source"], row["target"], weight=sim, shared=row.get("shared_context", ""))
    pos = nx.spring_layout(G, seed=42, weight="weight") if G.number_of_edges() else nx.spring_layout(G, seed=42)
    edge_x, edge_y = [], []
    for a, b in G.edges():
        x0, y0 = pos[a]; x1, y1 = pos[b]
        edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
    node_x, node_y, text, hover, size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        count = G.nodes[node].get("count", 1)
        node_x.append(x); node_y.append(y); text.append(node)
        hover.append(f"{node}<br>Compounds: {count}<br>Scaffold: {G.nodes[node].get('scaffold','')}<br>Class: {G.nodes[node].get('biosynthetic_class','')}")
        size.append(12 + min(count, 200) * 0.15)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.7), hoverinfo="none", showlegend=False))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", text=text, textposition="top center", hovertext=hover, hoverinfo="text", marker=dict(size=size, line=dict(width=1)), showlegend=False))
    fig.update_layout(title="NPSAA canonical motif network", height=720, xaxis=dict(showgrid=False, zeroline=False, visible=False), yaxis=dict(showgrid=False, zeroline=False, visible=False), margin=dict(l=20, r=20, t=60, b=20))
    return fig

def make_scaffold_network(scaffold_summary_df, scaffold_edges_df, min_edge_similarity=0.01):
    """Plot scaffold ontology network."""
    if not NETWORKX_AVAILABLE or scaffold_summary_df.empty:
        return None
    G = nx.Graph()
    for _, row in scaffold_summary_df.iterrows():
        scaffold = row["SCAFFOLD_CLASS"]
        G.add_node(
            scaffold,
            count=int(row.get("count", 1)),
            label=row.get("ontology_label", scaffold),
            level=row.get("biosynthetic_level", ""),
            motifs=row.get("canonical_motifs", ""),
        )
    for _, row in scaffold_edges_df.iterrows():
        sim = float(row.get("shared_feature_jaccard", 0))
        if sim >= float(min_edge_similarity):
            if row["source"] in G.nodes and row["target"] in G.nodes:
                G.add_edge(row["source"], row["target"], weight=sim, shared=row.get("shared_motifs", ""))

    if G.number_of_edges() == 0:
        # Still show isolated scaffold nodes.
        pos = nx.spring_layout(G, seed=42)
    else:
        pos = nx.spring_layout(G, seed=42, weight="weight")

    edge_x, edge_y = [], []
    for a, b in G.edges():
        x0, y0 = pos[a]
        x1, y1 = pos[b]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]

    node_x, node_y, text, hover, size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        count = G.nodes[node].get("count", 1)
        text.append(node)
        hover.append(
            f"{node}<br>Compounds: {count}<br>Level: {G.nodes[node].get('level','')}<br>Motifs: {G.nodes[node].get('motifs','')}"
        )
        size.append(16 + min(count, 200) * 0.18)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.8), hoverinfo="none", showlegend=False))
    fig.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text", text=text, textposition="top center",
        hovertext=hover, hoverinfo="text", marker=dict(size=size, line=dict(width=1)), showlegend=False
    ))
    fig.update_layout(
        title="NPSAA scaffold ontology network",
        height=720,
        xaxis=dict(showgrid=False, zeroline=False, visible=False),
        yaxis=dict(showgrid=False, zeroline=False, visible=False),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig

st.subheader("2g. Biosynthetic Architecture Analysis")
st.caption(
    "Architecture-level analysis represents each compound as conserved core anchors, informative partners and decorators. "
    "This is more robust than reconstructing full amino-acid sequences from noisy residue SMARTS."
)


# The architecture table is built first so that the ontology explorer can use actual results.
biosynthetic_architecture_df = build_biosynthetic_architecture_table(peptides, residue_cols, motif_cols)


st.subheader("2h. Scaffold Ontology Explorer")
st.caption(
    "Scaffold-first view of the database. This panel summarizes the ontology classes before motif-level architecture assignment."
)

with st.expander("NPSAA ontology and canonical motif library", expanded=False):
    ontology_df = pd.DataFrame([
        {
            "SCAFFOLD_CLASS": scaffold,
            "label": spec.get("label", ""),
            "biosynthetic_level": spec.get("level", ""),
            "core_tokens": "; ".join(spec.get("core", [])),
            "core_features": "; ".join(spec.get("core_features", [])),
            "family_terms": "; ".join(spec.get("families", [])),
            "allowed_partners": "; ".join(sorted(spec.get("allowed_partners", set()))),
            "feature_tokens": "; ".join(sorted(spec.get("feature_tokens", set()))),
            "canonical_motifs": "; ".join(spec.get("canonical_motifs", [])),
            "description": spec.get("description", ""),
        }
        for scaffold, spec in SCAFFOLD_ONTOLOGY.items()
    ])
    st.dataframe(ontology_df, use_container_width=True)
    st.download_button(
        "Download NPSAA scaffold ontology",
        data=csv_bytes(ontology_df),
        file_name="npsaa_scaffold_ontology.csv",
        mime="text/csv",
    )

    motif_library_df = pd.DataFrame([
        {
            "CANONICAL_MOTIF": motif,
            "SCAFFOLD_CLASS": spec.get("scaffold", ""),
            "family_terms": "; ".join(spec.get("families", [])),
        }
        for motif, spec in CANONICAL_MOTIF_LIBRARY.items()
    ])
    st.markdown("**Canonical motif library**")
    st.dataframe(motif_library_df, use_container_width=True)
    st.download_button(
        "Download canonical motif library",
        data=csv_bytes(motif_library_df),
        file_name="npsaa_canonical_motif_library.csv",
        mime="text/csv",
    )

scaffold_summary_df, scaffold_edges_df = build_scaffold_summary_and_edges(
    biosynthetic_architecture_df,
    min_count=1,
)

if len(scaffold_summary_df):
    col_scaf1, col_scaf2 = st.columns([2, 1])
    with col_scaf1:
        st.dataframe(scaffold_summary_df, use_container_width=True)
    with col_scaf2:
        selected_scaffold_for_explorer = st.selectbox(
            "Inspect scaffold class",
            scaffold_summary_df["SCAFFOLD_CLASS"].tolist(),
            index=0,
        )
        spec = SCAFFOLD_ONTOLOGY.get(selected_scaffold_for_explorer, {})
        st.markdown(f"**{spec.get('label', selected_scaffold_for_explorer)}**")
        st.write(spec.get("description", ""))
        st.write("**Level:**", spec.get("level", ""))
        st.write("**Core:**", ", ".join(spec.get("core", [])) or "—")
        st.write("**Families:**", ", ".join(spec.get("families", [])) or "—")
        st.write("**Allowed partners:**", ", ".join(sorted(spec.get("allowed_partners", set()))) or "—")
        st.write("**Feature tokens:**", ", ".join(sorted(spec.get("feature_tokens", set()))) or "—")

    selected_scaffold_rows = biosynthetic_architecture_df.loc[
        biosynthetic_architecture_df["SCAFFOLD_CLASS"].astype(str) == str(selected_scaffold_for_explorer)
    ].copy()
    if len(selected_scaffold_rows):
        st.markdown("**Representative compounds for selected scaffold**")
        rep_cols = [
            "compound_name", "family", "BIOSYNTHETIC_CLASS", "SCAFFOLD_CLASS", "CANONICAL_MOTIF",
            "FEATURE_SET", "BIOSYNTHETIC_ARCHITECTURE_CLASS", "BIOSYNTHETIC_ARCHITECTURE_CONFIDENCE", "SMILES"
        ]
        rep_cols = [c for c in rep_cols if c in selected_scaffold_rows.columns]
        st.dataframe(selected_scaffold_rows[rep_cols].head(300), use_container_width=True)

    scaf_dl1, scaf_dl2, scaf_dl3 = st.columns(3)
    with scaf_dl1:
        st.download_button(
            "Download scaffold summary",
            data=csv_bytes(scaffold_summary_df),
            file_name="npsaa_scaffold_summary.csv",
            mime="text/csv",
        )
    with scaf_dl2:
        st.download_button(
            "Download scaffold network edges",
            data=csv_bytes(scaffold_edges_df),
            file_name="npsaa_scaffold_network_edges.csv",
            mime="text/csv",
        )
    with scaf_dl3:
        st.download_button(
            "Download selected scaffold compounds",
            data=csv_bytes(selected_scaffold_rows.drop(columns=["mol"], errors="ignore")),
            file_name=f"npsaa_{selected_scaffold_for_explorer}_compounds.csv",
            mime="text/csv",
        )

    if NETWORKX_AVAILABLE:
        fig_scaffold_network = make_scaffold_network(scaffold_summary_df, scaffold_edges_df)
        if fig_scaffold_network is not None:
            st.plotly_chart(fig_scaffold_network, use_container_width=True)
else:
    st.info("No scaffold classes were detected.")



st.subheader("2i. Canonical Motif Explorer")
st.caption("Motif-first summary. This panel treats CANONICAL_MOTIF as the central NPSAA entity after biosynthetic class and scaffold assignment.")

motif_summary_df, motif_edges_df = build_canonical_motif_summary_and_edges(
    biosynthetic_architecture_df,
    min_count=1,
)

if len(motif_summary_df):
    st.dataframe(motif_summary_df, use_container_width=True)
    mot_dl1, mot_dl2 = st.columns(2)
    with mot_dl1:
        st.download_button(
            "Download canonical motif summary",
            data=csv_bytes(motif_summary_df),
            file_name="npsaa_canonical_motif_summary.csv",
            mime="text/csv",
        )
    with mot_dl2:
        st.download_button(
            "Download canonical motif network edges",
            data=csv_bytes(motif_edges_df),
            file_name="npsaa_canonical_motif_network_edges.csv",
            mime="text/csv",
        )
    if NETWORKX_AVAILABLE:
        fig_motif_network = make_canonical_motif_network(motif_summary_df, motif_edges_df)
        if fig_motif_network is not None:
            st.plotly_chart(fig_motif_network, use_container_width=True)
else:
    st.info("No canonical motifs were detected.")

# Merge architecture fields into peptides so downstream search and clustering become architecture-first.
biosyn_cols_to_merge = biosynthetic_architecture_df[[
    "compound_index", "BIOSYNTHETIC_CLASS", "NPSAA_CONFIDENCE_LEVEL", "SCAFFOLD_CORE_STATUS", "BIOSYNTHETIC_SCORE", "SCAFFOLD_CLASS", "SCAFFOLD_EVIDENCE", "SCAFFOLD_FEATURES", "FEATURE_SET", "CANONICAL_MOTIF", "BIOSYNTHETIC_SCAFFOLD", "BIOSYNTHETIC_CORE", "BIOSYNTHETIC_PARTNERS", "BIOSYNTHETIC_DECORATORS",
    "BIOSYNTHETIC_ARCHITECTURE", "BIOSYNTHETIC_CORE_PARTNERS", "BIOSYNTHETIC_ARCHITECTURE_KEY",
    "BIOSYNTHETIC_SCAFFOLD_FAMILY", "BIOSYNTHETIC_ARCHITECTURE_CLASS", "BIOSYNTHETIC_ARCHITECTURE_CONFIDENCE",
    "BIOSYNTHETIC_CORE_EVIDENCE", "BIOSYNTHETIC_PARTNER_EVIDENCE",
    "BIOSYNTHETIC_WEIGHTED_SIGNATURE", "biosynthetic_token_count", "biosynthetic_core_count",
    "biosynthetic_partner_count", "biosynthetic_decorator_count"
]].copy()

peptides = peptides.drop(
    columns=[
        "BIOSYNTHETIC_CLASS", "NPSAA_CONFIDENCE_LEVEL", "SCAFFOLD_CORE_STATUS", "BIOSYNTHETIC_SCORE", "SCAFFOLD_CLASS", "SCAFFOLD_EVIDENCE", "SCAFFOLD_FEATURES", "FEATURE_SET", "CANONICAL_MOTIF", "BIOSYNTHETIC_SCAFFOLD", "BIOSYNTHETIC_CORE", "BIOSYNTHETIC_PARTNERS", "BIOSYNTHETIC_DECORATORS",
        "BIOSYNTHETIC_ARCHITECTURE", "BIOSYNTHETIC_CORE_PARTNERS", "BIOSYNTHETIC_ARCHITECTURE_KEY",
        "BIOSYNTHETIC_SCAFFOLD_FAMILY", "BIOSYNTHETIC_ARCHITECTURE_CLASS", "BIOSYNTHETIC_ARCHITECTURE_CONFIDENCE",
        "BIOSYNTHETIC_CORE_EVIDENCE", "BIOSYNTHETIC_PARTNER_EVIDENCE",
        "BIOSYNTHETIC_WEIGHTED_SIGNATURE", "biosynthetic_token_count", "biosynthetic_core_count",
        "biosynthetic_partner_count", "biosynthetic_decorator_count"
    ],
    errors="ignore"
).merge(
    biosyn_cols_to_merge,
    left_index=True,
    right_on="compound_index",
    how="left"
).set_index("compound_index", drop=True)

for col in [
    "BIOSYNTHETIC_CLASS", "NPSAA_CONFIDENCE_LEVEL", "SCAFFOLD_CORE_STATUS", "SCAFFOLD_CLASS", "SCAFFOLD_EVIDENCE", "SCAFFOLD_FEATURES", "FEATURE_SET", "CANONICAL_MOTIF", "BIOSYNTHETIC_SCAFFOLD", "BIOSYNTHETIC_CORE", "BIOSYNTHETIC_PARTNERS", "BIOSYNTHETIC_DECORATORS",
    "BIOSYNTHETIC_ARCHITECTURE", "BIOSYNTHETIC_CORE_PARTNERS", "BIOSYNTHETIC_ARCHITECTURE_KEY",
    "BIOSYNTHETIC_SCAFFOLD_FAMILY", "BIOSYNTHETIC_ARCHITECTURE_CLASS", "BIOSYNTHETIC_ARCHITECTURE_CONFIDENCE",
    "BIOSYNTHETIC_CORE_EVIDENCE", "BIOSYNTHETIC_PARTNER_EVIDENCE", "BIOSYNTHETIC_WEIGHTED_SIGNATURE"
]:
    peptides[col] = peptides[col].fillna("")

for col in ["BIOSYNTHETIC_SCORE", "biosynthetic_token_count", "biosynthetic_core_count", "biosynthetic_partner_count", "biosynthetic_decorator_count"]:
    peptides[col] = peptides[col].fillna(0).astype(int)

# Overwrite NPPSA signature with the architecture-first representation.
peptides["NPPSA_signature"] = peptides.apply(choose_biosynthetic_nppsa_signature, axis=1)

baa_col1, baa_col2, baa_col3, baa_col4 = st.columns(4)
baa_col1.metric("Compounds with core anchor", int((peptides["biosynthetic_core_count"] > 0).sum()))
baa_col2.metric("Compounds with partners", int((peptides["biosynthetic_partner_count"] > 0).sum()))
baa_col3.metric("Compounds with decorators", int((peptides["biosynthetic_decorator_count"] > 0).sum()))
baa_col4.metric("Biosynthetic classes", int(peptides["BIOSYNTHETIC_CLASS"].nunique()) if "BIOSYNTHETIC_CLASS" in peptides.columns else 0)
st.metric("Scaffold classes", int(peptides["SCAFFOLD_CLASS"].nunique()) if "SCAFFOLD_CLASS" in peptides.columns else 0)
st.metric("High-confidence scaffold calls", int(peptides["BIOSYNTHETIC_ARCHITECTURE_CONFIDENCE"].astype(str).str.contains("high-confidence", na=False).sum()))

conf_counts = peptides["NPSAA_CONFIDENCE_LEVEL"].value_counts().reset_index()
conf_counts.columns = ["NPSAA_CONFIDENCE_LEVEL", "count"]
st.markdown("**NPSAA confidence audit**")
st.dataframe(conf_counts, use_container_width=True)
st.download_button(
    "Download NPSAA confidence audit",
    data=csv_bytes(conf_counts),
    file_name="npsaa_confidence_audit.csv",
    mime="text/csv",
)

architecture_show_cols = [
    "compound_name", "family", "BIOSYNTHETIC_CLASS", "NPSAA_CONFIDENCE_LEVEL", "SCAFFOLD_CORE_STATUS", "BIOSYNTHETIC_SCORE", "SCAFFOLD_CLASS", "CANONICAL_MOTIF", "SCAFFOLD_EVIDENCE", "SCAFFOLD_FEATURES", "FEATURE_SET",
    "BIOSYNTHETIC_ARCHITECTURE_CLASS", "BIOSYNTHETIC_ARCHITECTURE_CONFIDENCE",
    "BIOSYNTHETIC_ARCHITECTURE", "BIOSYNTHETIC_ARCHITECTURE_KEY", "BIOSYNTHETIC_SCAFFOLD_FAMILY",
    "BIOSYNTHETIC_CORE", "BIOSYNTHETIC_PARTNERS", "BIOSYNTHETIC_DECORATORS",
    "BIOSYNTHETIC_CORE_EVIDENCE", "BIOSYNTHETIC_PARTNER_EVIDENCE",
    "MODULE_signature", "cyanopeptide_signature_hits", "SMILES"
]
architecture_show_cols = [c for c in architecture_show_cols if c in peptides.columns]
st.dataframe(peptides[architecture_show_cols].head(500), use_container_width=True)

baa_s1, baa_s2, baa_s3 = st.columns(3)
with baa_s1:
    architecture_min_count = st.slider(
        "Minimum architecture support",
        min_value=1,
        max_value=50,
        value=2,
        step=1,
    )
with baa_s2:
    architecture_include_decorators = st.checkbox(
        "Include decorators in architecture network",
        value=False,
        help="Recommended off. Off compares core+partner architectures; on also considers NMe/Sugar/Halogen/Sulfate."
    )
with baa_s3:
    architecture_min_jaccard = st.slider(
        "Architecture network Jaccard threshold",
        min_value=0.10,
        max_value=1.00,
        value=0.34,
        step=0.01,
    )

architecture_summary_df, architecture_edges_df = build_architecture_summary_and_edges(
    biosynthetic_architecture_df,
    min_count=architecture_min_count,
    include_decorators=architecture_include_decorators,
    min_jaccard=architecture_min_jaccard,
)

st.markdown("**Biosynthetic architecture summary**")
st.dataframe(architecture_summary_df.head(200), use_container_width=True)

baa_dl1, baa_dl2, baa_dl3 = st.columns(3)
with baa_dl1:
    st.download_button(
        "Download biosynthetic architecture table",
        data=csv_bytes(peptides[architecture_show_cols]),
        file_name="biosynthetic_architecture_table.csv",
        mime="text/csv",
    )
with baa_dl2:
    st.download_button(
        "Download architecture summary",
        data=csv_bytes(architecture_summary_df),
        file_name="biosynthetic_architecture_summary.csv",
        mime="text/csv",
    )
with baa_dl3:
    st.download_button(
        "Download architecture network edges",
        data=csv_bytes(architecture_edges_df),
        file_name="biosynthetic_architecture_network_edges.csv",
        mime="text/csv",
    )

if NETWORKX_AVAILABLE and len(architecture_summary_df):
    fig_architecture_network = make_architecture_network(
        architecture_summary_df,
        architecture_edges_df,
        min_edge_similarity=architecture_min_jaccard,
    )
    if fig_architecture_network is not None:
        st.plotly_chart(fig_architecture_network, use_container_width=True)

######## 
st.subheader("2e. Sequence Explorer")
st.caption(
    "Explore compounds using residues, motifs, micropeptin signatures, "
    "diagnostic MS/MS-inspired fragments or compound families already detected by the app."
)

search_mode = st.radio(
    "Search mode",
    [
        "Biosynthetic architecture",
        "Residues / motifs",
        "Cyanopeptide signature",
        "Auto-built recurring signature",
        "Diagnostic MS/MS fragment",
        "Family"
    ],
    horizontal=True
)

result_cols = [
    "compound_name",
    "family",
    "BIOSYNTHETIC_CLASS",
    "NPSAA_CONFIDENCE_LEVEL",
    "SCAFFOLD_CORE_STATUS",
    "BIOSYNTHETIC_SCORE",
    "SCAFFOLD_CLASS",
    "CANONICAL_MOTIF",
    "SCAFFOLD_EVIDENCE",
    "SCAFFOLD_FEATURES",
    "FEATURE_SET",
    "BIOSYNTHETIC_ARCHITECTURE_CLASS",
    "BIOSYNTHETIC_ARCHITECTURE_CONFIDENCE",
    "BIOSYNTHETIC_ARCHITECTURE",
    "BIOSYNTHETIC_ARCHITECTURE_KEY",
    "BIOSYNTHETIC_CORE",
    "BIOSYNTHETIC_PARTNERS",
    "BIOSYNTHETIC_DECORATORS",
    "NPPSA_signature",
    "cyanopeptide_signature_hits",
    "SMILES"
]
result_cols = [c for c in result_cols if c in peptides.columns]

hits = pd.DataFrame()


# ==========================================================
# BIOSYNTHETIC ARCHITECTURE SEARCH
# ==========================================================

if search_mode == "Biosynthetic architecture":

    available_classes = sorted(
        peptides["BIOSYNTHETIC_ARCHITECTURE_CLASS"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    ) if "BIOSYNTHETIC_ARCHITECTURE_CLASS" in peptides.columns else []

    available_biosynthetic_classes = sorted(
        peptides["BIOSYNTHETIC_CLASS"].dropna().astype(str).unique().tolist()
    ) if "BIOSYNTHETIC_CLASS" in peptides.columns else []

    available_confidence_levels = sorted(
        peptides["NPSAA_CONFIDENCE_LEVEL"].dropna().astype(str).unique().tolist()
    ) if "NPSAA_CONFIDENCE_LEVEL" in peptides.columns else []

    available_scaffolds = sorted(
        peptides["SCAFFOLD_CLASS"].dropna().astype(str).unique().tolist()
    ) if "SCAFFOLD_CLASS" in peptides.columns else []

    available_motifs = sorted(
        peptides["CANONICAL_MOTIF"].dropna().astype(str).unique().tolist()
    ) if "CANONICAL_MOTIF" in peptides.columns else []

    col_arch0, col_arch1, col_arch2, col_arch3 = st.columns(4)
    with col_arch0:
        selected_biosynthetic_classes = st.multiselect(
            "Select biosynthetic classes",
            available_biosynthetic_classes,
            default=[]
        )
        selected_confidence_levels = st.multiselect(
            "Select confidence levels",
            available_confidence_levels,
            default=[]
        )
        selected_scaffolds = st.multiselect(
            "Select scaffold classes",
            available_scaffolds,
            default=[]
        )
        selected_canonical_motifs = st.multiselect(
            "Select canonical motifs",
            available_motifs,
            default=[]
        )
    with col_arch1:
        selected_arch_classes = st.multiselect(
            "Select architecture classes",
            available_classes,
            default=[]
        )
    with col_arch2:
        selected_arch_cores = st.multiselect(
            "Select core anchors",
            sorted(BIOSYNTHETIC_CORE_TOKENS),
            default=[]
        )
    with col_arch3:
        selected_arch_partners = st.multiselect(
            "Select partners/decorators/features",
            sorted(BIOSYNTHETIC_PARTNER_TOKENS | BIOSYNTHETIC_DECORATOR_TOKENS | SCAFFOLD_FEATURE_TOKENS),
            default=[]
        )

    if selected_biosynthetic_classes or selected_confidence_levels or selected_scaffolds or selected_canonical_motifs or selected_arch_classes or selected_arch_cores or selected_arch_partners:
        mask = pd.Series(True, index=peptides.index)
        if selected_biosynthetic_classes:
            mask &= peptides["BIOSYNTHETIC_CLASS"].isin(selected_biosynthetic_classes)
        if selected_confidence_levels:
            mask &= peptides["NPSAA_CONFIDENCE_LEVEL"].isin(selected_confidence_levels)
        if selected_scaffolds:
            mask &= peptides["SCAFFOLD_CLASS"].isin(selected_scaffolds)
        if selected_canonical_motifs:
            mask &= peptides["CANONICAL_MOTIF"].isin(selected_canonical_motifs)
        if selected_arch_classes:
            mask &= peptides["BIOSYNTHETIC_ARCHITECTURE_CLASS"].isin(selected_arch_classes)
        for core in selected_arch_cores:
            mask &= peptides["BIOSYNTHETIC_CORE"].astype(str).str.contains(core, case=False, na=False)
        for token in selected_arch_partners:
            mask &= (
                peptides["BIOSYNTHETIC_PARTNERS"].astype(str).str.contains(token, case=False, na=False)
                | peptides["BIOSYNTHETIC_DECORATORS"].astype(str).str.contains(token, case=False, na=False)
                | peptides["SCAFFOLD_FEATURES"].astype(str).str.contains(token, case=False, na=False)
            )
        hits = peptides.loc[mask, result_cols].copy()

# ==========================================================
# RESIDUE / MOTIF SEARCH
# ==========================================================

elif search_mode == "Residues / motifs":

    available_residues = sorted([
        col.replace("res_", "")
        for col in residue_cols
        if peptides[col].sum() > 0
    ])

    available_motifs = sorted([
        col.replace("motif_", "")
        for col in motif_cols
        if peptides[col].sum() > 0
    ])

    col_a, col_b = st.columns(2)

    with col_a:
        selected_residues = st.multiselect(
            "Select residues",
            available_residues,
            default=[]
        )

    with col_b:
        selected_motifs = st.multiselect(
            "Select motifs",
            available_motifs,
            default=[]
        )

    if selected_residues or selected_motifs:

        mask = pd.Series(True, index=peptides.index)

        for residue in selected_residues:
            mask &= peptides[f"res_{residue}"] > 0

        for motif in selected_motifs:
            mask &= peptides[f"motif_{motif}"] > 0

        hits = peptides.loc[mask, result_cols].copy()

# ==========================================================
# MICROPEPTIN SIGNATURE SEARCH
# ==========================================================

elif search_mode == "Cyanopeptide signature":

    selected_group = st.selectbox(
        "Select cyanopeptide class",
        list(CYANOPEPTIDE_SIGNATURE_GROUPS.keys())
    )

    selected_tiers = st.multiselect(
        "Select confidence tier",
        ["high-confidence structural", "name-supported", "exploratory"],
        default=["high-confidence structural", "name-supported"]
    )

    group_signatures = sorted([
        name for name, spec in CYANOPEPTIDE_SIGNATURE_GROUPS[selected_group].items()
        if spec.get("tier", "exploratory") in selected_tiers
    ])

    available_signatures = [
        sig for sig in group_signatures
        if f"sig_{sig}" in peptides.columns and peptides[f"sig_{sig}"].sum() > 0
    ]

    selected_signatures = st.multiselect(
        "Select signatures",
        available_signatures,
        default=[]
    )

    if selected_signatures:

        mask = pd.Series(True, index=peptides.index)

        for signature in selected_signatures:
            mask &= peptides[f"sig_{signature}"] > 0

        hits = peptides.loc[mask, result_cols].copy()


# ==========================================================
# AUTO-BUILT RECURRING SIGNATURE SEARCH
# ==========================================================

elif search_mode == "Auto-built recurring signature":

    if len(auto_signature_df):

        auto_options = (
            auto_signature_df
            .head(auto_top_n)
            .assign(label=lambda x: x["auto_signature"] + "  (" + x["count"].astype(str) + " compounds)")
        )

        selected_auto_labels = st.multiselect(
            "Select auto-built recurring signatures",
            auto_options["label"].tolist(),
            default=[]
        )

        if selected_auto_labels:

            selected_auto_signatures = auto_options.loc[
                auto_options["label"].isin(selected_auto_labels),
                ["auto_signature", "tokens"]
            ]

            # AND logic across selected signatures:
            # a compound must contain all tokens from all selected signatures.
            selected_tokens = sorted(set(
                token
                for token_list in selected_auto_signatures["tokens"]
                for token in token_list
            ))

            hits = filter_by_auto_signature(
                peptides,
                selected_tokens,
                residue_cols,
                motif_cols
            )[result_cols].copy()

            st.caption(
                "Selected token set: " + " + ".join(selected_tokens)
            )

    else:
        st.info("No auto-built signatures are available with the current settings.")


# ==========================================================
# DIAGNOSTIC FRAGMENT SEARCH
# ==========================================================

elif search_mode == "Diagnostic MS/MS fragment":

    DIAGNOSTIC_MAP = {
        "m/z 404 — Ahp-Phe-NMePhe core": "Ahp-Phe-NMePhe_core_like",
        "m/z 243 — Ahp-Phe core": "Ahp-Phe_core_like",
        "m/z 282 — BTA-Gln-Thr-like": "BTA-Gln-Thr_like",
        "m/z 209/370 — Leu/Ile-containing micropeptin-like": "Ahp-Phe-NMePhe_core_like"
    }

    selected_fragments = st.multiselect(
        "Select diagnostic MS/MS-inspired fragments",
        list(DIAGNOSTIC_MAP.keys()),
        default=[]
    )

    if selected_fragments:

        mask = pd.Series(True, index=peptides.index)

        for fragment in selected_fragments:
            signature = DIAGNOSTIC_MAP[fragment]
            col = f"sig_{signature}"

            if col in peptides.columns:
                mask &= peptides[col] > 0
            else:
                mask &= False

        hits = peptides.loc[mask, result_cols].copy()

# ==========================================================
# FAMILY SEARCH
# ==========================================================

elif search_mode == "Family":

    available_families = sorted(
        peptides["family"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_families = st.multiselect(
        "Select families",
        available_families,
        default=[]
    )

    if selected_families:

        hits = peptides.loc[
            peptides["family"].isin(selected_families),
            result_cols
        ].copy()

# ==========================================================
# RESULTS
# ==========================================================

st.metric("Matching compounds", len(hits))

if len(hits):

    st.dataframe(
        hits,
        use_container_width=True
    )

    st.download_button(
        "Download search results",
        data=csv_bytes(hits),
        file_name="sequence_explorer_results.csv",
        mime="text/csv"
    )

else:
    st.info("Select one or more options above to explore matching compounds.")


# Rebuild plot dataframe after module discovery so clustering can use MODULE/NPPSA signatures.
plot_df = peptides.head(max_items).copy()
labels = plot_df["compound_name"].astype(str).tolist()

st.subheader("3. Scaffold-first NPSAA clustering")
st.caption(
    "The primary clustering signature is now scaffold-first. It prioritizes SCAFFOLD_CLASS and CANONICAL_MOTIF through BIOSYNTHETIC_WEIGHTED_SIGNATURE, "
    "then curated cyanopeptide signatures, and only uses the full residue-derived AA_signature as a low-weight fallback. "
    "This reduces the influence of noisy residue SMARTS while preserving information for unresolved compounds."
)

seq_sim = compute_sequence_similarity(plot_df["NPPSA_signature"].tolist())
seq_pairs = matrix_to_long_table(seq_sim, labels, "npsaa_signature_jaccard")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(make_heatmap(seq_sim, labels, "Scaffold-first NPSAA similarity heatmap"), use_container_width=True)
with col2:
    fig_dendro_seq, Z_seq = make_dendrogram_figure(seq_sim, labels, "Scaffold-first NPSAA dendrogram")
    st.plotly_chart(fig_dendro_seq, use_container_width=True)

if NETWORKX_AVAILABLE:
    G_seq, fig_net_seq = make_network(seq_sim, labels, plot_df, sequence_network_threshold)
    st.plotly_chart(fig_net_seq, use_container_width=True)
else:
    st.warning("NetworkX is not installed. Network plots are disabled.")

with st.expander("Top biosynthetic architecture NPPSA similarities"):
    st.dataframe(seq_pairs.head(200), use_container_width=True)
    st.download_button(
        "Download NPPSA signature similarity pairs",
        data=csv_bytes(seq_pairs),
        file_name="npsaa_signature_similarity_pairs.csv",
        mime="text/csv"
    )

st.subheader("4. Structural clustering: Morgan/Tanimoto")
struct_sim = compute_morgan_similarity(plot_df["SMILES"].tolist(), radius=morgan_radius, n_bits=morgan_bits)
struct_pairs = matrix_to_long_table(struct_sim, labels, "morgan_tanimoto")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(make_heatmap(struct_sim, labels, "Morgan/Tanimoto structural similarity heatmap"), use_container_width=True)
with col2:
    fig_dendro_struct, Z_struct = make_dendrogram_figure(struct_sim, labels, "Morgan/Tanimoto dendrogram")
    st.plotly_chart(fig_dendro_struct, use_container_width=True)

if NETWORKX_AVAILABLE:
    G_struct, fig_net_struct = make_network(struct_sim, labels, plot_df, structural_network_threshold)
    st.plotly_chart(fig_net_struct, use_container_width=True)

with st.expander("Top structural similarities"):
    st.dataframe(struct_pairs.head(200), use_container_width=True)
    st.download_button(
        "Download Morgan/Tanimoto similarity pairs",
        data=csv_bytes(struct_pairs),
        file_name="morgan_tanimoto_similarity_pairs.csv",
        mime="text/csv"
    )

st.subheader("5. Sequence vs structure comparison")
comparison = seq_pairs.merge(struct_pairs, on=["source", "target"], how="inner")

if not comparison.empty:
    fig_compare = px.scatter(
        comparison,
        x="npsaa_signature_jaccard",
        y="morgan_tanimoto",
        hover_data=["source", "target"],
        title="Comparison between scaffold-first NPSAA similarity and structural similarity"
    )
    fig_compare.update_layout(height=600)
    st.plotly_chart(fig_compare, use_container_width=True)

    corr = comparison[["npsaa_signature_jaccard", "morgan_tanimoto"]].corr().iloc[0, 1]
    st.metric("Correlation: scaffold-first NPSAA vs Morgan/Tanimoto", f"{corr:.3f}")

    st.download_button(
        "Download sequence vs structure comparison",
        data=csv_bytes(comparison),
        file_name="npsaa_signature_vs_structure_similarity.csv",
        mime="text/csv"
    )


st.subheader("6. Download all results")
st.caption(
    "Baixa todos os resultados produzidos pelo app em um único arquivo ZIP. "
    "Todos os CSVs dentro do ZIP usam separador ';'."
)

all_result_tables = {
    "01_peptide_like_sequences.csv": peptides.drop(columns=["mol"], errors="ignore"),
    "02_cyanopeptide_signature_hits.csv": micro_hits_df,
    "03_signature_dictionary.csv": signature_dictionary_df,
    "04_auto_built_cyanopeptide_signatures.csv": auto_signature_df.drop(columns=["tokens"], errors="ignore"),
    "05_module_summary.csv": module_summary_df,
    "06_module_architecture_table.csv": module_architecture_df.drop(columns=["compound_index"], errors="ignore"),
    "07_compound_module_table.csv": compound_module_df,
    "08_module_network_edges.csv": module_edges_df,
    "09_npsaa_scaffold_ontology.csv": ontology_df,
    "10_npsaa_canonical_motif_library.csv": motif_library_df,
    "11_npsaa_scaffold_summary.csv": scaffold_summary_df,
    "12_npsaa_scaffold_network_edges.csv": scaffold_edges_df,
    "13_npsaa_canonical_motif_summary.csv": motif_summary_df,
    "14_npsaa_canonical_motif_network_edges.csv": motif_edges_df,
    "15_biosynthetic_architecture_table.csv": peptides[architecture_show_cols],
    "16_biosynthetic_architecture_summary.csv": architecture_summary_df,
    "17_biosynthetic_architecture_network_edges.csv": architecture_edges_df,
    "18_sequence_explorer_results.csv": hits,
    "19_npsaa_signature_similarity_pairs.csv": seq_pairs,
    "20_morgan_tanimoto_similarity_pairs.csv": struct_pairs,
    "21_npsaa_signature_vs_structure_similarity.csv": comparison,
}

st.download_button(
    "⬇️ Download all NPSAA results (.zip)",
    data=zip_results_bytes(all_result_tables),
    file_name="npsaa_all_results.zip",
    mime="application/zip",
    use_container_width=True,
)

st.subheader("6. Interpretation notes")
st.markdown(
    """
**Important interpretation:**

- The main NPSAA clustering now prioritizes **BIOSYNTHETIC_CLASS**, **SCAFFOLD_CLASS** and **CANONICAL_MOTIF**, not AA_signature or MODULE_signature.
- **NPSAA_CONFIDENCE_LEVEL** separates structural assignments from name-supported assignments: A = scaffold plus motif evidence, B = scaffold/core evidence, C = name/token-supported, D = unresolved.
- **SCAFFOLD_CORE_STATUS** flags cases where a name-supported scaffold is missing the expected structural core and should be interpreted as possible-like rather than fully confirmed.
- The scaffold ontology is evaluated before architecture assignment. High-confidence structural signatures resolve Ahp, Choi and Adda spaces; family/name rules resolve RiPP, PKS-NRPS, lipopeptide, glycolipopeptide, macrocyclic and linear peptide spaces.
- CANONICAL_MOTIF is treated as a central NPSAA entity, summarizing interpretable biosynthetic motifs such as Ahp-Phe, Ahp-Tyr, Choi-Arg, Adda-Glu, RiPP heterocycles, Microginin-Ahda/Ahdo and PKS-NRPS subclasses.
- Unknown scaffold compounds no longer receive biosynthetic partners or decorators. Raw residue/motif hits remain available only as low-weight exploratory evidence.
- Decorators are handled separately as NMe, Sugar, Halogen and Sulfate and do not define local modules or scaffold identity.
- Residue SMARTS are still retained and exported, but they are not the main representation because they can overcall generic peptide fragments.
- Morgan/Tanimoto clustering captures full structural similarity; disagreement between NPSAA and Morgan/Tanimoto can reveal shared biosynthetic logic with divergent decorations, or structural analogues with different scaffold classes.
"""
)
