# app.py
# Peptide-oriented clustering app for cyanobacterial metabolite databases
# Author: Ricardo M. Borges workflow draft

import io
import re
import itertools
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
    page_title="Cyano Peptide Clustering",
    page_icon="🧬",
    layout="wide"
)

st.title("🧬 Cyano Peptide Clustering")
st.caption(
    "Clustering de metabólitos peptídicos de cianobactérias baseado em sequência simplificada "
    "extraída do SMILES e comparação com clustering estrutural Morgan/Tanimoto."
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
        "microginin", "microviridin", "aerucyclamide", "patellamide"
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
    "Ahp_like": "N1C(=O)CC(O)CCC1",
    "Choi_like": "C1CCC2N(C1)CCCC2",
    "Adda_like": "C=C(C)C=CC=C",
    "NMe_amide": "C(=O)N(C)",
    "Guanidino": "NC(=N)N",
    "Sulfate": "OS(=O)(=O)O",
    "Halogenated": "[F,Cl,Br,I]",
    "Sugar_like": "O[C@H]1O[C@H](CO)[C@H](O)[C@@H](O)[C@H]1O",
}

# Micropeptin/cyanopeptolin signatures inspired by diagnostic MS/MS fragments
# from the Bertin workflow. These are intentionally "signature-like": they test
# whether the molecule contains the required residue/motif substructures, not
# whether the exact MS/MS ion will be formed.
MICROPEPTIN_SIGNATURES = {
    "Ahp-Phe-NMePhe_core_like": {
        "required": {"motif_Ahp_like": 1, "res_Phe": 2, "motif_NMe_amide": 1},
        "diagnostic_msms": "[Ahp-Phe-N-MePhe+H-H2O]+, m/z 404",
        "interpretation": "Core fragment typical of Phe-containing micropeptins/cyanopeptolins."
    },
    "Ahp-Phe_core_like": {
        "required": {"motif_Ahp_like": 1, "res_Phe": 1},
        "diagnostic_msms": "[Ahp-Phe+H-H2O]+, m/z 243",
        "interpretation": "Ahp-Phe fragment."
    },
    "BTA-Gln-Thr_like": {
        "required": {"res_Gln": 1, "res_Thr": 1},
        "any_of": {"BTA_or_short_acyl": ["CCCC(=O)", "CCCC(=O)N"]},
        "diagnostic_msms": "BTA-Gln-Thr-related ions, e.g. m/z 282",
        "interpretation": "Putative butyric acid starter plus Gln-Thr."
    },
    "BTA-Gln-Thr-Val-NMePhe_like": {
        "required": {"res_Gln": 1, "res_Thr": 1, "res_Val": 1, "res_Phe": 1, "motif_NMe_amide": 1},
        "any_of": {"BTA_or_short_acyl": ["CCCC(=O)", "CCCC(=O)N"]},
        "diagnostic_msms": "[BTA-Gln-Thr-Val-N-MePhe+H]+",
        "interpretation": "Side-chain/starter fragment reported for micropeptins."
    },
    "Met-Ahp-Phe-NMePhe_like": {
        "required": {"res_Met": 1, "motif_Ahp_like": 1, "res_Phe": 2, "motif_NMe_amide": 1},
        "diagnostic_msms": "[Met-Ahp-Phe-NMePhe+H-H2O]+",
        "interpretation": "Met-containing Ahp-Phe-NMePhe fragment."
    },
    "Met-Ahp-Phe_like": {
        "required": {"res_Met": 1, "motif_Ahp_like": 1, "res_Phe": 1},
        "diagnostic_msms": "[Met-Ahp-Phe+H-H2O]+",
        "interpretation": "Met-Ahp-Phe fragment."
    },
    "BTA-Gln-Thr-Met-Ahp-Phe-NMePhe-Val_like": {
        "required": {"res_Gln": 1, "res_Thr": 1, "res_Met": 1, "motif_Ahp_like": 1, "res_Phe": 2, "motif_NMe_amide": 1, "res_Val": 1},
        "any_of": {"BTA_or_short_acyl": ["CCCC(=O)", "CCCC(=O)N"]},
        "diagnostic_msms": "BTA-Gln-Thr-Met-Ahp-Phe-NMePhe-Val",
        "interpretation": "Micropeptin 950-like sequence signature."
    },
    "BTA-Gln-Thr-Trp-Ahp-Phe-NMePhe-Val_like": {
        "required": {"res_Gln": 1, "res_Thr": 1, "res_Trp": 1, "motif_Ahp_like": 1, "res_Phe": 2, "motif_NMe_amide": 1, "res_Val": 1},
        "any_of": {"BTA_or_short_acyl": ["CCCC(=O)", "CCCC(=O)N"]},
        "diagnostic_msms": "BTA-Gln-Thr-Trp-Ahp-Phe-NMePhe-Val",
        "interpretation": "Micropeptin 1005-like sequence signature."
    },
    "BTA-Gln-Thr-Hphe-Ahp-Phe-NMePhe-Val_like": {
        "required": {"res_Gln": 1, "res_Thr": 1, "motif_Ahp_like": 1, "res_Phe": 2, "motif_NMe_amide": 1, "res_Val": 1},
        "any_of": {"Hphe_like": ["N[C@H](CCC1=CC=CC=C1)C(=O)", "N[C@H](CCCc1ccccc1)C(=O)"],
                   "BTA_or_short_acyl": ["CCCC(=O)", "CCCC(=O)N"]},
        "diagnostic_msms": "BTA-Gln-Thr-Hphe-Ahp-Phe-NMePhe-Val",
        "interpretation": "Homologated phenylalanine-containing micropeptin-like signature."
    },
    "BTA-Gln-Thr-Hleu_or_Hile-Ahp-Phe-NMePhe-Val_like": {
        "required": {"res_Gln": 1, "res_Thr": 1, "motif_Ahp_like": 1, "res_Phe": 2, "motif_NMe_amide": 1, "res_Val": 1},
        "any_of": {"Hleu_or_Hile_like": ["N[C@H](CCC(C)C)C(=O)", "N[C@H](CC(C)CC)C(=O)"],
                   "BTA_or_short_acyl": ["CCCC(=O)", "CCCC(=O)N"]},
        "diagnostic_msms": "BTA-Gln-Thr-Hleu/Hile-Ahp-Phe-NMePhe-Val",
        "interpretation": "Homologated Leu/Ile micropeptin-like signature; Leu/Ile cannot be confidently separated by this heuristic."
    },
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


def detect_micropeptin_signatures(row, signatures):
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

        # any_of groups: at least one SMARTS inside each group must match.
        any_of = spec.get("any_of", {})
        for group_name, smarts_list in any_of.items():
            if not any(has_smarts(mol, smarts) for smarts in smarts_list):
                ok = False
                break

        if ok:
            hits.append(signature_name)

    return hits


def micropeptin_signature_table(row, signatures):
    hits = set(row.get("micropeptin_signature_hits_list", []))
    data = {}
    for signature_name in signatures:
        data[f"sig_{signature_name}"] = int(signature_name in hits)
    return pd.Series(data)


def build_signature_summary(row):
    hits = row.get("micropeptin_signature_hits_list", [])
    if not hits:
        return ""
    return "; ".join(hits)


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
    st.divider()

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

processed["micropeptin_signature_hits_list"] = processed.apply(
    lambda row: detect_micropeptin_signatures(row, MICROPEPTIN_SIGNATURES),
    axis=1
)
processed["micropeptin_signature_hits"] = processed.apply(build_signature_summary, axis=1)
signature_hit_table = processed.apply(
    lambda row: micropeptin_signature_table(row, MICROPEPTIN_SIGNATURES),
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
    "Microviridin", "Aerucyclamide", "Patellamide"
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
        + (" | Micropeptin_signatures: " + row["micropeptin_signature_hits"] if row["micropeptin_signature_hits"] else "")
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
    "detected_special_motif_count", "micropeptin_signature_hits", "MolWt", "AA_signature", "SMILES"
]
st.dataframe(peptides[show_cols].head(500), use_container_width=True)

csv_processed = csv_bytes(peptides.drop(columns=["mol"], errors="ignore"))
st.download_button(
    "Download peptide-like table with simplified sequences",
    data=csv_processed,
    file_name="cyano_peptide_like_sequences.csv",
    mime="text/csv"
)

st.subheader("2b. Micropeptin/cyanopeptolin substructure detection")
st.caption(
    "These signatures are structural proxies inspired by diagnostic MS/MS fragments. "
    "They use SMILES/RDKit substructure matching, not direct MS/MS ion detection."
)

signature_summary_cols = [
    "compound_name", "family", "micropeptin_signature_hits", "AA_signature", "SMILES"
] + signature_cols

micro_hits_df = peptides.loc[
    peptides["micropeptin_signature_hits"].astype(str).str.len() > 0,
    signature_summary_cols
].copy()

col_sig1, col_sig2 = st.columns(2)
col_sig1.metric("Compounds with micropeptin-like signatures", len(micro_hits_df))
col_sig2.metric("Signature types searched", len(MICROPEPTIN_SIGNATURES))

if len(micro_hits_df):
    st.dataframe(micro_hits_df, use_container_width=True)
else:
    st.info("No micropeptin/cyanopeptolin signatures were detected with the current SMARTS heuristics.")

st.download_button(
    "Download micropeptin/cyanopeptolin signature hits",
    data=csv_bytes(micro_hits_df),
    file_name="micropeptin_cyanopeptolin_signature_hits.csv",
    mime="text/csv"
)

with st.expander("Signature dictionary used for detection"):
    signature_dictionary_df = pd.DataFrame([
        {
            "signature": name,
            "required_columns": "; ".join(f"{k}>={v}" for k, v in spec.get("required", {}).items()),
            "optional_smarts_groups": "; ".join(spec.get("any_of", {}).keys()),
            "diagnostic_msms_reference": spec.get("diagnostic_msms", ""),
            "interpretation": spec.get("interpretation", "")
        }
        for name, spec in MICROPEPTIN_SIGNATURES.items()
    ])
    st.dataframe(signature_dictionary_df, use_container_width=True)
    st.download_button(
        "Download signature dictionary",
        data=csv_bytes(signature_dictionary_df),
        file_name="micropeptin_signature_dictionary.csv",
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
        st.markdown("**Micropeptin signatures**")
        st.write(selected_row.get("micropeptin_signature_hits", "None"))
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

st.subheader("3. Sequence-like clustering")
st.caption(
    "A sequência usada aqui é uma assinatura simplificada derivada de subestruturas no SMILES. "
    "Ela não representa necessariamente a ordem biossintética real dos resíduos."
)

seq_sim = compute_sequence_similarity(plot_df["AA_signature"].tolist())
seq_pairs = matrix_to_long_table(seq_sim, labels, "sequence_jaccard")

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(make_heatmap(seq_sim, labels, "Simplified sequence similarity heatmap"), use_container_width=True)
with col2:
    fig_dendro_seq, Z_seq = make_dendrogram_figure(seq_sim, labels, "Simplified sequence dendrogram")
    st.plotly_chart(fig_dendro_seq, use_container_width=True)

if NETWORKX_AVAILABLE:
    G_seq, fig_net_seq = make_network(seq_sim, labels, plot_df, sequence_network_threshold)
    st.plotly_chart(fig_net_seq, use_container_width=True)
else:
    st.warning("NetworkX is not installed. Network plots are disabled.")

with st.expander("Top sequence-like similarities"):
    st.dataframe(seq_pairs.head(200), use_container_width=True)
    st.download_button(
        "Download sequence similarity pairs",
        data=csv_bytes(seq_pairs),
        file_name="sequence_similarity_pairs.csv",
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
        x="sequence_jaccard",
        y="morgan_tanimoto",
        hover_data=["source", "target"],
        title="Comparison between simplified sequence similarity and structural similarity"
    )
    fig_compare.update_layout(height=600)
    st.plotly_chart(fig_compare, use_container_width=True)

    corr = comparison[["sequence_jaccard", "morgan_tanimoto"]].corr().iloc[0, 1]
    st.metric("Correlation: sequence-like similarity vs Morgan/Tanimoto", f"{corr:.3f}")

    st.download_button(
        "Download sequence vs structure comparison",
        data=csv_bytes(comparison),
        file_name="sequence_vs_structure_similarity.csv",
        mime="text/csv"
    )

st.subheader("6. Interpretation notes")
st.markdown(
    """
**Important interpretation:**

- The sequence-like clustering is based on residue and motif detection from SMILES.
- For non-ribosomal peptides, cyclic peptides and depsipeptides, this is better treated as a **residue signature** than as a true linear FASTA sequence.
- Morgan/Tanimoto clustering captures the full chemical structure and may group analogues better when modifications occur in fatty acid tails, sugars, halogens or N-methylations.
- Disagreement between sequence-like and structural clustering is biologically useful: it may reveal compounds with similar peptide cores but different decorations, or structurally similar analogues with different residue composition.
"""
)
