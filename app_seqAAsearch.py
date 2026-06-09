# app.py
# Peptide-oriented clustering app for cyanobacterial metabolite databases
# Author: Ricardo M. Borges workflow draft

import io
import re
import itertools
from collections import Counter

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
    left = str(sequence).split("|")[0]
    tokens = [t.strip() for t in re.split(r"[-; ,]+", left) if t.strip()]
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
from pathlib import Path
with st.sidebar:


    LOGO_PATH = Path(__file__).parent / "static" / "LAABio.png"

    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), use_container_width=True)

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

residue_cols = [c for c in processed.columns if c.startswith("res_")]
motif_cols = [c for c in processed.columns if c.startswith("motif_")]

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
    "detected_special_motif_count", "MolWt", "simplified_sequence", "SMILES"
]
st.dataframe(peptides[show_cols].head(500), use_container_width=True)

csv_processed = peptides.drop(columns=["mol"], errors="ignore").to_csv(index=False).encode("utf-8")
st.download_button(
    "Download peptide-like table with simplified sequences",
    data=csv_processed,
    file_name="cyano_peptide_like_sequences.csv",
    mime="text/csv"
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

seq_sim = compute_sequence_similarity(plot_df["simplified_sequence"].tolist())
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
        data=seq_pairs.to_csv(index=False).encode("utf-8"),
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
        data=struct_pairs.to_csv(index=False).encode("utf-8"),
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
        data=comparison.to_csv(index=False).encode("utf-8"),
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
