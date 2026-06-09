# NPPSA

## Natural Products Peptide Signature Analysis

A chemoinformatic framework for biosynthetic-oriented clustering, motif discovery and peptide signature mining in cyanobacterial natural products.

---

## Overview

NPPSA is a Streamlit application designed to analyze peptide natural products directly from molecular structures (SMILES).

Instead of comparing only whole-molecule similarity, NPPSA transforms molecules into residue-based compositional signatures and explores relationships between compounds from a biosynthetic perspective.

The framework is particularly useful for:

* Micropeptins
* Cyanopeptolins
* Aeruginosins
* Microcystins
* Nodularins
* Anabaenopeptins
* Microginins
* Microviridins

and other peptide-derived natural products.

---

## Why NPPSA?

Traditional cheminformatics asks:

> Which molecules are structurally similar?

NPPSA asks:

> Which molecules share similar biosynthetic modules?

For example:

Ahp–Phe–NMePhe–Val–Thr

Ahp–Tyr–NMePhe–Val–Thr

Ahp–Phe–NMeTyr–Val–Thr

may appear structurally different while sharing nearly identical biosynthetic logic.

---

## Main Features

### 1. Automatic Peptide Detection

The app automatically identifies peptide-like compounds using:

* amide bond counts
* known cyanobacterial peptide family names
* residue detection from SMILES

---

### 2. Residue Detection

Detected residues currently include:

* Gly
* Ala
* Val
* Leu/Ile
* Ser
* Thr
* Phe
* Tyr
* Trp
* Asp
* Glu
* Asn
* Gln
* Lys
* Arg
* His
* Pro

---

### 3. Cyanobacterial Motif Detection

Current motifs include:

* Ahp-like
* Choi-like
* Adda-like
* N-methyl amides
* Guanidino groups
* Sulfates
* Halogens
* Sugar-like motifs

---

### 4. Cyanopeptide Signature Detection

The app identifies structural signatures inspired by known cyanobacterial peptide families.

Examples:

* Ahp-Phe-NMePhe
* Ahp-Phe
* Adda-Glu
* Choi-Arg
* Lys-containing scaffolds

These signatures are structural proxies and are not direct MS/MS annotations.

---

### 5. Structure Inspector

Visual inspection of compounds with RDKit.

Features:

* structure rendering
* motif highlighting
* residue highlighting
* PNG export

---

### 6. CyanoPeptide Signature Builder

Automatically discovers recurring motifs from the uploaded database.

Example output:

| Signature   | Compounds |
| ----------- | --------- |
| Ahp-Phe-NMe | 47        |
| Adda-Glu    | 39        |
| Choi-Arg    | 22        |
| Lys-Val-Phe | 18        |

No manual curation is required.

---

### 7. Sequence Explorer

Search compounds using:

* residues
* motifs
* cyanopeptide signatures
* recurring signatures
* diagnostic MS/MS-inspired motifs
* compound families

---

### 8. Sequence-like Clustering

NPPSA converts detected residues and motifs into compositional signatures.

Example:

Val-Thr-Phe-Phe | Ahp | NMe

Similarity is calculated using:

* multiset Jaccard similarity

This produces:

* heatmaps
* dendrograms
* similarity networks

---

### 9. Structural Clustering

Standard cheminformatic clustering using:

* Morgan fingerprints
* Tanimoto similarity

Outputs:

* heatmaps
* dendrograms
* molecular networks

---

### 10. Sequence vs Structure Comparison

NPPSA directly compares:

Residue-based similarity

vs

Fingerprint-based similarity

allowing identification of compounds that:

* share peptide cores
* differ by decorations
* may belong to related biosynthetic spaces

---

## Input File

Required column:

| Column |
| ------ |
| SMILES |

Optional:

| Column        |
| ------------- |
| compound_name |
| InChI         |
| InChIKey      |

CSV, TSV and TXT files are supported.

---

## Downloadable Outputs

All exported tables use semicolon-separated CSV format.

Generated files include:

* cyano_peptide_like_sequences.csv
* auto_built_cyanopeptide_signatures.csv
* sequence_explorer_results.csv
* sequence_similarity_pairs.csv
* morgan_tanimoto_similarity_pairs.csv
* sequence_vs_structure_similarity.csv

---

## Example Workflow

1. Upload a metabolite database.
2. Detect peptide-like compounds.
3. Inspect residue signatures.
4. Explore recurring motifs.
5. Search for specific peptide signatures.
6. Build sequence-like clustering.
7. Build structural clustering.
8. Compare both clustering approaches.

---

## Scientific Applications

NPPSA can be used for:

* natural product dereplication
* cyanobacterial metabolomics
* chemotaxonomy
* biosynthetic studies
* genome–metabolome integration
* peptide family discovery
* NRPS module exploration
* natural product evolution studies

---

## Technology Stack

* Python
* Streamlit
* RDKit
* Plotly
* SciPy
* NetworkX
* Pandas
* NumPy

---

## Citation

If you use NPPSA in your research, please cite:

Borges RM et al.
Natural Products Peptide Signature Analysis (NPPSA):
A Chemoinformatic Framework for Biosynthetic-Oriented Clustering and Motif Discovery in Cyanobacterial Peptides.

Manuscript in preparation.

---

## Author

Ricardo Moreira Borges

Walter Mors Institute of Research on Natural Products (IPPN)

Federal University of Rio de Janeiro (UFRJ)

Brazil

ORCID: 0000-0002-7662-6734

---

## Future Developments

Planned features include:

* true peptide sequence reconstruction
* automatic motif mining
* BGC integration
* antiSMASH integration
* GNPS integration
* CyanoMetDB integration
* NPAtlas integration
* machine-learning-based signature discovery
* biosynthetic module prediction
