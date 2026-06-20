
# NPSAA

## Natural Products Scaffold and Amino Acid Analysis

A biosynthetic-oriented chemoinformatic framework for scaffold classification, motif discovery, architecture mining, and similarity analysis of peptide natural products.

---

## Overview

NPSAA is a Streamlit application designed to analyze peptide and peptide-derived natural products directly from molecular structures (SMILES).

Unlike traditional cheminformatics approaches that rely exclusively on molecular fingerprints, NPSAA attempts to reconstruct biosynthetic relationships through:

* Biosynthetic scaffold recognition
* Canonical motif identification
* Peptide module discovery
* Natural product ontology assignment
* NPSAA signature generation
* Structural similarity analysis

The framework is particularly focused on cyanobacterial peptides, but can be applied to peptide natural products from any biological source.

---

## Conceptual Framework

Traditional cheminformatics asks:

> Which molecules are structurally similar?

NPSAA asks:

> Which molecules belong to the same biosynthetic space?

The current version prioritizes:

```text
BIOSYNTHETIC_CLASS
        ↓
SCAFFOLD_CLASS
        ↓
CANONICAL_MOTIF
        ↓
NPSAA_SIGNATURE
        ↓
STRUCTURAL SIMILARITY
```

This hierarchy reflects biosynthetic logic more closely than residue-only similarity approaches.

---

# Main Features

## 1. Automatic Peptide Detection

Peptide-like compounds are automatically detected using:

* Amide bond counts
* Known peptide family names
* Structural motif recognition
* Biosynthetic scaffold assignment

---

## 2. Biosynthetic Ontology Assignment

Each compound is classified into biosynthetic categories.

Examples:

* Microcystin-like
* Cyanopeptolin-like
* Aeruginosin-like
* Anabaenopeptin-like
* Microginin-like
* Cyclamide-like
* Microviridin-like
* Laxaphycin-like
* Hassallidin-like
* PKS-NRPS hybrid lipopeptides

Generated fields:

| Field                  |
| ---------------------- |
| BIOSYNTHETIC_CLASS     |
| SCAFFOLD_CLASS         |
| CANONICAL_MOTIF        |
| NPSAA_CONFIDENCE_LEVEL |

---

## 3. Canonical Motif Recognition

The system recognizes diagnostic biosynthetic motifs.

Examples:

* Ahp
* Adda
* Choi
* Ureido linkage
* N-methyl amino acids
* Guanidino groups
* Sulfates
* Halogens
* Glycosylations
* Lipid tails

---

## 4. NPSAA Signature Generation

Each compound receives a biosynthetic signature.

Example:

```text
MICROCYSTIN | Adda-Glu-Mdha
```

or

```text
CYANOPEPTOLIN | Ahp-Phe-NMePhe
```

These signatures are used for clustering and similarity analysis.

---

## 5. Cyanopeptide Signature Discovery

NPSAA automatically discovers recurrent motif combinations in the uploaded database.

Example:

| Signature      | Count |
| -------------- | ----: |
| Ahp-Phe-NMePhe |    47 |
| Adda-Glu-Mdha  |    39 |
| Choi-Arg       |    22 |
| Ahp-Tyr-NMePhe |    18 |

---

## 6. Biosynthetic Architecture Analysis

The application reconstructs biosynthetic architectures from detected motifs.

Examples:

```text
PKS → NRPS → NRPS → Tailoring
```

```text
RiPP → Cyclization → Prenylation
```

```text
NRPS → Ahp Core → N-Methylation
```

Outputs include:

* Architecture tables
* Architecture summaries
* Architecture networks

---

## 7. Sequence Explorer

Compounds can be searched using:

* Biosynthetic classes
* Canonical motifs
* NPSAA signatures
* Structural families
* Cyanopeptide families
* Architecture types

---

## 8. NPSAA Signature Clustering

The primary clustering strategy uses biosynthetic signatures rather than residue composition.

Similarity is calculated using:

* Jaccard similarity
* Scaffold overlap
* Motif overlap

Outputs:

* Heatmaps
* Dendrograms
* Networks

---

## 9. Structural Clustering

Classical cheminformatics clustering based on:

* Morgan fingerprints
* Tanimoto similarity

Outputs:

* Structural heatmaps
* Structural dendrograms
* Molecular similarity networks

---

## 10. Sequence vs Structure Comparison

NPSAA compares:

```text
Biosynthetic Similarity
```

versus

```text
Structural Similarity
```

allowing identification of:

* Shared biosynthetic origins
* Divergent chemical decorations
* Scaffold conservation
* Analog series

---

# Confidence Levels

Each classification receives a confidence score.

| Level | Description                         |
| ----- | ----------------------------------- |
| A     | Scaffold + canonical motif evidence |
| B     | Scaffold/core evidence              |
| C     | Name-supported assignment           |
| D     | Unresolved                          |

---

# Current Performance (v21)

## Benchmark Dataset

| Metric                   | Value |
| ------------------------ | ----: |
| Total compounds analyzed |  1825 |
| Classified compounds     |   941 |
| Unresolved compounds     |   884 |
| Classification coverage  | 51.6% |

Compared with previous versions, unresolved compounds decreased from:

```text
930 → 884
```

representing 46 additional compounds successfully classified.

---

## Newly Recognized Superfamilies

The current ontology recognizes:

* Cyclamide RiPPs
* Microviridins
* Laxaphycins
* Hassallidins
* Marine PKS-NRPS lipopeptides
* NPSAA-rich depsipeptides

---

# Input File

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

Supported formats:

* CSV
* TSV
* TXT

---

# Downloadable Outputs

Current exports include:

```text
01_peptide_like_sequences.csv
02_cyanopeptide_signature_hits.csv
03_signature_dictionary.csv
04_auto_built_cyanopeptide_signatures.csv
05_module_summary.csv
06_module_architecture_table.csv
07_compound_module_table.csv
08_module_network_edges.csv
09_npsaa_scaffold_ontology.csv
10_npsaa_canonical_motif_library.csv
11_npsaa_scaffold_summary.csv
12_npsaa_scaffold_network_edges.csv
13_npsaa_canonical_motif_summary.csv
14_npsaa_canonical_motif_network_edges.csv
15_biosynthetic_architecture_table.csv
16_biosynthetic_architecture_summary.csv
17_biosynthetic_architecture_network_edges.csv
18_sequence_explorer_results.csv
19_npsaa_signature_similarity_pairs.csv
20_morgan_tanimoto_similarity_pairs.csv
21_npsaa_signature_vs_structure_similarity.csv
```

All CSV exports use semicolon separators.

---

# Scientific Applications

NPSAA can be applied to:

* Natural product dereplication
* Cyanobacterial metabolomics
* Chemotaxonomy
* Peptide family discovery
* Biosynthetic investigations
* NRPS studies
* RiPP studies
* Genome-metabolome integration
* Evolutionary studies
* Molecular networking interpretation

---

# Technology Stack

* Python
* Streamlit
* RDKit
* Plotly
* SciPy
* Pandas
* NumPy
* NetworkX

---

# Citation

If you use NPSAA in your research, please cite:

> Borges RM et al.
> Natural Products Scaffold and Amino Acid Analysis (NPSAA): A Biosynthetic-Oriented Framework for Scaffold Classification, Motif Discovery and Similarity Analysis in Peptide Natural Products.
> Manuscript in preparation.

---

# Author

**Ricardo Moreira Borges**

Institute of Research on Natural Products Walter Mors (IPPN)

Federal University of Rio de Janeiro (UFRJ)

Brazil

ORCID: 0000-0002-7662-6734

---

# Future Developments

Planned developments include:

* Expanded scaffold ontology
* Additional cyanobacterial peptide families
* BGC integration
* antiSMASH integration
* GNPS integration
* CyanoMetDB integration
* NPAtlas integration
* Biosynthetic module prediction
* Machine-learning-assisted motif discovery
* Automated scaffold inference
* Cross-linking with metabolomics and genomics datasets

---

Esse README agora está alinhado com a versão atual do aplicativo, incluindo as novas tabelas exportadas, ontologia biossintética, NPSAA signatures e a mudança de paradigma de "AA signatures" para "scaffold-first biosynthetic classification".
