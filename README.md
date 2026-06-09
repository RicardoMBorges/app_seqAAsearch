# Cyano Peptide Clustering

## Overview

Cyano Peptide Clustering is a Streamlit application designed for the exploration, annotation and clustering of cyanobacterial peptide natural products using chemical structures (SMILES).

The software combines:

* Peptide-like compound detection
* Amino acid residue identification
* Detection of cyanobacterial peptide motifs
* Detection of micropeptin/cyanopeptolin-inspired structural signatures
* Sequence-like clustering
* Structural clustering (Morgan/Tanimoto)
* Interactive structure inspection with highlighted substructures

The application is intended for metabolomics, natural products chemistry, cyanobacterial secondary metabolites and dereplication workflows.

---

# Input

The application accepts:

* CSV
* TSV
* TXT

Files must contain at least:

| Column        | Required |
| ------------- | -------- |
| SMILES        | Yes      |
| compound_name | Optional |

If compound names are absent, the software automatically generates identifiers.

---

# What the program does

## 1. Detects peptide-like compounds

The software identifies peptide-like molecules using:

### Amide bond count

A molecule is considered peptide-like if the number of amide bonds exceeds the user-defined threshold.

### Family recognition

Known cyanobacterial peptide families are automatically recognized:

* Microcystins
* Nodularins
* Aeruginosins
* Cyanopeptolins
* Micropeptins
* Anabaenopeptins
* Hassallidins
* Microginins
* Microviridins
* and others

---

## 2. Detects amino acid residues

The software searches for peptide-related substructures:

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

These residues are converted into a simplified residue signature.

Example:

Val-Thr-Phe-Phe

---

## 3. Detects cyanobacterial peptide motifs

The software also searches for:

* Ahp-like
* Choi-like
* Adda-like
* N-methyl amides
* Guanidino groups
* Sulfates
* Halogens
* Sugar-like motifs

Example:

Val-Thr-Phe-Phe | Ahp_like:1; NMe_amide:1

---

## 4. Detects micropeptin/cyanopeptolin signatures

The software contains a dedicated signature engine inspired by diagnostic MS/MS fragments reported in the literature.

Examples:

* Ahp-Phe-NMePhe core
* Ahp-Phe core
* BTA-Gln-Thr
* BTA-Gln-Thr-Val-NMePhe
* Met-Ahp-Phe
* Met-Ahp-Phe-NMePhe

Important:

These are NOT direct MS/MS detections.

They are structural proxies inferred from SMILES using RDKit.

---

## 5. Generates AA Signatures

The application combines:

* Residues
* Cyanobacterial motifs
* Micropeptin signatures

into a single annotation:

Example:

Val-Thr-Phe-Phe | Ahp_like:1; NMe_amide:1 | Micropeptin_signatures: Ahp-Phe-NMePhe_core_like

---

# Structure Inspector

The Structure Inspector allows visual inspection of detected motifs.

Features:

* Structure rendering with RDKit
* Highlighted residues
* Highlighted motifs
* Highlighted micropeptin-related regions
* PNG export

Example interpretation:

A structure displaying:

* Ahp_like
* Phe
* NMe_amide

highlighted simultaneously may support assignment to a micropeptin/cyanopeptolin-like scaffold.

---

# Clustering Approaches

## Sequence-like clustering

Uses:

AA_signature

and computes:

Multiset Jaccard similarity

Output:

* Heatmap
* Dendrogram
* Similarity network

Interpretation:

Compounds clustering together share similar residue composition and peptide motifs.

---

## Structural clustering

Uses:

Morgan fingerprints

and:

Tanimoto similarity

Output:

* Heatmap
* Dendrogram
* Similarity network

Interpretation:

Compounds clustering together share overall structural similarity.

---

## Sequence vs Structure Comparison

The software compares:

Sequence similarity

vs

Structural similarity

Interpretation:

### High sequence similarity + high structural similarity

Likely close analogues.

### High sequence similarity + low structural similarity

Similar peptide cores with different decorations.

Examples:

* Glycosylation
* Halogenation
* Sulfation
* Lipid modifications

### Low sequence similarity + high structural similarity

Potential scaffold analogues with different residue composition.

---

# Files Produced

## Peptide detection

cyano_peptide_like_sequences.csv

Contains:

* compound name
* family
* amide count
* detected residues
* detected motifs
* AA signatures

---

## Micropeptin signatures

micropeptin_cyanopeptolin_signature_hits.csv

Contains:

* detected signatures
* AA signatures
* family assignments

---

## Signature dictionary

micropeptin_signature_dictionary.csv

Contains:

* searched signatures
* required residues
* interpretation

---

## Sequence clustering

sequence_similarity_pairs.csv

Contains pairwise sequence-like similarity values.

---

## Structural clustering

morgan_tanimoto_similarity_pairs.csv

Contains pairwise structural similarity values.

---

## Sequence vs Structure

sequence_vs_structure_similarity.csv

Contains combined similarity metrics.

---

# Example Interpretation

Suppose a compound displays:

Ahp-Phe-NMePhe_core_like

and clusters with known micropeptins.

Interpretation:

The compound likely contains the characteristic Ahp-containing core typical of micropeptins/cyanopeptolins.

---

Suppose a compound shows:

High Morgan similarity
Low sequence similarity

Interpretation:

The molecule may be a structural analogue with substantial residue substitutions.

---

Suppose a compound shows:

High sequence similarity
Low Morgan similarity

Interpretation:

The peptide core is conserved but decorations differ significantly.

Examples:

* glycosylation
* halogenation
* sulfation
* lipid tails

---

# Disclaimer

This software does NOT reconstruct the true biosynthetic peptide sequence.

The generated sequence is a residue signature inferred from SMILES substructure matching and should be interpreted as a comparative chemoinformatic descriptor rather than a true peptide sequence.
