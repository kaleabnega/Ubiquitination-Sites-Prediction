# External dbPTM/PTMGPT2 validation v1

## Purpose

This protocol tests whether the already frozen exact-MMUbiPred/context
equal-probability ensemble generalizes beyond the PLMD cohort used throughout
model development. The external labels must not be used to change the
architecture, checkpoint, fusion weights, threshold, preprocessing, or
exclusion rules.

## Source and reconstruction

The released `benchmark.csv` contains 2,077 labelled, lysine-centred 21-mers
with historical UniProt entry names and one-based site positions. Each entry
name is resolved to a canonical UniProt accession and full protein. A record
is eligible only when its published 21-mer
matches the current UniProt sequence exactly at the published position. The
same verified protein is then used to construct the paper model's 49-residue
window and the context model's 257-residue window.

## Leakage controls

Before inference, the cohort builder removes:

1. every protein accession represented in the processed PLMD training set;
2. every protein accession represented in the historical PLMD test set;
3. every exact released site, central 21-mer, or 49-mer; and
4. every remaining external protein matched by CD-HIT-2D to a PLMD training
   protein at at least 30% global sequence identity and at least 80% coverage
   of the shorter sequence.

The builder records input checksums, UniProt release metadata, exclusion
counts, the exact CD-HIT-2D command, retained-accession hash, final cohort
hash, and class support. At least 80% of benchmark sites must map, at least
98% of PLMD training proteins must have reference sequences, and the final
cohort must contain at least 500 sites with at least 100 examples of each
class. It refuses to overwrite a frozen cohort.

## Frozen evaluation

The primary candidate is the already specified arithmetic mean of:

- the positive-class probability from the authors' exact released MMUbiPred
  H5 model; and
- the positive-class probability from the frozen seven-epoch, full-training
  257-residue LoRA-ESM-2 context model.

Both weights are 0.5 and the threshold is 0.5. The exact MMUbiPred and context
experts are secondary comparators on the identical retained cohort. MCC is the
primary metric. Accuracy, sensitivity, specificity, precision, F1, AUROC, and
AUPRC are secondary metrics. Paired stratified-bootstrap intervals use 10,000
replicates and seed 20260731.

The ensemble was designed after inspection of the historical PLMD test and is
therefore not a fresh architecture. This external cohort can nevertheless
provide an honest external validation because its predictions were not
inspected when this contract was frozen.
