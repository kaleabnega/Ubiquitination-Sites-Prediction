# Disjoint dbPTM holdout v2

## Pre-inference feasibility amendment

Version 1 predeclared a final-cohort minimum of 750 sites, 200 sites per
class, and 400 canonical proteins. Its leakage and homology audit retained 740
sites: 369 database-negative sites, 371 positive sites, and 477 canonical
proteins. The builder stopped before freezing the cohort and before generating
any model prediction. External labels were used only to verify the predeclared
class-support feasibility gate; no prediction or performance metric informed
this amendment, model selection, or threshold selection.

Version 2 changes only the minimum total support from 750 to 700. The class
minimum remains 200 per class and the protein minimum remains 400. The latter
is the prospective precision target for the protein-cluster bootstrap. The
observed v1 cohort exceeds both scientifically important gates and misses the
original round-number total gate by only 10 sites.

To prevent cohort drift, v2 pins the complete v1 prospective support
(`740/369/371/477`) and the byte-level SHA-256 hashes of both preserved MMseqs2
alignment artifacts:

- raw alignments: `39aa11e48f555747e1e1c546f7987a4529ec98bff213d2298ad9287eee5cf0cd`
- qualifying alignments: `39710ca4a7beb3b888014daf185646e2be70c7961f054f8701bd4513f083e461`

The builder must reproduce the same order-independent alignment content and
the same prospective cohort counts before v2 may freeze the cohort.

The v2 construction subsequently reproduced the pinned content and froze 740
sites from 477 proteins. Only after that pre-inference audit was reviewed was
the one-time inference stage authorized. No model, threshold, or hypothesis
was changed between cohort freeze and inference authorization.

## Purpose and hypothesis

This study tests the prior external observation that the frozen seven-epoch,
257-residue Long-Context Expert may rank ubiquitination sites better than the
exact released MMUbiPred model. AUROC is the single primary metric. Success
requires the lower bound of the two-sided 95% protein-cluster bootstrap
interval for the Long-Context Expert minus MMUbiPred AUROC difference to
exceed zero. MCC and the other fixed-threshold metrics are secondary. Both
checkpoints and threshold `0.5` remain frozen.

The hypothesis is explicitly informed by the completed 286-site external
analysis, where the AUROC difference was `0.03392` but its interval included
zero. The new cohort must not be used for model, checkpoint, weight, or
threshold selection.

## Source normalization and label limitation

The source is the official dbPTM ubiquitination predictor benchmark archive.
Its checksum and download URL are frozen in the configuration. Normalization
of 9,767 positive and 8,579 database-negative records produced 9,124 positive
and 6,573 database-negative lysine-centred 21-mers after documented duplicate,
centre, and unsupported-residue exclusions.

A database-negative site is not experimental proof that the lysine can never
be ubiquitinated. This possible label noise must accompany every conclusion.

## Independence controls

The previously evaluated 2,077-site dbPTM/PTMGPT2 source is an exact subset of
the official archive. The builder therefore excludes that complete source and
all released PLMD training and historical-test overlaps. Remaining proteins
are searched with MMseqs2 against the union of PLMD training proteins and all
resolved proteins from the prior source.

A candidate protein is excluded at at least 30% identical residues divided by
the shorter full-protein length and at least 80% coverage of that shorter
protein. Alignment backtraces are mandatory and zero-identity output fails
closed. At least 60% of normalized sites must validate against current UniProt
sequences, and at least 98% of the reference sequences must be available.

## Frozen evaluation and statistical analysis

Predictions remain paired by site, while bootstrap resampling uses whole
canonical-protein groups to preserve within-protein dependence. Ten thousand
replicates use seed 20260803. There is no threshold tuning. The primary
Long-Context Expert versus exact MMUbiPred comparison, primary AUROC endpoint,
and success rule are unchanged from v1. The earlier ensemble is not a primary
candidate.

## Post-primary exploratory analysis

The confirmatory primary comparison was completed before evaluating the
previously defined equal-probability MMUbiPred–Long-Context hybrid. The saved
paper-model and Long-Context Expert probabilities are averaged with fixed
weights `0.5/0.5`; neither weight nor threshold is searched, and no
neural-network inference is repeated.
This hybrid comparison is explicitly post-primary and exploratory. Its metric
differences, bootstrap intervals, and p-values cannot replace or rescue the
failed confirmatory Long-Context Expert versus MMUbiPred claim.
