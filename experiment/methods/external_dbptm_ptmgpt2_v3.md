# External dbPTM/PTMGPT2 validation v3

## Feasibility amendment before inference

The corrected v2 homology audit retained 286 sites from 231 canonical proteins:
191 negative and 95 positive sites. This failed the original operational guard
of 500 total and 100 per class, so v2 did not freeze a cohort. No external
model prediction or performance metric was generated or inspected.

Version 3 transparently amends only the feasibility guards to 250 total sites
and 75 sites per class. The exact v2 raw- and qualifying-alignment hashes and
the prospective cohort counts are pinned in the configuration and must be
reproduced before the builder can freeze v3. The smaller cohort provides less
precision than originally planned, so conclusions must be based on confidence
intervals and described as limited-size external validation.

MMseqs2 may emit identical alignments in a different row order across repeated
runs. V3 therefore verifies the preserved v2 files against their original byte
hashes, then compares v2 and v3 alignment content using sorted-line canonical
hashes. It also requires the exact prospective site, class, and protein counts.
This order-independent correction was made after a v3 construction failure and
before any external model inference.

After the frozen primary evaluation, the context expert may be compared with
exact MMUbiPred using the already saved site-paired predictions and the same
canonical-accession cluster bootstrap. The context expert was a predeclared
secondary comparator, but this follow-up cannot replace the ensemble as the
primary candidate. Its intervals and p-values are reported as unadjusted
secondary analyses, and the script performs no new inference.

## Unchanged scientific contract

The source benchmark, UniProt mapping, exact-overlap filters, 30% identity over
80% shorter-protein coverage rule, frozen checkpoints, equal `0.5/0.5`
probability ensemble, threshold `0.5`, and primary MCC endpoint are unchanged.
External labels still cannot select the model, weights, or threshold.

Because 286 sites belong to 231 proteins, inferential bootstrap replicates
resample whole canonical-accession groups rather than individual sites. This
preserves within-protein dependence. Site-level McNemar counts are descriptive
only; the paired protein-cluster bootstrap is the primary uncertainty analysis.
