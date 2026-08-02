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

## Unchanged scientific contract

The source benchmark, UniProt mapping, exact-overlap filters, 30% identity over
80% shorter-protein coverage rule, frozen checkpoints, equal `0.5/0.5`
probability ensemble, threshold `0.5`, and primary MCC endpoint are unchanged.
External labels still cannot select the model, weights, or threshold.

Because 286 sites belong to 231 proteins, inferential bootstrap replicates
resample whole canonical-accession groups rather than individual sites. This
preserves within-protein dependence. Site-level McNemar counts are descriptive
only; the paired protein-cluster bootstrap is the primary uncertainty analysis.
