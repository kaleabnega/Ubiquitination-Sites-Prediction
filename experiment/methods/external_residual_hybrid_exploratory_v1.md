# External residual-hybrid transportability assessment v1

## Scope

This analysis evaluates the already fitted MMUbiPred–Context Residual Hybrid
v1 on the two previously inspected external cohorts. It is explicitly
post-primary and exploratory. It cannot rescue, replace, or modify either
external cohort's completed primary analysis.

The candidate is unchanged from its historical-test evaluation: the
five-epoch full-data 49-residue Short-Range Expert refit, the frozen
seven-epoch 257-residue Long-Context Expert refit, and the nonnegative residual
logit stacker fitted only to corrected training out-of-fold predictions. Its
weights, intercept, and threshold `0.5` are immutable.

## Reused predictions and new inference

The exact released MMUbiPred and Long-Context Expert probabilities are read from each
cohort's checksum-locked completed evaluation. They are not recomputed. The
only new inference is the frozen Short-Range Expert on the same cohort rows.
The residual probability is then generated with the frozen stacker. No model
is trained and no threshold, weight, feature, checkpoint, or cohort is selected
using external labels.

The two cohorts are:

1. 286 sites from 231 proteins in the dbPTM/PTMGPT2 v3 cohort; and
2. 740 sites from 477 proteins in the disjoint dbPTM v2 cohort.

## Reporting

For each cohort, all metrics are reported at threshold `0.5`. Candidate-minus-
exact-MMUbiPred differences use 10,000 paired nonparametric bootstrap
replicates that resample whole canonical-accession protein clusters. Accuracy
discordance is additionally reported by exact McNemar counts as a descriptive
site-level diagnostic. Neither cohort supplies a confirmatory primary claim,
and failure or success on one cohort cannot tune the analysis on the other.
