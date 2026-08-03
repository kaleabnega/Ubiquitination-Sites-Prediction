# MMUbiPred–Context prediction complementarity diagnostic v1

## Question

This analysis asks whether the corrected MMUbiPred-compatible local expert and
the long-context LoRA-ESM2 expert make sufficiently complementary development
errors to justify spending compute on a more flexible gated fusion model.

It is a saved-prediction diagnostic. It does not train a model, rerun inference,
select a threshold, or access a dataset that has not already been evaluated.

## Primary evidence

The only evidence allowed to control the recommendation is the corrected
five-fold development record from MMUbiPred–Context Residual v1:

- every site is predicted out of fold;
- folds are grouped by protein accession;
- the local expert uses the uniform multi-initialization stabilization rule;
- the context expert predictions are unchanged; and
- the residual stack is cross-fitted, so each held-out fold is predicted by
  fusion weights learned from the other four folds.

The analysis records SHA-256 hashes and verifies the recorded metrics of
`stabilized_summary.json` and `stabilized_oof_predictions.npz` before reporting
new diagnostics.

## Diagnostics

At the fixed threshold `0.5`, the script reports:

- sites both experts classify correctly or incorrectly;
- sites corrected only by the local or only by the context expert;
- those disagreement counts separately for positive and negative labels;
- probability and correctness correlations;
- the mean absolute probability difference;
- a label-informed oracle accuracy ceiling;
- site-level descriptive McNemar counts;
- the same behavior separately in each held-out fold; and
- the incremental performance of the equal average and cross-fitted residual
  stack over context alone.

The oracle ceiling is not a deployable model. It uses the true label to choose
the correct expert and therefore describes only the maximum error diversity
available to a hypothetical perfect selector.

## Gated-fusion rule

Raw disagreement is necessary but not sufficient evidence for gating. A
sample-dependent gate has more capacity than the already evaluated
nonnegative residual stack and therefore has greater overfitting risk.

The existing frozen development decision controls this diagnostic:

- if the cross-fitted residual stack achieved the predeclared material gain
  over context alone, the output may recommend considering one separately
  predeclared gated-fusion screen;
- otherwise, the output is `DO_NOT_TRAIN_GATED_FUSION`.

Historical and external saved predictions are optional descriptive inputs.
They are permanently excluded from model, fusion-weight, threshold, and
architecture selection and cannot reverse the development recommendation.

## Interpretation

Complementary error counts show whether useful information exists in
principle. Cross-fitted fusion performance shows whether a realistic,
label-free combination can actually capture it. The latter is the stronger
criterion for deciding whether a more flexible gate is scientifically and
computationally justified.
