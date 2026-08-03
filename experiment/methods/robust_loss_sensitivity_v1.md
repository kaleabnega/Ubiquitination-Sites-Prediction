# MMUbiPred robust-loss sensitivity v1

## Objective

This is a narrow development-only test of whether conservative label-robust
objectives improve the MMUbiPred-compatible architecture. Architecture, input
window, released training records, protein-grouped split, seed, optimizer,
learning rate, regularization, batch size, epoch limit, and fixed threshold are
identical. Only the training objective changes.

The released independent test and all external cohorts remain inaccessible.
This screen cannot support a test-set performance claim.

## Frozen conditions

All three conditions use seed 42, a 90/10 protein-grouped development split,
Adam with learning rate `0.001` and epsilon `1e-7`, batch size 32, and all 15
epochs. Early-stopping patience equals the epoch limit, so no condition is
given a shorter optimization opportunity.

1. **BCE reference:** ordinary binary cross entropy with logits.
2. **Asymmetric negative smoothing:** observed positive targets remain `1.0`;
   observed negative targets become `0.05`. This is a conservative sensitivity
   value, not an estimate that 5% of PLMD negatives are false.
3. **Generalized cross entropy:** binary GCE with fixed `q=0.7`, reducing the
   gradient contribution of examples that remain confidently inconsistent
   with the model. The value is frozen for this one-condition screen.

No smoothing strength, GCE parameter, class weight, threshold, or epoch count
is searched.

## Predeclared screen

Fixed-threshold MCC at `0.5` is primary. Relative to the paired BCE run, a
candidate advances only if all conditions hold:

- fixed MCC improves by at least `0.01`;
- AUROC does not decrease;
- sensitivity decreases by no more than `0.03`;
- specificity decreases by no more than `0.03`; and
- neither the candidate nor the comparison suite collapses to one class.

If both robust losses pass, the one with higher fixed-threshold MCC advances.
Passing authorizes only a three-seed development confirmation. If neither
passes, BCE is retained and the label-robustness track stops.
