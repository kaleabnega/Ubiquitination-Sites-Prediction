# MultiScale CenterLoRA-ESM2 v2

## Scientific question

CenterLoRA-ESM2 v1 improved development AUROC and AUPRC over the stabilized
MMUbiPred-compatible baseline, but its mean fixed-threshold MCC gain was only
0.0041. V1 classifies only the contextual representation at the candidate
lysine. V2 tests whether explicitly retaining contextualized local motif
information improves MCC without changing the pretrained backbone or training
protocol.

## Controlled architectural change

The input conversion, 49-residue window, ESM-2 backbone, rank-8 LoRA adapters,
and optimization settings are identical to the full-15 v1 experiment. From
the final ESM-2 residue representations, v2 constructs three components:

1. the central candidate-lysine representation;
2. a masked mean over the central residue and two residues on either side
   (radius 2, at most 5 residues); and
3. a masked mean over the central residue and five residues on either side
   (radius 5, at most 11 residues).

Terminal padding, CLS, and EOS tokens are excluded from both means. Close to a
protein terminus, each component averages only real residues.

All three components pass through the same layer-normalized 256-dimensional
GELU projection. A shared scalar gate scores each projected component, and a
softmax produces sample-dependent component weights. Their weighted sum passes
through dropout and one binary output layer.

The shared projection prevents parameter growth from being confused with the
effect of multi-scale pooling. Returning the three component weights also
provides a diagnostic of whether the model actually uses the additional
context.

## Predeclared seed-42 screen

- Seed: 42.
- Split: the same protein-grouped split as v1 seed 42.
- Epochs: all 15 epochs; retain the best development checkpoint by MCC at 0.5.
- Primary comparison: fixed-threshold MCC against full-15 v1 seed 42.
- Material-improvement target: v2 minus v1 fixed MCC at least 0.005.
- Ranking safeguard: AUROC and AUPRC may not decrease by more than 0.002.
- Stability safeguard: predictions may not collapse to one class.
- The independent test remains locked and no full-data refit is performed.

A passing screen advances the model only to a paired three-seed development
comparison. A borderline outcome requires review before further computation;
a failed screen ends the version without test-set evaluation.

## Interpretation boundary

The pooling radii are a single biologically motivated choice rather than a
hyperparameter search. No alternative radii may be selected using the seed-42
result. This restriction limits adaptation to one validation split and keeps
the v1-to-v2 comparison interpretable.

## Recorded outcome

The seed-42 run completed all 15 epochs and selected epoch 6. Fixed MCC
improved from 0.56894 to 0.57238, a gain of 0.00345 that did not reach the
predeclared 0.005 target. AUROC, AUPRC, and selected-threshold MCC were lower
than v1. The center, radius-2, and radius-5 components received mean weights of
0.6070, 0.2210, and 0.1721, respectively, showing that the additional context
was used but was not beneficial overall.

The screen was classified as borderline and stopped. No additional seeds were
run, and the independent test remained locked.
