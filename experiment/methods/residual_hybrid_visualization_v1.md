# MMUbiPred–Context Residual Hybrid visualization protocol

## Scope

This post hoc visualization workflow describes the frozen residual hybrid on
the 12,288-site context-matched historical-test cohort. It performs no model
training, checkpoint selection, threshold selection, calibration, or fusion
fitting. ROC, precision–recall, and confusion-matrix figures reuse the saved
probabilities from the completed matched evaluation.

## Representation extraction

One deterministic inference pass extracts two hidden representations from the
frozen checkpoints:

- the six-dimensional ReLU output of the MMUbiPred-compatible local expert's
  score-fusion layer; and
- the 256-dimensional GELU output immediately before the final dropout and
  binary layer of the seven-epoch long-context LoRA-ESM-2 expert.

The extraction script verifies checkpoint, sequence-cache, cohort-index, and
prediction alignment before saving the representation matrix. Recomputed
probabilities must match the frozen saved probabilities within numerical
tolerance.

## t-SNE

A deterministic class-balanced subset contains at most 6,000 sites, selected
with seed 42 before either representation is embedded. Each representation is
standardized independently and reduced by PCA to at most 50 dimensions. t-SNE
then uses PCA initialization, perplexity 30, automatic learning rate, 1,500
iterations, Barnes–Hut optimization, angle 0.5, and seed 42. The same site
indices are used for both panels. t-SNE is qualitative and is not evidence of
classification superiority.

## Performance figures

ROC and precision–recall curves compare exact released MMUbiPred, the
MMUbiPred-compatible local expert, the context expert, and the residual hybrid
on aligned saved labels and probabilities. Confusion matrices use the frozen
threshold 0.5 and report counts with row-normalized percentages.

## Learning curves

Training objective and inner-validation MCC at threshold 0.5 are shown for all
five protein-grouped development folds. Local curves use the initialization
selected by the uniform stabilization protocol for each fold. Context curves
use the unchanged context-fold histories. These curves document development
behavior and are not derived from the final full-data refits, which had no
validation partitions.

## Output

Each figure is displayed in the notebook and exported as editable PDF and SVG,
plus a 600-DPI PNG. Figure manifests and source-artifact hashes preserve the
post hoc, non-confirmatory interpretation boundary.
