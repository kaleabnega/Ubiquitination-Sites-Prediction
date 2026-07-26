# LoRA-ESM2 Hybrid v1

## Scientific question

CenterLoRA-ESM2 v1 showed that task-adapting ESM-2 is useful, but the
ProtBERT screen and its historical-test evaluation suggest that a central
language-model representation alone may not recover enough explicit local
motif and physicochemical evidence. This experiment asks whether complementary
site-level experts can improve discrimination while retaining the much smaller
and faster ESM-2 35M backbone.

This is a new development experiment, not a revision chosen from the released
MMUbiPred test labels. The previously evaluated MMUbiPred test is treated as a
historical benchmark and is not read by the screen.

## Architecture

All three experts receive the same 49-residue window centred on the candidate
lysine:

1. **Contextual ESM-2 expert.** `facebook/esm2_t12_35M_UR50D` is adapted with
   rank-8 LoRA modules on the query and value projections. The central-residue
   representation and a masked mean of the residue representations are
   projected to 128 dimensions.
2. **Local motif expert.** The 20-channel one-hot sequence passes through
   parallel dilated convolutions with kernel/dilation pairs `(3,1)`, `(5,2)`,
   and `(7,3)`. Their receptive fields are 3, 9, and 19 residues. A residual
   convolution block followed by masked max and mean pooling yields a
   128-dimensional motif vector.
3. **Physicochemical expert.** The paper's same 31 normalized AAindex values
   pass through a 64-unit bidirectional GRU. The central state and masked mean
   state are projected to 128 dimensions.

A sample-dependent softmax gate weights the three expert vectors. A residual
projection of their concatenation is added to the gated representation before
the final classifier. This allows both specialization and direct
cross-expert interactions.

During training, a small auxiliary binary-classification head is attached to
each expert. The mean branch loss is weighted by `0.2` and added to the fused
binary cross-entropy. This deep supervision is intended to prevent the gate
from prematurely suppressing a branch before that branch learns useful
features. Auxiliary heads are not used for the final prediction.

## Controlled development protocol

- Seed: 42 for the first screen.
- Data: the released training set only.
- Split: the identical seed-42 protein-grouped split used by CenterLoRA-ESM2
  v1, verified by validation-index hash.
- Schedule: all 15 epochs; the best checkpoint is selected by development MCC
  at threshold 0.5.
- Optimizer: AdamW, learning rate `0.0002`, weight decay `0.0001`.
- Effective and physical batch size: 128.
- Automatic mixed precision: enabled on CUDA.
- Resume: exact epoch-boundary state is stored in Google Drive.

The architecture is evaluated at the fixed threshold of 0.5. A threshold
selected on the development fold is secondary and is never substituted for
the fixed-threshold comparison.

## Predeclared decision rule

The direct ESM-2 v1 seed-42 reference is:

- fixed MCC `0.5689370097`;
- selected-threshold MCC `0.5803138265`;
- AUROC `0.8799274072`; and
- AUPRC `0.8964866895`.

The strongest completed seed-42 development candidate, ProtBERT v1, has fixed
MCC `0.5827515028`.

The result is classified prospectively:

- **GO:** fixed MCC is at least `0.5827515028`, selected-threshold MCC is not
  below `0.5803138265`, AUROC and AUPRC are each no more than `0.002` below
  ESM-2 v1, and predictions do not collapse to one class.
- **PROMISING BUT NOT CHAMPION:** the safety conditions hold and fixed MCC
  improves over ESM-2 v1 by at least `0.010`, but remains below the ProtBERT
  fixed-MCC result.
- **STOP:** any other result.

A GO result is permission for multi-seed development confirmation, not
permission to reuse the historical test for architecture selection.

## Evaluation-status disclosure

The released MMUbiPred independent test has already been evaluated for the
frozen ProtBERT candidate. It therefore cannot serve as an untouched
independent test for this later hybrid. The development screen does not load
it. Notebook 08 contains a separate opt-in evaluation of the frozen
development-best checkpoint solely for exploratory historical comparison.
That result cannot reverse a STOP decision or guide another architecture. If
this architecture were otherwise frozen after development, publication-grade
confirmation would require a newly curated, homology-aware blind external test
set. The released test may only be reported as a transparently labelled
historical benchmark.

## External-pretraining disclosure

ESM-2 supplies external unsupervised pretraining. Although the labelled
training data remain identical to the paper's, the eventual manuscript must
disclose the pretrained checkpoint and assess homology-related leakage or
performance inflation.
