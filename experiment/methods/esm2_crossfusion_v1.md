# ESM2-CrossFusion v1

## Scientific question

Can pretrained protein-sequence context improve ubiquitination-site
prediction beyond the stabilized MMUbiPred-compatible development baseline,
while the labelled training and independent-test samples remain exactly those
released by the authors?

## Architecture

The input is the same 49-residue candidate-site window used by MMUbiPred. Gap
padding is removed only for the ESM-2 call and the candidate lysine position is
tracked after compaction.

1. **Context branch:** frozen `facebook/esm2_t6_8M_UR50D` representations at
   the candidate lysine and over the masked mean of the residue window.
2. **Local-motif branch:** explicit one-hot amino-acid identities processed by
   three dilated convolutions, followed by a residual block and max/mean
   pooling.
3. **Biochemical branch:** the same 31 normalized AAindex properties used by
   the paper, summarized at the candidate residue and across valid residues.
4. **Fusion:** a learned softmax gate assigns sample-specific weights to the
   three projected branches. A projection of their concatenation is added as
   a residual path before binary classification.

The initial screen freezes all ESM-2 weights. This limits trainable capacity,
GPU memory, and the risk of overfitting. Fine-tuning is a later, explicitly
versioned experiment only if the frozen screen is competitive.

## Evaluation contract

- The authors' released training set supplies all supervised fitting data.
- Seed 42 uses the existing protein-grouped development split.
- The first screen selects by development MCC at threshold 0.5.
- The independent test remains locked.
- A competitive screen must be repeated over seeds 42, 123, and 2026 and
  compared on identical split hashes before any final refit.
- The fixed 0.5 threshold remains the primary paper-comparable operating point.

## Pretraining disclosure

ESM-2 was pretrained without the ubiquitination labels used here, but it is an
external unsupervised data source. Manuscript reporting therefore distinguishes
the unchanged labelled train/test split from the large protein-sequence corpus
used to pretrain the representation model. The primary same-data comparison is
accompanied by a protein-homology-aware robustness analysis.

## References

- [Hugging Face ESM documentation](https://huggingface.co/docs/transformers/model_doc/esm)
- [Official ESM-2 8M model card](https://huggingface.co/facebook/esm2_t6_8M_UR50D)
- [Lin et al., *Science* (2023)](https://www.science.org/doi/10.1126/science.ade2574)
- [Hermann, PMLR (2024): pretraining-data leakage in protein language models](https://proceedings.mlr.press/v261/hermann24a.html)
