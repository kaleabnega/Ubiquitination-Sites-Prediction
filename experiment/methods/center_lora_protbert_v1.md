# CenterLoRA-ProtBERT v1

## Scientific question

CenterLoRA-ESM2 v1 remains the development champion after the multi-scale v2
head failed its predeclared screen. This final architecture-exploration
experiment asks whether the result depends on the selected pretrained protein
language model or whether ProtBERT-BFD provides a stronger central
candidate-site representation.

Only the pretrained backbone changes. The independent test remains locked.
After this screen and any authorized paired confirmation, architecture
selection closes permanently.

## Architecture and controlled comparison

The experiment replaces `facebook/esm2_t12_35M_UR50D` with
`Rostlab/prot_bert_bfd`. ProtBERT-BFD is a 30-layer BERT model with
1024-dimensional residue representations, pretrained without labels on the
BFD protein sequence collection.

All task-specific choices are retained from CenterLoRA-ESM2 v1:

1. the same 49-residue window centred on the candidate lysine;
2. rank-8 LoRA adapters on attention query and value projections;
3. LoRA alpha 16 and adapter dropout 0.1;
4. the final contextual representation at the central lysine;
5. layer normalization, a 256-unit GELU projection, 0.3 dropout, and one
   binary logit; and
6. ordinary binary cross-entropy and AdamW with learning rate 0.0002.

ProtBERT uses CLS and SEP tokens instead of ESM's CLS and EOS convention. The
implementation constructs the token IDs directly, excludes terminal padding,
and verifies the compacted central-lysine position exactly as in v1. It loads
ProtBERT's released slow WordPiece tokenizer explicitly (`use_fast=false`);
this avoids an unsupported fast-tokenizer conversion and does not change token
IDs or model inputs.

## Memory and numerical protocol

ProtBERT-BFD is substantially larger than ESM-2 35M. Gradient checkpointing
reduces activation memory but does not change the model or objective. A
physical batch size of 16 is accumulated for 8 steps, preserving v1's
effective batch size of 128. Automatic mixed precision remains enabled.

The run saves trainable LoRA/head state, optimizer state, scaler state, random
number generator state, data-loader generator state, history, and best state
after every completed epoch. A disconnected Colab runtime can therefore resume
at the next epoch without repeating completed epochs. `num_workers=0` is
intentional so epoch-boundary data order is reproducible after a restart.

## Predeclared seed-42 screen

The frozen v1 seed-42 reference values are:

- fixed MCC: 0.5689370097;
- selected-threshold MCC: 0.5803138265;
- AUROC: 0.8799274072; and
- AUPRC: 0.8964866895.

ProtBERT passes only if all conditions hold:

- fixed MCC is at least 0.5739370097, a gain of at least 0.005;
- selected-threshold MCC does not decrease;
- AUROC is at least 0.8779274072;
- AUPRC is at least 0.8944866895; and
- predictions do not collapse to one class.

Seed 42 uses the identical protein-grouped split and split hash as v1. All 15
epochs run, while the best development checkpoint by MCC at threshold 0.5 is
retained. No hyperparameter is changed after observing this result.

A pass authorizes a paired three-seed development confirmation. Any other
outcome retains ESM-2 v1 and closes architecture exploration without further
ProtBERT runs.

## External-pretraining disclosure

Both candidates use external unsupervised protein pretraining, but their
pretraining corpora and model sizes differ. This screen is a backbone ablation,
not a parameter-matched comparison. Any eventual publication must disclose
those differences and cannot attribute a ProtBERT gain solely to architecture.
