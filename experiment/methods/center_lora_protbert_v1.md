# CenterLoRA-ProtBERT v1

## Scientific question

CenterLoRA-ESM2 v1 remains the development champion after the multi-scale v2
head failed its predeclared screen. This final architecture-exploration
experiment asks whether the result depends on the selected pretrained protein
language model or whether ProtBERT-BFD provides a stronger central
candidate-site representation.

Only the pretrained backbone changes. The independent test remained locked
throughout the screen. After the passing result, architecture selection closed
and a one-time evaluation of the frozen seed-42 checkpoint was authorized.

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
and verifies the compacted central-lysine position exactly as in v1. It reads
ProtBERT's released `vocab.txt` directly and validates every canonical residue
plus PAD, UNK, CLS, and SEP. No Transformers tokenizer backend is instantiated,
which avoids unsupported legacy-to-fast conversion while preserving the exact
published token IDs.

The released ProtBERT configuration predates the `model_type` field required
by modern `AutoModel`. The implementation reads the released `config.json`
directly into `BertConfig` and loads `BertModel` explicitly; model dimensions
and pretrained weights still come from the released checkpoint.

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

## Frozen seed-42 final-evaluation protocol

The seed-42 screen passed every predeclared condition on the identical
development split:

- fixed-threshold MCC: 0.5827515028;
- selected-threshold MCC: 0.5890905043;
- AUROC: 0.8871901712;
- AUPRC: 0.9024115065; and
- no single-class prediction collapse.

The fixed MCC gain over CenterLoRA-ESM2 v1 was 0.0138144930. Because the
approximately six-hour ProtBERT run made immediate three-seed confirmation
impractical, the architecture was frozen after this passing screen. This is a
resource-driven deviation from the preferred three-seed confirmation and must
be disclosed as a limitation.

Before test access, the resource-constrained final protocol was revised to
evaluate the already selected seed-42 checkpoint directly:

1. use `development_best.pt`, the epoch-4 checkpoint selected without test
   access;
2. retain the 82,535-record fitting subset and 9,188-record development
   validation subset exactly as used during selection;
3. perform no additional training or full-data refit;
4. report threshold 0.5 as the primary result;
5. report the development-selected threshold 0.555 only as secondary; and
6. evaluate once on the released 12,598-record independent test.

This is a valid train/development/test protocol, but it uses fewer labelled
training records than a complete-training refit. Both the single-seed evidence
and the 10% development holdout must therefore be disclosed when comparing
with the authors' model.

No architecture, optimization setting, epoch count, or reporting threshold may
change after this evaluation. Later seeds may repeat the identical frozen
protocol, but all such results must be reported rather than selecting the best
seed.

## External-pretraining disclosure

Both candidates use external unsupervised protein pretraining, but their
pretraining corpora and model sizes differ. This screen is a backbone ablation,
not a parameter-matched comparison. Any eventual publication must disclose
those differences and cannot attribute a ProtBERT gain solely to architecture.
