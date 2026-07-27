# MMUbiPred–Context Residual v1

## Motivation

MMUbiPred already captures strong signal within its 49-residue input. Earlier
feature-level hybrids sometimes suppressed useful branches, and longer
training did not repair their generalization. This experiment therefore
preserves the paper-compatible local model as an expert and adds information
it cannot observe: wider, target-centred protein context.

This is a development experiment. The released independent test has already
become a historical benchmark and is not read by this pipeline.

## Inputs and validation

The labels, protein accessions, lysine positions, 49-residue inputs, and
AAindex table remain those released with MMUbiPred. No labelled examples are
added.

Full sequences are retrieved through the
[documented UniProt REST query API](https://www.uniprot.org/help/api_queries).
Every site must have a
valid position, a lysine at that position, and a 49-residue crop that exactly
reproduces the released input. The released PLMD/MMUbiPred header coordinates
are zero-based and are converted explicitly to UniProt's one-based residue
positions. Failures are excluded from both experts, reported by reason, and
hashed. The experiment stops below 95% coverage.

## Architecture

The local expert is the complete MMUbiPred-compatible topology: its AAindex
LSTM, one-hot CNN, learned-embedding CNN, and score-level dense fusion.

The context expert applies rank-8 LoRA adapters (`alpha=16`, dropout `0.1`) to
the query and value projections of `facebook/esm2_t12_35M_UR50D`. It consumes
a 257-residue target-centred window and extracts the contextual central
lysine, a masked radius-24 mean, and a masked global mean. A shared
256-dimensional projection transforms each component. Their concatenation is
classified by a compact dropout-regularized MLP. Concatenation is deliberate:
unlike a competitive softmax gate, it cannot silently discard a scale early
in training.

LoRA uses learning rate `1e-4`; the task head uses `3e-4`. Training uses
AdamW, weight decay `1e-4`, mixed precision, gradient clipping at 1.0, and an
effective batch size of 128. It is capped at 15 epochs with patience 4.

The experts are combined in logit space:

```text
z = b + w_local * logit(p_local) + w_context * logit(p_context)
```

Both weights are nonnegative. L2 regularization shrinks them toward `(1, 0)`,
so context must demonstrate complementary signal rather than destabilizing a
strong local prediction.

## Fold-safe evaluation

Five outer folds are made with `StratifiedGroupKFold`, keeping every site from
one UniProt accession in one fold. Within each outer-training partition, a
protein-grouped inner 90/10 split selects the expert checkpoint. The selected
expert then predicts the untouched outer fold.

After both out-of-fold expert vectors are complete, the residual stacker is
itself cross-fitted: a fold's stacked predictions come from weights fitted
only on the other four folds. This cross-fitted stack is the primary
development result. Threshold 0.5 is primary; OOF-selected thresholds are
exploratory.

## Predeclared decision rule

The architecture advances only if the cross-fitted stack:

- improves fixed-threshold MCC over the fold-matched local expert by at least
  `0.01`;
- does not reduce accuracy by more than `0.005`; and
- does not reduce AUROC or AUPRC by more than `0.002`.

Individual folds and fitted weights must also be checked for failure. Passing
freezes the architecture and hyperparameters; it does not authorize tuning on
the historical MMUbiPred test.

## Compute and restart behavior

There are ten expert-fold jobs: five local and five context. Notebook 09 can
run one outer fold per Colab session. State is saved after every epoch, and
completed outer predictions are reused with `--resume`. The OOF summary is
generated automatically after all jobs exist.
