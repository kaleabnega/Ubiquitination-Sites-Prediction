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

## Recorded protocol amendment: local-expert stabilization

The original five-fold run completed with strong context-expert performance
but local fold 4 predicted every outer sample as positive (`MCC=0`,
`sensitivity=1`, `specificity=0`). Consequently, the original automatic `GO`
was invalid: its reference OOF vector contained a collapsed fold. No context
model or context prediction is changed by this amendment.

Before any final refit or historical-test use, all five inexpensive local
folds are repeated under one uniform multi-initialization rule:

1. keep the original outer and inner protein-grouped partitions fixed;
2. train local candidates with initialization seeds `42`, `123`, and `2026`;
3. reject candidates producing one-class inner-validation predictions;
4. select maximum inner fixed-threshold MCC, breaking ties by AUPRC and then
   ascending seed; and
5. evaluate only the selected candidate on the outer fold.

The outer fold is never used for initialization selection. Applying the same
rule to every fold avoids a fold-4-only retry. The corrected summary reads the
unchanged context predictions, uses the stabilized local OOF vector, and
automatically returns `INVALID_COLLAPSED_EXPERT` if either expert still has a
one-class outer fold.

The amendment also prevents a weak fusion from advancing merely because it
beats the local baseline. Residual fusion is selected only if its fixed MCC
exceeds context-only MCC by at least `0.005`, with the same accuracy and
ranking-metric safeguards. Otherwise, context alone advances if it improves
fixed MCC over the stabilized local expert by at least `0.01` while satisfying
those safeguards. These outcomes are reported as `ADVANCE_RESIDUAL_STACK`,
`ADVANCE_CONTEXT_ONLY`, `STOP`, or `INVALID_COLLAPSED_EXPERT`.

## Frozen corrected result and final refit

The uniform stabilization completed without collapsed outer folds. At the
fixed `0.5` threshold, the corrected local expert obtained OOF MCC `0.55626`,
accuracy `0.75375`, AUROC `0.85650`, and AUPRC `0.88590`. The unchanged
context expert obtained MCC `0.57901`, accuracy `0.78695`, AUROC `0.88123`,
and AUPRC `0.90243`. The residual stack reached MCC `0.58172`, only `0.00271`
above context, so it did not meet the `0.005` minimum fusion gain. The frozen
decision is therefore `ADVANCE_CONTEXT_ONLY`.

The context inner-validation best epochs across folds 0–4 were
`[8, 8, 7, 6, 7]`. The final refit epoch count is their integer median:
seven. A fresh context-only model is refitted with the unchanged optimizer,
learning rates, LoRA configuration, 257-residue input, and seed 42 on all
89,551 released training records that passed the frozen sequence validation.
No validation set, threshold search, early stopping, fusion fitting, or test
loader is used in this refit. The primary future reporting threshold remains
`0.5`.

The refit writes an exact epoch-boundary resume checkpoint after every epoch.
Its final compact checkpoint stores the trainable LoRA and task-head
parameters plus immutable model, cache, decision-summary, and eligible-record
hashes. Re-running with `--resume` continues from the next unfinished epoch.

The frozen checkpoint may then be evaluated once on reconstructed
independent-test contexts at the primary threshold `0.5`. Test sequence
retrieval uses a separate hashed Drive cache and applies the identical
coordinate, central-lysine, released-49-mer, and residue-alphabet validation.
Coverage and every exclusion reason are reported. A direct comparison with
the reproduced full-test MMUbiPred aggregate is valid only if every processed
test record passes context validation. Otherwise, the primary comparison
requires MMUbiPred predictions restricted to the exact saved context-valid
test indices.

## Post-test exploratory epoch extension

After the epoch-7 historical-test result was inspected, an explicitly
exploratory duration ablation was added to test whether the final refit had
been undertrained. It restores the exact epoch-7 model, AdamW optimizer, AMP
scaler, DataLoader generator, and random-number-generator states, then
continues epochs 8–15. The frozen run is never modified, and no learning rate,
loss, model, seed, data, batch, accumulation, or threshold setting changes.

Because training duration was reconsidered after observing the historical
test, the epoch-15 result cannot replace the epoch-7 primary result or support
a confirmatory superiority claim. It is recorded as test-informed exploratory
evidence and would require confirmation on a new external test set.

## Post-test exploratory residual-hybrid refit

The residual stack originally failed the predeclared advancement margin by
improving fixed-threshold OOF MCC only from `0.57901` to `0.58172`. After the
context-only historical-test results were inspected, a separate exploratory
pipeline was added to measure—not reselect—this previously specified hybrid.
The frozen seven-epoch context checkpoint and the final stacker fitted to all
corrected OOF predictions are reused without modification.

Only the MMUbiPred-compatible local expert is refitted. Its initialization is
the most frequently selected non-collapsed seed across the five stabilized
folds, with an ascending-seed tie-break. Its duration is the integer median of
the corresponding five inner-validation best epochs. It trains on the same
89,551 context-validated released training records as the context refit and
never reads validation or test data.

Historical-test evaluation uses the exact 12,288-record context-valid cohort,
the already saved frozen context probabilities, and threshold `0.5`. The
stacker intercept and nonnegative local/context weights cannot be refitted,
threshold-tuned, or selected on the test. Results are compared side by side
with the released MMUbiPred checkpoint, the newly refitted local expert, and
the frozen context expert. Because the decision to revisit a candidate that
failed its original advancement margin occurred after historical-test
inspection, every output is permanently marked post-test exploratory and
cannot support a confirmatory superiority claim.
