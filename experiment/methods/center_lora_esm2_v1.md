# CenterLoRA-ESM2 v1

## Scientific question

The frozen ESM2-CrossFusion pilot was dominated by its ESM branch but
generalized poorly. This experiment asks whether parameter-efficient,
task-specific adaptation of an intermediate ESM-2 model can produce a useful
candidate-site representation without a complex hand-crafted fusion network.

## Architecture

1. Use the same 49-residue windows and central candidate lysine as MMUbiPred.
2. Remove terminal gap padding before ESM tokenization while preserving the
   compacted position of the central lysine.
3. Load `facebook/esm2_t12_35M_UR50D`.
4. Freeze the pretrained weights and attach rank-8 LoRA adapters to the query
   and value projections in its attention layers. LoRA alpha is 16 and adapter
   dropout is 0.1.
5. Pass the final central-residue representation through layer normalization,
   a 256-unit GELU projection, dropout, and one binary logit.

No CNN, AAindex, fusion gate, focal loss, or additional labelled data is used.
The released training data are approximately balanced, so ordinary binary
cross-entropy is retained.

The Colab notebook removes the optional preinstalled `torchao` package before
training. Colab currently supplies `torchao 0.10.0`, while PEFT 0.19.1 rejects
TorchAO versions below 0.16 during adapter injection. This experiment does not
use quantization or TorchAO, so removing that optional package changes neither
the architecture nor the numerical training protocol.

## Selection contract

- Seed: 42 for the first screen.
- Split: the existing protein-grouped development split.
- Maximum epochs: 15.
- Early-stopping patience: 4 epochs.
- Primary screen metric: MCC at threshold 0.5.
- Go criterion: fixed-threshold MCC at least 0.56.
- The independent test remains locked.

A passing screen is only permission to run seeds 42, 123, and 2026. It is not
evidence of superiority by itself. A failing screen ends this version without
test-set evaluation.

## Full-15 three-seed protocol

The early-stopped seed-42 screen passed its go criterion with fixed-threshold
MCC 0.5689. The confirmatory development benchmark is versioned separately as
`center_lora_esm2_full15_v1`:

- seeds 42, 123, and 2026 each train for all 15 epochs;
- early-stopping patience equals the 15-epoch maximum, so it cannot terminate
  a run early;
- the best checkpoint by development MCC at threshold 0.5 is still retained;
- every run uses the same protein-grouped split rule and seed as the stabilized
  compatible baseline; and
- the independent test remains locked.

Running every epoch provides complete learning curves. Retaining the best
development checkpoint avoids knowingly selecting an overfitted epoch merely
because it was last.

## External-pretraining disclosure

The labelled comparison uses the authors' exact released train/test split, but
ESM-2 contributes external unsupervised pretraining. This must be disclosed in
the eventual paper and accompanied by a homology-aware robustness analysis.

## References

- [Official ESM documentation](https://huggingface.co/docs/transformers/model_doc/esm)
- [Official PEFT LoRA reference](https://huggingface.co/docs/peft/package_reference/lora)
- [Official PEFT feature-extraction model example](https://huggingface.co/docs/peft/en/package_reference/peft_model)
- [Parameter-efficient fine-tuning of protein language models](https://pmc.ncbi.nlm.nih.gov/articles/PMC10659351/)
