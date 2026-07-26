# Publication experiments

This directory contains the clean experimental pipeline for developing and
evaluating models that improve on MMUbiPred while using the authors' exact
released training and independent-test samples.

The completed replication remains frozen under `replication/`. Code in this
directory must not modify the upstream `replication/MMUbiPred` submodule.

## Primary comparison contract

The headline comparison uses only these released labelled samples:

- Training positives: `Positive_90_percent_training_set_DeepUBI.fasta`
- Training negatives: `Negative_90_percent_training_set_DeepUBI.fasta`
- Test positives: `Positive_10_percent_independent_test_set_DeepUBI.fasta`
- Test negatives: `Negative_10_percent_independent_test_set_DeepUBI.fasta`

The released test was locked during the original architecture-selection
sequence and then evaluated once for the frozen ProtBERT candidate. It is now a
historical benchmark. Later architecture selection uses only released training
data, and a new blind external set is required for a genuinely untouched final
evaluation. The fixed 0.5 threshold remains the primary direct-comparison
threshold.

## Pilot model: UbiFusionNet v1

UbiFusionNet v1 is an architecture-only experiment using the same three input
sources as MMUbiPred:

1. amino-acid identity through a trainable embedding and transformer;
2. one-hot sequence features through a residual multi-kernel CNN; and
3. the same 31 normalized AAindex properties through a bidirectional GRU.

Each branch produces a site representation rather than an independent class
score. A learned softmax gate fuses these representations before the final
classifier. This directly tests whether feature-level, sample-dependent fusion
is better than MMUbiPred's score-level fusion without adding labelled data.

The corrected three-seed validation benchmark rejected this pilot: its mean
fixed-threshold MCC was 0.5167 versus 0.5639 for the stabilized compatible
baseline. The independent test was not accessed. See
[`results/2026-07-22-validation-pilot-v1`](results/2026-07-22-validation-pilot-v1/README.md).

## Rejected screen: ESM2-CrossFusion v1

This candidate combined contextual representations from the frozen
`facebook/esm2_t6_8M_UR50D` protein language model with a dilated local-motif
CNN and the paper's normalized AAindex properties. Three site-level vectors
are combined by sample-dependent gates plus a residual fusion projection.

The seed-42 screen was rejected at fixed-threshold MCC 0.4981. Training loss
continued to improve after validation MCC peaked, and the fusion gate was
dominated by the frozen ESM branch. The independent test was not accessed. See
[`results/2026-07-22-esm2-crossfusion-v1-screen`](results/2026-07-22-esm2-crossfusion-v1-screen/README.md).

## Three-seed ESM-2 reference: CenterLoRA-ESM2 v1

This ablation tests a single, sharper hypothesis: task adaptation of an
intermediate ESM-2 model. Rank-8 LoRA adapters modify the query and value
attention projections of `facebook/esm2_t12_35M_UR50D`, and a compact head
classifies the representation at the central candidate lysine. It deliberately
removes the CNN, AAindex, and learned branch gate.

The full-15 paired development comparison completed for seeds 42, 123, and
2026 without collapsed runs. Against the stabilized compatible baseline, v1
improved mean fixed-threshold MCC from 0.5639 to 0.5680, AUROC from 0.8628 to
0.8795, and AUPRC from 0.8773 to 0.8971. The test remained locked during this
comparison. The frozen compact record is in
[`results/2026-07-25-center-lora-esm2-v1-development`](results/2026-07-25-center-lora-esm2-v1-development/README.md).

The completed benchmark can be resumed without retraining:

```bash
python experiment/scripts/run_validation_benchmark.py \
  --suite experiment/configs/center_lora_esm2_full15_three_seed_v1.json \
  --resume
```

[Open the full-15 three-seed benchmark in Google Colab](https://colab.research.google.com/github/kaleabnega/Ubiquitination-Sites-Prediction/blob/main/experiment/notebooks/05_CenterLoRA_ESM2_Full15_ThreeSeed_Colab.ipynb)

The notebook uninstalls Colab's incompatible optional `torchao 0.10.0` package
after installing the pinned Transformers/PEFT versions. CenterLoRA does not use
TorchAO or quantized weights.

The benchmark runner accepts `--output-root` for persistent Google Drive
storage and `--resume` to reuse completed seeds after a runtime disconnection.
The Colab notebook then merges the three candidate runs with the already
preserved stabilized baseline and verifies identical split hashes. At that
stage, only a candidate selected without seeing the released test was eligible
for its one-time evaluation.

## Stopped screen: MultiScale CenterLoRA-ESM2 v2

V2 preserves the same ESM-2 backbone, LoRA adapters, optimizer, and full
15-epoch schedule as v1. Its only controlled change is the classification
head. It combines the contextual central lysine with masked mean pools over
radius-2 and radius-5 neighbourhoods through a shared projection and learned
softmax gate.

The predeclared seed-42 screen compares v2 directly with the saved full-15 v1
seed-42 result on the identical protein-grouped split. V2 must improve fixed
MCC by at least 0.005 without reducing AUROC or AUPRC by more than 0.002.
Details are frozen in
[`methods/center_lora_esm2_multiscale_v2.md`](methods/center_lora_esm2_multiscale_v2.md).
The released test remained locked at that stage.

[Open the MultiScale CenterLoRA-ESM2 v2 screen in Google Colab](https://colab.research.google.com/github/kaleabnega/Ubiquitination-Sites-Prediction/blob/main/experiment/notebooks/06_MultiScale_CenterLoRA_ESM2_v2_Screen_Colab.ipynb)

The screen completed at fixed MCC 0.5724, a gain of 0.00345 over v1 seed 42.
It missed the material-gain target, and its AUROC, AUPRC, and
selected-threshold MCC were lower than v1. The contextual pools received 39.3%
of mean fusion weight but did not improve discrimination. V2 was therefore
stopped without additional seeds or test access. See the
[`frozen screen result`](results/2026-07-26-center-lora-esm2-multiscale-v2-screen/README.md).

## Backbone screen and historical-test result: CenterLoRA-ProtBERT v1

This experiment changes only the pretrained backbone of CenterLoRA v1 from
ESM-2 35M to `Rostlab/prot_bert_bfd`. The central-residue head, rank-8 LoRA
protocol, seed-42 protein-grouped split, full 15-epoch schedule, optimizer, and
effective batch size remain fixed.

ProtBERT-BFD is much larger, so the physical batch is 16 with eight-step
gradient accumulation, preserving an effective batch of 128. Gradient
checkpointing controls activation memory. Lightweight trainable-state and RNG
checkpoints are written after each epoch so a Drive-backed Colab run can resume
without repeating completed epochs.

The fixed development-screen contract is documented in
[`methods/center_lora_protbert_v1.md`](methods/center_lora_protbert_v1.md).
ProtBERT must improve fixed MCC over ESM-2 v1 by at least 0.005, preserve
selected-threshold MCC, and satisfy the AUROC/AUPRC safeguards. Any failure
would have retained ESM-2 v1.

[Open the CenterLoRA-ProtBERT v1 screen in Google Colab](https://colab.research.google.com/github/kaleabnega/Ubiquitination-Sites-Prediction/blob/main/experiment/notebooks/07_CenterLoRA_ProtBERT_v1_Screen_Colab.ipynb)

The seed-42 screen passed at fixed development MCC `0.58275`. Its frozen
epoch-4 checkpoint subsequently obtained MCC `0.54997` on the released test at
threshold 0.5, compared with `0.54584` for the reproduced paper model. The
small `0.00413` numerical gain is accompanied by lower accuracy and
sensitivity, is based on one seed, and does not establish robust superiority.
The released test is no longer untouched for future architecture work.

## Current screen: LoRA-ESM2 Hybrid v1

This candidate uses the faster `facebook/esm2_t12_35M_UR50D` backbone with the
same rank-8 attention LoRA protocol. It combines three feature-level experts:

1. contextual central and mean-pooled ESM-2 representations;
2. multi-receptive-field one-hot motif convolutions; and
3. a bidirectional GRU over the paper's normalized AAindex features.

Sample-dependent gates and a residual projection fuse the experts. Auxiliary
branch losses provide deep supervision during training so that an initially
weak expert is not immediately suppressed by the gate.

The predeclared seed-42 screen uses the identical protein-grouped development
split as ESM-2 v1 and runs all 15 epochs. A performance GO requires matching or
exceeding ProtBERT's fixed development MCC `0.58275`, while preserving the
ESM-2 reference's selected MCC and ranking metrics. It does not load the
released test.

[Open the LoRA-ESM2 Hybrid v1 screen in Google Colab](https://colab.research.google.com/github/kaleabnega/Ubiquitination-Sites-Prediction/blob/main/experiment/notebooks/08_LoRA_ESM2_Hybrid_v1_Screen_Colab.ipynb)

The complete contract is in
[`methods/lora_esm2_hybrid_v1.md`](methods/lora_esm2_hybrid_v1.md).

## Layout

```text
experiment/
├── configs/                 Versioned experiment configurations
├── methods/                 Architecture provenance and implementation notes
├── notebooks/               Thin Colab orchestration notebooks
├── scripts/                 Auditing, training, and evaluation entry points
├── src/ubipred/             Reusable data, model, metric, and training code
├── tests/                   Unit and smoke tests
├── outputs/                 Local/Colab run artifacts (not committed)
└── results/                 Accepted compact result records (committed)
```

## Local audit

The dataset audit uses only the Python standard library:

```bash
python3 experiment/scripts/audit_dataset.py \
  --data-dir replication/MMUbiPred \
  --output experiment/outputs/dataset_audit.json
```

## Training

Training is intended for Google Colab. From the project root:

[Open UbiFusionNet v1 in Google Colab](https://colab.research.google.com/github/kaleabnega/Ubiquitination-Sites-Prediction/blob/main/experiment/notebooks/01_UbiFusionNet_v1_Colab.ipynb)

For a private repository, add a fine-grained GitHub token with read-only access
to this repository to the Colab Secrets panel as `GITHUB_TOKEN`, then enable
notebook access to that secret. The notebook uses a temporary Git askpass
helper, so the token is not embedded in the clone URL or saved Git remote.

```bash
python experiment/scripts/train.py \
  --config experiment/configs/ubifusion_v1.json
```

The development split selects the epoch count and optional reporting
threshold. The pipeline then initializes a fresh model and refits it for that
many epochs on all 91,723 released training samples. It writes the final
checkpoint, histories, validation predictions, and provenance manifest to the
configured output directory. It does **not** access the locked independent
test set.

## Paired development benchmark

Before evaluating another architecture on the independent test set, compare it
against the [MMUbiPred-compatible reimplementation](methods/mmubipred_compatible.md)
on identical protein-grouped development splits. The default suite runs both
models for seeds 42, 123, and 2026 and reports mean, standard deviation, and
paired candidate-minus-baseline differences.

[Open the paired benchmark in Google Colab](https://colab.research.google.com/github/kaleabnega/Ubiquitination-Sites-Prediction/blob/main/experiment/notebooks/02_Paired_Validation_Benchmark_Colab.ipynb)

From the repository root, the equivalent command is:

```bash
python experiment/scripts/run_validation_benchmark.py \
  --suite experiment/configs/paired_baseline_v1.json
```

All suite runs pass `--development-only`, which prevents full-data refitting
and does not expose the independent test. A second suite,
`experiment/configs/ubifusion_ablation.json`, compares gated UbiFusionNet v1
against its no-context and fixed-mean-fusion variants after the baseline stage.

If a reconstructed baseline run collapses to one-class predictions, do not use
the aggregate comparison. The focused stability suite reruns only the baseline
with Keras-compatible initialization and numerical settings:

```bash
python experiment/scripts/run_validation_benchmark.py \
  --suite experiment/configs/mmubipred_stability_v2.json
```

Combine those corrected reference runs with the already completed UbiFusionNet
runs without retraining the candidate:

```bash
python experiment/scripts/merge_validation_benchmarks.py \
  --reference-summary experiment/outputs/benchmarks/mmubipred_stability_v2/summary.json \
  --reference-model mmubipred_compatible \
  --candidate-summary experiment/outputs/benchmarks/paired_baseline_v1/summary.json \
  --candidate-model ubifusion_v1 \
  --benchmark-name paired_baseline_stabilized_v2 \
  --output-dir experiment/outputs/benchmarks/paired_baseline_stabilized_v2
```

## Historical released-test evaluation

The evaluator below was used only after the original ProtBERT architecture and
hyperparameters were selected using training/validation data:

```bash
python experiment/scripts/evaluate.py \
  --run-dir experiment/outputs/SELECTED_FINAL_RUN \
  --data-dir replication/MMUbiPred \
  --allow-locked-test
```

Both fixed-threshold and validation-selected-threshold results are saved. The
fixed-threshold MMUbiPred reference is `0.5458386140298045`. Because this
released test has now been accessed, do not use it to choose or revise the
hybrid architecture. A new blind external test is required for the eventual
confirmatory claim.

## Released-data audit findings

The authors' alphabet filter retains 91,723 training samples (46,587 positive,
45,136 negative) and 12,598 test samples (7,578 positive, 5,020 negative).
There is no protein-ID or site-ID overlap between training and test. There are,
however, three identical cropped 49-residue windows across different train and
test proteins, and one duplicated training window has conflicting labels.

These records are retained in the primary same-data comparison. Removing them
would make our evaluated dataset differ from the paper. They will be addressed
separately in the homology-aware robustness track.
