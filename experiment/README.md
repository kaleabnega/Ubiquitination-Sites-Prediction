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

The independent test set is locked. Architecture selection, early stopping,
and threshold selection use only the released training data. The fixed 0.5
threshold is always reported for direct comparison with MMUbiPred.

## First proposed model: UbiFusionNet v1

UbiFusionNet v1 is an architecture-only experiment using the same three input
sources as MMUbiPred:

1. amino-acid identity through a trainable embedding and transformer;
2. one-hot sequence features through a residual multi-kernel CNN; and
3. the same 31 normalized AAindex properties through a bidirectional GRU.

Each branch produces a site representation rather than an independent class
score. A learned softmax gate fuses these representations before the final
classifier. This directly tests whether feature-level, sample-dependent fusion
is better than MMUbiPred's score-level fusion without adding labelled data.

Protein-language-model, structure, and positive-unlabelled extensions belong
in later experiments after this same-data architecture baseline is measured.

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

## Locked evaluation

Run this only after the architecture and hyperparameters have been selected
using training/validation data:

```bash
python experiment/scripts/evaluate.py \
  --run-dir experiment/outputs/ubifusion_v1_seed42 \
  --data-dir replication/MMUbiPred \
  --allow-locked-test
```

Both fixed-threshold and validation-selected-threshold results are saved. The
fixed-threshold MCC target to beat is `0.5458386140298045`.

## Released-data audit findings

The authors' alphabet filter retains 91,723 training samples (46,587 positive,
45,136 negative) and 12,598 test samples (7,578 positive, 5,020 negative).
There is no protein-ID or site-ID overlap between training and test. There are,
however, three identical cropped 49-residue windows across different train and
test proteins, and one duplicated training window has conflicting labels.

These records are retained in the primary same-data comparison. Removing them
would make our evaluated dataset differ from the paper. They will be addressed
separately in the homology-aware robustness track.
