# Ubiquitination Sites Prediction

This repository is a reproducible research project built around **MMUbiPred**,
the multimodal deep-learning method introduced in:

> Subash C. Pakhrin, Moriah R. Beck, Punjan Subedi, Rabina Lama, and Simonsha
> Shrestha. “Multimodal deep learning for predicting protein ubiquitination
> sites.” *Bioinformatics Advances*, 2025, vbaf200.
> [https://doi.org/10.1093/bioadv/vbaf200](https://doi.org/10.1093/bioadv/vbaf200)

The project has two goals:

1. reproduce the authors’ released independent-test result from their model and
   data; and
2. develop and rigorously evaluate a new architecture that improves predictive
   performance under a fair comparison protocol.

The replication is complete. Architecture development is ongoing, and this
repository does **not** yet claim a new model that outperforms the published
MMUbiPred result on the locked independent test set.

## Current status

| Stage | Model or task | Result | Status |
|---|---|---:|---|
| Replication | Authors’ pretrained MMUbiPred model | MCC `0.54584` on the released independent test | Exactly reproduced |
| Development baseline | Stabilized MMUbiPred-compatible implementation | MCC `0.56389 ± 0.01116` | Reference across three development seeds |
| Candidate 1 | UbiFusionNet v1 | MCC `0.51671 ± 0.00538` | Rejected on development data |
| Candidate 2 | ESM2-CrossFusion v1 | MCC `0.49815`, seed 42 | Rejected on development data |
| Current candidate | CenterLoRA-ESM2 v1 | Seed-42 MCC `0.56894`; three-seed confirmation pending | Competitive development screen |

Candidate results above use protein-grouped development splits drawn only from
the released training set. They are not independent-test results.

## Exact replication result

The authors’ pretrained model was evaluated on the released general PLMD
independent test set in Google Colab. Compatibility changes were limited to
loading the legacy Keras model under a modern runtime; weights, inputs,
encodings, labels, threshold, and metric calculations were unchanged.

| Metric | Published result | Replication | Difference |
|---|---:|---:|---:|
| Matthews correlation coefficient | 0.5458386140 | 0.5458386140 | 0 |
| Accuracy | 0.7725035720 | 0.7725035720 | 0 |
| Sensitivity | 0.7498020586 | 0.7498020586 | 0 |
| Specificity | 0.8067729084 | 0.8067729084 | 0 |

The replicated confusion matrix is:

```text
[[4050  970]
 [1896 5682]]
```

See the full [replication record](replication/results/2026-07-21-general-plmd/RESULTS.md)
and the
[executed replication notebook](replication/results/2026-07-21-general-plmd/MMUbiPred_General_PLMD_Replication_Executed.ipynb).

## Scientific evaluation contract

The primary comparison retains the authors’ released labelled datasets:

- 46,587 positive and 45,136 negative training sites;
- 7,578 positive and 5,020 negative independent-test sites; and
- the same 49-residue candidate-site windows and normalized AAindex inputs.

Architecture selection, early stopping, threshold selection, and ablations use
only the released training set. Models are compared on identical
protein-grouped development splits using fixed seeds and split hashes.

The independent test remains locked until one architecture and its
hyperparameters have been selected. The fixed threshold of `0.5` is always
reported for direct comparison with MMUbiPred. A validation-selected threshold
may be reported separately but cannot replace the fixed-threshold comparison.

Protein-language models introduce external unsupervised pretraining even when
the labelled train/test split is unchanged. Experiments using such models
record this explicitly and will require a homology-aware robustness analysis
before publication.

## Current architecture

**CenterLoRA-ESM2 v1** tests whether task-specific adaptation is more effective
than combining frozen embeddings with several hand-crafted branches.

It uses:

- `facebook/esm2_t12_35M_UR50D`;
- rank-8 LoRA adapters on attention query and value projections;
- the final representation of the central candidate lysine; and
- a compact binary-classification head.

The first seed-42 screen reached fixed-threshold development MCC `0.56894`
without accessing the independent test. The confirmatory protocol reruns seeds
42, 123, and 2026 for all 15 epochs and retains each run's best development
checkpoint.

Read the complete
[CenterLoRA-ESM2 method specification](experiment/methods/center_lora_esm2_v1.md).

## Repository layout

```text
.
├── experiment/
│   ├── audits/             Immutable released-data audit
│   ├── configs/            Versioned model and benchmark configurations
│   ├── methods/            Architecture and evaluation specifications
│   ├── notebooks/          Thin Google Colab workflows
│   ├── results/            Compact, accepted development records
│   ├── scripts/            Audit, training, benchmark, and evaluation tools
│   ├── src/ubipred/        Reusable Python implementation
│   └── tests/              Unit and smoke tests
├── replication/
│   ├── MMUbiPred/          Authors’ repository as a Git submodule
│   └── results/            Executed replication evidence
└── research-documents/
    └── vbaf200.pdf         Reference paper
```

The upstream submodule is kept separate from new experimental code. Publication
experiments must not modify `replication/MMUbiPred`.

## Running the current experiment in Google Colab

Training is designed for a Colab GPU runtime. The current workflow is:

[05_CenterLoRA_ESM2_Full15_ThreeSeed_Colab.ipynb](experiment/notebooks/05_CenterLoRA_ESM2_Full15_ThreeSeed_Colab.ipynb)

The notebook:

1. mounts Google Drive;
2. clones or updates this repository and its submodule;
3. installs the pinned experiment dependencies;
4. removes Colab’s incompatible optional `torchao` package, which the model
   does not use;
5. runs the test suite; and
6. writes training history, metrics, provenance, predictions, and the best
   development checkpoint directly to Google Drive.

For this private repository, create a fine-grained GitHub token with read-only
access, save it in Colab Secrets as `GITHUB_TOKEN`, and enable notebook access
to the secret. The notebook uses a temporary Git askpass helper and does not
store the token in the clone URL or Git remote.

## Cloning the repository

Clone recursively so the authors’ MMUbiPred submodule is available:

```bash
git clone --recurse-submodules \
  https://github.com/kaleabnega/Ubiquitination-Sites-Prediction.git
cd Ubiquitination-Sites-Prediction
```

If the repository was cloned without submodules:

```bash
git submodule update --init --recursive
```

Authentication is required while the repository remains private.

## Auditing and testing

Run the deterministic released-data audit:

```bash
python3 experiment/scripts/audit_dataset.py \
  --data-dir replication/MMUbiPred \
  --output experiment/outputs/dataset_audit.json
```

Run the experiment tests after installing the Colab requirements in an
environment that already provides PyTorch:

```bash
python3 -m pip install -r experiment/requirements-colab.txt
python3 -m unittest discover -s experiment/tests -v
```

The committed [dataset audit](experiment/audits/released_data_audit.json)
records artifact hashes, retained sample counts, train/test overlap checks, and
the unsupported-residue filtering inherited from the authors’ workflow.

## Reproducibility and artifacts

- Experiment configurations, source code, compact metrics, and provenance are
  version controlled.
- Large checkpoints and prediction arrays are stored in Google Drive rather
  than Git.
- Generated local outputs belong under `experiment/outputs/` and are ignored.
- Every accepted result should record its configuration, Git revision, package
  versions, data hashes, split hashes, seed, and whether the locked test was
  accessed.

Additional implementation and benchmark details are available in the
[experiment documentation](experiment/README.md).

## Upstream project and attribution

The released MMUbiPred code and datasets are incorporated through the
[`PakhrinLab/MMUbiPred`](https://github.com/PakhrinLab/MMUbiPred) submodule.
Third-party code, pretrained models, and datasets remain subject to their
respective licences and terms. Please cite the original MMUbiPred paper when
using its model, data, or methodology.
