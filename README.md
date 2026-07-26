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

The replication is complete. Architecture development is ongoing. One frozen
single-seed candidate produced a small numerical MCC gain on the released test,
but the evidence is not yet strong enough to claim robust superiority over
MMUbiPred.

## Current status

| Stage | Model or task | Result | Status |
|---|---|---:|---|
| Replication | Authors’ pretrained MMUbiPred model | MCC `0.54584` on the released independent test | Exactly reproduced |
| Development baseline | Stabilized MMUbiPred-compatible implementation | MCC `0.56389 ± 0.01116` | Reference across three development seeds |
| Candidate 1 | UbiFusionNet v1 | MCC `0.51671 ± 0.00538` | Rejected on development data |
| Candidate 2 | ESM2-CrossFusion v1 | MCC `0.49815`, seed 42 | Rejected on development data |
| Three-seed ESM-2 reference | CenterLoRA-ESM2 v1 | MCC `0.56800 ± 0.00444` | Valid three-seed comparison |
| Candidate 4 | MultiScale CenterLoRA-ESM2 v2 | MCC `0.57238`, seed 42 | Stopped by predeclared screen |
| Backbone screen | CenterLoRA-ProtBERT v1 | MCC `0.58275`, seed 42 | Passed development screen |
| Historical-test evaluation | CenterLoRA-ProtBERT v1 | MCC `0.54997`, seed 42 | Numerical gain of `0.00413`; inconclusive |
| Current screen | LoRA-ESM2 Hybrid v1 | Seed-42 development protocol | Ready to run |

Architecture-screen results above use protein-grouped development splits drawn
only from the released training set. The separately labelled historical-test
row is the only new-model result from the released test.

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

The released test was held out during the original architecture-selection
sequence and then evaluated once for the frozen ProtBERT candidate. It is now a
historical benchmark and must not guide later architecture changes. The fixed
threshold of `0.5` is always reported for direct comparison with MMUbiPred. A
validation-selected threshold may be reported separately but cannot replace
the fixed-threshold comparison. A later publication-grade final claim requires
a newly curated, homology-aware blind external test set.

Protein-language models introduce external unsupervised pretraining even when
the labelled train/test split is unchanged. Experiments using such models
record this explicitly and will require a homology-aware robustness analysis
before publication.

## Current evidence and next architecture

**CenterLoRA-ESM2 v1** is the stable three-seed ESM-2 reference. It uses:

- the `facebook/esm2_t12_35M_UR50D` protein language model;
- rank-8 LoRA adapters on attention query and value projections;
- the final contextual representation of the central candidate lysine; and
- a compact 256-dimensional binary-classification head.

V1 completed its full-15 three-seed comparison at fixed-threshold MCC
`0.56800 ± 0.00444`, compared with `0.56389 ± 0.01116` for the stabilized
baseline.

MultiScale v2 added radius-2 and radius-5 contextual mean pools. On seed 42,
it improved fixed MCC over v1 by only `0.00345`, below the predeclared `0.005`
target, while AUROC, AUPRC, and selected-threshold MCC decreased. It was
therefore stopped without additional seeds. The released test remained locked
at that stage.

The subsequent backbone experiment changed only the pretrained
backbone from ESM-2 35M to ProtBERT-BFD while retaining the central-residue
head and LoRA protocol. It passed every predeclared seed-42 screen condition:
fixed MCC `0.58275`, selected MCC `0.58909`, AUROC `0.88719`, and AUPRC
`0.90241`, without prediction collapse.

Because one ProtBERT development run required approximately six Colab GPU
hours, immediate three-seed confirmation and a new full-data refit were
deferred. Its frozen epoch-4 seed-42 checkpoint was evaluated on the released
test at threshold 0.5. It obtained MCC `0.54997`, accuracy `0.76266`,
sensitivity `0.69332`, and specificity `0.86733`. Relative to the reproduced
paper model, MCC increased by only `0.00413` while accuracy and sensitivity
decreased. This is a numerical MCC win, not compelling evidence of a generally
better model, especially because it is one seed fitted on only 82,535 of the
91,723 released training records.

The current development-only experiment returns to the faster ESM-2 35M
backbone and adds two explicitly complementary experts: a dilated local-motif
CNN and a bidirectional-AAindex encoder. Learned feature-level gating, residual
fusion, and auxiliary branch supervision combine them. The seed-42 screen uses
only the established protein-grouped development split and does not read the
released test. It must materially beat CenterLoRA-ESM2 v1 and match the
ProtBERT development MCC before it can become the new performance candidate.

Read the complete
[CenterLoRA-ESM2 v1 method specification](experiment/methods/center_lora_esm2_v1.md)
and the
[MultiScale v2 screen result](experiment/results/2026-07-26-center-lora-esm2-multiscale-v2-screen/README.md).
The hybrid protocol is specified in
[LoRA-ESM2 Hybrid v1](experiment/methods/lora_esm2_hybrid_v1.md).

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

Training and evaluation are designed for a Colab GPU runtime. The current
development workflow is:

[08_LoRA_ESM2_Hybrid_v1_Screen_Colab.ipynb](experiment/notebooks/08_LoRA_ESM2_Hybrid_v1_Screen_Colab.ipynb)

The notebook:

1. mounts Google Drive;
2. clones or updates this repository and its submodule;
3. installs the pinned experiment dependencies;
4. removes Colab’s incompatible optional `torchao` package, which the model
   does not use;
5. runs the test suite;
6. resumes or runs the 15-epoch seed-42 development screen;
7. verifies that the comparison uses the identical validation split; and
8. applies the predeclared decision rule without loading the released test.

After the screen, a separate opt-in cell can evaluate the frozen
development-best checkpoint once on the released test. Because Hybrid v1 was
stopped on development and that test has already been examined for ProtBERT,
the notebook labels this result exploratory and does not allow it to alter the
selection decision.

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
