# Context-aware prediction of protein ubiquitination sites

This repository contains a reproducible replication and extension of
**MMUbiPred**, the multimodal deep-learning method described by Pakhrin et al.
The study evaluates whether protein-language-model context can improve
ubiquitination-site prediction while preserving a controlled comparison with
the released MMUbiPred data and model.

> Subash C. Pakhrin, Moriah R. Beck, Punjan Subedi, Rabina Lama, and Simonsha
> Shrestha. “Multimodal deep learning for predicting protein ubiquitination
> sites.” *Bioinformatics Advances* (2025), vbaf200.
> [https://doi.org/10.1093/bioadv/vbaf200](https://doi.org/10.1093/bioadv/vbaf200)

## Study status

The authors’ released independent-test result has been reproduced exactly.
Subsequent development examined local convolutional models, LoRA-adapted
protein language models, robust losses, contextual fusion, external
transportability, and structure-data feasibility.

The strongest current candidate is the **MMUbiPred–Context Residual Hybrid**,
a heterogeneous stacked ensemble comprising:

- a 49-residue MMUbiPred-compatible local expert;
- a 257-residue rank-8 LoRA-ESM-2 context expert; and
- a regularized nonnegative logistic stacker fitted to protein-grouped
  out-of-fold predictions.

The hybrid improved Matthews correlation coefficient (MCC) over exact
MMUbiPred on the historical matched cohort and on two external cohorts. Its limitation is lower specificity, reflecting a shift toward greater
positive-site sensitivity.

### Replication result

| Metric | Published MMUbiPred | Replication | Difference |
|---|---:|---:|---:|
| MCC | 0.5458386140 | 0.5458386140 | 0 |
| Accuracy | 0.7725035720 | 0.7725035720 | 0 |
| Sensitivity | 0.7498020586 | 0.7498020586 | 0 |
| Specificity | 0.8067729084 | 0.8067729084 | 0 |

The replicated confusion matrix is `[[4050, 970], [1896, 5682]]` over 12,598 processed test sites. Complete provenance is recorded in the
[replication result](replication/results/2026-07-21-general-plmd/RESULTS.md).

### Residual-hybrid evidence

All comparisons below use threshold `0.5` and identical cohorts for the two models.

| Cohort | Sites | Exact MMUbiPred MCC | Residual hybrid MCC | Difference |
|---|---:|---:|---:|---:|
| Released historical test, context-matched subset | 12,288 | 0.5464 | **0.5536** | **+0.0072** |
| dbPTM/PTMGPT2 external cohort | 286 | 0.1458 | **0.1694** | **+0.0236** |
| Protein-disjoint dbPTM holdout | 740 | 0.1477 | **0.2040** | **+0.0563** |

On the 740-site protein-disjoint cohort, paired protein-cluster bootstrap
analysis supported improvements in MCC, sensitivity, F1, AUROC, and AUPRC.
Specificity decreased from 0.3360 to 0.3035. On the smaller 286-site cohort,
the AUROC improvement was statistically supported, whereas the MCC interval included zero.

These evaluations are post-primary exploratory analyses of already inspected
cohorts. They support transportability but do not constitute a new untouched
confirmatory test. This distinction is encoded in the result manifests and is
retained throughout the repository.

## Model architecture

The local expert reproduces the released MMUbiPred topology in PyTorch:

1. an AAindex representation processed by a 64-unit LSTM;
2. a one-hot convolutional branch; and
3. a trainable-embedding convolutional branch.

Their class probabilities undergo score-level fusion. The implementation
matches the published layer dimensions, activations, dropout rates, and L1
penalties, but its weights are independently trained and are not the authors’
released Keras checkpoint.

The context expert retrieves the full UniProt sequence associated with each
released site and requires exact reconstruction of the released 49-mer. A
257-residue window is then formed from 128 residues on either side of the
central lysine. ESM-2 representations are summarized at three scales:

- the central lysine embedding;
- a radius-24 local mean pool, corresponding to 49 residues; and
- a padding-masked global mean pool across the 257-residue context.

The three projected representations are concatenated and classified by a
compact nonlinear head. LoRA adapters target the ESM-2 attention query and
value projections; the pretrained backbone is otherwise frozen.

The frozen residual fusion is

```text
logit(p) = 0.319899
         + 0.643761 × logit(p_local)
         + 0.557711 × logit(p_context)
```

with L2 strength `0.01`. Nonnegative weights prevent either expert from being
used as an inverted predictor.

## Experimental design

The released labelled data remain the basis of the controlled comparison:

- 46,587 positive and 45,136 negative training sites;
- 7,578 retained positive and 5,020 negative independent-test sites; and
- 49-residue lysine-centred windows with normalized AAindex descriptors.

Context reconstruction validated 89,551 of 91,723 released training records.
Protein-grouped development folds prevent proteins from appearing in both the
training and validation partitions of a fold. Architecture selection,
checkpoint selection, threshold analysis, and fusion fitting are separated
from locked-test evaluation.

Protein language models and UniProt sequences introduce external unlabeled
information even though the supervised labels are unchanged. External
pretraining, sequence-retrieval provenance, coverage exclusions, split hashes,
and checkpoint hashes are therefore recorded explicitly.

## Repository structure

```text
.
├── experiment/
│   ├── audits/          Deterministic released-data audits
│   ├── configs/         Versioned experiment configurations
│   ├── methods/         Protocols, claim boundaries, and decision rules
│   ├── notebooks/       Google Colab orchestration notebooks
│   ├── outputs/         Ignored local artifacts; README retained as a marker
│   ├── results/         Compact accepted development records
│   ├── scripts/         Training, evaluation, retrieval, and audit entry points
│   ├── src/ubipred/     Reusable model and analysis implementation
│   └── tests/           Unit and protocol tests
└── replication/
    ├── MMUbiPred/       Unmodified upstream repository as a Git submodule
    └── results/         Executed replication evidence
```

Large model checkpoints, sequence caches, and prediction arrays are stored
outside Git. Their paths and SHA-256 hashes are recorded in manifests. The upstream `replication/MMUbiPred` submodule remains unmodified.

## Notebook index

| Notebook | Purpose | Outcome or role |
|---:|---|---|
| [01](experiment/notebooks/01_UbiFusionNet_v1_Colab.ipynb) | UbiFusionNet development workflow | Initial candidate |
| [02](experiment/notebooks/02_Paired_Validation_Benchmark_Colab.ipynb) | Paired MMUbiPred-compatible benchmark | Stabilized reference |
| [03](experiment/notebooks/03_ESM2_CrossFusion_Screen_Colab.ipynb) | ESM2-CrossFusion screen | Rejected |
| [04](experiment/notebooks/04_CenterLoRA_ESM2_Screen_Colab.ipynb) | CenterLoRA-ESM2 screen | Advanced |
| [05](experiment/notebooks/05_CenterLoRA_ESM2_Full15_ThreeSeed_Colab.ipynb) | Three-seed ESM-2 benchmark | Completed |
| [06](experiment/notebooks/06_MultiScale_CenterLoRA_ESM2_v2_Screen_Colab.ipynb) | Multi-scale context screen | Rejected |
| [07](experiment/notebooks/07_CenterLoRA_ProtBERT_v1_Screen_Colab.ipynb) | ProtBERT backbone screen | Exploratory test result |
| [08](experiment/notebooks/08_LoRA_ESM2_Hybrid_v1_Screen_Colab.ipynb) | Three-branch ESM-2 hybrid | Rejected |
| [09](experiment/notebooks/09_MMUbiPred_Context_Residual_v1_Colab.ipynb) | Context and residual development | Primary architecture workflow |
| [10](experiment/notebooks/10_External_dbPTM_PTMGPT2_Validation_Colab.ipynb) | First external cohort | Completed |
| [11](experiment/notebooks/11_Disjoint_dbPTM_Holdout_v1_Colab.ipynb) | Protein-disjoint external cohort | Completed |
| [12](experiment/notebooks/12_Training_Label_Noise_Audit_Colab.ipynb) | Label-noise audit | Completed |
| [13](experiment/notebooks/13_MMUbiPred_Robust_Loss_Sensitivity_Colab.ipynb) | Robust-loss sensitivity | Rejected |
| [14](experiment/notebooks/14_MMUbiPred_Context_Complementarity_Diagnostic_Colab.ipynb) | Expert complementarity | Gated fusion not supported |
| [15](experiment/notebooks/15_Structure_Context_Feasibility_Audit_Colab.ipynb) | Structure coverage audit | Coverage gate failed |
| [16](experiment/notebooks/16_Structure_Identifier_Repair_Audit_Colab.ipynb) | Exact identifier repair | Coverage gate remained unmet |
| [17](experiment/notebooks/17_Residual_Hybrid_External_Assessment_Colab.ipynb) | Residual-hybrid external assessment | Completed |
| [18](experiment/notebooks/18_MMUbiPred_Context_Residual_Fixed30_Colab.ipynb) | Fixed-30-epoch residual rebuild | Ongoing exploratory analysis |
| [19](experiment/notebooks/19_Residual_Hybrid_Visualizations_Colab.ipynb) | Frozen residual-hybrid performance and biological-interpretation figures | Post hoc visualization |

Negative and stopped experiments remain versioned because they document the
model-selection path and reduce selective reporting.

## Reproducibility

### Clone

```bash
git clone --recurse-submodules \
  https://github.com/kaleabnega/Ubiquitination-Sites-Prediction.git
cd Ubiquitination-Sites-Prediction
```

For an existing clone without the upstream data:

```bash
git submodule update --init --recursive
```

Private-repository Colab access is mediated through a fine-grained read-only
token stored as `GITHUB_TOKEN` in Colab Secrets. Protein-model downloads can
use an optional `HF_TOKEN`. Notebook code supplies credentials through a
temporary askpass process and does not embed them in clone URLs or Git remotes.

### Environment and tests

The notebooks target a standard Google Colab runtime with PyTorch already
available.

```bash
python3 -m pip install -r experiment/requirements-colab.txt
python3 -m unittest discover -s experiment/tests -v
```

The deterministic released-data audit is generated with:

```bash
python3 experiment/scripts/audit_dataset.py \
  --data-dir replication/MMUbiPred \
  --output experiment/outputs/dataset_audit.json
```

The committed [audit](experiment/audits/released_data_audit.json) records data
hashes, retained counts, overlap checks, and unsupported-residue filtering.

### Artifact policy

Version-controlled artifacts include source code, configurations, protocol
documents, compact metrics, and provenance records. Generated checkpoints,
prediction arrays, downloaded sequence caches, and local logs remain outside
Git. Each accepted result records the configuration, Git revision, package
versions, data and split hashes, random seed, checkpoint hash, and test-access
status.

## Attribution

The released model, code, and datasets are incorporated through the
[`PakhrinLab/MMUbiPred`](https://github.com/PakhrinLab/MMUbiPred) submodule.
Third-party software, pretrained models, and datasets remain subject to their
respective licences and terms. Research using these assets should cite the
original MMUbiPred publication and the corresponding upstream resources.
