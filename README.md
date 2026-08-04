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
| Candidate 5 | LoRA-ESM2 Hybrid v1 | Development MCC `0.57165`; historical-test MCC `0.53616` | Rejected |
| Current architecture | Long-context LoRA-ESM2 candidate | OOF MCC `0.57901` | Advanced to frozen 7-epoch full-data refit |
| Robust-loss sensitivity | Negative smoothing and GCE | Neither improved BCE development MCC or AUROC | Rejected; retain BCE |
| External validation | dbPTM/PTMGPT2 benchmark | Cohort and inference protocol frozen | Awaiting leakage-audited cohort construction |

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

The current experiment compared the complete paper-compatible MMUbiPred
topology with a rank-8 LoRA ESM-2 expert over validated 257-residue UniProt
context. After the uniform local-expert stabilization amendment, the context
expert reached fixed-threshold OOF MCC `0.57901`, versus `0.55626` for the
fold-matched local expert. It also improved accuracy from `0.75375` to
`0.78695`, AUROC from `0.85650` to `0.88123`, and AUPRC from `0.88590` to
`0.90243`. The residual stack added only `0.00271` MCC over context alone,
below the frozen `0.005` fusion requirement, so the simpler context-only
candidate advanced.

The five inner-validation best epochs were `[8, 8, 7, 6, 7]`. Their frozen
integer median fixes the final training duration at seven epochs.
The next run refits a fresh context-only model on all 89,551 released training
sites that passed exact UniProt-context validation. It performs no validation
or test evaluation and can resume exactly after each completed epoch.

Read the complete
[CenterLoRA-ESM2 v1 method specification](experiment/methods/center_lora_esm2_v1.md)
and the
[MultiScale v2 screen result](experiment/results/2026-07-26-center-lora-esm2-multiscale-v2-screen/README.md).
The hybrid protocol is specified in
[LoRA-ESM2 Hybrid v1](experiment/methods/lora_esm2_hybrid_v1.md).
The current protocol is specified in
[MMUbiPred–Context Residual v1](experiment/methods/mmubipred_context_residual_v1.md).
Notebook 09 also contains guarded post-test exploratory workflows for the
epoch-15 duration ablation and for a full-data refit of the previously
specified residual hybrid. These runs preserve the frozen primary result and
are explicitly excluded from confirmatory claims.
The same notebook also provides a fixed equal-probability ensemble of the
authors' exact saved MMUbiPred predictions and the frozen seven-epoch context
predictions on their identical matched cohort.

## External dbPTM/PTMGPT2 validation

The next evaluation uses the released 2,077-site dbPTM/PTMGPT2 benchmark.
Before inference, every historical UniProt entry name is resolved to its
canonical accession and every published 21-mer must map exactly to its stated
position. The pipeline removes PLMD training and historical-test
proteins, exact released sites and windows, and proteins matched to PLMD
training proteins by high-sensitivity MMseqs2 at 30% global identity with 80%
shorter-sequence coverage. The corrected v2 search explicitly requests
MMseqs2 alignment backtraces so its exported identical-residue counts are
valid, and refuses to freeze a cohort if all such counts are zero. The v1
cohort attempt was invalidated before model inference and is preserved only
as provenance. The corrected v2 feasibility audit retained 286 sites from 231
proteins; v3 transparently lowers only the pre-inference feasibility guards
and uses protein-cluster bootstrap intervals.

The candidate was frozen before this external evaluation: it is the
equal-weight probability average of the authors' exact H5 model and the
seven-epoch context model, evaluated at threshold `0.5`. MCC is primary and
10,000 paired protein-cluster bootstrap replicates quantify candidate-minus-
paper uncertainty. The notebook deliberately stops after writing the cohort
lock; external inference requires a separate explicit acknowledgement.
After the one-time evaluation, a guarded saved-prediction analysis provides the
predeclared context-versus-paper secondary comparison without rerunning either
model or replacing the ensemble primary endpoint.

[Open the external dbPTM/PTMGPT2 validation notebook](experiment/notebooks/10_External_dbPTM_PTMGPT2_Validation_Colab.ipynb)

Read the frozen
[external-validation contract](experiment/methods/external_dbptm_ptmgpt2_v3.md).

## Disjoint larger dbPTM holdout

Notebook 11 constructs a larger follow-up cohort from the official dbPTM
ubiquitination benchmark archive. The earlier 2,077-site external source is an
exact subset of that archive, so the new protocol excludes the complete prior
source and adds its resolved proteins to the MMseqs2 homology reference along
with PLMD training proteins. Source normalization, UniProt coordinate
validation, exact-overlap filtering, and the 30% identity/80% shorter-coverage
rule all occur before prediction.

The frozen, prior-informed hypothesis compares the seven-epoch context model
directly with exact MMUbiPred using AUROC as the single primary metric. The v1
pre-inference audit retained 740 sites (369 negative, 371 positive) from 477
proteins and generated no predictions. A documented v2 feasibility amendment
changes only the round-number total gate from 750 to 700; the 200-per-class and
400-protein gates, cohort filters, models, hypothesis, and statistics remain
unchanged. The notebook verifies the preserved v1 alignments and exact counts
before freezing v2. After that audit passed review, a one-time frozen inference
section was added for the predeclared direct context-versus-MMUbiPred analysis.

[Open the disjoint dbPTM holdout notebook](experiment/notebooks/11_Disjoint_dbPTM_Holdout_v1_Colab.ipynb)

Read the
[disjoint-holdout contract](experiment/methods/external_dbptm_disjoint_holdout_v2.md).

## Training-label noise audit

Notebook 12 performs a CPU-only, prediction-free audit before any robust-loss
or positive–unlabeled experiment is selected. It measures contradictory
training sites and windows, mixed-label protein structure, negative-site
proximity to known positives, and exact conflicts between PLMD training labels
and the already-inspected official dbPTM source. It does not access the
released independent test or modify any labels.

[Open the training-label noise audit notebook](experiment/notebooks/12_Training_Label_Noise_Audit_Colab.ipynb)

Read the
[label-noise audit contract](experiment/methods/label_noise_audit_v1.md).

The follow-up development-only sensitivity screen holds the
MMUbiPred-compatible architecture and training protocol fixed while comparing
BCE, asymmetric negative-label smoothing at `0.05`, and generalized cross
entropy at `q=0.7`. A predeclared rule determines whether either robust loss
earns a three-seed confirmation; no test set is loaded.

[Open the robust-loss sensitivity notebook](experiment/notebooks/13_MMUbiPred_Robust_Loss_Sensitivity_Colab.ipynb)

Read the
[robust-loss sensitivity protocol](experiment/methods/robust_loss_sensitivity_v1.md).

The screen returned `STOP_ROBUST_LOSS`. Negative smoothing reduced fixed MCC
by `0.00583` and AUROC by `0.01222`; generalized cross entropy reduced fixed
MCC by `0.00168` and AUROC by `0.00460`. The next CPU-only diagnostic uses the
already saved corrected out-of-fold local, context, average, and residual-stack
predictions to determine whether their complementary errors justify a gated
fusion screen. Historical and external saved predictions are descriptive only
and cannot change the development recommendation.

[Open the prediction-complementarity diagnostic notebook](experiment/notebooks/14_MMUbiPred_Context_Complementarity_Diagnostic_Colab.ipynb)

Read the
[prediction-complementarity protocol](experiment/methods/prediction_complementarity_v1.md).

The next controlled candidate is a structure-aware bounded residual on the
long-context LoRA-ESM2 model. Phase 1 performs no training: it retrieves exact
full-sequence AlphaFold DB metadata and candidate-site pLDDT for the validated
training cohort, checks overall and class-specific coverage, and freezes the
eligible-site artifact. Structural training is designed only if every
predeclared coverage gate passes.

[Open the structure-feasibility audit notebook](experiment/notebooks/15_Structure_Context_Feasibility_Audit_Colab.ipynb)

The frozen Phase-1 audit retained 81,928 of 89,551 context-valid training
sites: 91.49% overall, 93.14% positive, and 89.77% negative coverage. It
therefore failed all three unchanged 95% gates. Notebook 16 performs the one
permitted training-only identifier-repair amendment in a separate directory.
It accepts an AlphaFold alias or current UniProt primary accession only when
the complete sequence exactly matches the frozen cache; it does not use fuzzy
matching, alter the gates, fit a model, or access a test set.

[Open the identifier-repair audit notebook](experiment/notebooks/16_Structure_Identifier_Repair_Audit_Colab.ipynb)

Read the
[structure-aware context residual protocol](experiment/methods/structure_context_residual_v1.md).

Notebook 17 performs the missing external transportability check for the
previously frozen MMUbiPred–Context Residual Hybrid. It reuses the immutable
exact-MMUbiPred and context probabilities from the completed 286-site and
740-site evaluations, runs only the small frozen local expert, and applies the
unchanged OOF-fitted stacker. Both comparisons are explicitly post-primary;
paired protein-cluster bootstrap intervals are reported against exact
MMUbiPred.

[Open the residual-hybrid external assessment](experiment/notebooks/17_Residual_Hybrid_External_Assessment_Colab.ipynb)

Read the
[external residual-hybrid assessment protocol](experiment/methods/external_residual_hybrid_exploratory_v1.md).

Notebook 18 performs a separate fixed-duration sensitivity analysis requested
after all historical and external results were inspected. Both the
MMUbiPred-compatible local expert and 257-residue LoRA-ESM2 context expert run
for exactly 30 epochs in each of five protein-grouped folds, without early
stopping or best-epoch selection. Epoch-30 OOF probabilities fit a new residual
stacker, after which both components can be refitted on all eligible training
records for 30 epochs. Every epoch is Drive-checkpointed for cross-account
resume. The workflow is permanently post-test exploratory and does not load a
test set.

[Open the fixed-30-epoch residual-hybrid notebook](experiment/notebooks/18_MMUbiPred_Context_Residual_Fixed30_Colab.ipynb)

Read the
[fixed-30-epoch rebuild protocol](experiment/methods/mmubipred_context_residual_fixed30_exploratory_v1.md).

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

[09_MMUbiPred_Context_Residual_v1_Colab.ipynb](experiment/notebooks/09_MMUbiPred_Context_Residual_v1_Colab.ipynb)

The notebook:

1. mounts Google Drive;
2. clones or updates this repository and its submodule;
3. installs the pinned experiment dependencies;
4. removes Colab’s incompatible optional `torchao` package, which the model
   does not use;
5. runs the test suite;
6. retrieves and validates full UniProt sequences into a Drive cache;
7. runs or resumes one or more of five outer folds;
8. creates fold-safe OOF predictions for both experts; and
9. evaluates the residual stack through a second cross-fit without loading the
   released test; and
10. runs the frozen seven-epoch context-only full-data refit after the corrected
    decision is `ADVANCE_CONTEXT_ONLY`; and
11. provides a guarded one-time historical-test evaluation that validates test
    context coverage before reporting the fixed-threshold result.

The separate external-validation workflow is:

[10_External_dbPTM_PTMGPT2_Validation_Colab.ipynb](experiment/notebooks/10_External_dbPTM_PTMGPT2_Validation_Colab.ipynb)

The CPU-only label-noise audit workflow is:

[12_Training_Label_Noise_Audit_Colab.ipynb](experiment/notebooks/12_Training_Label_Noise_Audit_Colab.ipynb)

The paired robust-loss development screen is:

[13_MMUbiPred_Robust_Loss_Sensitivity_Colab.ipynb](experiment/notebooks/13_MMUbiPred_Robust_Loss_Sensitivity_Colab.ipynb)

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
