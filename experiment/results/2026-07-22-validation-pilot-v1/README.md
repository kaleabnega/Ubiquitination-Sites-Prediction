# UbiFusionNet v1 validation decision

Status: **rejected as the final candidate** after the corrected three-seed
development benchmark. The locked independent test was not accessed.

The first benchmark contained a collapsed compatible-baseline run. That run
was discarded. The baseline was rerun with the corrected Keras-like
initialization and numerical settings, then merged with the unchanged
UbiFusionNet runs. The merged comparison reported `comparison_valid: true` and
no collapsed runs.

| Development metric | MMUbiPred-compatible | UbiFusionNet v1 | Candidate minus baseline |
|---|---:|---:|---:|
| MCC at 0.5 | 0.5639 ± 0.0112 | 0.5167 ± 0.0054 | -0.0472 ± 0.0112 |
| Selected-threshold MCC | 0.5724 ± 0.0052 | 0.5269 ± 0.0055 | -0.0455 ± 0.0102 |
| AUROC | 0.8628 ± 0.0062 | 0.8467 ± 0.0039 | -0.0161 ± 0.0094 |
| AUPRC | 0.8773 ± 0.0178 | 0.8683 ± 0.0088 | -0.0090 ± 0.0093 |
| Sensitivity at 0.5 | 0.5699 ± 0.0817 | 0.6410 ± 0.0558 | +0.0711 ± 0.0327 |
| Specificity at 0.5 | 0.9464 ± 0.0571 | 0.8606 ± 0.0447 | -0.0858 ± 0.0149 |

Values are mean ± sample standard deviation over seeds 42, 123, and 2026.
UbiFusionNet's higher sensitivity did not compensate for its loss of
specificity and MCC. This is a useful negative pilot, but it must not be tested
on the independent set or presented as the proposed final architecture.

The compact machine-readable aggregate copied from the Colab output is in
`aggregates.json`. Full per-run manifests, metrics, and split hashes are kept
in the project's Google Drive benchmark archive.
