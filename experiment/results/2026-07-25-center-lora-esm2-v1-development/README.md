# CenterLoRA-ESM2 v1 development benchmark

This record freezes the valid full-15, three-seed development comparison
completed before MultiScale CenterLoRA-ESM2 v2 was specified.

CenterLoRA-ESM2 v1 was compared with the stabilized MMUbiPred-compatible
baseline on paired protein-grouped splits for seeds 42, 123, and 2026. All
runs completed without one-class collapse, and the independent test was not
accessed.

| Metric | MMUbiPred-compatible | CenterLoRA-ESM2 v1 | Paired difference |
|---|---:|---:|---:|
| MCC at 0.5 | 0.5639 ± 0.0112 | 0.5680 ± 0.0044 | +0.0041 |
| Selected-threshold MCC | 0.5724 ± 0.0052 | 0.5788 ± 0.0053 | +0.0064 |
| Accuracy | 0.7600 ± 0.0079 | 0.7811 ± 0.0042 | +0.0210 |
| Sensitivity | 0.5699 ± 0.0817 | 0.7079 ± 0.0157 | +0.1380 |
| Specificity | 0.9464 ± 0.0571 | 0.8538 ± 0.0119 | -0.0926 |
| AUROC | 0.8628 ± 0.0062 | 0.8795 ± 0.0020 | +0.0167 |
| AUPRC | 0.8773 ± 0.0178 | 0.8971 ± 0.0020 | +0.0198 |

Values are mean ± sample standard deviation across the three seeds. The
candidate improved fixed-threshold MCC in two of three paired runs and
selected-threshold MCC in all three.

These are development results, not a direct comparison with the paper's
independent-test MCC. V1 was retained as the development champion and the
reference for the v2 comparison.
