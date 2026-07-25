# MultiScale CenterLoRA-ESM2 v2 seed-42 screen

This development-only screen tested contextual radius-2 and radius-5 mean
pooling against the frozen full-15 CenterLoRA-ESM2 v1 seed-42 result. Both runs
used the identical protein-grouped development split. V2 completed all 15
epochs, retained epoch 6, did not collapse, and did not access the independent
test.

| Metric | CenterLoRA v1 | MultiScale v2 | V2 minus v1 |
|---|---:|---:|---:|
| MCC at 0.5 | 0.56894 | 0.57238 | +0.00345 |
| Accuracy | 0.78287 | 0.78265 | -0.00022 |
| Sensitivity | 0.72438 | 0.69871 | -0.02567 |
| Specificity | 0.84042 | 0.86526 | +0.02483 |
| AUROC | 0.87993 | 0.87843 | -0.00150 |
| AUPRC | 0.89649 | 0.89466 | -0.00183 |
| Selected-threshold MCC | 0.58031 | 0.57296 | -0.00735 |

The fixed-MCC gain did not reach the predeclared 0.005 material-improvement
target. Accuracy was unchanged, both ranking metrics declined, and
selected-threshold MCC was lower. The fixed-threshold change primarily traded
sensitivity for specificity.

Mean component weights were 0.6070 for the central representation, 0.2210 for
radius 2, and 0.1721 for radius 5. The pooled components were therefore used,
but they did not add better discriminative information.

The notebook classified the outcome as borderline. Under the predeclared
contract, v2 was stopped without a three-seed run. CenterLoRA-ESM2 v1 remains
the development champion.
