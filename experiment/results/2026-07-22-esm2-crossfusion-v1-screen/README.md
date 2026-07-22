# ESM2-CrossFusion v1 seed-42 screen

Status: **rejected**. The locked independent test was not accessed, and no
additional seeds should be run for this version.

The model reached its best validation MCC at epoch 7 and stopped after epoch
12 following five non-improving epochs. Training loss continued falling from
0.4558 at the best epoch to 0.4073 at epoch 12 while validation MCC failed to
improve, indicating overfitting rather than insufficient training time.

| Metric | Threshold 0.5 | Validation-selected threshold 0.525 |
|---|---:|---:|
| MCC | 0.4981 | 0.5040 |
| Accuracy | 0.7455 | 0.7460 |
| Sensitivity | 0.6546 | 0.6305 |
| Specificity | 0.8350 | 0.8596 |
| AUROC | 0.8361 | 0.8361 |
| AUPRC | 0.8524 | 0.8524 |

At the selected epoch, mean gates were 0.8073 for ESM-2, 0.0667 for the local
motif branch, and 0.1260 for AAindex. The residual fusion path means these are
not complete feature-importance estimates, but they show strong ESM dominance.

The next experiment isolates task adaptation with CenterLoRA-ESM2 instead of
adding more fusion components.
