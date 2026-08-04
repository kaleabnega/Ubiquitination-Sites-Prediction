# Fixed-30-epoch MMUbiPred–Context Residual Hybrid rebuild

## Role and claim boundary

This is a post-test exploratory fixed-duration rebuild of the previously
specified MMUbiPred–Context Residual Hybrid. The released historical test and
both external cohorts were inspected before this protocol was created.
Consequently, neither development nor future test results from this run can
replace the frozen 5/7-epoch result or support a confirmatory superiority
claim. The existing experiment directories and checkpoints are immutable.

## Fixed-duration outer-fold protocol

The eligible cohort remains the 89,551 released training records whose current
UniProt sequence exactly reproduces the released 49-mer and supports a
lysine-centred 257-residue context. The existing seed-42 five-fold
`StratifiedGroupKFold` assignment is reproduced so that a protein cannot occur
in both an outer-training and outer-validation partition.

For every outer fold, two fresh experts are trained on the complete
outer-training partition:

- MMUbiPred-compatible 49-residue local expert, initialization seed 123 plus
  the fold number;
- rank-8 LoRA ESM-2 257-residue context expert, initialization seed 42 plus the
  fold number.

Both experts complete exactly 30 epochs. There is no inner validation, early
stopping, best-epoch selection, learning-rate selection, threshold selection,
or outer-fold access during training. Only the epoch-30 checkpoint predicts
the outer fold. The ten final-epoch prediction vectors are assembled into
complete out-of-fold local and context probabilities.

As in v1, a nonnegative L2-regularized residual logit stacker is cross-fitted
for unbiased development reporting. A final stacker is then fitted to all OOF
predictions for a possible full-data refit. The reporting threshold remains
`0.5`; OOF-selected thresholds are descriptive only.

## Full-data refit

After all ten expert-fold predictions exist and neither expert has collapsed,
both components are freshly refitted on all 89,551 eligible training records
for exactly 30 epochs. The final residual weights come only from the fixed-30
OOF predictions. The refit has no validation or test loader and performs no
threshold search.

## Resume and account transfer

The training engine writes `refit_resume.pt` after every complete epoch. It
contains trainable model parameters, optimizer state, AMP scaler, Python,
NumPy, CPU/CUDA RNG states, DataLoader-generator state, and complete history.
`num_workers` is locked to zero. A rerun with `--resume` skips completed
expert-fold predictions or continues an interrupted component after its last
completed epoch. It can also finalize a run interrupted immediately after
epoch 30 but before the compact final checkpoint was written.

Cross-account continuation requires copying the complete experiment directory
and the immutable UniProt sequence cache into identical Drive paths. Copying
only `best.pt` is insufficient. A hardware change can still introduce small
floating-point differences even though the protocol and stochastic state are
preserved.

The Colab command does not name individual folds. On every invocation it scans
folds 0–4 in order, reuses completed `outer_predictions.npz` artifacts, resumes
the first interrupted component, and proceeds through all remaining work.
Consequently, the same cell is used before and after a runtime interruption or
account transfer; no fold-selection variable is edited by the user.

## Test isolation

The notebook and both training scripts load only the released training split
and its frozen sequence cache. They do not load the historical independent
test, the 286-site cohort, or the 740-site cohort. Any later evaluation must be
explicitly labelled post-test exploratory.
