# Baseline-anchored context correction: development protocol

This screen evaluates whether the Long-Context Expert adds reproducible
information to the authors' dataset-specific model architecture on hCKSAAP or
UbiComb. It is a **post-historical-test exploration**: both published test
splits were already inspected in earlier experiments. This screen does not
load either test split and cannot by itself establish independent superiority.

## Model and data boundary

The previously prepared, protein-grouped training cohort and the Long-Context
Expert's five held-out-fold predictions are immutable inputs. The released
authors' checkpoint supplies the exact Keras architecture and is checked by
SHA-256. Its trained weights and its predictions on its own training data are
**not** used to fit the correction. Instead, each outer fold trains fresh
copies of that architecture on the other four folds. Three initializations
(42, 123, 2026) train for 15 epochs each without early stopping. An inner
protein-grouped validation partition chooses the epoch and initialization;
the selected checkpoint predicts the held-out outer fold. This gives one
training-only out-of-fold anchor probability per site.

The hCKSAAP screen inherits its documented 10,996-site verified-context
training subset; UbiComb inherits its 5,402-site verified training cohort.
The checkpoint clone matches the released model topology, but the newly
trained anchor is not asserted to reproduce the authors' released weights or
their original training trajectory.

The bounded correction is

\[
\operatorname{logit}(p_{\mathrm{corrected}})
=\operatorname{logit}(p_{\mathrm{anchor}})+a
+b\,\frac{\operatorname{clip}(\operatorname{logit}(p_{\mathrm{context}})
-\operatorname{logit}(p_{\mathrm{anchor}}),-4,4)}{4},
\]

with \(a\in[-0.5,0.5]\), \(b\in[0,1]\), and fixed L2 strength 0.1.
The baseline-logit coefficient is fixed at one; \(a=b=0\) is the untouched
anchor. An intercept-only control fits \(a\) with \(b=0\). The correction is
fitted on four folds' out-of-fold probabilities and scored on the fifth,
then repeated for all five folds. The reporting threshold remains 0.5.

The screen advances only when the pooled cross-fitted correction exceeds the
anchor by at least 0.01 MCC, exceeds intercept-only by at least 0.005 MCC,
improves MCC on at least four of five held-out folds, and loses no more than
0.005 AUROC or 0.02 sensitivity/specificity relative to the anchor.

This is a development gate, not fully nested end-to-end model assessment:
first-level models used for other folds' out-of-fold probabilities may have
trained on examples in the fold used to assess the second-level correction.
No significance claim or released-test evaluation follows automatically from
an `ADVANCE_TO_REFIT_PLANNING` result. A separate full-training refit and
genuinely untouched evaluation would be required for confirmation.

## Kaggle artifacts

The [Kaggle notebook](../notebooks/25_Anchored_Context_Correction_Development_Kaggle.ipynb)
requires the corresponding completed notebook-23 or notebook-24 Output as an
attached input. It checks all prior cohort, fold, and context-prediction
hashes before training. Its run directory contains per-epoch resumable exact
anchor checkpoints, one held-out anchor prediction file per fold, and
`development_screen.json`. An interrupted candidate resumes after the last
complete epoch with its saved optimizer state; bitwise replay across runtime
images is not guaranteed.
