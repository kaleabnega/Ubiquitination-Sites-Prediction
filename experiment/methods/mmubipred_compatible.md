# MMUbiPred-compatible validation baseline

This baseline reconstructs the topology in the released training notebook
`replication/MMUbiPred/Human Ubiquitination CPLM 4.0 Dataset Experiment
Results.ipynb`. It is intended for controlled development-fold comparisons,
not as a claim of bit-for-bit reproduction of the released Keras checkpoint.

## Released topology represented here

- AAindex branch: 31-property input, 64-unit LSTM with sequence output, 0.3
  dropout, flatten, 32-unit ReLU layer, 0.3 dropout, and two-class output.
- One-hot branch: same-padded 16-filter Conv1D with kernel size 3, max pooling,
  flatten, 512-unit ReLU layer, 0.5 dropout, and two-class output.
- Integer-embedding branch: 23-by-21 embedding table, 256-filter Conv1D with
  kernel size 3, max pooling, 0.4 dropout, 256-filter Conv1D with kernel size
  7, max pooling, 0.4 dropout, then 768- and 256-unit ReLU layers with 0.5
  dropout and a two-class output.
- Score-level fusion: concatenate the three two-class probability vectors,
  apply a six-unit ReLU layer, and produce the final two-class output.

The released L1 coefficient of `1e-4` is applied to the AAindex dense layers
and the one-hot convolution/dense layers. The configuration also follows the
released Adam learning rate of `1e-3`, batch size 32, and 15-epoch limit.

## Controlled-comparison differences

The reimplementation uses PyTorch with mixed precision and gradient clipping,
so framework initializers, recurrent kernels, optimization details, and
floating-point behavior can differ from Keras. A scalar binary
logit equal to `class_1_logit - class_0_logit` is used for training; binary
cross-entropy on that value is mathematically equivalent to two-class softmax
cross-entropy.

For leakage-aware model selection, all compared architectures use the same
protein-grouped development indices for each seed. Checkpoint selection uses
fixed-threshold development MCC, and the independent test set is inaccessible
to benchmark runs. These controls were not part of the released fixed-epoch
training cell but are required for a defensible paired comparison.
