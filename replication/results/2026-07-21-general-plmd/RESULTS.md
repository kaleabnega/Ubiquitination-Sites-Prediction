# General PLMD inference replication

Status: **exactly reproduced**

Date: 2026-07-21 (Asia/Dubai)

## Scope

This run evaluated the authors' pretrained MMUbiPred model on the released
general PLMD independent test set. It verifies the released model,
preprocessing, test artifacts, and evaluation pipeline. It does not yet verify
that training the architecture from scratch reproduces the pretrained model's
performance.

Executed notebook:
`MMUbiPred_General_PLMD_Replication_Executed.ipynb`

Notebook SHA-256:
`b0bc69b2c5e50175a977b320feb72f63cf25c1244f25c57291411f2651852c87`

Upstream MMUbiPred commit:
`fb45202d0a91c58e74d73a9f8e4ad43dff4e48e5`

## Runtime

- Google Colab
- Python 3.12.13
- TensorFlow 2.19.1
- Legacy Keras (`tf-keras`) 2.19.0

The notebook's original metadata still names the authors' older Python kernel;
the actual replication runtime is recorded in the executed cell output above.

## Compatibility changes

The following compatibility-only changes were required for modern Colab:

1. Installed `tf-keras==2.19.0` and set `TF_USE_LEGACY_KERAS=1` before
   importing TensorFlow. This supports the old serialized LSTM
   `time_major=False` configuration.
2. Replaced the authors' machine-specific working directory with
   `/content/MMUbiPred`.
3. Loaded the HDF5 model with `compile=False`; training configuration is not
   needed for inference.

No model weights, input sequences, encodings, decision threshold, labels, or
metric calculations were changed.

## Results

| Metric | Published/saved baseline | Replication | Difference |
|---|---:|---:|---:|
| MCC | 0.5458386140298045 | 0.5458386140298045 | 0 |
| Accuracy | 0.7725035719955549 | 0.7725035719955549 | 0 |
| Sensitivity | 0.7498020585906572 | 0.7498020585906572 | 0 |
| Specificity | 0.8067729083665338 | 0.8067729083665338 | 0 |

Confusion matrix:

```text
[[4050  970]
 [1896 5682]]
```

The evaluated set contains 5,020 negative and 7,578 positive sites (12,598
total). The three-site difference from the 7,581 positives listed in the paper
is consistent with the notebook filtering sequences containing unsupported
residue symbols.

## Artifact checksums

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| `DeepUBI_AAindex_One_Hot_Emb_drop_out2423.h5` | 36,608,352 | `a9393c05635f8019d08fe916885bdf3f31225859b2d0cb4130ec4c2bd89a5001` |
| `aaindex31.txt` | 3,357 | `26ddb1bff5bcca51d1e1fb0d77b76da01564e6269bd9cac62774e36ac673524d` |
| `Positive_10_percent_independent_test_set_DeepUBI.fasta` | 726,805 | `c6a5107096fa0a60581a7e81e622ef1b865cf9b5db6823f81d6275e273f1359d` |
| `Negative_10_percent_independent_test_set_DeepUBI.fasta` | 481,693 | `659dad50721cacce097b174c41c05517ca064faf54defa21f8722649ce2ab2da` |

## Conclusion

The general PLMD pretrained-model inference result is exactly reproducible
from the released artifacts under a modern Colab runtime with legacy-Keras
deserialization enabled.
