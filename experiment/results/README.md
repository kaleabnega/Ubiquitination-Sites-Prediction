# Accepted experiment results

Each accepted run has a dated directory. Acceptance requires configuration
selection without consulting the independent test set. Versioned artifacts
include compact metrics, configuration, provenance, and the executed notebook.
Large checkpoints and prediction arrays remain in external artifact storage;
their hashes and storage locations are recorded in the corresponding result
README.

The directory naming convention is `YYYY-MM-DD-model-seed`, for example
`2026-07-22-ubifusion-v1-seed42`.
