# Structure-aware context residual v1

## Motivation

The completed experiments do not support further fusion of sequence-derived
MMUbiPred and ESM-2 experts. The Long-Context Expert consistently improves positive
site recognition, but the residual stack captures only a small and
fold-heterogeneous gain. Robust losses also fail to improve BCE.

The next controlled hypothesis introduces information not explicitly present
in either sequence architecture: the predicted three-dimensional environment
of the candidate lysine. Ubiquitination requires physical access to the
lysine, making local confidence, solvent exposure, secondary structure, and
spatial contacts biologically motivated candidate features.

## Phase 1: frozen structural feasibility audit

No model is fitted in Phase 1. Only the released training split and the
existing hashed UniProt training-sequence cache are used. For every record
that already passed the 257-residue context validation, the pipeline:

1. queries the official AlphaFold Protein Structure Database API by canonical
   UniProt accession;
2. rejects isoforms, fragments, and predictions whose complete sequence does
   not exactly match the frozen UniProt sequence;
3. retrieves the versioned per-residue confidence document;
4. verifies that the one-based candidate-lysine position has a pLDDT value;
5. records site and global confidence without excluding low-pLDDT labels; and
6. reports overall, positive-class, negative-class, and protein-level coverage.

Retrieval is concurrent but conservative, retries transient HTTP failures, and
writes an atomic Drive-backed cache every 25 completed proteins. Rerunning the
same command resumes unresolved accessions. The released independent and
external tests are not queried.

Structural training is authorized only if overall, positive, and negative
site coverage each reach `95%`. Low-confidence residues are retained with an
explicit mask rather than removed, because excluding disordered regions could
selectively discard biologically meaningful ubiquitination sites.

### Identifier-repair amendment

The frozen Phase-1 audit failed the three 95% coverage gates. Before rejecting
the structural residual, one training-only identifier-repair audit is allowed.
The original cache and report remain unchanged. A failed accession may be
replaced by a current UniProt primary accession only when the official UniProt
record contains the exact complete cached sequence. An AlphaFold record returned
under an accession alias is accepted under the same exact full-sequence rule.
Fuzzy sequence matching, partial structures, changes to the coverage gates, and
access to either released test cohort are prohibited. If the amended audit still
fails any frozen coverage gate, Phase 2 is not authorized.

## Planned Phase 2 architecture

Phase 2 is designed only if the audit passes. It retains the frozen
257-residue Long-Context Expert as the base and adds a small
lysine-centred structural encoder. The final score is a bounded correction:

```text
z_final = z_context + alpha * tanh(z_structure)
```

The first screen freezes the Long-Context Expert and cross-fits only the structural
encoder and bounded residual head. This isolates structural incremental value
and prevents the new branch from replacing the existing Long-Context signal.
BCE remains the objective. A graph encoder is considered only after coordinate
coverage and local structural confidence are known.

The structural residual must improve fixed-threshold OOF MCC over the
Long-Context Expert by at least `0.01`, preserve AUROC and AUPRC, keep accuracy
within `0.01` and both sensitivity and specificity within `0.02`, and improve
MCC in at least four of five protein-grouped folds. Those thresholds will be
frozen in the Phase 2
configuration before its predictions are produced.

## Comparison scope

The labels and candidate sites remain the MMUbiPred/PLMD release. AlphaFold DB
is an external, unlabeled predicted-structure resource and must be disclosed as
such. Any comparison is restricted to an identical structurally eligible
cohort, with exact MMUbiPred recomputed on those same indices. Previously seen
historical and external labels cannot select this architecture, its features,
its threshold, or its training duration.
