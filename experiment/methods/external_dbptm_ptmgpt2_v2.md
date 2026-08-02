# External dbPTM/PTMGPT2 validation v2

## Pre-inference amendment

Version 1 was invalidated before any external model prediction was generated.
Its MMseqs2 search returned 18,196 alignments with meaningful coverage values
but zero identical residues for every alignment because the command did not
request alignment backtraces. The v1 artifacts remain unchanged as an audit
trail.

Version 2 preserves the v1 benchmark, mapping rules, leakage exclusions,
homology thresholds, frozen models, ensemble weights, decision threshold,
metrics, and bootstrap plan. The only protocol correction is that MMseqs2 is
run with `-a` so the exported `nident` field is calculated from alignment
backtraces. The builder also fails closed when alignments are reported but all
identical-residue counts are zero, and it records the maximum observed count
and global identity in the cohort lock.

Because v1 stopped before inference and no external predictions or metrics
were inspected, this amendment does not use external labels to tune the model
or evaluation protocol.

If the corrected homology filter leaves fewer records than the frozen minimum,
the builder writes a non-frozen `cohort_feasibility.json` before stopping. It
contains only mapping, homology, retained-site, unique-protein, and class-
support counts; it contains no model predictions or performance metrics. Any
subsequent feasibility amendment must be versioned before inference.

The completed v2 feasibility audit retained 286 sites from 231 proteins (191
negative and 95 positive). It did not freeze a cohort or generate predictions;
the resulting pre-inference amendment is specified in v3.

## Cohort construction and evaluation

All remaining procedures follow the v1 contract: current UniProt sequences
must reproduce each published 21-mer at its stated site; released PLMD
accessions, sites, and exact windows are removed; remaining proteins are
excluded at at least 30% identity over at least 80% of the shorter full protein.
The frozen primary candidate remains the equal `0.5/0.5` probability ensemble
of the exact released MMUbiPred H5 model and the frozen seven-epoch context
model, classified at threshold `0.5`. MCC remains primary, with the same
secondary metrics and paired uncertainty analysis.
