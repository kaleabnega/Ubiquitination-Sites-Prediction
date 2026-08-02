# Disjoint dbPTM holdout v1

## Purpose and hypothesis

This study tests the prior external observation that the frozen seven-epoch
257-residue context model may rank ubiquitination sites better than the exact
released MMUbiPred model. AUROC is the single primary metric. Success requires
the lower bound of the two-sided 95% protein-cluster bootstrap interval for
context-minus-MMUbiPred AUROC to exceed zero. MCC and the other fixed-threshold
metrics are secondary. Both checkpoints and threshold `0.5` remain frozen.

The hypothesis is explicitly informed by the completed 286-site external
analysis, where the AUROC difference was `0.03392` but its interval included
zero. The new cohort must not be used for model, checkpoint, weight, or
threshold selection.

## Source normalization

The source is the official dbPTM ubiquitination predictor benchmark archive.
Its checksum and download URL are frozen in the configuration. The archive
contains 9,767 positive and 8,579 database-negative records. Normalization
removes every repeated identifier, every record not centred on lysine, and
records containing unsupported residues, producing 9,124 positive and 6,573
database-negative 21-mers. All removals are recorded before UniProt mapping.

The negative label means that dbPTM supplied the site as a benchmark negative;
it is not experimental proof that the lysine can never be ubiquitinated. This
label-noise limitation must accompany all conclusions.

## Independence controls

The previously evaluated 2,077-site dbPTM/PTMGPT2 benchmark is an exact subset
of this official archive. Therefore, the builder excludes the entire prior
source: identifiers, 21-mers, source entry names, resolved canonical
accessions, and the frozen 286-site cohort. It also removes all PLMD training
and historical-test accessions, sites, 21-mers, and 49-mers.

Remaining proteins are searched with MMseqs2 against the union of PLMD training
proteins and all resolved proteins from the prior 2,077-site source. A candidate
protein is excluded at at least 30% identical residues divided by the shorter
full-protein length and at least 80% coverage of that shorter protein. Alignment
backtraces are mandatory and zero-identity output fails closed.

Before inference, at least 60% of normalized sites must validate against current
UniProt sequences. The final cohort must contain at least 750 sites, at least
200 of each class, and at least 400 canonical proteins. Failure creates only a
feasibility audit; it cannot unlock prediction.

The 400-protein gate is a prospective precision target. Scaling the prior
231-protein cluster-bootstrap AUROC interval by the square-root ratio suggests
that 400 independent protein groups would reduce its approximate half-width
from about `0.0363` to `0.0276`, below the prior point difference of `0.0339` if
the effect and dependence structure persist. This is a planning approximation,
not a guarantee of significance.

## Statistical analysis

Predictions remain paired by site, while bootstrap resampling uses whole
canonical-protein groups to preserve within-protein dependence. Ten thousand
replicates use seed 20260803. The primary context-versus-MMUbiPred comparison
is predeclared before cohort construction or prediction. The earlier ensemble
is not the primary candidate in this new, prior-informed study.
