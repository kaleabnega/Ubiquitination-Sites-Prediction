# Training-label noise audit v1

## Purpose

This audit tests whether asymmetric negative-label uncertainty is measurable
before changing the MMUbiPred architecture or objective. It fits no model,
uses no prediction, and does not access the released independent test set.

The released PLMD training labels are compared internally and against the
already-inspected official dbPTM source. Because dbPTM has already been used in
external evaluation, this analysis is diagnostic and may guide development;
neither the 286-site nor 740-site cohort can serve as an untouched final test
for any method developed from the audit.

## Internal checks

The audit reports repeated and contradictory protein-site keys, exact
21-residue windows, and exact 49-residue windows. It also measures how many
proteins contain both positive and negative records, how many negative records
occur on such proteins, and the distance from each eligible negative site to
the nearest known positive site on the same protein. Terminal padding is
reported separately to reveal possible class-correlated sampling artifacts.

## Cross-source checks

The normalized official dbPTM archive and its frozen UniProt mapping are used
to reconstruct valid current protein coordinates and 49-residue windows. The
strongest direct evidence is a released PLMD training-negative site that has
the same canonical accession and residue coordinate as a validated dbPTM
positive. The reverse disagreement is also reported because database-negative
labels may be uncertain in either source.

Exact 21-mer and 49-mer disagreements are weaker evidence: identical windows
can occur at different sites or proteins. They are reported but must not be
interpreted as confirmed site relabeling.

## Interpretation boundary

A conflict between databases demonstrates label inconsistency, not biological
ground truth. Ubiquitination is condition-dependent, database versions differ,
and absence of an annotation is not proof of a true negative. The audit will
therefore inform whether positive-unlabeled or robust-loss experiments are
scientifically motivated, but it will not estimate a definitive false-negative
rate by itself.
