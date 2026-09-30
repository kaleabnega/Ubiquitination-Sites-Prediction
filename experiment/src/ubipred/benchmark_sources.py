"""Dataset-specific source adapters for the five MMUbiPred benchmarks.

Source coordinates are resolved using training sequence reconstruction, then
frozen for test preprocessing. Short peptide-only records are never padded
to manufacture unobserved 49/257-residue context.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from .fasta import ALPHABET_SET, iter_fasta


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=path.parent, delete=False) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write('\n')
        temporary = handle.name
    os.replace(temporary, path)


def resolve_source(spec, source_dir, upstream_dir):
    for root in (source_dir, upstream_dir):
        path = Path(root) / spec['filename']
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Required published source missing: {spec['filename']}. "
        f"Place it in {source_dir}; see the benchmark configuration's source_url."
    )


def load_sources(config, split, source_dir, upstream_dir):
    if config.get('blocked_reason'):
        raise ValueError(config['blocked_reason'])
    rows, provenance = [], []
    for spec in config['sources'][split]:
        path = resolve_source(spec, source_dir, upstream_dir)
        provenance.append({'filename': path.name, 'sha256': digest(path)})
        if spec['format'] == 'fasta':
            for header, sequence in iter_fasta(path):
                accession, separator, position = header.rpartition('|')
                if not separator or not position.isdigit() or not accession:
                    raise ValueError(f'{path.name}: missing accession/site in {header}')
                rows.append({'source_id': f'{path.name}:{len(rows)}',
                             'accession': accession, 'position': int(position),
                             'sequence': sequence, 'label': spec['label']})
        elif spec['format'] == 'ubicomb_csv':
            with path.open() as handle:
                for row in csv.DictReader(handle):
                    rows.append({'source_id': f'{path.name}:{len(rows)}',
                                 'accession': row['PID'].strip(),
                                 'position': int(row['Position']),
                                 'sequence': row['Peptide_of_length_81'].upper(),
                                 'label': int(row['Label'])})
        else:
            raise ValueError(f"Unsupported source format: {spec['format']}")
    counts = Counter()
    eligible = []
    for row in rows:
        seq = row['sequence']
        if row['label'] not in (0, 1):
            raise ValueError('Labels must be binary')
        if len(seq) != 81:
            counts['invalid_length'] += 1
        elif set(seq) - ALPHABET_SET:
            counts['unsupported_residue'] += 1
        elif seq[40] != 'K':
            counts['invalid_center'] += 1
        else:
            row['window_49'] = seq[16:65]
            eligible.append(row)
    return eligible, {'raw_sites': len(rows), 'preprocessed_sites': len(eligible),
                      'exclusions': dict(counts), 'source_files': provenance,
                      'raw_support': dict(Counter(str(r['label']) for r in rows))}


def window(sequence, position, width):
    """Construct a one-based target window, padding only protein termini."""
    if not 1 <= position <= len(sequence):
        return None
    radius = width // 2
    start, stop = position - 1 - radius, position + radius
    return '-' * max(0, -start) + sequence[max(0, start):stop] + '-' * max(0, stop-len(sequence))


def coordinate_offset(rows, sequences, configured='infer_from_training'):
    """Infer a single coordinate convention from exact training 49-mer matches."""
    counts = {}
    for offset in (0, 1):
        counts[str(offset)] = sum(
            window(sequences.get(r['accession'], ''), r['position'] + offset, 49)
            == r['window_49'] for r in rows
        )
    if configured == 'infer_from_training':
        # A one-position shift of a poly-lysine repeat must not resolve a tie.
        if counts['0'] == counts['1']:
            raise ValueError(f'Ambiguous coordinate convention: {counts}')
        selected = max((0, 1), key=lambda offset: counts[str(offset)])
    else:
        selected = int(configured)
        if selected not in (0, 1):
            raise ValueError('Coordinate offset must be 0 or 1')
    return selected, counts


def validate_rows(rows, sequences, offset):
    retained, reasons = [], Counter()
    for row in rows:
        seq = sequences.get(row['accession'])
        position = row['position'] + offset
        if seq is None:
            reason = 'accession_not_retrieved'
        elif not 1 <= position <= len(seq):
            reason = 'position_out_of_range'
        elif seq[position-1] != 'K':
            reason = 'center_is_not_lysine'
        elif window(seq, position, 49) != row['window_49']:
            reason = 'released_49mer_mismatch'
        else:
            context = window(seq, position, 257)
            if set(context) - set('ARNDCQEGHILKMFPSTWYVXBUZO-'):
                reason = 'unsupported_context_residue'
            else:
                reason = 'validated'
                retained.append({**row, 'position_one_based': position,
                                 'context_257': context})
        reasons[reason] += 1
    return retained, {'total': len(rows), 'retained': len(retained),
                      'coverage': len(retained) / max(1, len(rows)),
                      'reasons': dict(reasons),
                      'support': dict(Counter(str(r['label']) for r in retained))}


def overlap_report(train_rows, test_rows):
    proteins = {r['accession'] for r in train_rows}
    sites = {(r['accession'], r['position_one_based']) for r in train_rows}
    windows = {r['window_49'] for r in train_rows}
    return {
        'test_sites_with_training_protein': sum(r['accession'] in proteins for r in test_rows),
        'test_sites_with_training_site': sum((r['accession'], r['position_one_based']) in sites for r in test_rows),
        'test_sites_with_training_49mer': sum(r['window_49'] in windows for r in test_rows),
        'policy': 'published split preserved; overlap reported, not silently filtered',
    }
