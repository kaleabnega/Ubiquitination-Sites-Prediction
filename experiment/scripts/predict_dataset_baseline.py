#!/usr/bin/env python3
"""Run a frozen dataset-specific MMUbiPred H5 baseline in a separate process."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'experiment/src'))
from ubipred.benchmark_sources import digest  # noqa: E402
from ubipred.fasta import ALPHABET  # noqa: E402


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--model-file',type=Path,required=True)
    parser.add_argument('--aaindex',type=Path,required=True)
    args=parser.parse_args()
    lock=json.loads((args.run_dir/'model_lock.json').read_text())
    protocol=json.loads((args.run_dir/'protocol.json').read_text())
    test_contract=json.loads((args.run_dir/'test_contract.json').read_text())
    if digest(args.run_dir/'model_lock.json')!=test_contract['model_lock_sha256']:
        raise ValueError('Model lock changed')
    if digest(args.run_dir/'protocol.json')!=lock['protocol_sha256']:
        raise ValueError('Training protocol changed')
    if digest(args.model_file)!=lock['baseline_sha256']:
        raise ValueError('Dataset-specific released model checksum changed')
    if digest(args.aaindex)!=protocol['aaindex_sha256']:
        raise ValueError('AAindex checksum changed')
    if digest(args.run_dir/'test_cohort.json')!=test_contract['test_cohort_sha256']:
        raise ValueError('Test cohort changed')
    output=args.run_dir/'test_baseline.npz'
    if output.exists():
        raise FileExistsError(f'Baseline predictions already exist: {output}')
    rows=json.loads((args.run_dir/'test_cohort.json').read_text())
    tokens=np.array([[ALPHABET.index(a) for a in r['window_49']] for r in rows])
    raw=np.loadtxt(args.aaindex,dtype=np.float64)
    normalized=((raw-raw.min(1,keepdims=True))/(raw.max(1,keepdims=True)-raw.min(1,keepdims=True))).T
    lookup=np.concatenate([normalized,np.zeros((1,31))])
    os.environ['TF_USE_LEGACY_KERAS']='1'
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    import tf_keras
    model=tf_keras.models.load_model(args.model_file,compile=False)
    expected=[(None,49,31),(None,49,21),(None,49)]
    shapes=[tuple(t.shape) for t in model.inputs]
    if shapes!=expected:
        raise ValueError(f'Published baseline requires a different input adapter: {shapes}')
    predicted=model.predict([lookup[tokens],np.eye(21)[tokens],tokens],batch_size=32,verbose=1)
    if predicted.shape!=(len(rows),2) or not np.isfinite(predicted).all():
        raise ValueError('Invalid two-class baseline outputs')
    if not np.allclose(predicted.sum(axis=1),1,atol=1e-5) or (predicted<0).any() or (predicted>1).any():
        raise ValueError('Baseline must output class probabilities, not logits')
    np.savez_compressed(output,labels=np.array([r['label'] for r in rows]),
                        probabilities=predicted[:,1],indices=np.arange(len(rows)),
                        model_sha256=np.asarray(digest(args.model_file)),
                        contract_sha256=np.asarray(digest(args.run_dir/'test_contract.json')))


if __name__=='__main__':
    main()
