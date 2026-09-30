#!/usr/bin/env python3
"""Dataset-specific residual-hybrid development, refit, and test orchestration.

Each stage has an explicit artifact contract; test stages require a completed
frozen refit. All large computation and downloads run in Colab.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'experiment/src'))
from ubipred.benchmark_sources import (  # noqa: E402
    atomic_json, coordinate_offset, digest, load_sources, object_digest,
    overlap_report, resolve_source, validate_rows,
)


def read(path):
    return json.loads(Path(path).read_text())


def bind(path, contract):
    if path.exists():
        if read(path) != contract:
            raise ValueError(f'Artifact contract changed: {path}. Use a new run directory.')
    else:
        atomic_json(path, contract)


def implementation_hashes():
    paths = [
        'experiment/scripts/run_dataset_benchmark.py',
        'experiment/scripts/predict_dataset_baseline.py',
        'experiment/src/ubipred/benchmark_sources.py',
        'experiment/src/ubipred/model.py',
        'experiment/src/ubipred/engine.py',
        'experiment/src/ubipred/data.py',
        'experiment/src/ubipred/stacking.py',
    ]
    return {name: digest(ROOT/name) for name in paths}


def audit(args, config):
    result = {'benchmark': config['benchmark'], 'training_started': False,
              'test_predictions_generated': False, 'files': {},
              'source_url': config['source_url'], 'blockers': []}
    for split in ('train', 'test'):
        files = []
        for spec in config['sources'][split]:
            try:
                path = resolve_source(spec, args.source_dir, args.upstream_dir)
                files.append({'filename': path.name, 'available': True,
                              'sha256': digest(path)})
            except FileNotFoundError:
                files.append({'filename': spec['filename'], 'available': False})
                result['blockers'].append(f"Missing {split} source: {spec['filename']}")
        result['files'][split] = files
    if config.get('blocked_reason'):
        result['blockers'].append(config['blocked_reason'])
    baseline = Path(args.model_file)
    result['baseline'] = {'expected_filename': config['baseline_filename'],
                          'available': baseline.is_file()}
    if not baseline.is_file():
        result['blockers'].append(f'Dataset-specific baseline checkpoint missing: {baseline}')
    elif baseline.name != config['baseline_filename']:
        result['blockers'].append('Baseline filename differs from published dataset mapping')
    elif config.get('baseline_sha256') and digest(baseline) != config['baseline_sha256']:
        result['blockers'].append('Baseline checkpoint checksum differs from verified replication')
    elif not config.get('blocked_reason'):
        try:
            import h5py
            with h5py.File(baseline, 'r') as handle:
                architecture = json.loads(handle.attrs['model_config'])['config']
            layers = {layer['name']: layer for layer in architecture['layers']}
            shapes = []
            for entry in architecture['input_layers']:
                layer = layers[entry[0]]['config']
                shapes.append(layer.get('batch_input_shape', layer.get('batch_shape')))
            result['baseline']['input_shapes'] = shapes
            if shapes != [[None, 49, 31], [None, 49, 21], [None, 49]]:
                result['blockers'].append(f'Released baseline needs a reviewed input adapter: {shapes}')
            result['baseline']['sha256'] = digest(baseline)
            result['baseline']['dataset_assignment_evidence'] = 'authors upstream README/notebook filename mapping'
        except (ImportError, OSError, KeyError, TypeError, ValueError) as error:
            result['blockers'].append(f'Baseline architecture preflight failed: {error}')
    result['source_preflight_passed'] = not result['blockers']
    atomic_json(args.run_dir / 'source_audit.json', result)
    print(json.dumps(result, indent=2))
    return result


def initialize(args, config):
    validate_protocol(config)
    # File presence and checksums only: no test labels or predictions are read.
    result = audit(args, config)
    if result['blockers']:
        raise RuntimeError('Source preflight failed. Resolve the listed source/provenance gaps first.')
    if config['baseline_window'] != 49:
        raise ValueError('This workflow currently requires a verified 49-residue baseline adapter')
    bind(args.run_dir / 'protocol.json', {
        'config': config, 'source_files': result['files'],
        'baseline_sha256': digest(args.model_file),
        'aaindex_sha256': digest(args.upstream_dir / 'aaindex31.txt'),
        'implementation_sha256': implementation_hashes(),
    })


def validate_protocol(config):
    if config['outer_folds'] != 5 or config['epochs'] != 15 or config['early_stopping']:
        raise ValueError('This study requires five folds and 15 epochs without early stopping')
    if config['threshold'] != 0.5 or config['stacker_l2'] != 0.01:
        raise ValueError('Frozen threshold or fusion regularization changed')
    if config['local_expert']['initialization_seeds'] != [42, 123, 2026]:
        raise ValueError('Short-Range initialization protocol changed')
    if config['context_expert']['initialization_seeds'] != [42]:
        raise ValueError('Long-Context initialization protocol changed')
    for name in ('local', 'context'):
        if config[f'{name}_expert']['inner_validation_fraction'] != 0.1:
            raise ValueError('Inner validation fraction must remain 0.1')


def prepare(args, config, split):
    if split == 'test':
        verify_freeze(args)
    else:
        initialize(args, config)
    rows, preprocessing = load_sources(config, split, args.source_dir, args.upstream_dir)
    protocol = read(args.run_dir / 'protocol.json')
    expected = [{'filename': x['filename'], 'sha256': x['sha256']}
                for x in protocol['source_files'][split]]
    if preprocessing['source_files'] != expected:
        raise ValueError('Published source files changed after protocol initialization')
    cache_path = args.run_dir / f'{split}_sequences.json'
    cache = read(cache_path) if cache_path.exists() else {'schema_version': 1, 'sequences': {}}
    output = args.run_dir / f'{split}_cohort.json'
    manifest = args.run_dir / f'{split}_preparation.json'
    if manifest.exists():
        stored = read(manifest)
        if (digest(output) != stored['cohort_sha256'] or digest(cache_path) != stored['cache_sha256']
                or stored['preprocessing']['source_files'] != preprocessing['source_files']):
            raise ValueError('Prepared cohort/cache checksum changed')
        print(f'{split} preparation complete; reused {output}', flush=True)
        return
    if args.sequence_cache and not cache_path.exists():
        imported = read(args.sequence_cache)
        cache['sequences'].update(imported['sequences'])
        cache['imported_cache_sha256'] = digest(args.sequence_cache)
    required = sorted({r['accession'] for r in rows})
    missing = [p for p in required if p not in cache['sequences']]
    accession_pattern = re.compile(r'^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$')
    unsupported = [p for p in missing if not accession_pattern.fullmatch(p)]
    if unsupported:
        raise ValueError(f'Accession/isoform resolution requires review: {unsupported[:10]}')
    if missing:
        from fetch_uniprot_context import fetch_batch, make_session
        session = make_session()
        for start in range(0, len(missing), 40):
            found, provenance = fetch_batch(session, missing[start:start+40], 120)
            cache['sequences'].update(found)
            cache['response_provenance'] = provenance
            atomic_json(cache_path, cache)
            print(f'{split} UniProt: attempted {min(start+40, len(missing))}/{len(missing)} missing accessions', flush=True)
    atomic_json(cache_path, cache)
    if split == 'train':
        offset, counts = coordinate_offset(rows, cache['sequences'], config['coordinate_offset'])
    else:
        offset = read(args.run_dir / 'train_preparation.json')['coordinate_offset']
        counts = None
    retained, validation = validate_rows(rows, cache['sequences'], offset)
    report = {'preprocessing': preprocessing, 'validation': validation,
              'coordinate_offset': offset, 'training_coordinate_match_counts': counts,
              'cache_sha256': digest(cache_path), 'test_labels_used_for_selection': False}
    if split == 'test':
        report['overlap'] = overlap_report(read(args.run_dir / 'train_cohort.json'), retained)
    atomic_json(args.run_dir / f'{split}_feasibility.json', report)
    if validation['coverage'] < config['minimum_context_coverage']:
        raise RuntimeError(f'Context coverage {validation["coverage"]:.4f} below protocol minimum; see {split}_feasibility.json')
    if set(r['label'] for r in retained) != {0, 1}:
        raise ValueError('Both classes must survive preprocessing')
    atomic_json(output, retained)
    report['cohort_sha256'] = digest(output)
    atomic_json(manifest, report)
    print(json.dumps(report, indent=2))


def runtime(args, split='train'):
    import numpy as np
    import torch
    from ubipred.fasta import SiteRecord
    from ubipred.context import LongContextSiteDataset
    from ubipred.data import SiteDataset, load_normalized_aaindex
    report = read(args.run_dir / f'{split}_preparation.json')
    cohort_path = args.run_dir / f'{split}_cohort.json'
    if digest(cohort_path) != report['cohort_sha256']:
        raise ValueError('Prepared cohort checksum changed')
    rows = read(cohort_path)
    records = [SiteRecord(r['source_id'], r['accession'], r['position_one_based']-1,
                          r['window_49'], r['label'], split) for r in rows]
    datasets = {'local': SiteDataset(records),
                'context': LongContextSiteDataset(records, [r['context_257'] for r in rows])}
    aaindex = load_normalized_aaindex(args.upstream_dir / 'aaindex31.txt')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return rows, records, datasets, aaindex, device


def train_kwargs(expert):
    return {key: expert[key] for key in ('learning_rate', 'weight_decay',
            'gradient_clip_norm', 'gradient_accumulation_steps', 'use_amp')} | {
                'optimizer_name': expert['optimizer'],
                'optimizer_epsilon': expert['optimizer_epsilon']}


def release(model):
    import torch
    model.to('cpu')
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def require_gpu(device):
    if device.type != 'cuda':
        raise RuntimeError('Select a Colab GPU runtime before training')


def development(args, config):
    import numpy as np
    import torch
    from sklearn.model_selection import StratifiedGroupKFold
    from ubipred.data import make_loader, make_train_validation_indices, seed_everything
    from ubipred.engine import train_model, predict, load_checkpoint_model_state
    from ubipred.model import build_model
    from ubipred.metrics import compute_metrics
    from ubipred.stacking import fit_residual_stacker
    rows, records, datasets, aaindex, device = runtime(args)
    require_gpu(device)
    labels = np.array([r.label for r in records])
    groups = np.array([r.protein_id for r in records])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=config['split_seed'])
    assignments = np.full(len(rows), -1, dtype=int)
    folds = list(splitter.split(np.zeros(len(rows)), labels, groups))
    for fold, (_, held) in enumerate(folds):
        assignments[held] = fold
    split_path = args.run_dir / 'fold_assignments.json'
    bind(split_path, assignments.tolist())
    base_contract = {'protocol_sha256': digest(args.run_dir / 'protocol.json'),
                     'cohort_sha256': digest(args.run_dir / 'train_cohort.json'),
                     'preparation_sha256': digest(args.run_dir / 'train_preparation.json'),
                     'fold_sha256': digest(split_path)}
    completed_summary = args.run_dir/'development_summary.json'
    if completed_summary.exists():
        completed = read(completed_summary)
        if completed['contract'] != base_contract:
            raise ValueError('Completed development contract changed')
        for relative, sha in completed['fold_predictions_sha256'].items():
            if digest(args.run_dir/relative) != sha:
                raise ValueError(f'Completed fold artifact changed: {relative}')
        print(json.dumps(completed,indent=2))
        print('All development folds and fusion already complete; skipped.')
        return
    all_probabilities = {name: np.full(len(rows), np.nan) for name in ('local', 'context')}
    selections = {'local': [], 'context': []}
    for fold, (outer_train, held) in enumerate(folds):
        training_records = [records[i] for i in outer_train]
        inner_train, inner_val = make_train_validation_indices(
            training_records, 0.1, config['split_seed'] + fold, 'protein_grouped')
        inner_train, inner_val = outer_train[inner_train], outer_train[inner_val]
        for indices in (inner_train, inner_val, held):
            if len(np.unique(labels[indices])) != 2:
                raise ValueError('A fold partition has only one class; review split feasibility')
        if set(groups[inner_train]) & set(groups[inner_val]) or set(groups[outer_train]) & set(groups[held]):
            raise AssertionError('Protein overlap within grouped development split')
        for name in ('local', 'context'):
            expert = config[f'{name}_expert']
            expert_dir = args.run_dir / 'folds' / f'fold_{fold}' / name
            expert_dir.mkdir(parents=True, exist_ok=True)
            prediction_path = expert_dir / 'outer_predictions.npz'
            selection_path = expert_dir / 'selection.json'
            contract = base_contract | {'expert': name, 'fold': fold,
                         'inner_train_hash': object_digest(inner_train.tolist()),
                         'inner_validation_hash': object_digest(inner_val.tolist())}
            bind(expert_dir / 'contract.json', contract)
            if prediction_path.exists() and selection_path.exists():
                saved = np.load(prediction_path)
                if not np.array_equal(saved['indices'], held) or not np.array_equal(saved['labels'], labels[held]):
                    raise ValueError('Completed outer predictions have changed alignment')
                if digest(prediction_path) != read(selection_path)['prediction_sha256']:
                    raise ValueError('Completed prediction checksum changed')
                all_probabilities[name][held] = saved['probabilities']
                selections[name].append(read(selection_path))
                print(f'fold={fold} expert={name}: completed; skipped', flush=True)
                continue
            candidates = []
            for seed in expert['initialization_seeds']:
                candidate_dir = expert_dir / f'seed_{seed}'
                metadata = contract | {'seed': seed, 'expert_config': expert}
                summary_path = candidate_dir / 'candidate_summary.json'
                seed_everything(seed)
                if summary_path.exists():
                    candidate = read(summary_path)
                    if candidate['contract'] != metadata:
                        raise ValueError('Completed candidate contract mismatch')
                    candidates.append(candidate)
                    continue
                model = build_model(aaindex_lookup=aaindex, window_size=49 if name=='local' else 257,
                                    model_config=expert['model'])
                def loader(indices, shuffle):
                    return make_loader(datasets[name], indices, expert['batch_size'], shuffle, 0, seed)
                print(f'fold={fold} expert={name} seed={seed} budget={config["epochs"]}', flush=True)
                summary = train_model(model=model, train_loader=loader(inner_train, True),
                    validation_loader=loader(inner_val, False), device=device, output_dir=candidate_dir,
                    epochs=config['epochs'], patience=config['epochs']+1,
                    checkpoint_metadata=metadata, resume=True, **train_kwargs(expert))
                y, p, _, _ = predict(model, loader(inner_val, False), device)
                metrics = compute_metrics(y, p, 0.5)
                collapsed = len(np.unique(p >= 0.5)) < 2
                candidate = {'contract': metadata, 'seed': seed, 'summary': summary,
                             'metrics': metrics, 'collapsed': collapsed,
                             'checkpoint_sha256': digest(candidate_dir/'development_best.pt')}
                atomic_json(summary_path, candidate)
                candidates.append(candidate)
                release(model)
                del model
            viable = [c for c in candidates if not c['collapsed']]
            if not viable:
                raise RuntimeError(f'All inner-validation candidates collapsed: fold={fold} expert={name}')
            best = max(viable, key=lambda c:(c['metrics']['mcc'], c['metrics']['auprc'], -c['seed']))
            seed_everything(best['seed'])
            model = build_model(aaindex_lookup=aaindex, window_size=49 if name=='local' else 257,
                                model_config=expert['model'])
            checkpoint = torch.load(expert_dir / f'seed_{best["seed"]}' / 'development_best.pt',
                                    map_location='cpu', weights_only=False)
            if digest(expert_dir / f'seed_{best["seed"]}' / 'development_best.pt') != best['checkpoint_sha256']:
                raise ValueError('Selected development checkpoint checksum changed')
            if checkpoint['metadata'] != best['contract']:
                raise ValueError('Selected checkpoint contract changed')
            load_checkpoint_model_state(model, checkpoint)
            model.to(device)
            outer_loader = make_loader(datasets[name], held, expert['batch_size'], False, 0, best['seed'])
            y, p, indices, _ = predict(model, outer_loader, device)
            if not np.array_equal(indices, held):
                raise ValueError('Outer prediction order mismatch')
            np.savez_compressed(prediction_path, indices=held, labels=y, probabilities=p)
            selection = {'seed': best['seed'], 'best_epoch': best['summary']['best_epoch'],
                         'outer_metrics': compute_metrics(y,p,0.5),
                         'outer_collapsed': len(np.unique(p>=0.5))<2,
                         'prediction_sha256': digest(prediction_path)}
            atomic_json(selection_path, selection)
            selections[name].append(selection)
            all_probabilities[name][held] = p
            release(model)
            del model
    if any(not np.isfinite(p).all() for p in all_probabilities.values()):
        raise ValueError('Incomplete OOF probability vectors')
    crossfit = np.empty(len(labels))
    for fold in range(5):
        other = assignments != fold
        stacker = fit_residual_stacker(labels[other], all_probabilities['local'][other],
                all_probabilities['context'][other], l2_strength=config['stacker_l2'])
        crossfit[~other] = stacker.predict_proba(all_probabilities['local'][~other], all_probabilities['context'][~other])
    stacker = fit_residual_stacker(labels, all_probabilities['local'], all_probabilities['context'],
                                  l2_strength=config['stacker_l2'])
    plans = {}
    for name, values in selections.items():
        seed_counts = Counter(s['seed'] for s in values)
        plans[name] = {'seed': min(seed_counts, key=lambda seed:(-seed_counts[seed],seed)),
                       'epochs': int(statistics.median(s['best_epoch'] for s in values))}
    result = {'benchmark': config['benchmark'], 'selection': selections, 'refit_plan': plans,
              'contract':base_contract,
              'fold_predictions_sha256':{str(p.relative_to(args.run_dir)):digest(p)
                  for p in sorted((args.run_dir/'folds').glob('fold_*/*/outer_predictions.npz'))},
              'comparison_valid': not any(s['outer_collapsed'] for v in selections.values() for s in v),
              'stacker': stacker.as_dict(), 'independent_test_accessed': False,
              'oof_metrics': {name:compute_metrics(labels,p,0.5) for name,p in
                             (all_probabilities | {'residual_crossfit':crossfit}).items()},
              'crossfit_caveat': 'Fusion cross-fit over first-level OOF predictions; not fully nested end-to-end assessment'}
    np.savez_compressed(args.run_dir/'oof_predictions.npz', labels=labels, folds=assignments,
                        local=all_probabilities['local'], context=all_probabilities['context'], residual=crossfit)
    atomic_json(args.run_dir/'development_summary.json', result)
    print(json.dumps(result, indent=2))


def refit(args, config):
    import torch
    from ubipred.data import make_loader, seed_everything
    from ubipred.engine import refit_model
    from ubipred.model import build_model
    summary = read(args.run_dir/'development_summary.json')
    if not summary['comparison_valid']:
        raise ValueError('Collapsed development expert: final refit is blocked')
    if (args.run_dir/'model_lock.json').exists():
        verify_freeze(args)
        print('Frozen refit already complete; skipped.')
        return
    rows, records, datasets, aaindex, device = runtime(args)
    require_gpu(device)
    for name in ('local', 'context'):
        expert, plan = config[f'{name}_expert'], summary['refit_plan'][name]
        directory = args.run_dir/'refit'/name
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {'protocol_sha256':digest(args.run_dir/'protocol.json'),
                    'development_summary_sha256':digest(args.run_dir/'development_summary.json'),
                    'training_cohort_sha256':digest(args.run_dir/'train_cohort.json'),
                    'expert':name, 'expert_config':expert, 'plan':plan}
        bind(directory/'contract.json', metadata)
        done = directory/'completed.json'
        if done.exists():
            if read(done)['checkpoint_sha256'] != digest(directory/'best.pt'):
                raise ValueError('Completed refit checkpoint changed')
            continue
        seed_everything(plan['seed'])
        model = build_model(aaindex_lookup=aaindex, window_size=49 if name=='local' else 257,
                            model_config=expert['model'])
        loader = make_loader(datasets[name], None, expert['batch_size'], True, 0, plan['seed'])
        print(f'full refit expert={name} epochs={plan["epochs"]} sites={len(rows)}', flush=True)
        result = refit_model(model=model, train_loader=loader, device=device,
            output_dir=directory, epochs=plan['epochs'], validation_selected_threshold=0.5,
            checkpoint_metadata=metadata, resume=True, compact_checkpoint=True, **train_kwargs(expert))
        atomic_json(done, result | {'checkpoint_sha256':digest(directory/'best.pt')})
        release(model)
        del model
    lock = {'benchmark':config['benchmark'], 'threshold':0.5, 'stacker':summary['stacker'],
            'protocol_sha256':digest(args.run_dir/'protocol.json'),
            'development_summary_sha256':digest(args.run_dir/'development_summary.json'),
            'training_cohort_sha256':digest(args.run_dir/'train_cohort.json'),
            'training_preparation_sha256':digest(args.run_dir/'train_preparation.json'),
            'checkpoints':{name:digest(args.run_dir/'refit'/name/'best.pt') for name in ('local','context')},
            'baseline_sha256':digest(args.model_file), 'test_inference_performed':False,
            'analysis_role':config['analysis_role'], 'refit_plan':summary['refit_plan']}
    atomic_json(args.run_dir/'model_lock.json', lock)
    print(json.dumps(lock,indent=2))


def verify_freeze(args):
    lock = read(args.run_dir/'model_lock.json')
    for filename,key in [('protocol.json','protocol_sha256'),
                         ('development_summary.json','development_summary_sha256'),
                         ('train_cohort.json','training_cohort_sha256'),
                         ('train_preparation.json','training_preparation_sha256')]:
        if digest(args.run_dir/filename) != lock[key]:
            raise ValueError(f'Frozen artifact changed: {filename}')
    for name,sha in lock['checkpoints'].items():
        if digest(args.run_dir/'refit'/name/'best.pt') != sha:
            raise ValueError(f'Frozen {name} checkpoint changed')
    if digest(args.model_file) != lock['baseline_sha256']:
        raise ValueError('Frozen dataset-specific baseline changed')
    return lock


def evaluate(args, config):
    import numpy as np
    import torch
    from ubipred.data import make_loader
    from ubipred.engine import load_checkpoint_model_state, predict
    from ubipred.model import build_model
    from ubipred.metrics import compute_metrics
    from ubipred.stacking import ResidualStacker
    from ubipred.paired_statistics import paired_cluster_bootstrap
    lock = verify_freeze(args)
    if (args.run_dir/'test_results.json').exists():
        print(json.dumps(read(args.run_dir/'test_results.json'),indent=2))
        return
    rows, records, datasets, aaindex, device = runtime(args, 'test')
    labels = np.array([r.label for r in records])
    bind(args.run_dir/'test_contract.json', {'model_lock_sha256':digest(args.run_dir/'model_lock.json'),
         'test_cohort_sha256':digest(args.run_dir/'test_cohort.json')})
    test_contract_hash = digest(args.run_dir/'test_contract.json')
    probabilities = {}
    for name in ('local','context'):
        path = args.run_dir/f'test_{name}.npz'
        if path.exists():
            saved = np.load(path)
            if not np.array_equal(saved['labels'], labels) or not np.array_equal(saved['indices'],np.arange(len(labels))):
                raise ValueError('Saved test prediction alignment mismatch')
            if saved['contract_sha256'].item() != test_contract_hash:
                raise ValueError('Saved test predictions belong to another frozen contract')
            probabilities[name] = saved['probabilities']
            continue
        expert=config[f'{name}_expert']
        model=build_model(aaindex_lookup=aaindex,window_size=49 if name=='local' else 257,model_config=expert['model'])
        checkpoint=torch.load(args.run_dir/'refit'/name/'best.pt',map_location='cpu',weights_only=False)
        load_checkpoint_model_state(model,checkpoint)
        model.to(device)
        loader=make_loader(datasets[name],None,expert['batch_size'],False,0,42)
        y,p,indices,_=predict(model,loader,device)
        if not np.array_equal(y,labels) or not np.array_equal(indices,np.arange(len(labels))):
            raise ValueError('Test inference order mismatch')
        np.savez_compressed(path,labels=y,probabilities=p,indices=indices,
                            contract_sha256=np.asarray(test_contract_hash))
        probabilities[name]=p
        release(model)
        del model
    baseline_path=args.run_dir/'test_baseline.npz'
    if not baseline_path.exists():
        subprocess.run([sys.executable,str(ROOT/'experiment/scripts/predict_dataset_baseline.py'),
            '--run-dir',str(args.run_dir),'--model-file',str(args.model_file),
            '--aaindex',str(args.upstream_dir/'aaindex31.txt')],check=True)
    baseline=np.load(baseline_path)
    if (not np.array_equal(baseline['labels'],labels)
            or not np.array_equal(baseline['indices'],np.arange(len(labels)))
            or baseline['model_sha256'].item() != lock['baseline_sha256']
            or baseline['contract_sha256'].item() != test_contract_hash):
        raise ValueError('Baseline labels differ from paired cohort')
    probabilities['paper']=baseline['probabilities']
    probabilities['hybrid']=ResidualStacker(**lock['stacker']).predict_proba(probabilities['local'],probabilities['context'])
    metrics={name:compute_metrics(labels,p,0.5) for name,p in probabilities.items()}
    paired=paired_cluster_bootstrap(labels,probabilities['hybrid'],probabilities['paper'],
        np.array([r.protein_id for r in records]),threshold=0.5,replicates=10000,
        seed=42,confidence_level=0.95,progress_every=1000)
    results={'benchmark':config['benchmark'],'analysis_role':config['analysis_role'],
        'comparison':'dataset-specific released checkpoint versus separately trained residual hybrid',
        'cohort':'identical context-valid published test rows', 'metrics':metrics,
        'paired_hybrid_vs_paper':paired,'test_preparation':read(args.run_dir/'test_preparation.json'),
        'model_lock_sha256':digest(args.run_dir/'model_lock.json'),
        'five_benchmark_multiplicity':'Report all five; no study-wide superiority claim from unadjusted per-dataset intervals'}
    np.savez_compressed(args.run_dir/'test_predictions.npz',labels=labels,
        groups=np.array([r.protein_id for r in records]),**probabilities)
    atomic_json(args.run_dir/'test_results.json',results)
    print(json.dumps(results,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['audit','prepare-train','develop','refit','prepare-test','evaluate'])
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--upstream-dir',type=Path,default=ROOT/'replication/MMUbiPred')
    parser.add_argument('--model-file',type=Path,required=True)
    parser.add_argument('--sequence-cache',type=Path)
    parser.add_argument('--allow-test',action='store_true')
    args=parser.parse_args()
    config=read(args.config)
    validate_protocol(config)
    args.run_dir.mkdir(parents=True,exist_ok=True)
    if args.stage in ('prepare-test','evaluate') and not args.allow_test:
        parser.error('Frozen test stages require --allow-test')
    if args.stage not in ('audit','prepare-train'):
        protocol=read(args.run_dir/'protocol.json')
        if protocol['config']!=config:
            raise ValueError('Configuration changed after training initialization')
        if protocol['implementation_sha256']!=implementation_hashes():
            raise ValueError('Training implementation changed; resume with the original code version')
        if digest(args.upstream_dir/'aaindex31.txt')!=protocol['aaindex_sha256']:
            raise ValueError('AAindex changed after initialization')
        if digest(args.model_file)!=protocol['baseline_sha256']:
            raise ValueError('Dataset-specific baseline changed after initialization')
    if args.stage=='audit': audit(args,config)
    elif args.stage.startswith('prepare-'): prepare(args,config,args.stage.split('-')[1])
    elif args.stage=='develop': development(args,config)
    elif args.stage=='refit': refit(args,config)
    elif args.stage=='evaluate': evaluate(args,config)


if __name__=='__main__':
    main()
