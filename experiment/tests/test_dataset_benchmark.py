"""Protocol guards and source adapters; no model downloads or GPU required."""
import json
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'experiment/src'))
sys.path.insert(0,str(ROOT/'experiment/scripts'))
from ubipred.benchmark_sources import (
    atomic_json, coordinate_offset, digest, load_sources, overlap_report,
    validate_rows, window,
)
from run_dataset_benchmark import bind, validate_protocol, verify_freeze


class DatasetBenchmarkTests(unittest.TestCase):
    def row(self):
        protein='A'*140+'K'+'G'*140
        row={'accession':'P12345','position':140,'label':1,
             'window_49':window(protein,141,49),'source_id':'example:0'}
        return protein,row

    def test_training_coordinates_are_resolved_by_exact_reconstruction(self):
        protein,row=self.row()
        offset,counts=coordinate_offset([row],{'P12345':protein})
        self.assertEqual(offset,1)
        self.assertEqual(counts,{'0':0,'1':1})
        retained,report=validate_rows([row],{'P12345':protein},offset)
        self.assertEqual(len(retained[0]['context_257']),257)
        self.assertEqual(retained[0]['context_257'][128],'K')
        self.assertEqual(report['coverage'],1)

    def test_coordinate_ambiguity_is_not_guessed(self):
        with self.assertRaisesRegex(ValueError,'Ambiguous'):
            coordinate_offset([{'accession':'P12345','position':80,
                                'window_49':'K'*49}],{'P12345':'K'*200})

    def test_test_uses_frozen_coordinate_offset(self):
        protein,row=self.row()
        retained,report=validate_rows([row],{'P12345':protein},0)
        self.assertEqual(retained,[])
        self.assertEqual(report['reasons']['center_is_not_lysine'],1)

    def test_missing_or_mismatched_context_is_excluded(self):
        protein,row=self.row()
        retained,report=validate_rows([row],{},1)
        self.assertEqual(report['reasons'],{'accession_not_retrieved':1})
        changed={**row,'window_49':'G'*24+'K'+'A'*24}
        retained,report=validate_rows([changed],{'P12345':protein},1)
        self.assertEqual(report['reasons'],{'released_49mer_mismatch':1})

    def test_overlap_is_reported_without_reassigning_the_published_split(self):
        protein,row=self.row()
        rows,_=validate_rows([row],{'P12345':protein},1)
        report=overlap_report(rows,rows)
        self.assertEqual(report['test_sites_with_training_site'],1)
        self.assertEqual(report['test_sites_with_training_49mer'],1)
        self.assertEqual(len(rows),1)

    def test_dbptm_cannot_train_from_unmapped_short_peptides(self):
        config=json.loads((ROOT/'experiment/configs/benchmarks/dbptm_residual_v1.json').read_text())
        with self.assertRaisesRegex(ValueError,'without protein IDs'):
            load_sources(config,'train',ROOT,ROOT)

    def test_plant_and_human_sources_have_preserved_labels_and_identifiers(self):
        for name,expected in [('ubicomb',5410),('hcksaap',12199),('plmd',91750)]:
            config=json.loads((ROOT/f'experiment/configs/benchmarks/{name}_residual_v1.json').read_text())
            rows,report=load_sources(config,'train',ROOT/'missing',ROOT/'replication/MMUbiPred')
            self.assertEqual(report['raw_sites'],expected)
            self.assertTrue(all(r['accession'] and len(r['window_49'])==49 for r in rows))
            self.assertEqual({r['label'] for r in rows},{0,1})

    def test_frozen_protocol_cannot_change_in_place(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'contract.json'
            bind(path,{'epochs':15})
            bind(path,{'epochs':15})
            with self.assertRaisesRegex(ValueError,'contract changed'):
                bind(path,{'epochs':30})

    def test_all_configs_enforce_the_selected_training_budget(self):
        for path in (ROOT/'experiment/configs/benchmarks').glob('*.json'):
            config=json.loads(path.read_text())
            validate_protocol(config)
            config['epochs']=30
            with self.assertRaises(ValueError): validate_protocol(config)

    def test_test_stage_requires_intact_frozen_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=SimpleNamespace(run_dir=root,model_file=root/'baseline.json')
            with self.assertRaises(FileNotFoundError): verify_freeze(args)
            for name in ['protocol.json','development_summary.json','train_cohort.json','train_preparation.json','baseline.json']:
                atomic_json(root/name,{'sentinel':name})
            checks={}
            for name in ('local','context'):
                path=root/'refit'/name/'best.pt'
                atomic_json(path,{'component':name})
                checks[name]=digest(path)
            lock={'protocol_sha256':digest(root/'protocol.json'),
                  'development_summary_sha256':digest(root/'development_summary.json'),
                  'training_cohort_sha256':digest(root/'train_cohort.json'),
                  'training_preparation_sha256':digest(root/'train_preparation.json'),
                  'baseline_sha256':digest(args.model_file),'checkpoints':checks}
            atomic_json(root/'model_lock.json',lock)
            self.assertEqual(verify_freeze(args),lock)
            atomic_json(root/'refit/local/best.pt',{'component':'changed'})
            with self.assertRaisesRegex(ValueError,'checkpoint changed'): verify_freeze(args)

    def test_tiny_five_fold_refit_and_resume(self):
        """Exercise the real training engine/stacker on a tiny synthetic cohort."""
        try:
            import sklearn
        except ImportError:
            self.skipTest('Integration smoke test runs in Colab with scikit-learn installed')
        import numpy as np
        import torch
        from ubipred.fasta import SiteRecord
        import run_dataset_benchmark as runner

        class TinyDataset(torch.utils.data.Dataset):
            def __len__(self): return 100
            def __getitem__(self,index):
                return {'tokens':torch.tensor([float(index%2)]),
                        'label':torch.tensor(float(index%2)),
                        'index':torch.tensor(index)}

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head=torch.nn.Linear(1,1)
                with torch.no_grad():
                    self.head.weight.fill_(2)
                    self.head.bias.fill_(-1)
            def forward(self,tokens,return_gates=False):
                logits=self.head(tokens).squeeze(-1)
                return (logits,torch.ones((len(tokens),1))) if return_gates else logits

        config=json.loads((ROOT/'experiment/configs/benchmarks/ubicomb_residual_v1.json').read_text())
        records=[SiteRecord(str(i),f'protein_{i}',0,'K',i%2,'synthetic') for i in range(100)]
        datasets={'local':TinyDataset(),'context':TinyDataset()}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            args=SimpleNamespace(run_dir=root,model_file=root/'baseline.h5')
            atomic_json(args.model_file,{'synthetic':True})
            atomic_json(root/'protocol.json',config)
            atomic_json(root/'train_cohort.json',[{'row':i} for i in range(100)])
            atomic_json(root/'train_preparation.json',{'synthetic':True})
            with patch.object(runner,'runtime',return_value=(list(range(100)),records,datasets,np.zeros((21,31)),torch.device('cpu'))), \
                 patch.object(runner,'require_gpu'), \
                 patch('ubipred.model.build_model',side_effect=lambda **kwargs:TinyModel()), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.development(args,config)
                summary=json.loads((root/'development_summary.json').read_text())
                self.assertTrue(summary['comparison_valid'])
                self.assertEqual(len(summary['selection']['local']),5)
                first_hash=digest(root/'folds/fold_0/local/outer_predictions.npz')
                with patch('ubipred.engine.train_model',side_effect=AssertionError('Completed run retrained')):
                    runner.development(args,config)
                self.assertEqual(digest(root/'folds/fold_0/local/outer_predictions.npz'),first_hash)
                runner.refit(args,config)
                self.assertTrue((root/'model_lock.json').exists())
                runner.verify_freeze(args)
                with patch('ubipred.engine.refit_model',side_effect=AssertionError('Completed refit repeated')):
                    runner.refit(args,config)


if __name__=='__main__':
    unittest.main()
