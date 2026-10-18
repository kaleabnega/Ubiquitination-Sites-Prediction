import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiment/src"))
from ubipred.benchmark_sources import atomic_json, digest

HAS_SKLEARN = importlib.util.find_spec("sklearn") is not None
if HAS_SKLEARN:
    SPEC = importlib.util.spec_from_file_location(
        "anchored_screen", ROOT / "experiment/scripts/run_anchored_context_screen.py"
    )
    SCREEN = importlib.util.module_from_spec(SPEC)
    SPEC.loader.exec_module(SCREEN)


@unittest.skipUnless(HAS_SKLEARN, "scikit-learn is installed in the Kaggle environment")
class AnchoredContextScreenTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.prior = root / "prior"
        self.prior.mkdir()
        self.output = root / "new"
        self.model = root / "released.h5"
        self.model.write_bytes(b"dummy checkpoint")
        self.aaindex = root / "aaindex.txt"
        self.aaindex.write_text("dummy aaindex")
        self.config = ROOT / "experiment/configs/anchored_context_correction_screen_v1.json"
        self.labels = np.asarray([0, 1] * 5)
        self.assignments = np.repeat(np.arange(5), 2)
        self.rows = [
            {"label": int(label), "accession": f"P{index:05d}", "window_49": "A" * 49}
            for index, label in enumerate(self.labels)
        ]
        atomic_json(self.prior / "train_cohort.json", self.rows)
        atomic_json(self.prior / "train_preparation.json", {"training": True})
        atomic_json(self.prior / "fold_assignments.json", self.assignments.tolist())
        atomic_json(self.prior / "protocol.json", {
            "config": {"benchmark": "hcksaap"},
            "baseline_sha256": digest(self.model),
            "aaindex_sha256": digest(self.aaindex),
        })
        context_hashes = {}
        for fold in range(5):
            indices = np.where(self.assignments == fold)[0]
            path = self.prior / "folds" / f"fold_{fold}" / "context" / "outer_predictions.npz"
            path.parent.mkdir(parents=True)
            np.savez_compressed(path, indices=indices, labels=self.labels[indices],
                                probabilities=np.array([0.35, 0.65]))
            context_hashes[str(path.relative_to(self.prior))] = digest(path)
        atomic_json(self.prior / "development_summary.json", {
            "benchmark": "hcksaap",
            "comparison_valid": True,
            "independent_test_accessed": False,
            "contract": {
                "cohort_sha256": digest(self.prior / "train_cohort.json"),
                "preparation_sha256": digest(self.prior / "train_preparation.json"),
                "fold_sha256": digest(self.prior / "fold_assignments.json"),
                "protocol_sha256": digest(self.prior / "protocol.json"),
            },
            "fold_predictions_sha256": context_hashes,
        })
        # This deliberately malformed test artifact must never be opened.
        (self.prior / "test_cohort.json").write_text("not valid JSON")
        self.args = SimpleNamespace(config=self.config, prior_run_dir=self.prior,
                                    released_model=self.model, aaindex=self.aaindex,
                                    output_dir=self.output)

    def tearDown(self):
        self.temp.cleanup()

    def test_contract_uses_only_development_artifacts(self):
        _, rows, labels, _, assignments, context, contract = SCREEN.load_contract(self.args)
        self.assertEqual(len(rows), 10)
        np.testing.assert_array_equal(labels, self.labels)
        np.testing.assert_array_equal(assignments, self.assignments)
        np.testing.assert_allclose(context, np.tile([0.35, 0.65], 5))
        self.assertEqual(contract["benchmark"], "hcksaap")

    def test_analysis_rejects_changed_anchor_predictions(self):
        config, _, labels, _, assignments, context, contract = SCREEN.load_contract(self.args)
        for fold in range(5):
            indices = np.where(assignments == fold)[0]
            folder = self.output / "folds" / f"fold_{fold}"
            folder.mkdir(parents=True)
            path = folder / "anchor_outer_predictions.npz"
            np.savez_compressed(path, indices=indices, labels=labels[indices],
                                probabilities=np.array([0.3, 0.7]))
            atomic_json(folder / "selection.json", {"fold": fold, "prediction_sha256": digest(path)})
        SCREEN.analyze(self.args, config, labels, assignments, context, contract)
        report = json.loads((self.output / "development_screen.json").read_text())
        self.assertFalse(report["released_independent_test_accessed"])
        self.assertEqual(report["metrics"]["exact_architecture_anchor"]["support"]["total"], 10)
        path = self.output / "folds/fold_0/anchor_outer_predictions.npz"
        with path.open("ab") as handle:
            handle.write(b"changed")
        with self.assertRaises(ValueError):
            SCREEN.analyze(self.args, config, labels, assignments, context, contract)


if __name__ == "__main__":
    unittest.main()
