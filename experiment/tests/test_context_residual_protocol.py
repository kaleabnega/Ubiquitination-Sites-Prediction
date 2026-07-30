from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "scripts"))

from run_context_residual_cv import finish_oof_summary  # noqa: E402
from refit_context_residual_hybrid import derive_local_refit_plan  # noqa: E402
from stabilize_context_residual_local import (  # noqa: E402
    candidate_sort_key,
    is_collapsed,
)
from ubipred.fasta import SiteRecord  # noqa: E402


def record(index: int, label: int) -> SiteRecord:
    return SiteRecord(
        header=f"P{index}|0",
        protein_id=f"P{index}",
        position=0,
        sequence="A" * 24 + "K" + "A" * 24,
        label=label,
        source="synthetic",
    )


class ContextResidualProtocolTests(unittest.TestCase):
    def test_single_class_metrics_are_collapsed(self) -> None:
        self.assertTrue(is_collapsed({"sensitivity": 1.0, "specificity": 0.0}))
        self.assertTrue(is_collapsed({"sensitivity": 0.0, "specificity": 1.0}))
        self.assertFalse(is_collapsed({"sensitivity": 0.7, "specificity": 0.8}))

    def test_candidate_selection_uses_mcc_then_auprc_then_seed(self) -> None:
        candidates = [
            {
                "initialization_seed": 123,
                "inner_fixed_threshold": {"mcc": 0.5, "auprc": 0.8},
            },
            {
                "initialization_seed": 42,
                "inner_fixed_threshold": {"mcc": 0.5, "auprc": 0.9},
            },
            {
                "initialization_seed": 2026,
                "inner_fixed_threshold": {"mcc": 0.4, "auprc": 0.95},
            },
        ]
        selected = sorted(candidates, key=candidate_sort_key)[0]
        self.assertEqual(selected["initialization_seed"], 42)

    def test_hybrid_local_refit_plan_uses_mode_seed_and_median_epoch(
        self,
    ) -> None:
        selected_seeds = [42, 123, 42, 2026, 42]
        best_epochs = [8, 4, 6, 10, 7]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fold, (seed, best_epoch) in enumerate(
                zip(selected_seeds, best_epochs)
            ):
                target = (
                    root
                    / "folds"
                    / f"fold_{fold}"
                    / "local_stabilized"
                )
                target.mkdir(parents=True)
                candidates = [
                    {
                        "initialization_seed": candidate_seed,
                        "inner_collapsed": candidate_seed != seed,
                        "development_selection": {
                            "best_epoch": (
                                best_epoch
                                if candidate_seed == seed
                                else best_epoch + 1
                            )
                        },
                    }
                    for candidate_seed in (42, 123, 2026)
                ]
                (target / "selection.json").write_text(
                    json.dumps(
                        {
                            "fold": fold,
                            "selected_initialization_seed": seed,
                            "outer_fold_accessed_for_selection": False,
                            "candidates": candidates,
                        }
                    ),
                    encoding="utf-8",
                )

            plan = derive_local_refit_plan(root, fold_count=5)

        self.assertEqual(
            plan["fold_selected_initialization_seeds"], selected_seeds
        )
        self.assertEqual(plan["fold_selected_best_epochs"], best_epochs)
        self.assertEqual(plan["selected_seed"], 42)
        self.assertEqual(plan["selected_epochs"], 7)

    def test_collapsed_outer_fold_invalidates_go(self) -> None:
        labels = np.asarray([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int64)
        assignments = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        records = [record(index, int(label)) for index, label in enumerate(labels)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for fold in (0, 1):
                indices = np.flatnonzero(assignments == fold)
                for expert in ("local", "context"):
                    target = root / "folds" / f"fold_{fold}" / expert
                    target.mkdir(parents=True)
                    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9])
                    if fold == 0 and expert == "local":
                        probabilities = np.full(4, 0.9)
                    np.savez_compressed(
                        target / "outer_predictions.npz",
                        dataset_indices=indices,
                        labels=labels[indices],
                        probabilities=probabilities,
                    )

            summary = finish_oof_summary(
                output_dir=root,
                records=records,
                fold_assignments=assignments,
                l2_strength=0.01,
            )

        self.assertFalse(summary["comparison_valid"])
        self.assertEqual(
            summary["predeclared_decision"], "INVALID_COLLAPSED_EXPERT"
        )
        self.assertEqual(
            summary["amended_candidate_decision"],
            "INVALID_COLLAPSED_EXPERT",
        )
        self.assertEqual(
            summary["collapsed_expert_folds"],
            [{"fold": 0, "expert": "local"}],
        )


if __name__ == "__main__":
    unittest.main()
