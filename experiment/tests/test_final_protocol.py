from __future__ import annotations

import copy
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "experiment" / "src"
sys.path.insert(0, str(SRC_ROOT))

from ubipred.engine import (  # noqa: E402
    load_checkpoint_model_state,
    refit_model,
)


def load_script(name: str, relative_path: str):
    path = PROJECT_ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


refit_script = load_script(
    "refit_frozen_candidate",
    "experiment/scripts/refit_frozen_candidate.py",
)
evaluate_script = load_script(
    "evaluate_locked_test",
    "experiment/scripts/evaluate.py",
)


class TinyDataset(Dataset):
    def __init__(self) -> None:
        self.tokens = torch.arange(8, dtype=torch.float32).unsqueeze(1)
        self.labels = (self.tokens.squeeze(1) % 2).float()

    def __len__(self) -> int:
        return len(self.tokens)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": self.tokens[index],
            "label": self.labels[index],
            "index": torch.tensor(index),
        }


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.linear(tokens).squeeze(-1)


def make_tiny_loader() -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(42)
    return DataLoader(
        TinyDataset(),
        batch_size=2,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )


class FinalProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config_path = (
            PROJECT_ROOT
            / "experiment/configs/center_lora_protbert_final_seed42.json"
        )
        cls.config = json.loads(config_path.read_text(encoding="utf-8"))

    def valid_selection(self) -> tuple[dict[str, object], dict[str, object]]:
        contract = self.config["selection_contract"]
        manifest = {
            "config": {
                "experiment_name": contract["experiment_name"],
                "seed": contract["seed"],
                "model": self.config["model"],
            },
            "development_only": True,
            "locked_test_accessed": False,
            "development_split": {
                "validation_indices_sha256": contract[
                    "validation_indices_sha256"
                ]
            },
        }
        metrics = {
            "development_selection": {
                "epochs_completed": contract["epochs_completed"],
                "best_epoch": contract["best_epoch"],
                "validation_mcc_at_0_5": contract[
                    "validation_mcc_at_0_5"
                ],
                "validation_selected_threshold": contract[
                    "validation_selected_threshold"
                ],
            },
            "prediction_diagnostics": {
                "single_class_predictions_at_0_5": contract[
                    "single_class_predictions_at_0_5"
                ]
            },
        }
        return manifest, metrics

    def test_frozen_selection_is_accepted(self) -> None:
        manifest, metrics = self.valid_selection()
        refit_script.validate_frozen_selection(
            self.config, manifest, metrics
        )

    def test_model_drift_is_rejected(self) -> None:
        manifest, metrics = self.valid_selection()
        manifest = copy.deepcopy(manifest)
        manifest["config"]["model"]["dropout"] = 0.9
        with self.assertRaisesRegex(ValueError, "model configs differ"):
            refit_script.validate_frozen_selection(
                self.config, manifest, metrics
            )

    def test_changed_best_epoch_is_rejected(self) -> None:
        manifest, metrics = self.valid_selection()
        metrics["development_selection"]["best_epoch"] = 5
        with self.assertRaisesRegex(ValueError, "best_epoch"):
            refit_script.validate_frozen_selection(
                self.config, manifest, metrics
            )

    def test_locked_test_cannot_be_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "locked_test_metrics.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(FileExistsError, "Refusing to repeat"):
                evaluate_script.ensure_locked_outputs_absent(run_dir)

    def test_refit_checkpoint_is_compact_and_reloadable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            torch.manual_seed(7)
            model = TinyModel()
            refit_model(
                model=model,
                train_loader=make_tiny_loader(),
                device=torch.device("cpu"),
                output_dir=directory,
                epochs=1,
                learning_rate=0.01,
                weight_decay=0.0,
                optimizer_name="adamw",
                optimizer_epsilon=1e-8,
                gradient_clip_norm=1.0,
                gradient_accumulation_steps=1,
                use_amp=False,
                validation_selected_threshold=0.555,
                checkpoint_metadata={"test": "metadata"},
                resume=True,
            )
            checkpoint = torch.load(
                Path(directory) / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                checkpoint["state_dict_scope"], "trainable_parameters"
            )
            restored = TinyModel()
            load_checkpoint_model_state(restored, checkpoint)
            for expected, actual in zip(
                model.parameters(), restored.parameters()
            ):
                self.assertTrue(torch.equal(expected, actual))

            refit_model(
                model=TinyModel(),
                train_loader=make_tiny_loader(),
                device=torch.device("cpu"),
                output_dir=directory,
                epochs=1,
                learning_rate=0.01,
                weight_decay=0.0,
                optimizer_name="adamw",
                optimizer_epsilon=1e-8,
                gradient_clip_norm=1.0,
                gradient_accumulation_steps=1,
                use_amp=False,
                validation_selected_threshold=0.555,
                checkpoint_metadata={"test": "metadata"},
                resume=True,
            )


if __name__ == "__main__":
    unittest.main()
