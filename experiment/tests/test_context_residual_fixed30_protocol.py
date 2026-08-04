from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "scripts"))

from ubipred.fixed30 import (  # noqa: E402
    FIXED_EPOCHS,
    fixed30_training_arguments,
    validate_fixed30_protocol,
)
class TinyDataset(Dataset):
    def __init__(self) -> None:
        self.tokens = torch.tensor([[0.0], [1.0], [2.0], [3.0]])
        self.labels = torch.tensor([0.0, 0.0, 1.0, 1.0])

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": self.tokens[index],
            "label": self.labels[index],
            "index": torch.tensor(index),
        }


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Linear(1, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.classifier(tokens).squeeze(-1)


class Fixed30ProtocolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config_path = (
            PROJECT_ROOT
            / "experiment"
            / "configs"
            / "mmubipred_context_residual_fixed30_exploratory_v1.json"
        )
        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))

    def test_configuration_locks_exact_30_without_early_stopping(self) -> None:
        validate_fixed30_protocol(self.config)
        self.assertEqual(self.config["fixed_epochs"], FIXED_EPOCHS)
        self.assertEqual(self.config["num_workers"], 0)
        for expert_name in ("local_expert", "context_expert"):
            expert = self.config[expert_name]
            arguments = fixed30_training_arguments(expert)
            self.assertNotIn("patience", arguments)
            self.assertNotIn("epochs", arguments)
            self.assertNotIn("early_stopping_patience", expert)

    def test_protocol_rejects_any_other_duration(self) -> None:
        changed = dict(self.config)
        changed["fixed_epochs"] = 29
        with self.assertRaisesRegex(ValueError, "exactly 30"):
            validate_fixed30_protocol(changed)

    def test_completed_epoch_resume_can_finalize_after_disconnect(self) -> None:
        try:
            from ubipred.data import make_loader
            from ubipred.engine import refit_model
        except ModuleNotFoundError as error:
            if error.name == "sklearn":
                self.skipTest("scikit-learn is not installed in the local runtime")
            raise
        metadata = {"experiment": "tiny_fixed30_resume_test"}
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            dataset = TinyDataset()
            loader = make_loader(
                dataset,
                indices=None,
                batch_size=2,
                shuffle=True,
                num_workers=0,
                seed=7,
            )
            refit_model(
                model=TinyModel(),
                train_loader=loader,
                device=torch.device("cpu"),
                output_dir=output_dir,
                epochs=1,
                learning_rate=0.01,
                weight_decay=0.0,
                optimizer_name="adam",
                optimizer_epsilon=1e-8,
                gradient_clip_norm=None,
                gradient_accumulation_steps=1,
                use_amp=False,
                validation_selected_threshold=0.5,
                checkpoint_metadata=metadata,
                resume=True,
                compact_checkpoint=True,
            )
            (output_dir / "best.pt").unlink()
            (output_dir / "refit_summary.json").unlink()

            resumed_loader = make_loader(
                dataset,
                indices=None,
                batch_size=2,
                shuffle=True,
                num_workers=0,
                seed=7,
            )
            summary = refit_model(
                model=TinyModel(),
                train_loader=resumed_loader,
                device=torch.device("cpu"),
                output_dir=output_dir,
                epochs=1,
                learning_rate=0.01,
                weight_decay=0.0,
                optimizer_name="adam",
                optimizer_epsilon=1e-8,
                gradient_clip_norm=None,
                gradient_accumulation_steps=1,
                use_amp=False,
                validation_selected_threshold=0.5,
                checkpoint_metadata=metadata,
                resume=True,
                compact_checkpoint=True,
            )

            self.assertTrue(summary["resumed"])
            self.assertTrue((output_dir / "best.pt").exists())
            self.assertTrue((output_dir / "refit_summary.json").exists())


if __name__ == "__main__":
    unittest.main()
