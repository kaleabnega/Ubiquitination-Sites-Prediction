from __future__ import annotations

import importlib.util
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


SCRIPT_PATH = PROJECT_ROOT / "experiment" / "scripts" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("evaluate_locked_test", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import {SCRIPT_PATH}")
evaluate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate
SPEC.loader.exec_module(evaluate)

CONTEXT_SCRIPT_PATH = (
    PROJECT_ROOT / "experiment" / "scripts" / "evaluate_context_candidate.py"
)
CONTEXT_SPEC = importlib.util.spec_from_file_location(
    "evaluate_context_candidate", CONTEXT_SCRIPT_PATH
)
if CONTEXT_SPEC is None or CONTEXT_SPEC.loader is None:
    raise RuntimeError(f"Could not import {CONTEXT_SCRIPT_PATH}")
evaluate_context = importlib.util.module_from_spec(CONTEXT_SPEC)
sys.modules[CONTEXT_SPEC.name] = evaluate_context
CONTEXT_SPEC.loader.exec_module(evaluate_context)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(1, 1)
        self.trainable = nn.Linear(1, 1)
        for parameter in self.frozen.parameters():
            parameter.requires_grad = False


class TinyRefitDataset(Dataset):
    def __len__(self) -> int:
        return 12

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": torch.tensor([index % 3, (index + 1) % 3]),
            "label": torch.tensor(float(index % 2)),
            "index": torch.tensor(index),
        }


class TinyRefitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(1, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        features = tokens.float().mean(dim=1, keepdim=True)
        return self.classifier(self.dropout(features)).squeeze(-1)


def tiny_loader(seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        TinyRefitDataset(),
        batch_size=3,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )


def run_tiny_refit(
    model: nn.Module,
    loader: DataLoader,
    output_dir: Path,
    epochs: int,
    resume: bool,
) -> dict[str, object]:
    return refit_model(
        model=model,
        train_loader=loader,
        device=torch.device("cpu"),
        output_dir=output_dir,
        epochs=epochs,
        learning_rate=0.01,
        weight_decay=0.0,
        optimizer_name="adam",
        optimizer_epsilon=1e-8,
        gradient_clip_norm=1.0,
        gradient_accumulation_steps=2,
        use_amp=False,
        validation_selected_threshold=0.5,
        checkpoint_metadata={"protocol": "tiny_exact_resume", "target_epochs": 3},
        resume=resume,
        compact_checkpoint=True,
    )


class FinalProtocolTests(unittest.TestCase):
    def test_locked_test_cannot_be_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "locked_test_metrics.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(FileExistsError, "Refusing to repeat"):
                evaluate.ensure_locked_outputs_absent(run_dir)

    def test_context_test_cannot_be_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "context_test_metrics.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(FileExistsError, "Refusing to repeat"):
                evaluate_context.ensure_outputs_absent(run_dir)

    def test_development_trainable_checkpoint_loads(self) -> None:
        source = TinyModel()
        checkpoint = {
            "state_dict_scope": "trainable_parameters",
            "model_state_dict": {
                name: value.detach().clone()
                for name, value in source.state_dict().items()
                if name.startswith("trainable.")
            },
        }
        restored = TinyModel()
        load_checkpoint_model_state(restored, checkpoint)
        self.assertTrue(
            torch.equal(source.trainable.weight, restored.trainable.weight)
        )
        self.assertTrue(
            torch.equal(source.trainable.bias, restored.trainable.bias)
        )

    def test_refit_epoch_resume_matches_uninterrupted_training(self) -> None:
        torch.manual_seed(5)
        initial_state = {
            name: value.detach().clone()
            for name, value in TinyRefitModel().state_dict().items()
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            torch.manual_seed(17)
            uninterrupted = TinyRefitModel()
            uninterrupted.load_state_dict(initial_state)
            run_tiny_refit(
                uninterrupted,
                tiny_loader(17),
                root / "uninterrupted",
                epochs=3,
                resume=False,
            )

            torch.manual_seed(17)
            interrupted = TinyRefitModel()
            interrupted.load_state_dict(initial_state)
            run_tiny_refit(
                interrupted,
                tiny_loader(17),
                root / "resumed",
                epochs=2,
                resume=False,
            )

            torch.manual_seed(999)
            resumed = TinyRefitModel()
            run_tiny_refit(
                resumed,
                tiny_loader(999),
                root / "resumed",
                epochs=3,
                resume=True,
            )

            uninterrupted_checkpoint = torch.load(
                root / "uninterrupted" / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            resumed_checkpoint = torch.load(
                root / "resumed" / "best.pt",
                map_location="cpu",
                weights_only=False,
            )
            self.assertEqual(
                resumed_checkpoint["state_dict_scope"],
                "trainable_parameters",
            )
            for name, value in uninterrupted_checkpoint[
                "model_state_dict"
            ].items():
                self.assertTrue(
                    torch.equal(
                        value,
                        resumed_checkpoint["model_state_dict"][name],
                    ),
                    msg=name,
                )
            self.assertEqual(
                len(
                    torch.load(
                        root / "resumed" / "refit_resume.pt",
                        map_location="cpu",
                        weights_only=False,
                    )["history"]
                ),
                3,
            )


if __name__ == "__main__":
    unittest.main()
