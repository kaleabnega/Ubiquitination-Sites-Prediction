from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = PROJECT_ROOT / "experiment" / "src"
sys.path.insert(0, str(SRC_ROOT))

from ubipred.engine import load_checkpoint_model_state  # noqa: E402


SCRIPT_PATH = PROJECT_ROOT / "experiment" / "scripts" / "evaluate.py"
SPEC = importlib.util.spec_from_file_location("evaluate_locked_test", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import {SCRIPT_PATH}")
evaluate = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate
SPEC.loader.exec_module(evaluate)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.frozen = nn.Linear(1, 1)
        self.trainable = nn.Linear(1, 1)
        for parameter in self.frozen.parameters():
            parameter.requires_grad = False


class FinalProtocolTests(unittest.TestCase):
    def test_locked_test_cannot_be_repeated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "locked_test_metrics.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(FileExistsError, "Refusing to repeat"):
                evaluate.ensure_locked_outputs_absent(run_dir)

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


if __name__ == "__main__":
    unittest.main()
