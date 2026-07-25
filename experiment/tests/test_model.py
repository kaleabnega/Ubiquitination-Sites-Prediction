from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "experiment" / "src"))

from ubipred.model import (  # noqa: E402
    CenterLoRAESM2,
    CenterLoRAProtBERT,
    ESM2CrossFusion,
    MMUbiPredCompatible,
    MultiScaleCenterLoRAESM2,
    UbiFusionNet,
    build_model,
)


class FakeESMBackbone(torch.nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = torch.nn.Embedding(32, hidden_size)
        self.gradient_checkpointing_enabled = False
        self.input_grads_enabled = False

    def gradient_checkpointing_enable(self) -> None:
        self.gradient_checkpointing_enabled = True

    def enable_input_require_grads(self) -> None:
        self.input_grads_enabled = True

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> SimpleNamespace:
        del attention_mask
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


class FakeESMTokenizer:
    cls_token_id = 1
    eos_token_id = 2
    pad_token_id = 0
    unk_token_id = 3

    @classmethod
    def from_pretrained(cls, name: str) -> "FakeESMTokenizer":
        del name
        return cls()

    def convert_tokens_to_ids(self, residue: str) -> int:
        return 4 + CenterLoRAESM2.alphabet.index(residue)


class FakeProtBERTTokenizer(FakeESMTokenizer):
    eos_token_id = None
    sep_token_id = 2
    received_kwargs: dict[str, object] = {}

    @classmethod
    def from_pretrained(
        cls, name: str, **kwargs: object
    ) -> "FakeProtBERTTokenizer":
        del name
        cls.received_kwargs = kwargs
        return cls()


class FakeAutoModel:
    @staticmethod
    def from_pretrained(name: str, **kwargs: object) -> FakeESMBackbone:
        del name, kwargs
        return FakeESMBackbone(hidden_size=16)


class ModelTests(unittest.TestCase):
    def test_output_and_gate_shapes(self) -> None:
        model = UbiFusionNet(
            aaindex_lookup=np.zeros((21, 31), dtype=np.float32),
            window_size=49,
            embedding_dim=32,
            branch_dim=64,
            conv_channels=8,
            transformer_heads=4,
            transformer_layers=1,
            transformer_ff_dim=64,
            dropout=0.0,
        )
        tokens = torch.zeros((3, 49), dtype=torch.long)
        tokens[:, 24] = 11  # K in ARNDCQEGHILKMFPSTWYV-
        logits, gates = model(tokens, return_gates=True)
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(tuple(gates.shape), (3, 3))
        torch.testing.assert_close(gates.sum(dim=1), torch.ones(3))

    def test_no_context_ablation_has_zero_context_gate(self) -> None:
        model = UbiFusionNet(
            aaindex_lookup=np.zeros((21, 31), dtype=np.float32),
            window_size=49,
            embedding_dim=32,
            branch_dim=64,
            conv_channels=8,
            transformer_heads=4,
            transformer_layers=1,
            transformer_ff_dim=64,
            dropout=0.0,
            active_branches=("one_hot", "aaindex"),
        )
        tokens = torch.zeros((2, 49), dtype=torch.long)
        tokens[:, 24] = 11
        _, gates = model(tokens, return_gates=True)
        torch.testing.assert_close(gates[:, 0], torch.zeros(2))
        torch.testing.assert_close(gates.sum(dim=1), torch.ones(2))

    def test_mmubipred_compatible_outputs(self) -> None:
        model = MMUbiPredCompatible(
            aaindex_lookup=np.zeros((21, 31), dtype=np.float32),
            window_size=49,
        )
        tokens = torch.zeros((2, 49), dtype=torch.long)
        tokens[:, 24] = 11
        logits, branch_probabilities = model(tokens, return_gates=True)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(branch_probabilities.shape), (2, 3))
        self.assertTrue(torch.all(branch_probabilities >= 0))
        self.assertTrue(torch.all(branch_probabilities <= 1))
        regularization = model.regularization_loss()
        self.assertGreater(float(regularization.detach()), 0.0)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, torch.tensor([0.0, 1.0])
        ) + regularization
        loss.backward()
        self.assertIsNotNone(model.fusion_output.weight.grad)

    def test_mmubipred_uses_keras_style_initialization(self) -> None:
        torch.manual_seed(123)
        model = MMUbiPredCompatible(
            aaindex_lookup=np.zeros((21, 31), dtype=np.float32),
            window_size=49,
        )
        torch.testing.assert_close(
            model.fusion_dense.bias, torch.zeros_like(model.fusion_dense.bias)
        )
        self.assertLessEqual(
            float(model.sequence_embedding.weight.detach().abs().max()), 0.05
        )
        forget_bias = model.aaindex_lstm.bias_ih_l0.detach()[64:128]
        torch.testing.assert_close(forget_bias, torch.ones_like(forget_bias))

        model.eval()
        tokens = torch.randint(0, 20, (32, 49))
        tokens[:, 24] = 11
        logits = model(tokens)
        self.assertGreater(float(logits.detach().std()), 1e-4)

    def test_esm2_crossfusion_outputs_and_compacts_padding(self) -> None:
        backbone = FakeESMBackbone()
        model = ESM2CrossFusion(
            aaindex_lookup=np.zeros((21, 31), dtype=np.float32),
            backbone=backbone,
            residue_token_lookup=list(range(4, 24)) + [0],
            cls_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            window_size=49,
            branch_dim=32,
            conv_channels=8,
            conv_kernels=(3, 5),
            conv_dilations=(1, 2),
            dropout=0.0,
            freeze_backbone=True,
        )
        tokens = torch.full((2, 49), 20, dtype=torch.long)
        tokens[:, 10:39] = 0
        tokens[:, 24] = 11

        esm_ids, attention_mask, center_positions = model._esm_inputs(tokens)
        self.assertEqual(attention_mask.sum(dim=1).tolist(), [31, 31])
        self.assertEqual(center_positions.tolist(), [15, 15])
        self.assertEqual(esm_ids[:, 30].tolist(), [2, 2])

        model.train()
        self.assertFalse(backbone.training)
        logits, gates = model(tokens, return_gates=True)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertEqual(tuple(gates.shape), (2, 3))
        torch.testing.assert_close(gates.sum(dim=1), torch.ones(2))
        self.assertFalse(any(parameter.requires_grad for parameter in backbone.parameters()))

    def test_center_lora_esm2_uses_candidate_residue(self) -> None:
        backbone = FakeESMBackbone()
        model = CenterLoRAESM2(
            backbone=backbone,
            backbone_hidden_size=16,
            residue_token_lookup=list(range(4, 24)) + [0],
            cls_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            window_size=49,
            classifier_hidden_dim=8,
            dropout=0.0,
        )
        tokens = torch.full((3, 49), 20, dtype=torch.long)
        tokens[:, 8:41] = 0
        tokens[:, 24] = 11

        esm_ids, attention_mask, center_positions = model._esm_inputs(tokens)
        self.assertEqual(attention_mask.sum(dim=1).tolist(), [35, 35, 35])
        self.assertEqual(center_positions.tolist(), [17, 17, 17])
        self.assertEqual(esm_ids[:, 17].tolist(), [15, 15, 15])

        logits, weights = model(tokens, return_gates=True)
        self.assertEqual(tuple(logits.shape), (3,))
        torch.testing.assert_close(weights, torch.ones((3, 1)))
        logits.sum().backward()
        self.assertIsNotNone(backbone.embedding.weight.grad)

    def test_center_lora_factory_configures_feature_extraction_lora(self) -> None:
        fake_peft = ModuleType("peft")
        fake_peft.LoraConfig = lambda **kwargs: SimpleNamespace(**kwargs)

        def attach_lora(backbone: FakeESMBackbone, config: SimpleNamespace):
            backbone.lora_config = config
            return backbone

        fake_peft.get_peft_model = attach_lora
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModel = FakeAutoModel
        fake_transformers.AutoTokenizer = FakeESMTokenizer

        with patch.dict(
            sys.modules,
            {"peft": fake_peft, "transformers": fake_transformers},
        ):
            model = CenterLoRAESM2.from_pretrained(
                pretrained_model_name="fake-esm",
                lora_rank=8,
                lora_alpha=16,
                lora_dropout=0.1,
                lora_target_modules=("query", "value"),
                window_size=49,
                classifier_hidden_dim=8,
                dropout=0.0,
            )
            multiscale_model = MultiScaleCenterLoRAESM2.from_pretrained(
                pretrained_model_name="fake-esm",
                lora_rank=8,
                lora_alpha=16,
                lora_dropout=0.1,
                lora_target_modules=("query", "value"),
                window_size=49,
                classifier_hidden_dim=8,
                dropout=0.0,
                pooling_radii=(2, 5),
            )

        self.assertEqual(model.esm_backbone.lora_config.task_type, "FEATURE_EXTRACTION")
        self.assertEqual(model.esm_backbone.lora_config.target_modules, ["query", "value"])
        self.assertEqual(model.esm_backbone.lora_config.r, 8)
        self.assertIsInstance(multiscale_model, MultiScaleCenterLoRAESM2)
        self.assertEqual(multiscale_model.pooling_radii, (2, 5))

    def test_center_lora_factory_explains_incompatible_torchao(self) -> None:
        fake_peft = ModuleType("peft")
        fake_peft.LoraConfig = lambda **kwargs: SimpleNamespace(**kwargs)

        def reject_torchao(*args: object, **kwargs: object) -> None:
            del args, kwargs
            raise ImportError(
                "Found an incompatible version of torchao. Found version 0.10.0"
            )

        fake_peft.get_peft_model = reject_torchao
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModel = FakeAutoModel
        fake_transformers.AutoTokenizer = FakeESMTokenizer

        with patch.dict(
            sys.modules,
            {"peft": fake_peft, "transformers": fake_transformers},
        ):
            with self.assertRaisesRegex(ImportError, "pip uninstall -y torchao"):
                CenterLoRAESM2.from_pretrained(
                    pretrained_model_name="fake-esm",
                    lora_rank=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    lora_target_modules=("query", "value"),
                    window_size=49,
                    classifier_hidden_dim=8,
                    dropout=0.0,
                )

    def test_multiscale_center_lora_fuses_masked_context(self) -> None:
        backbone = FakeESMBackbone()
        model = MultiScaleCenterLoRAESM2(
            backbone=backbone,
            backbone_hidden_size=16,
            residue_token_lookup=list(range(4, 24)) + [0],
            cls_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            window_size=49,
            classifier_hidden_dim=8,
            dropout=0.0,
            pooling_radii=(2, 5),
        )
        tokens = torch.full((3, 49), 20, dtype=torch.long)
        tokens[0, 22:27] = 0
        tokens[1, 19:30] = 0
        tokens[2, 8:41] = 0
        tokens[:, 24] = 11

        logits, weights = model(tokens, return_gates=True)
        self.assertEqual(tuple(logits.shape), (3,))
        self.assertEqual(tuple(weights.shape), (3, 3))
        self.assertEqual(
            model.branch_names,
            ("esm2_center", "esm2_radius_2", "esm2_radius_5"),
        )
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(3))
        logits.sum().backward()
        self.assertIsNotNone(backbone.embedding.weight.grad)
        self.assertIsNotNone(model.component_gate.weight.grad)

    def test_multiscale_pool_excludes_special_and_padding_tokens(self) -> None:
        features = torch.arange(7, dtype=torch.float32).view(1, 7, 1)
        pooled = MultiScaleCenterLoRAESM2._centered_mean_pool(
            features=features,
            center_positions=torch.tensor([2]),
            residue_lengths=torch.tensor([3]),
            radius=2,
        )
        # Positions 1, 2, and 3 are residues; position 0 is CLS and 4 is EOS.
        torch.testing.assert_close(pooled, torch.tensor([[2.0]]))

    def test_multiscale_build_model_routes_frozen_configuration(self) -> None:
        model_config = {
            "architecture": "center_lora_esm2_multiscale_v2",
            "pretrained_model_name": "fake-esm",
            "classifier_hidden_dim": 256,
            "dropout": 0.3,
            "pooling_radii": [2, 5],
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_target_modules": ["query", "value"],
        }
        sentinel = object()
        with patch.object(
            MultiScaleCenterLoRAESM2,
            "from_pretrained",
            return_value=sentinel,
        ) as factory:
            model = build_model(
                aaindex_lookup=np.zeros((21, 31), dtype=np.float32),
                window_size=49,
                model_config=model_config,
            )

        self.assertIs(model, sentinel)
        factory.assert_called_once_with(
            pretrained_model_name="fake-esm",
            window_size=49,
            classifier_hidden_dim=256,
            dropout=0.3,
            pooling_radii=(2, 5),
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.1,
            lora_target_modules=("query", "value"),
        )

    def test_protbert_factory_uses_sep_and_gradient_checkpointing(self) -> None:
        fake_peft = ModuleType("peft")
        fake_peft.LoraConfig = lambda **kwargs: SimpleNamespace(**kwargs)

        def attach_lora(backbone: FakeESMBackbone, config: SimpleNamespace):
            backbone.lora_config = config
            return backbone

        fake_peft.get_peft_model = attach_lora
        fake_transformers = ModuleType("transformers")
        fake_transformers.AutoModel = FakeAutoModel
        fake_transformers.AutoTokenizer = FakeProtBERTTokenizer

        with patch.dict(
            sys.modules,
            {"peft": fake_peft, "transformers": fake_transformers},
        ):
            model = CenterLoRAProtBERT.from_pretrained(
                pretrained_model_name="fake-protbert",
                lora_rank=8,
                lora_alpha=16,
                lora_dropout=0.1,
                lora_target_modules=("query", "value"),
                gradient_checkpointing=True,
                tokenizer_do_lower_case=False,
                window_size=49,
                classifier_hidden_dim=8,
                dropout=0.0,
            )

        self.assertIsInstance(model, CenterLoRAProtBERT)
        self.assertEqual(model.eos_token_id, FakeProtBERTTokenizer.sep_token_id)
        self.assertEqual(
            FakeProtBERTTokenizer.received_kwargs, {"do_lower_case": False}
        )
        self.assertTrue(model.esm_backbone.gradient_checkpointing_enabled)
        self.assertTrue(model.esm_backbone.input_grads_enabled)
        self.assertEqual(model.branch_names, ("protbert_center",))

    def test_protbert_build_model_routes_frozen_configuration(self) -> None:
        model_config = {
            "architecture": "center_lora_protbert_v1",
            "pretrained_model_name": "Rostlab/prot_bert_bfd",
            "classifier_hidden_dim": 256,
            "dropout": 0.3,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "lora_target_modules": ["query", "value"],
            "gradient_checkpointing": True,
            "tokenizer_do_lower_case": False,
        }
        sentinel = object()
        with patch.object(
            CenterLoRAProtBERT,
            "from_pretrained",
            return_value=sentinel,
        ) as factory:
            model = build_model(
                aaindex_lookup=np.zeros((21, 31), dtype=np.float32),
                window_size=49,
                model_config=model_config,
            )

        self.assertIs(model, sentinel)
        factory.assert_called_once_with(
            pretrained_model_name="Rostlab/prot_bert_bfd",
            window_size=49,
            classifier_hidden_dim=256,
            dropout=0.3,
            lora_rank=8,
            lora_alpha=16,
            lora_dropout=0.1,
            lora_target_modules=("query", "value"),
            gradient_checkpointing=True,
            tokenizer_do_lower_case=False,
        )


if __name__ == "__main__":
    unittest.main()
