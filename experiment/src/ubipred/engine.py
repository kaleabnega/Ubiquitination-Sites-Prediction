"""Training and prediction loops for publication experiments."""

from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import compute_metrics, select_mcc_threshold, write_json


def _autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


def _build_optimizer(
    model: nn.Module,
    name: str,
    learning_rate: float,
    weight_decay: float,
    epsilon: float,
) -> torch.optim.Optimizer:
    if epsilon <= 0:
        raise ValueError("optimizer epsilon must be positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=epsilon,
        )
    if name == "adamw":
        return torch.optim.AdamW(
            parameters,
            lr=learning_rate,
            weight_decay=weight_decay,
            eps=epsilon,
        )
    raise ValueError("optimizer must be 'adam' or 'adamw'")


def _loss_with_regularization(
    model: nn.Module,
    criterion: nn.Module,
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    loss = criterion(logits, labels)
    regularization_loss = getattr(model, "regularization_loss", None)
    if callable(regularization_loss):
        loss = loss + regularization_loss()
    return loss


def _trainable_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in trainable_names
    }


def _load_trainable_state(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Unexpected trainable checkpoint keys: {incompatible.unexpected_keys}"
        )


def load_checkpoint_model_state(
    model: nn.Module, checkpoint: dict[str, object]
) -> None:
    """Load either a legacy full checkpoint or a compact trainable-only one."""

    state_dict = checkpoint["model_state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint model_state_dict must be a dictionary")
    scope = str(checkpoint.get("state_dict_scope", "full_model"))
    if scope == "full_model":
        model.load_state_dict(state_dict)
    elif scope == "trainable_parameters":
        _load_trainable_state(model, state_dict)
    else:
        raise ValueError(f"Unsupported checkpoint state_dict_scope: {scope}")


def _optimizer_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    gradient_clip_norm: float | None,
) -> None:
    scaler.unscale_(optimizer)
    if gradient_clip_norm is not None:
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=gradient_clip_norm
        )
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    gates: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            logits, batch_gates = model(tokens, return_gates=True)
            labels.append(batch["label"].cpu().numpy())
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            indices.append(batch["index"].cpu().numpy())
            gates.append(batch_gates.cpu().numpy())

    return (
        np.concatenate(labels),
        np.concatenate(probabilities),
        np.concatenate(indices),
        np.concatenate(gates),
    )


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
    optimizer_epsilon: float,
    gradient_clip_norm: float | None,
    gradient_accumulation_steps: int,
    use_amp: bool,
    checkpoint_metadata: dict[str, object],
    resume: bool = False,
) -> dict[str, object]:
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive or None")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if resume and bool(getattr(train_loader, "persistent_workers", False)):
        raise ValueError(
            "Exact epoch-boundary resume requires num_workers=0 so DataLoader "
            "generator consumption is reproducible"
        )
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    optimizer = _build_optimizer(
        model,
        name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epsilon=optimizer_epsilon,
    )
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)

    history: list[dict[str, object]] = []
    best_mcc = float("-inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_threshold = 0.5
    epochs_without_improvement = 0
    start_epoch = 1
    resume_path = run_dir / "development_resume.pt"

    if resume and resume_path.exists():
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        if resume_checkpoint["metadata"] != checkpoint_metadata:
            raise ValueError(
                "Resume checkpoint metadata does not match this code/configuration"
            )
        _load_trainable_state(model, resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        history = list(resume_checkpoint["history"])
        best_mcc = float(resume_checkpoint["best_mcc"])
        best_epoch = int(resume_checkpoint["best_epoch"])
        best_state = resume_checkpoint["best_model_state_dict"]
        best_threshold = float(resume_checkpoint["best_threshold"])
        epochs_without_improvement = int(
            resume_checkpoint["epochs_without_improvement"]
        )
        start_epoch = int(resume_checkpoint["completed_epoch"]) + 1
        random.setstate(resume_checkpoint["python_rng_state"])
        np.random.set_state(resume_checkpoint["numpy_rng_state"])
        torch.set_rng_state(resume_checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and resume_checkpoint["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(resume_checkpoint["cuda_rng_states"])
        loader_generator = getattr(train_loader, "generator", None)
        if (
            loader_generator is not None
            and resume_checkpoint["loader_generator_state"] is not None
        ):
            loader_generator.set_state(
                resume_checkpoint["loader_generator_state"]
            )
        print(
            f"resumed_after_epoch={start_epoch - 1:03d} "
            f"best_epoch={best_epoch:03d}",
            flush=True,
        )

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        print(f"epoch={epoch:03d} started", flush=True)
        model.train()
        total_loss = 0.0
        total_examples = 0
        optimizer.zero_grad(set_to_none=True)
        batch_count = len(train_loader)

        for batch_index, batch in enumerate(train_loader, start=1):
            tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            group_start = (
                (batch_index - 1) // gradient_accumulation_steps
            ) * gradient_accumulation_steps
            group_size = min(
                gradient_accumulation_steps, batch_count - group_start
            )

            with _autocast_context(device, enabled=amp_enabled):
                logits = model(tokens)
                loss = _loss_with_regularization(model, criterion, logits, labels)

            scaler.scale(loss / group_size).backward()
            if (
                batch_index % gradient_accumulation_steps == 0
                or batch_index == batch_count
            ):
                _optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    gradient_clip_norm=gradient_clip_norm,
                )

            batch_size = labels.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_examples += batch_size

        validation_labels, validation_probabilities, _, validation_gates = predict(
            model, validation_loader, device
        )
        fixed_metrics = compute_metrics(
            validation_labels, validation_probabilities, threshold=0.5
        )
        selected_threshold, selected_mcc = select_mcc_threshold(
            validation_labels, validation_probabilities
        )
        diagnostic_name = str(getattr(model, "diagnostic_name", "branch_diagnostics"))
        diagnostic_key = f"mean_{diagnostic_name}"
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_examples, 1),
            "validation_fixed": fixed_metrics,
            "validation_selected_threshold": selected_threshold,
            "validation_selected_mcc": selected_mcc,
            "branch_diagnostic_name": diagnostic_name,
            diagnostic_key: validation_gates.mean(axis=0).tolist(),
            "seconds": time.time() - started,
        }
        history.append(epoch_record)
        write_json(run_dir / "development_history.json", history)

        current_mcc = float(fixed_metrics["mcc"])
        print(
            f"epoch={epoch:03d} "
            f"loss={epoch_record['train_loss']:.5f} "
            f"val_mcc@0.5={current_mcc:.5f} "
            f"val_auprc={fixed_metrics['auprc']:.5f} "
            f"{diagnostic_key}={epoch_record[diagnostic_key]}"
        )

        if current_mcc > best_mcc:
            best_mcc = current_mcc
            best_epoch = epoch
            best_threshold = selected_threshold
            best_state = _trainable_state_dict(model)
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": best_state,
                "state_dict_scope": "trainable_parameters",
                "best_epoch": best_epoch,
                "validation_mcc_at_0_5": best_mcc,
                "validation_selected_threshold": best_threshold,
                "metadata": checkpoint_metadata,
            }
            torch.save(checkpoint, run_dir / "development_best.pt")
        else:
            epochs_without_improvement += 1

        loader_generator = getattr(train_loader, "generator", None)
        resume_checkpoint = {
            "completed_epoch": epoch,
            "model_state_dict": _trainable_state_dict(model),
            "best_model_state_dict": best_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "best_mcc": best_mcc,
            "best_epoch": best_epoch,
            "best_threshold": best_threshold,
            "epochs_without_improvement": epochs_without_improvement,
            "metadata": checkpoint_metadata,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "loader_generator_state": (
                loader_generator.get_state()
                if loader_generator is not None
                else None
            ),
        }
        torch.save(resume_checkpoint, resume_path)

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs")
            break

    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint")
    _load_trainable_state(model, best_state)

    summary = {
        "best_epoch": best_epoch,
        "validation_mcc_at_0_5": best_mcc,
        "validation_selected_threshold": best_threshold,
        "epochs_completed": len(history),
    }
    write_json(run_dir / "development_summary.json", summary)
    return summary


def refit_model(
    model: nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    output_dir: str | Path,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    optimizer_name: str,
    optimizer_epsilon: float,
    gradient_clip_norm: float | None,
    gradient_accumulation_steps: int,
    use_amp: bool,
    validation_selected_threshold: float,
    checkpoint_metadata: dict[str, object],
    resume: bool = False,
) -> dict[str, object]:
    """Refit a fresh model on every released training sample.

    The number of epochs and optional reporting threshold must already have
    been selected on development data. This function has no validation or test
    loader, so the final checkpoint cannot adapt to either evaluation set.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive or None")
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be positive")
    if resume and bool(getattr(train_loader, "persistent_workers", False)):
        raise ValueError(
            "Exact epoch-boundary resume requires num_workers=0 so DataLoader "
            "generator consumption is reproducible"
        )

    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = _build_optimizer(
        model,
        name=optimizer_name,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        epsilon=optimizer_epsilon,
    )
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    history: list[dict[str, object]] = []
    start_epoch = 1
    resume_path = run_dir / "refit_resume.pt"

    if resume and resume_path.exists():
        resume_checkpoint = torch.load(
            resume_path, map_location="cpu", weights_only=False
        )
        if resume_checkpoint["metadata"] != checkpoint_metadata:
            raise ValueError(
                "Refit resume checkpoint metadata does not match this "
                "code/configuration"
            )
        if int(resume_checkpoint["target_epochs"]) != epochs:
            raise ValueError("Refit resume checkpoint target epoch count changed")
        if (
            float(resume_checkpoint["validation_selected_threshold"])
            != float(validation_selected_threshold)
        ):
            raise ValueError(
                "Refit resume checkpoint reporting threshold changed"
            )
        _load_trainable_state(model, resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scaler.load_state_dict(resume_checkpoint["scaler_state_dict"])
        history = list(resume_checkpoint["history"])
        start_epoch = int(resume_checkpoint["completed_epoch"]) + 1
        random.setstate(resume_checkpoint["python_rng_state"])
        np.random.set_state(resume_checkpoint["numpy_rng_state"])
        torch.set_rng_state(resume_checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and resume_checkpoint["cuda_rng_states"]:
            torch.cuda.set_rng_state_all(resume_checkpoint["cuda_rng_states"])
        loader_generator = getattr(train_loader, "generator", None)
        if (
            loader_generator is not None
            and resume_checkpoint["loader_generator_state"] is not None
        ):
            loader_generator.set_state(
                resume_checkpoint["loader_generator_state"]
            )
        print(
            f"refit_resumed_after_epoch={start_epoch - 1:03d}",
            flush=True,
        )

    for epoch in range(start_epoch, epochs + 1):
        started = time.time()
        print(f"refit_epoch={epoch:03d}/{epochs:03d} started", flush=True)
        model.train()
        total_loss = 0.0
        total_examples = 0
        optimizer.zero_grad(set_to_none=True)
        batch_count = len(train_loader)

        for batch_index, batch in enumerate(train_loader, start=1):
            tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            group_start = (
                (batch_index - 1) // gradient_accumulation_steps
            ) * gradient_accumulation_steps
            group_size = min(
                gradient_accumulation_steps, batch_count - group_start
            )

            with _autocast_context(device, enabled=amp_enabled):
                logits = model(tokens)
                loss = _loss_with_regularization(model, criterion, logits, labels)

            scaler.scale(loss / group_size).backward()
            if (
                batch_index % gradient_accumulation_steps == 0
                or batch_index == batch_count
            ):
                _optimizer_step(
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    gradient_clip_norm=gradient_clip_norm,
                )

            batch_size = labels.shape[0]
            total_loss += float(loss.detach()) * batch_size
            total_examples += batch_size

        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_examples, 1),
            "samples": total_examples,
            "seconds": time.time() - started,
        }
        history.append(epoch_record)
        write_json(run_dir / "refit_history.json", history)
        print(
            f"refit_epoch={epoch:03d}/{epochs:03d} "
            f"loss={epoch_record['train_loss']:.5f}"
        )

        loader_generator = getattr(train_loader, "generator", None)
        resume_checkpoint = {
            "completed_epoch": epoch,
            "target_epochs": epochs,
            "model_state_dict": _trainable_state_dict(model),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "history": history,
            "validation_selected_threshold": float(
                validation_selected_threshold
            ),
            "metadata": checkpoint_metadata,
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
            ),
            "loader_generator_state": (
                loader_generator.get_state()
                if loader_generator is not None
                else None
            ),
        }
        torch.save(resume_checkpoint, resume_path)

    if len(history) != epochs:
        raise RuntimeError(
            f"Refit history has {len(history)} epochs; expected {epochs}"
        )
    checkpoint = {
        "model_state_dict": _trainable_state_dict(model),
        "state_dict_scope": "trainable_parameters",
        "best_epoch": epochs,
        "refit_epochs": epochs,
        "validation_selected_threshold": float(validation_selected_threshold),
        "metadata": checkpoint_metadata,
    }
    torch.save(checkpoint, run_dir / "best.pt")
    summary = {
        "epochs": epochs,
        "samples": len(train_loader.dataset),
        "checkpoint": "best.pt",
    }
    write_json(run_dir / "refit_summary.json", summary)
    return summary
