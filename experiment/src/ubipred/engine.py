"""Training and prediction loops for publication experiments."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import compute_metrics, select_mcc_threshold, write_json


def _autocast_context(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled)


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
    use_amp: bool,
    checkpoint_metadata: dict[str, object],
) -> dict[str, object]:
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
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

    for epoch in range(1, epochs + 1):
        started = time.time()
        print(f"epoch={epoch:03d} started", flush=True)
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with _autocast_context(device, enabled=amp_enabled):
                logits = model(tokens)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

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
        epoch_record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_examples, 1),
            "validation_fixed": fixed_metrics,
            "validation_selected_threshold": selected_threshold,
            "validation_selected_mcc": selected_mcc,
            "mean_branch_gates": validation_gates.mean(axis=0).tolist(),
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
            f"gate_mean={epoch_record['mean_branch_gates']}"
        )

        if current_mcc > best_mcc:
            best_mcc = current_mcc
            best_epoch = epoch
            best_threshold = selected_threshold
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            checkpoint = {
                "model_state_dict": best_state,
                "best_epoch": best_epoch,
                "validation_mcc_at_0_5": best_mcc,
                "validation_selected_threshold": best_threshold,
                "metadata": checkpoint_metadata,
            }
            torch.save(checkpoint, run_dir / "development_best.pt")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs")
            break

    if best_state is None:
        raise RuntimeError("Training completed without a valid checkpoint")
    model.load_state_dict(best_state)

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
    use_amp: bool,
    validation_selected_threshold: float,
    checkpoint_metadata: dict[str, object],
) -> dict[str, object]:
    """Refit a fresh model on every released training sample.

    The number of epochs and optional reporting threshold must already have
    been selected on development data. This function has no validation or test
    loader, so the final checkpoint cannot adapt to either evaluation set.
    """

    if epochs <= 0:
        raise ValueError("epochs must be positive")

    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.BCEWithLogitsLoss()
    amp_enabled = bool(use_amp and device.type == "cuda")
    scaler = torch.amp.GradScaler(device.type, enabled=amp_enabled)
    history: list[dict[str, object]] = []

    for epoch in range(1, epochs + 1):
        started = time.time()
        print(f"refit_epoch={epoch:03d}/{epochs:03d} started", flush=True)
        model.train()
        total_loss = 0.0
        total_examples = 0

        for batch in train_loader:
            tokens = batch["tokens"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with _autocast_context(device, enabled=amp_enabled):
                logits = model(tokens)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

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

    checkpoint = {
        "model_state_dict": copy.deepcopy(model.state_dict()),
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
