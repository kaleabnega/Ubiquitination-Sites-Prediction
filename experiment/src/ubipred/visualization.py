"""Frozen-representation extraction and publication plotting utilities."""

from __future__ import annotations

import inspect
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .model import LongContextLoRAESM2, MMUbiPredCompatible


MODEL_COLORS = {
    "Exact MMUbiPred": "#3B4CC0",
    "Local expert": "#2A9D8F",
    "Context expert": "#7B2CBF",
    "Residual hybrid": "#E76F51",
}
CLASS_COLORS = {0: "#2878B5", 1: "#D9534F"}
AMINO_ACIDS = tuple("ACDEFGHIKLMNPQRSTVWY")
COMPLEMENTARITY_COLORS = {
    "Both correct": "#4C78A8",
    "Local only correct": "#59A14F",
    "Context only correct": "#E15759",
    "Both wrong": "#B8B8B8",
}


def _validate_aligned_sequences(
    sequences: Sequence[str], masks: Sequence[np.ndarray]
) -> tuple[list[str], int]:
    normalized = [str(sequence).upper() for sequence in sequences]
    if not normalized:
        raise ValueError("At least one sequence is required")
    width = len(normalized[0])
    if width <= 0 or any(len(sequence) != width for sequence in normalized):
        raise ValueError("All sequences must have one identical positive length")
    for mask in masks:
        values = np.asarray(mask)
        if values.shape != (len(normalized),):
            raise ValueError("Every selection mask must align with the sequences")
    return normalized, width


def positional_log2_enrichment(
    sequences: Sequence[str],
    numerator_mask: np.ndarray,
    denominator_mask: np.ndarray,
    *,
    pseudocount: float = 0.5,
    alphabet: Sequence[str] = AMINO_ACIDS,
) -> np.ndarray:
    """Return smoothed position-wise residue enrichment between two groups."""

    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    numerator_mask = np.asarray(numerator_mask, dtype=bool)
    denominator_mask = np.asarray(denominator_mask, dtype=bool)
    normalized, width = _validate_aligned_sequences(
        sequences, (numerator_mask, denominator_mask)
    )
    if not numerator_mask.any() or not denominator_mask.any():
        raise ValueError("Both enrichment groups must contain at least one sequence")
    residues = tuple(str(residue).upper() for residue in alphabet)
    if len(set(residues)) != len(residues) or any(len(value) != 1 for value in residues):
        raise ValueError("The residue alphabet must contain unique single characters")
    residue_set = set(residues)

    encoded = np.asarray([list(sequence) for sequence in normalized], dtype="U1")
    frequencies: list[np.ndarray] = []
    for selected in (numerator_mask, denominator_mask):
        group = encoded[selected]
        valid = np.isin(group, residues)
        valid_counts = valid.sum(axis=0).astype(np.float64)
        counts = np.stack(
            [(group == residue).sum(axis=0) for residue in residues]
        ).astype(np.float64)
        if not set(np.unique(group[valid])).issubset(residue_set):
            raise AssertionError("Internal residue-alphabet error")
        frequencies.append(
            (counts + pseudocount)
            / (valid_counts[np.newaxis, :] + pseudocount * len(residues))
        )
    if frequencies[0].shape != (len(residues), width):
        raise AssertionError("Internal enrichment-shape error")
    return np.log2(frequencies[0] / frequencies[1])


def radial_band_log2_enrichment(
    sequences: Sequence[str],
    numerator_mask: np.ndarray,
    denominator_mask: np.ndarray,
    bands: Sequence[tuple[int, int]],
    *,
    pseudocount: float = 0.5,
    alphabet: Sequence[str] = AMINO_ACIDS,
) -> np.ndarray:
    """Return residue enrichment aggregated over symmetric distance bands."""

    if pseudocount <= 0:
        raise ValueError("pseudocount must be positive")
    numerator_mask = np.asarray(numerator_mask, dtype=bool)
    denominator_mask = np.asarray(denominator_mask, dtype=bool)
    normalized, width = _validate_aligned_sequences(
        sequences, (numerator_mask, denominator_mask)
    )
    if width % 2 == 0:
        raise ValueError("Radial enrichment requires odd-length centered sequences")
    if not numerator_mask.any() or not denominator_mask.any():
        raise ValueError("Both enrichment groups must contain at least one sequence")
    radius = width // 2
    normalized_bands = [(int(start), int(stop)) for start, stop in bands]
    if not normalized_bands or any(
        start < 1 or stop < start or stop > radius
        for start, stop in normalized_bands
    ):
        raise ValueError("Distance bands must lie between 1 and the sequence radius")
    residues = tuple(str(residue).upper() for residue in alphabet)
    encoded = np.asarray([list(sequence) for sequence in normalized], dtype="U1")
    matrices: list[np.ndarray] = []
    offsets = np.arange(-radius, radius + 1)
    for selected in (numerator_mask, denominator_mask):
        group = encoded[selected]
        band_frequencies: list[np.ndarray] = []
        for start, stop in normalized_bands:
            positions = (np.abs(offsets) >= start) & (np.abs(offsets) <= stop)
            values = group[:, positions].reshape(-1)
            valid = np.isin(values, residues)
            valid_count = float(valid.sum())
            counts = np.asarray(
                [(values == residue).sum() for residue in residues],
                dtype=np.float64,
            )
            band_frequencies.append(
                (counts + pseudocount)
                / (valid_count + pseudocount * len(residues))
            )
        matrices.append(np.stack(band_frequencies, axis=1))
    return np.log2(matrices[0] / matrices[1])


def expert_correctness_masks(
    labels: np.ndarray,
    local_probabilities: np.ndarray,
    context_probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, np.ndarray]:
    """Partition aligned predictions by local/context correctness."""

    labels = np.asarray(labels, dtype=np.int64)
    local = np.asarray(local_probabilities, dtype=np.float64)
    context = np.asarray(context_probabilities, dtype=np.float64)
    if labels.ndim != 1 or local.shape != labels.shape or context.shape != labels.shape:
        raise ValueError("Labels and expert probabilities must be aligned vectors")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("Binary labels are required")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must lie between zero and one")
    local_correct = (local >= threshold).astype(np.int64) == labels
    context_correct = (context >= threshold).astype(np.int64) == labels
    masks = {
        "Both correct": local_correct & context_correct,
        "Local only correct": local_correct & ~context_correct,
        "Context only correct": ~local_correct & context_correct,
        "Both wrong": ~local_correct & ~context_correct,
    }
    total = np.sum(np.stack(list(masks.values()), axis=0), axis=0)
    if not np.all(total == 1):
        raise AssertionError("Correctness groups must form an exact partition")
    return masks


def _representation_layer(
    model: nn.Module,
) -> tuple[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    if isinstance(model, MMUbiPredCompatible):
        return model.fusion_dense, F.relu
    if isinstance(model, LongContextLoRAESM2):
        return model.classifier[3], lambda value: value
    raise TypeError(
        "Representation extraction supports only MMUbiPredCompatible and "
        "LongContextLoRAESM2"
    )


def predict_with_representations(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return labels, probabilities, indices, and frozen hidden vectors."""

    layer, transform = _representation_layer(model)
    captured: list[torch.Tensor] = []

    def capture_output(
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        captured.append(transform(output).detach().cpu())

    hook = layer.register_forward_hook(capture_output)
    labels: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    indices: list[np.ndarray] = []
    representations: list[np.ndarray] = []
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                captured.clear()
                tokens = batch["tokens"].to(device, non_blocking=True)
                logits = model(tokens)
                if len(captured) != 1:
                    raise RuntimeError(
                        "The representation hook must run exactly once per batch"
                    )
                labels.append(batch["label"].cpu().numpy())
                probabilities.append(torch.sigmoid(logits).cpu().numpy())
                indices.append(batch["index"].cpu().numpy())
                representations.append(captured[0].numpy())
    finally:
        hook.remove()

    return (
        np.concatenate(labels),
        np.concatenate(probabilities),
        np.concatenate(indices),
        np.concatenate(representations),
    )


def stratified_visualization_indices(
    labels: np.ndarray,
    maximum_points: int,
    seed: int,
) -> np.ndarray:
    """Select a deterministic class-balanced subset for nonlinear embedding."""

    labels = np.asarray(labels, dtype=np.int64)
    if labels.ndim != 1 or set(np.unique(labels).tolist()) != {0, 1}:
        raise ValueError("Binary one-dimensional labels are required")
    if maximum_points < 2:
        raise ValueError("maximum_points must be at least two")
    if len(labels) <= maximum_points:
        return np.arange(len(labels), dtype=np.int64)
    rng = np.random.default_rng(seed)
    target_negative = maximum_points // 2
    target_positive = maximum_points - target_negative
    negative = np.flatnonzero(labels == 0)
    positive = np.flatnonzero(labels == 1)
    if len(negative) < target_negative or len(positive) < target_positive:
        minority = min(len(negative), len(positive), maximum_points // 2)
        target_negative = minority
        target_positive = minority
    selected = np.concatenate(
        [
            rng.choice(negative, size=target_negative, replace=False),
            rng.choice(positive, size=target_positive, replace=False),
        ]
    )
    return np.sort(selected.astype(np.int64))


def compute_tsne(
    representations: np.ndarray,
    *,
    seed: int,
    perplexity: float = 30.0,
    iterations: int = 1500,
) -> np.ndarray:
    """Apply standardization, optional PCA, and a fixed t-SNE configuration."""

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler

    values = np.asarray(representations, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 3:
        raise ValueError("Representations must be a two-dimensional sample matrix")
    if not np.isfinite(values).all():
        raise ValueError("Representations contain non-finite values")
    if not 0 < perplexity < values.shape[0]:
        raise ValueError("perplexity must be below the sample count")
    standardized = StandardScaler().fit_transform(values)
    components = min(50, standardized.shape[1], standardized.shape[0] - 1)
    reduced = PCA(n_components=components, random_state=seed).fit_transform(
        standardized
    )
    arguments: dict[str, object] = {
        "n_components": 2,
        "perplexity": perplexity,
        "learning_rate": "auto",
        "init": "pca",
        "random_state": seed,
        "method": "barnes_hut",
        "angle": 0.5,
    }
    iteration_name = (
        "max_iter" if "max_iter" in inspect.signature(TSNE).parameters else "n_iter"
    )
    arguments[iteration_name] = iterations
    return TSNE(**arguments).fit_transform(reduced)


def publication_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "figure.dpi": 120,
            "savefig.dpi": 600,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(figure: object, output_dir: str | Path, stem: str) -> list[Path]:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for suffix in ("pdf", "svg", "png"):
        path = directory / f"{stem}.{suffix}"
        figure.savefig(path, dpi=600 if suffix == "png" else None)
        paths.append(path)
    return paths


def plot_tsne_comparison(
    local_embedding: np.ndarray,
    context_embedding: np.ndarray,
    labels: np.ndarray,
) -> object:
    import matplotlib.pyplot as plt

    publication_style()
    labels = np.asarray(labels, dtype=np.int64)
    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    panels = (
        (local_embedding, "A", "Local expert representation (6D)"),
        (context_embedding, "B", "Context expert representation (256D)"),
    )
    for axis, (embedding, panel, title) in zip(axes, panels):
        for label, name in ((0, "Non-ubiquitinated"), (1, "Ubiquitinated")):
            selected = labels == label
            axis.scatter(
                embedding[selected, 0],
                embedding[selected, 1],
                s=8,
                alpha=0.42,
                c=CLASS_COLORS[label],
                edgecolors="none",
                rasterized=True,
                label=name,
            )
        axis.set_title(f"{panel}. {title}", loc="left", fontweight="bold")
        axis.set_xlabel("t-SNE 1")
        axis.set_ylabel("t-SNE 2")
        axis.spines[["top", "right"]].set_visible(False)
    axes[1].legend(frameon=False, loc="best")
    return figure


def plot_roc_curves(
    labels: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
) -> object:
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    publication_style()
    figure, axis = plt.subplots(figsize=(5.3, 4.6), constrained_layout=True)
    for name, scores in probabilities.items():
        false_positive, true_positive, _ = roc_curve(labels, scores)
        area = auc(false_positive, true_positive)
        axis.plot(
            false_positive,
            true_positive,
            lw=2.0,
            color=MODEL_COLORS.get(name),
            label=f"{name} (AUROC = {area:.3f})",
        )
    axis.plot([0, 1], [0, 1], linestyle="--", color="#777777", lw=1.0)
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.01),
        xlabel="False-positive rate",
        ylabel="True-positive rate",
    )
    axis.set_title("Receiver operating characteristic", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower right")
    return figure


def plot_precision_recall_curves(
    labels: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
) -> object:
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve

    publication_style()
    figure, axis = plt.subplots(figsize=(5.3, 4.6), constrained_layout=True)
    for name, scores in probabilities.items():
        precision, recall, _ = precision_recall_curve(labels, scores)
        area = average_precision_score(labels, scores)
        axis.plot(
            recall,
            precision,
            lw=2.0,
            color=MODEL_COLORS.get(name),
            label=f"{name} (AUPRC = {area:.3f})",
        )
    prevalence = float(np.mean(labels))
    axis.axhline(
        prevalence,
        linestyle="--",
        color="#777777",
        lw=1.0,
        label=f"Prevalence = {prevalence:.3f}",
    )
    axis.set(xlim=(0, 1), ylim=(0, 1.01), xlabel="Recall", ylabel="Precision")
    axis.set_title("Precision–recall curve", fontweight="bold")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, loc="lower left")
    return figure


def plot_confusion_matrix_comparison(
    labels: np.ndarray,
    probabilities: Mapping[str, np.ndarray],
    threshold: float = 0.5,
) -> object:
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    publication_style()
    items = list(probabilities.items())
    columns = 2
    rows = int(np.ceil(len(items) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(8.0, 3.6 * rows),
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(-1)
    image = None
    for panel_index, (axis, (name, scores)) in enumerate(zip(axes, items)):
        matrix = confusion_matrix(
            labels,
            np.asarray(scores) >= threshold,
            labels=[0, 1],
        )
        normalized = matrix / matrix.sum(axis=1, keepdims=True)
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
        for row in range(2):
            for column in range(2):
                color = "white" if normalized[row, column] > 0.55 else "black"
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:,}\n({normalized[row, column] * 100:.1f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                )
        axis.set_xticks([0, 1], ["Non-ubiquitinated", "Ubiquitinated"])
        axis.set_yticks([0, 1], ["Non-ubiquitinated", "Ubiquitinated"])
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
        panel = chr(ord("A") + panel_index)
        axis.set_title(f"{panel}. {name}", loc="left", fontweight="bold")
    for axis in axes[len(items) :]:
        axis.set_visible(False)
    if image is not None:
        figure.colorbar(
            image,
            ax=axes[: len(items)].tolist(),
            shrink=0.75,
            label="Row-normalized fraction",
        )
    figure.suptitle(
        f"Confusion matrices at the fixed threshold {threshold:.1f}",
        fontweight="bold",
    )
    return figure


def plot_training_validation_curves(
    histories: Mapping[str, Sequence[Sequence[Mapping[str, object]]]],
) -> object:
    import matplotlib.pyplot as plt

    publication_style()
    experts = list(histories)
    figure, axes = plt.subplots(
        len(experts),
        2,
        figsize=(10.0, 3.7 * len(experts)),
        constrained_layout=True,
        squeeze=False,
    )
    fold_colors = plt.get_cmap("tab10")
    panel_index = 0
    for row, expert in enumerate(experts):
        for fold, history in enumerate(histories[expert]):
            epochs = [int(record["epoch"]) for record in history]
            losses = [float(record["train_loss"]) for record in history]
            mcc = [float(record["validation_fixed"]["mcc"]) for record in history]
            color = fold_colors(fold)
            axes[row, 0].plot(epochs, losses, color=color, lw=1.6, label=f"Fold {fold}")
            axes[row, 1].plot(epochs, mcc, color=color, lw=1.6, label=f"Fold {fold}")
        for column, (ylabel, descriptor) in enumerate(
            (
                ("Training objective", "training loss"),
                ("Validation MCC", "validation MCC at 0.5"),
            )
        ):
            panel = chr(ord("A") + panel_index)
            panel_index += 1
            axis = axes[row, column]
            axis.set_title(
                f"{panel}. {expert}: {descriptor}",
                loc="left",
                fontweight="bold",
            )
            axis.set_xlabel("Epoch")
            axis.set_ylabel(ylabel)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(alpha=0.2, linewidth=0.6)
    axes[0, 0].legend(frameon=False, ncol=3)
    return figure


def _draw_enrichment_heatmap(
    axis: object,
    matrix: np.ndarray,
    xlabels: Sequence[object],
    *,
    title: str,
    panel: str,
    display_limit: float = 2.0,
) -> object:
    image = axis.imshow(
        np.clip(np.asarray(matrix, dtype=np.float64), -display_limit, display_limit),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-display_limit,
        vmax=display_limit,
        interpolation="nearest",
    )
    axis.set_yticks(np.arange(len(AMINO_ACIDS)), AMINO_ACIDS)
    tick_indices = np.linspace(0, len(xlabels) - 1, min(9, len(xlabels)), dtype=int)
    axis.set_xticks(tick_indices, [str(xlabels[index]) for index in tick_indices])
    axis.set_title(f"{panel}. {title}", loc="left", fontweight="bold")
    axis.set_xlabel("Position relative to central lysine")
    axis.set_ylabel("Amino acid")
    return image


def plot_local_sequence_enrichment(
    sequences: Sequence[str], labels: np.ndarray
) -> object:
    """Plot positive-versus-negative residue enrichment in the 49-mer."""

    import matplotlib.pyplot as plt

    labels = np.asarray(labels, dtype=np.int64)
    normalized, width = _validate_aligned_sequences(sequences, (labels,))
    if width % 2 == 0 or not np.isin(labels, (0, 1)).all():
        raise ValueError("Odd-length sequences and binary labels are required")
    enrichment = positional_log2_enrichment(
        normalized, labels == 1, labels == 0
    )
    positions = np.arange(-(width // 2), width // 2 + 1)
    publication_style()
    figure, axis = plt.subplots(figsize=(11.0, 4.8), constrained_layout=True)
    image = _draw_enrichment_heatmap(
        axis,
        enrichment,
        positions,
        title="Ubiquitinated versus non-ubiquitinated local sequence",
        panel="A",
    )
    axis.axvline(width // 2, color="black", lw=0.8, linestyle="--")
    colorbar = figure.colorbar(image, ax=axis, shrink=0.88)
    colorbar.set_label(
        "log2 residue-frequency ratio\n(positive / negative; clipped at ±2)"
    )
    return figure


def plot_expert_complementarity_biology(
    sequences: Sequence[str],
    labels: np.ndarray,
    local_probabilities: np.ndarray,
    context_probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> tuple[object, dict[str, dict[str, int]]]:
    """Plot correctness partition and sequence signatures of expert-only wins."""

    import matplotlib.pyplot as plt

    labels = np.asarray(labels, dtype=np.int64)
    normalized, width = _validate_aligned_sequences(sequences, (labels,))
    masks = expert_correctness_masks(
        labels,
        local_probabilities,
        context_probabilities,
        threshold=threshold,
    )
    counts = {
        class_name: {
            category: int(np.sum(mask & (labels == label)))
            for category, mask in masks.items()
        }
        for label, class_name in ((0, "Non-ubiquitinated"), (1, "Ubiquitinated"))
    }
    for label in (0, 1):
        if not np.any(masks["Context only correct"] & (labels == label)) or not np.any(
            masks["Local only correct"] & (labels == label)
        ):
            raise ValueError(
                "Both expert-only correctness groups are required within each class"
            )
    positions = np.arange(-(width // 2), width // 2 + 1)
    positive_enrichment = positional_log2_enrichment(
        normalized,
        masks["Context only correct"] & (labels == 1),
        masks["Local only correct"] & (labels == 1),
    )
    negative_enrichment = positional_log2_enrichment(
        normalized,
        masks["Context only correct"] & (labels == 0),
        masks["Local only correct"] & (labels == 0),
    )

    publication_style()
    figure = plt.figure(figsize=(12.0, 9.0), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(0.72, 1.0))
    count_axis = figure.add_subplot(grid[0, :])
    classes = ("Non-ubiquitinated", "Ubiquitinated")
    left = np.zeros(2, dtype=np.float64)
    support = np.asarray([sum(counts[name].values()) for name in classes])
    for category, color in COMPLEMENTARITY_COLORS.items():
        values = np.asarray([counts[name][category] for name in classes])
        fractions = values / support
        count_axis.barh(
            classes,
            fractions,
            left=left,
            color=color,
            label=category,
            height=0.58,
        )
        for row, (start, fraction, value) in enumerate(zip(left, fractions, values)):
            if fraction >= 0.055:
                count_axis.text(
                    start + fraction / 2,
                    row,
                    f"{value:,}\n{fraction * 100:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if category != "Both wrong" else "black",
                )
        left += fractions
    count_axis.set_xlim(0, 1)
    count_axis.set_xlabel("Fraction within true class")
    count_axis.set_title(
        f"A. Expert correctness at threshold {threshold:.1f}",
        loc="left",
        fontweight="bold",
    )
    count_axis.spines[["top", "right", "left"]].set_visible(False)
    count_axis.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))

    positive_axis = figure.add_subplot(grid[1, 0])
    negative_axis = figure.add_subplot(grid[1, 1])
    positive_image = _draw_enrichment_heatmap(
        positive_axis,
        positive_enrichment,
        positions,
        title="Positive sites: context-only versus local-only successes",
        panel="B",
    )
    _draw_enrichment_heatmap(
        negative_axis,
        negative_enrichment,
        positions,
        title="Negative sites: context-only versus local-only successes",
        panel="C",
    )
    for axis in (positive_axis, negative_axis):
        axis.axvline(width // 2, color="black", lw=0.8, linestyle="--")
    colorbar = figure.colorbar(
        positive_image, ax=[positive_axis, negative_axis], shrink=0.86
    )
    colorbar.set_label(
        "log2 residue-frequency ratio\n(context-only / local-only; clipped at ±2)"
    )
    return figure, counts


def plot_context_radial_enrichment(
    contexts: Sequence[str],
    labels: np.ndarray,
    bands: Sequence[tuple[int, int]] = ((1, 5), (6, 24), (25, 64), (65, 128)),
) -> object:
    """Plot positive-versus-negative composition across context distance bands."""

    import matplotlib.pyplot as plt

    labels = np.asarray(labels, dtype=np.int64)
    enrichment = radial_band_log2_enrichment(
        contexts,
        labels == 1,
        labels == 0,
        bands,
    )
    band_labels = [f"±{start}–{stop}" for start, stop in bands]
    publication_style()
    figure, axis = plt.subplots(figsize=(7.2, 5.0), constrained_layout=True)
    image = axis.imshow(
        np.clip(enrichment, -2.0, 2.0),
        aspect="auto",
        cmap="RdBu_r",
        vmin=-2.0,
        vmax=2.0,
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(band_labels)), band_labels)
    axis.set_yticks(np.arange(len(AMINO_ACIDS)), AMINO_ACIDS)
    axis.set_xlabel("Absolute residue distance from central lysine")
    axis.set_ylabel("Amino acid")
    axis.set_title(
        "A. Ubiquitinated versus non-ubiquitinated broad-context composition",
        loc="left",
        fontweight="bold",
    )
    colorbar = figure.colorbar(image, ax=axis, shrink=0.88)
    colorbar.set_label(
        "log2 residue-frequency ratio\n(positive / negative; clipped at ±2)"
    )
    return figure
