"""Frozen invariants for the post-test fixed-30-epoch rebuild."""

from __future__ import annotations


FIXED_EPOCHS = 30
PROTOCOL_VERSION = 1


def validate_fixed30_protocol(config: dict[str, object]) -> None:
    if int(config["fixed_epochs"]) != FIXED_EPOCHS:
        raise ValueError(
            f"This frozen rebuild requires exactly {FIXED_EPOCHS} epochs"
        )
    if int(config["num_workers"]) != 0:
        raise ValueError(
            "Cross-account exact epoch-boundary resume requires num_workers=0"
        )
    if bool(config.get("confirmatory_claim_allowed", True)):
        raise ValueError("The post-test 30-epoch rebuild cannot be confirmatory")
    if not bool(config.get("historical_test_informed", False)):
        raise ValueError("The configuration must acknowledge prior test access")


def fixed30_training_arguments(
    config: dict[str, object],
) -> dict[str, object]:
    """Return optimizer arguments only; duration and stopping are immutable."""

    clip = config.get("gradient_clip_norm")
    return {
        "learning_rate": float(config["learning_rate"]),
        "weight_decay": float(config["weight_decay"]),
        "optimizer_name": str(config["optimizer"]),
        "optimizer_epsilon": float(config["optimizer_epsilon"]),
        "gradient_clip_norm": None if clip is None else float(clip),
        "gradient_accumulation_steps": int(
            config.get("gradient_accumulation_steps", 1)
        ),
        "use_amp": bool(config.get("use_amp", False)),
    }
