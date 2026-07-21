"""Reusable components for the UbiFusionNet experiments."""

# Keep package initialization dependency-light so the dataset audit can run
# without importing PyTorch or scikit-learn.
from .fasta import ALPHABET, SiteRecord, load_labelled_fasta, prepare_records

__all__ = [
    "ALPHABET",
    "SiteRecord",
    "load_labelled_fasta",
    "prepare_records",
]
