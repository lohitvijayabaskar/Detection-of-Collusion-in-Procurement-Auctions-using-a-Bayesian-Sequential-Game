"""Detector variants. Importing this package registers every variant module."""

from bidrig.detectors import previous, stat_auc  # noqa: F401  (import registers the variants)
from bidrig.detectors.base import REGISTRY, Detector, make, register

__all__ = ["REGISTRY", "Detector", "make", "register"]
