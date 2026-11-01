"""Detector variants. Importing this package registers every variant module."""

from bidrig.detectors import previous  # noqa: F401  (import registers the variant)
from bidrig.detectors.base import REGISTRY, Detector, make, register

__all__ = ["REGISTRY", "Detector", "make", "register"]
