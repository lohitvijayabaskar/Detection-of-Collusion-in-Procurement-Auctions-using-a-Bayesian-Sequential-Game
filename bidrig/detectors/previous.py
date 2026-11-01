"""Variant ``previous``: the detector before iteration I1.

Same model and feature groups as ``full_gbm`` (lit, dep, change, seq), with every column that
ends in ``_t`` removed. Those columns are the dependence statistics standardised against the
second, within-firm null. No training code differs, so the only change is the feature set.
"""

from __future__ import annotations

from bidrig.detectors.base import GBMOn, register
from bidrig.features import FEATURE_GROUPS

GROUPS = ("lit", "dep", "change", "seq")  # the groups of full_gbm


def full_gbm_features() -> tuple[str, ...]:
    return tuple(f for g in GROUPS for f in FEATURE_GROUPS[g])


def drop_time_null(features) -> tuple[str, ...]:
    """Remove the second-null columns (suffix ``_t``)."""
    return tuple(f for f in features if not f.endswith("_t"))


@register("previous")
class Previous(GBMOn):
    name = "previous"

    def __init__(self):
        super().__init__(drop_time_null(full_gbm_features()))
