"""The reference family: gradient boosting on a union of feature groups.

``full_gbm`` is the main detector; the others are ablations and baselines built from subsets of
the groups. The monotone designs constrain the model along ``priors.TAIL_DIRECTIONS``.
"""

from __future__ import annotations

from bidrig.detectors.base import GBMOn, register
from bidrig.detectors.priors import TAIL_DIRECTIONS
from bidrig.features import group_features

DESIGNS = {
    "lit_gbm": ("lit",),
    "dep_gbm": ("dep",),
    "change_gbm": ("change",),
    "seq_gbm": ("seq",),
    "knowledge_gbm": ("dep", "change", "seq"),
    "full_gbm": ("lit", "dep", "change", "seq"),
    "full_minus_dep": ("lit", "change", "seq"),
    "full_minus_change": ("lit", "dep", "seq"),
    "full_minus_seq": ("lit", "dep", "change"),
    "full_part_gbm": ("lit", "dep", "part", "change", "seq"),
    "lit_dep_part_gbm": ("lit", "dep", "part"),
}
MONOTONE = {
    "full_part_mono": ("lit", "dep", "part", "change", "seq"),
    "lit_dep_part_mono": ("lit", "dep", "part"),
}


class GBM(GBMOn):
    """Gradient boosting on the union of ``groups`` (optionally monotone)."""

    def __init__(self, groups, monotone: dict | None = None):
        self.groups = tuple(groups)
        super().__init__(group_features(self.groups), monotone=monotone)


for _name, _groups in DESIGNS.items():
    register(_name)(lambda g=_groups: GBM(g))
for _name, _groups in MONOTONE.items():
    register(_name)(lambda g=_groups: GBM(g, monotone=dict(TAIL_DIRECTIONS)))
