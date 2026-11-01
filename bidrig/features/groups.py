"""Feature groups, in the column order of every feature table."""

from __future__ import annotations

from bidrig.features.change import CHANGE_FEATURES
from bidrig.features.dependence import DEP_FEATURES
from bidrig.features.literature import LIT_FEATURES
from bidrig.features.participation import PART_FEATURES
from bidrig.features.sequential import SEQ_FEATURES

FEATURE_GROUPS = {
    "lit": LIT_FEATURES,
    "dep": DEP_FEATURES,
    "part": PART_FEATURES,
    "change": CHANGE_FEATURES,
    "seq": SEQ_FEATURES,
}
ALL_FEATURES = tuple(f for g in FEATURE_GROUPS.values() for f in g)


def group_features(groups) -> tuple[str, ...]:
    """Feature names of the given groups, in the order the groups are listed."""
    return tuple(f for g in groups for f in FEATURE_GROUPS[g])
