"""Multi-detector combinations: simultaneously initialize detectors with varying metric subsets.

Logical divisions of metrics:
- All Metrics (Tier 5): Detector using ALL 5 defined feature groups (lit, dep, part, change, seq),
  spanning all 66 collusion features.
- Single-Metric (Tier 1): Detectors using exactly 1 metric group (lit, dep, part, change, seq).
- Pairwise Metrics (Tier 2): Detectors using 2 metric groups, combining pairs of complementary signals:
  * lit + dep: Empirical IO foundation (price distribution shape + bidder identity correlation).
  * lit + part: Price shape + participation & exclusivity patterns.
  * lit + change: Price shape + structural break detection.
  * lit + seq: Price shape + sequential HMM dynamic tracking.
  * dep + part: Bidder identity correlation + exclusivity (pure behavioral/structural).
  * dep + change: Identity correlation + structural change.
  * dep + seq: Identity correlation + sequential dynamics.
  * part + change: Participation exclusivity + structural breaks.
  * part + seq: Participation exclusivity + sequential tracking.
  * change + seq: Pure dynamic and temporal monitoring.
- Triplet Metrics (Tier 3): Detectors using 3 metric groups:
  * lit + dep + part: Complete static structural suite (prices + identity + participation).
  * lit + dep + seq: Prices + identity + sequential dynamics.
  * lit + dep + change: Prices + identity + structural breaks.
  * lit + part + seq: Prices + participation + sequential monitoring.
  * lit + change + seq: Market-level price dynamics without identity.
  * dep + change + seq: Non-literature behavioral & temporal signals.
  * dep + part + seq: Identity + participation + sequential dynamics.
  * part + change + seq: Participation + structural breaks + sequential dynamics.
- Quad Metrics (Tier 4): Detectors using 4 metric groups (leave-one-group-out ablations):
  * lit + dep + change + seq: All except participation (the standard full_gbm design).
  * lit + dep + part + seq: All except structural change.
  * lit + dep + part + change: All except sequential dynamics.
  * dep + part + change + seq: All behavioral & temporal signals (excluding raw literature screens).
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from bidrig.detectors.base import REGISTRY, GBMOn, register
from bidrig.features import group_features

# Logical groupings of metric combinations by tier (number of metric groups used)
METRIC_COMBINATIONS_BY_TIER: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "all_metrics": {
        "all_metrics_gbm": ("lit", "dep", "part", "change", "seq"),
    },
    "single_metric": {
        "single_lit_gbm": ("lit",),
        "single_dep_gbm": ("dep",),
        "single_part_gbm": ("part",),
        "single_change_gbm": ("change",),
        "single_seq_gbm": ("seq",),
        "part_gbm": ("part",),
    },
    "pair_metrics": {
        "pair_lit_dep_gbm": ("lit", "dep"),
        "pair_lit_part_gbm": ("lit", "part"),
        "pair_lit_change_gbm": ("lit", "change"),
        "pair_lit_seq_gbm": ("lit", "seq"),
        "pair_dep_part_gbm": ("dep", "part"),
        "pair_dep_change_gbm": ("dep", "change"),
        "pair_dep_seq_gbm": ("dep", "seq"),
        "pair_part_change_gbm": ("part", "change"),
        "pair_part_seq_gbm": ("part", "seq"),
        "pair_change_seq_gbm": ("change", "seq"),
    },
    "triplet_metrics": {
        "triplet_lit_dep_part_gbm": ("lit", "dep", "part"),
        "triplet_lit_dep_seq_gbm": ("lit", "dep", "seq"),
        "triplet_lit_dep_change_gbm": ("lit", "dep", "change"),
        "triplet_lit_part_seq_gbm": ("lit", "part", "seq"),
        "triplet_lit_change_seq_gbm": ("lit", "change", "seq"),
        "triplet_dep_change_seq_gbm": ("dep", "change", "seq"),
        "triplet_dep_part_seq_gbm": ("dep", "part", "seq"),
        "triplet_part_change_seq_gbm": ("part", "change", "seq"),
    },
    "quad_metrics": {
        "quad_lit_dep_change_seq_gbm": ("lit", "dep", "change", "seq"),
        "quad_lit_dep_part_seq_gbm": ("lit", "dep", "part", "seq"),
        "quad_lit_dep_part_change_gbm": ("lit", "dep", "part", "change"),
        "quad_dep_part_change_seq_gbm": ("dep", "part", "change", "seq"),
    },
}

# Flat dictionary of all newly defined combinations
ALL_COMBINATIONS: Dict[str, Tuple[str, ...]] = {
    name: groups for tier in METRIC_COMBINATIONS_BY_TIER.values() for name, groups in tier.items()
}


class MultiMetricGBM(GBMOn):
    """Gradient boosting on a specified combination of metric groups."""

    def __init__(self, groups: Sequence[str], name: str = "multi_metric_gbm"):
        self.groups = tuple(groups)
        self.name = name
        super().__init__(group_features(self.groups))


# Register new designs that are not already in REGISTRY
for _name, _groups in ALL_COMBINATIONS.items():
    if _name not in REGISTRY:
        register(_name)(lambda g=_groups, n=_name: MultiMetricGBM(g, name=n))


class MultiDetectorSuite:
    """Simultaneously initializes and coordinates multiple collusion detectors.

    Allows instantiating all combinations at once, fitting them simultaneously
    on the same training data, and scoring on identical test markets.
    """

    def __init__(
        self,
        combinations: Mapping[str, Sequence[str]] | None = None,
        tiers: Sequence[str] | None = None,
    ):
        if combinations is not None:
            self.combinations = {k: tuple(v) for k, v in combinations.items()}
        elif tiers is not None:
            self.combinations = {}
            for t in tiers:
                if t in METRIC_COMBINATIONS_BY_TIER:
                    self.combinations.update(METRIC_COMBINATIONS_BY_TIER[t])
                else:
                    raise KeyError(
                        f"unknown tier {t!r}; known: {list(METRIC_COMBINATIONS_BY_TIER)}"
                    )
        else:
            self.combinations = dict(ALL_COMBINATIONS)

        # Simultaneously initialize all detectors
        self.detectors: Dict[str, MultiMetricGBM] = {
            name: MultiMetricGBM(groups, name=name) for name, groups in self.combinations.items()
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultiDetectorSuite":
        """Simultaneously fit all initialized detectors on the same training data."""
        for d in self.detectors.values():
            d.fit(X, y)
        return self

    def score(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Simultaneously score evaluation data across all initialized detectors."""
        return {name: d.score(X) for name, d in self.detectors.items()}

    def describe(self) -> List[Dict[str, object]]:
        """Return metadata for all initialized detectors in the suite."""
        out = []
        for name, d in self.detectors.items():
            tier = "unknown"
            for t_name, t_dict in METRIC_COMBINATIONS_BY_TIER.items():
                if name in t_dict:
                    tier = t_name
                    break
            out.append(
                {
                    "name": name,
                    "tier": tier,
                    "num_metric_groups": len(d.groups),
                    "metric_groups": d.groups,
                    "num_features": len(d.features),
                }
            )
        return out
