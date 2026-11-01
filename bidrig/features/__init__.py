"""Market-level features computed from public data only (bids, bidder ids, estimates).

One module per group: ``literature`` (lit), ``dependence`` (dep), ``participation`` (part),
``change`` (change) and ``sequential`` (seq). ``groups`` fixes the column order and
``extract`` computes every feature of a market. The names below are the package's API.
"""

from bidrig.features.change import (
    CHANGE_FEATURES,
    SERIES,
    bridge_scan,
    change_features,
    tender_series,
)
from bidrig.features.common import (
    CLOSE_TOL,
    EPS,
    MIN_BIDS,
    MIN_CO,
    MIN_WINS,
    N_PERM,
    TIE_TOL,
    residuals,
)
from bidrig.features.dependence import (
    DEP_FEATURES,
    DEP_STATS,
    RAW_STATS,
    cobid_features,
    dependence_features,
    identity_stats,
    pair_correlation_z,
    permute_within_firms,
    permute_within_tenders,
)
from bidrig.features.extract import market_features
from bidrig.features.groups import ALL_FEATURES, FEATURE_GROUPS, group_features
from bidrig.features.literature import (
    LIT_FEATURES,
    LIT_SCREENS,
    literature_features,
    tender_screens,
)
from bidrig.features.participation import PART_FEATURES, exclusive_group, participation_features
from bidrig.features.sequential import (
    SEQ_FEATURES,
    cusum_max,
    hmm_filter,
    sequential_features,
    suspicion_series,
)

__all__ = [
    "ALL_FEATURES",
    "CHANGE_FEATURES",
    "CLOSE_TOL",
    "DEP_FEATURES",
    "DEP_STATS",
    "EPS",
    "FEATURE_GROUPS",
    "LIT_FEATURES",
    "LIT_SCREENS",
    "MIN_BIDS",
    "MIN_CO",
    "MIN_WINS",
    "N_PERM",
    "PART_FEATURES",
    "RAW_STATS",
    "SEQ_FEATURES",
    "SERIES",
    "TIE_TOL",
    "bridge_scan",
    "change_features",
    "cobid_features",
    "cusum_max",
    "dependence_features",
    "exclusive_group",
    "group_features",
    "hmm_filter",
    "identity_stats",
    "literature_features",
    "market_features",
    "pair_correlation_z",
    "participation_features",
    "permute_within_firms",
    "permute_within_tenders",
    "residuals",
    "sequential_features",
    "suspicion_series",
    "tender_screens",
    "tender_series",
]
