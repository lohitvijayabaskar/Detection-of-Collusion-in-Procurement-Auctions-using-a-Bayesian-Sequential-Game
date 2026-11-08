"""Compatibility layer: the detector designs now live in ``bidrig.detectors``.

This module holds no logic. It keeps the old import paths (``from bidrig.models import
make_design, DESIGNS, ...``) working for the frozen protocol code, which resolves designs by
name.
"""

from bidrig.detectors import (
    ALL_COMBINATIONS,
    BASELINES,
    METRIC_COMBINATIONS_BY_TIER,
    MultiDetectorSuite,
    MultiMetricGBM,
)
from bidrig.firm_identification import FirmCollusionIdentifier, FirmProfile
from bidrig.sequential_game import (
    ALL_SEQ_GAME_COMBINATIONS,
    BayesianSequentialFilter,
    OnlineGradientLearner,
    SEQ_GAME_COMBINATIONS_BY_TIER,
    SEQ_GAME_PILLARS,
    TemporalVolatilityTracker,
    extract_sequential_game_features,
)
from bidrig.detectors import make as make_design
from bidrig.detectors.config import GBM_PARAMS
from bidrig.detectors.gbm_groups import DESIGNS, GBM, MONOTONE
from bidrig.detectors.priors import TAIL_DIRECTIONS, TAIL_MARGIN
from bidrig.detectors.single_screen import SINGLE_SCREENS, SingleScreen
from bidrig.detectors.tail_fused import FUSED, Fused, TailScore

__all__ = [
    "ALL_COMBINATIONS",
    "ALL_SEQ_GAME_COMBINATIONS",
    "BASELINES",
    "BayesianSequentialFilter",
    "DESIGNS",
    "FUSED",
    "FirmCollusionIdentifier",
    "FirmProfile",
    "GBM",
    "GBM_PARAMS",
    "METRIC_COMBINATIONS_BY_TIER",
    "MONOTONE",
    "MultiDetectorSuite",
    "MultiMetricGBM",
    "OnlineGradientLearner",
    "SEQ_GAME_COMBINATIONS_BY_TIER",
    "SEQ_GAME_PILLARS",
    "SINGLE_SCREENS",
    "TAIL_DIRECTIONS",
    "TAIL_MARGIN",
    "TemporalVolatilityTracker",
    "Fused",
    "SingleScreen",
    "TailScore",
    "extract_sequential_game_features",
    "make_design",
]
