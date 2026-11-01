"""Detector designs. Importing this package registers every design by name.

One module per detector family, all depending only on ``base`` (interface and registry),
``config`` / ``priors`` (shared settings) and ``bidrig.features``; no module imports another:

  gbm_groups     ``full_gbm`` and its ablations / baselines (gradient boosting on feature groups)
  single_screen  one literature screen on its own
  tail_fused     knowledge-directed tail score, and its fusion with the GBM
  previous       variant: ``full_gbm`` without the within-firm null columns
  stat_auc       variant: validation-tuned logistic regression
  rl_screens     variant: screen-subset selection by a bandit

Use ``make(name)`` to build a detector; every detector has ``fit(X, y)`` and ``score(X)``.
"""

from bidrig.detectors import (  # noqa: F401  (importing a module registers its designs)
    gbm_groups,
    previous,
    rl_screens,
    single_screen,
    stat_auc,
    tail_fused,
)
from bidrig.detectors.base import REGISTRY, Detector, GBMOn, make, register
from bidrig.detectors.single_screen import SINGLE_SCREENS

BASELINES = ("lit_gbm",) + tuple(f"single:{f}" for f in SINGLE_SCREENS)

__all__ = ["BASELINES", "REGISTRY", "Detector", "GBMOn", "make", "register"]
