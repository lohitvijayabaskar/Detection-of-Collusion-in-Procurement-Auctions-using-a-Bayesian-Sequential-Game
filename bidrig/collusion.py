"""Library of imposed collusion mechanisms.

Each mechanism is parameterised and randomised per market (``sample_params``) and edits the
bids of cartel members tender by tender (``apply``). Cartels are imposed: they need not be
profitable. ``in_scope=False`` marks negative controls that are collusive but not
identifiable from bids alone (uniform markup collusion).

Mechanisms never look at future tenders. A tender with fewer than two participating
members is left honest (``active=False``), except where noted.
"""

from __future__ import annotations

import numpy as np

from bidrig.simulator import MarketConfig, TenderContext


class Mechanism:
    name = "honest"
    label = 0
    in_scope = True
    share_range = (0.2, 1.0)  # cartel size as a share of the market

    def sample_members(self, cfg: MarketConfig, rng: np.random.Generator) -> np.ndarray:
        share = rng.uniform(*self.share_range)
        k = int(np.clip(round(share * cfg.n_firms), 2, cfg.n_firms))
        return np.sort(rng.choice(cfg.n_firms, size=k, replace=False))

    def sample_params(self, cfg: MarketConfig, rng: np.random.Generator) -> dict:
        return {}

    def init_state(self, cfg: MarketConfig, params: dict, rng: np.random.Generator) -> dict:
        return {}

    def apply(self, ctx: TenderContext, cfg, params: dict, state: dict):
        return ctx.bids, False


HONEST = Mechanism()


def _cover_params(rng: np.random.Generator) -> dict:
    infl_lo = rng.uniform(0.0, 0.15)
    gap_lo = rng.uniform(0.005, 0.05)
    return {
        "infl_lo": infl_lo,
        "infl_hi": infl_lo + rng.uniform(0.02, 0.25),
        "gap_lo": gap_lo,
        "gap_hi": gap_lo + rng.uniform(0.01, 0.15),
    }


def _participating(ctx: TenderContext, members: np.ndarray) -> np.ndarray:
    return members[ctx.part[members]]


def _lowest_cost(ctx: TenderContext, present: np.ndarray) -> int:
    return int(present[np.argmin(ctx.cost[present])])


def _rig(ctx: TenderContext, params: dict, present: np.ndarray, winner: int, cover_noise=0.0):
    """Designated winner inflates its honest bid; other present members cover above it."""
    rng = ctx.rng
    bids = ctx.bids.copy()
    bids[winner] = ctx.honest[winner] * (1.0 + rng.uniform(params["infl_lo"], params["infl_hi"]))
    for j in present:
        if j == winner:
            continue
        gap = rng.uniform(params["gap_lo"], params["gap_hi"])
        bids[j] = bids[winner] * (1.0 + gap) * np.exp(rng.normal(0.0, cover_noise))
        bids[j] = max(bids[j], bids[winner] * 1.001)
    return bids


class CoverBidding(Mechanism):
    """Lowest-cost present member is the designated winner; the rest submit cover bids."""

    name = "cover_bidding"
    label = 1

    def sample_params(self, cfg, rng):
        return {"members": self.sample_members(cfg, rng), **_cover_params(rng)}

    def apply(self, ctx, cfg, params, state):
        present = _participating(ctx, params["members"])
        if len(present) < 2:
            return ctx.bids, False
        return _rig(ctx, params, present, _lowest_cost(ctx, present)), True


class Rotation(Mechanism):
    """Designated winner rotates through members in a fixed order, regardless of cost."""

    name = "rotation"
    label = 1

    def sample_params(self, cfg, rng):
        members = self.sample_members(cfg, rng)
        return {"members": members, "order": rng.permutation(members), **_cover_params(rng)}

    def init_state(self, cfg, params, rng):
        return {"pointer": 0}

    def apply(self, ctx, cfg, params, state):
        present = _participating(ctx, params["members"])
        if len(present) < 2:
            return ctx.bids, False
        order = params["order"]
        for step in range(len(order)):
            cand = int(order[(state["pointer"] + step) % len(order)])
            if ctx.part[cand]:
                state["pointer"] = (state["pointer"] + step + 1) % len(order)
                return _rig(ctx, params, present, cand), True
        return ctx.bids, False


class Complementary(Mechanism):
    """Losing members bid a fixed firm-specific multiple of the public estimate."""

    name = "complementary"
    label = 1

    def sample_params(self, cfg, rng):
        members = self.sample_members(cfg, rng)
        return {
            "members": members,
            "multiple": rng.uniform(1.05, 1.40, len(members)),
            "noise": rng.uniform(0.005, 0.03),
            **_cover_params(rng),
        }

    def apply(self, ctx, cfg, params, state):
        members = params["members"]
        present = _participating(ctx, members)
        if len(present) < 2:
            return ctx.bids, False
        rng = ctx.rng
        winner = _lowest_cost(ctx, present)
        bids = ctx.bids.copy()
        bids[winner] = ctx.honest[winner] * (
            1.0 + rng.uniform(params["infl_lo"], params["infl_hi"])
        )
        for j in present:
            if j == winner:
                continue
            mult = params["multiple"][int(np.searchsorted(members, j))]
            bid = ctx.estimate * mult * np.exp(rng.normal(0.0, params["noise"]))
            bids[j] = max(bid, bids[winner] * 1.005)
        return bids, True


PRICE_LEVELS = {"low": (0.95, 1.10), "mid": (1.10, 1.30), "high": (1.30, 1.60)}


class PriceFixing(Mechanism):
    """Members bid an agreed multiple of the estimate (never below own cost)."""

    name = "price_fixing"
    label = 1

    def sample_params(self, cfg, rng):
        level = ("low", "mid", "high")[int(rng.integers(0, 3))]
        return {
            "members": self.sample_members(cfg, rng),
            "level_name": level,
            "level": rng.uniform(*PRICE_LEVELS[level]),
            "noise": rng.uniform(0.002, 0.02),
        }

    def apply(self, ctx, cfg, params, state):
        present = _participating(ctx, params["members"])
        if len(present) < 2:
            return ctx.bids, False
        bids = ctx.bids.copy()
        target = params["level"] * ctx.estimate
        noise = np.exp(ctx.rng.normal(0.0, params["noise"], len(present)))
        bids[present] = np.maximum(ctx.cost[present], target * noise)
        return bids, True


class PartialFringe(CoverBidding):
    """A small cover-bidding cartel (2-3 firms) inside a competitive fringe."""

    name = "partial_fringe"

    def sample_members(self, cfg, rng):
        k = int(min(rng.integers(2, 4), cfg.n_firms))
        return np.sort(rng.choice(cfg.n_firms, size=k, replace=False))


class NoisyDefecting(Mechanism):
    """Cover bidding with noisy covers; each member defects (undercuts) with probability q."""

    name = "noisy_defecting"
    label = 1

    def sample_params(self, cfg, rng):
        return {
            "members": self.sample_members(cfg, rng),
            "defect_prob": rng.uniform(0.1, 0.4),
            "cover_noise": rng.uniform(0.01, 0.06),
            **_cover_params(rng),
        }

    def apply(self, ctx, cfg, params, state):
        present = _participating(ctx, params["members"])
        if len(present) < 2:
            return ctx.bids, False
        bids = _rig(ctx, params, present, _lowest_cost(ctx, present), params["cover_noise"])
        defect = ctx.rng.random(len(present)) < params["defect_prob"]
        for j in present[defect]:
            bids[j] = ctx.honest[j] * (1.0 - ctx.rng.uniform(0.0, 0.03))
        return bids, True


class Intermittent(CoverBidding):
    """Cover bidding that is active only in one contiguous window of tenders."""

    name = "intermittent"

    def sample_params(self, cfg, rng):
        params = super().sample_params(cfg, rng)
        length = max(3, int(round(rng.uniform(0.3, 0.6) * cfg.n_tenders)))
        start = int(rng.integers(0, cfg.n_tenders - length + 1))
        params.update({"start": start, "stop": start + length})
        return params

    def apply(self, ctx, cfg, params, state):
        if not (params["start"] <= ctx.t < params["stop"]):
            return ctx.bids, False
        return super().apply(ctx, cfg, params, state)


class BidSuppression(Mechanism):
    """Non-designated members withdraw with probability s, otherwise they cover."""

    name = "bid_suppression"
    label = 1

    def sample_params(self, cfg, rng):
        return {
            "members": self.sample_members(cfg, rng),
            "withdraw_prob": rng.uniform(0.6, 1.0),
            **_cover_params(rng),
        }

    def apply(self, ctx, cfg, params, state):
        present = _participating(ctx, params["members"])
        if len(present) < 2:
            return ctx.bids, False
        winner = _lowest_cost(ctx, present)
        bids = _rig(ctx, params, present, winner)
        withdraw = ctx.rng.random(len(present)) < params["withdraw_prob"]
        for j, w in zip(present, withdraw):
            if w and j != winner:
                bids[j] = np.nan
        return bids, True


class UniformMarkup(Mechanism):
    """Negative control: every firm bids cost with a common 30-60 % markup.

    Without cost data this is indistinguishable from honest firms with higher costs, so
    it is out of scope and reported separately.
    """

    name = "uniform_markup"
    label = 1
    in_scope = False

    def sample_params(self, cfg, rng):
        return {"members": np.arange(cfg.n_firms), "markup": rng.uniform(0.30, 0.60)}

    def apply(self, ctx, cfg, params, state):
        return ctx.bids * (1.0 + params["markup"]) / (1.0 + cfg.markup), True


MECHANISMS: dict[str, Mechanism] = {
    m.name: m
    for m in (
        HONEST,
        Rotation(),
        CoverBidding(),
        Complementary(),
        PriceFixing(),
        PartialFringe(),
        NoisyDefecting(),
        Intermittent(),
        BidSuppression(),
        UniformMarkup(),
    )
}

IN_SCOPE = tuple(sorted(n for n, m in MECHANISMS.items() if m.label == 1 and m.in_scope))
