"""Market simulator: repeated first-price procurement tenders with persistent bidders.

A market is a fixed set of firms that meet in ``T`` sequential tenders. Each firm has a
persistent cost efficiency, type-specific specialisation offsets and a capacity backlog
that raises its cost after recent wins (an honest source of rotation-like patterns).
The buyer publishes a noisy engineering estimate per tender, built from the project size
only, never from realised costs. Honest firms bid cost times a uniform market markup
(0-25 % by default) plus small firm-specific markups and competition shading.

All randomness comes from the ``rng`` argument; nothing uses the global ``random`` module.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

Intervals = tuple[tuple[float, float], ...]

MIN_BIDDERS = 3


def draw_from(intervals: Intervals, rng: np.random.Generator, integer: bool = False) -> float:
    """Draw uniformly from a union of closed intervals, weighting intervals by length.

    Integer draws treat ``(a, b)`` as the inclusive set ``{a, ..., b}``.
    """
    if integer:
        widths = np.array([b - a + 1 for a, b in intervals], dtype=float)
    else:
        widths = np.array([b - a for a, b in intervals], dtype=float)
        if widths.sum() == 0:
            widths = np.ones(len(intervals))
    idx = int(rng.choice(len(intervals), p=widths / widths.sum()))
    a, b = intervals[idx]
    if integer:
        return int(rng.integers(int(a), int(b) + 1))
    return float(rng.uniform(a, b))


@dataclass(frozen=True)
class MarketSpec:
    """Ranges (unions of intervals) from which a market's parameters are drawn."""

    n_firms: Intervals = ((10, 22),)
    n_tenders: Intervals = ((20, 60),)
    cost_spread: Intervals = ((0.06, 0.18),)  # sd of log persistent efficiency
    idio_sd: Intervals = ((0.03, 0.10),)  # sd of per-tender log cost noise
    estimate_noise: Intervals = ((0.05, 0.20),)  # sd of log engineering-estimate error
    markup: Intervals = ((0.0, 0.25),)  # uniform honest markup
    capacity: Intervals = ((0.0, 0.15),)  # cost increase per unit of backlog
    participation: Intervals = ((0.4, 0.9),)
    specialization: Intervals = ((0.0, 0.15),)  # sd of firm x tender-type cost offsets

    def with_(self, **kwargs) -> "MarketSpec":
        return replace(self, **kwargs)


@dataclass(frozen=True)
class MarketConfig:
    n_firms: int
    n_tenders: int
    cost_spread: float
    idio_sd: float
    estimate_noise: float
    markup: float
    capacity: float
    participation: float
    specialization: float
    n_types: int
    type_participation_bias: float
    backlog_decay: float
    shading: float
    bid_noise: float


def sample_config(spec: MarketSpec, rng: np.random.Generator) -> MarketConfig:
    return MarketConfig(
        n_firms=int(draw_from(spec.n_firms, rng, integer=True)),
        n_tenders=int(draw_from(spec.n_tenders, rng, integer=True)),
        cost_spread=draw_from(spec.cost_spread, rng),
        idio_sd=draw_from(spec.idio_sd, rng),
        estimate_noise=draw_from(spec.estimate_noise, rng),
        markup=draw_from(spec.markup, rng),
        capacity=draw_from(spec.capacity, rng),
        participation=draw_from(spec.participation, rng),
        specialization=draw_from(spec.specialization, rng),
        n_types=int(rng.integers(1, 4)),
        type_participation_bias=float(rng.uniform(0.0, 1.0)),
        backlog_decay=float(rng.uniform(0.5, 0.9)),
        shading=float(rng.uniform(0.0, 0.10)),
        bid_noise=float(rng.uniform(0.0, 0.02)),
    )


@dataclass
class Market:
    """One simulated market. ``bids`` is (T, N) with NaN where a firm did not bid."""

    bids: np.ndarray
    estimate: np.ndarray
    label: int
    mechanism: str
    members: tuple[int, ...]
    active: np.ndarray  # (T,) bool: cartel behaviour applied in this tender
    config: MarketConfig
    params: dict = field(default_factory=dict)


@dataclass
class TenderContext:
    """What a collusion mechanism may see and change in one tender."""

    t: int
    n_tenders: int
    estimate: float
    cost: np.ndarray  # (N,) realised costs (private)
    honest: np.ndarray  # (N,) honest bids
    part: np.ndarray  # (N,) bool participation
    bids: np.ndarray  # (N,) current bids, NaN for non-participants
    rng: np.random.Generator


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def simulate_market(
    cfg: MarketConfig, mechanism, rng: np.random.Generator, params: dict | None = None
) -> Market:
    """Simulate one market. ``mechanism`` is a ``collusion.Mechanism`` (``HONEST`` for none)."""
    n, n_t = cfg.n_firms, cfg.n_tenders
    eff = np.exp(rng.normal(0.0, cfg.cost_spread, n))
    offsets = (
        rng.normal(0.0, cfg.specialization, (n, cfg.n_types))
        if cfg.specialization > 0
        else (np.zeros((n, cfg.n_types)))
    )
    p_mean = np.clip(cfg.participation + rng.uniform(-0.15, 0.15, n), 0.05, 0.98)
    firm_markup = cfg.markup + rng.uniform(0.0, 0.03, n)
    size = 100.0 * np.exp(rng.normal(0.0, 0.5, n_t))
    types = rng.integers(0, cfg.n_types, n_t)
    estimate = size * np.exp(rng.normal(0.0, cfg.estimate_noise, n_t))

    if params is None:
        params = mechanism.sample_params(cfg, rng)
    state = mechanism.init_state(cfg, params, rng)

    bids_all = np.full((n_t, n), np.nan)
    active = np.zeros(n_t, dtype=bool)
    backlog = np.zeros(n)
    spec_scale = cfg.specialization if cfg.specialization > 0 else 1.0
    for t in range(n_t):
        z = offsets[:, types[t]] / spec_scale
        p = _sigmoid(np.log(p_mean / (1 - p_mean)) - cfg.type_participation_bias * z)
        part = rng.random(n) < p
        if part.sum() < MIN_BIDDERS:
            outside = np.flatnonzero(~part)
            extra = rng.choice(outside, size=MIN_BIDDERS - int(part.sum()), replace=False)
            part[np.sort(extra)] = True
        cost = (
            size[t]
            * eff
            * np.exp(offsets[:, types[t]] + rng.normal(0.0, cfg.idio_sd, n))
            * (1.0 + cfg.capacity * backlog)
        )
        noise = np.exp(rng.normal(0.0, cfg.bid_noise, n))
        honest = cost * (1.0 + firm_markup + cfg.shading / part.sum()) * noise
        bids = np.where(part, honest, np.nan)
        ctx = TenderContext(t, n_t, float(estimate[t]), cost, honest, part.copy(), bids, rng)
        bids, active[t] = mechanism.apply(ctx, cfg, params, state)
        bids_all[t] = bids
        winner = int(np.nanargmin(bids))
        backlog *= cfg.backlog_decay
        backlog[winner] += 1.0

    return Market(
        bids=bids_all,
        estimate=estimate,
        label=int(mechanism.label),
        mechanism=mechanism.name,
        members=tuple(int(m) for m in params.get("members", ())),
        active=active,
        config=cfg,
        params=params,
    )
