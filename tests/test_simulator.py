import numpy as np

from bidrig.collusion import HONEST, IN_SCOPE, MECHANISMS
from bidrig.simulator import MIN_BIDDERS, MarketSpec, draw_from, sample_config, simulate_market


def make(mech="honest", seed=0, **spec):
    rng = np.random.default_rng(seed)
    cfg = sample_config(MarketSpec().with_(**spec), rng)
    return simulate_market(cfg, MECHANISMS[mech], rng)


def test_draw_from_respects_union_of_intervals():
    rng = np.random.default_rng(0)
    v = [draw_from(((6, 7), (25, 30)), rng, integer=True) for _ in range(500)]
    assert set(v) <= {6, 7, 25, 26, 27, 28, 29, 30} and {6, 7, 30} <= set(v)
    f = [draw_from(((0.02, 0.04), (0.22, 0.30)), rng) for _ in range(500)]
    assert all(0.02 <= x <= 0.04 or 0.22 <= x <= 0.30 for x in f)


def test_honest_market_shapes_and_min_bidders():
    for seed in range(20):
        m = make(seed=seed)
        t, n = m.bids.shape
        assert t == m.config.n_tenders and n == m.config.n_firms
        assert 20 <= t <= 60 and 10 <= n <= 22
        assert ((~np.isnan(m.bids)).sum(axis=1) >= MIN_BIDDERS).all()
        assert m.label == 0 and m.members == () and not m.active.any()
        assert (m.bids[~np.isnan(m.bids)] > 0).all()


def test_estimate_independent_of_mechanism():
    """The public estimate depends only on project size and noise, not on costs or bids."""
    a, b = make("honest", seed=3), make("price_fixing", seed=3)
    assert np.array_equal(a.estimate, b.estimate)


def test_same_seed_identical():
    for mech in MECHANISMS:
        a, b = make(mech, seed=5), make(mech, seed=5)
        assert np.array_equal(a.bids, b.bids, equal_nan=True)


def test_markup_range_scales_honest_bids():
    lo = make(seed=1, markup=((0.0, 0.0),))
    hi = make(seed=1, markup=((0.25, 0.25),))
    # identical draws except the uniform markup -> bids scale by a near-constant factor
    ratio = hi.bids / lo.bids
    ratio = ratio[np.isfinite(ratio)]
    assert np.allclose(ratio[0], 1.25, rtol=0.05)


def test_all_in_scope_mechanisms_act_and_label():
    assert len(IN_SCOPE) >= 8
    for mech in IN_SCOPE:
        acted = [make(mech, seed=s).active.any() for s in range(10)]
        m = make(mech, seed=0)
        assert m.label == 1 and len(m.members) >= 2 and any(acted), mech


def _member_bids(m):
    mem = np.array(m.members)
    return m.bids[:, mem]


def test_cover_bids_sit_above_designated_winner():
    m = make("cover_bidding", seed=2)
    mb = _member_bids(m)
    for t in np.flatnonzero(m.active):
        row = mb[t][~np.isnan(mb[t])]
        assert (np.sort(row)[1:] >= np.sort(row)[0] * 1.0009).all()


def test_rotation_cycles_through_members():
    m = make("rotation", seed=4, participation=((0.98, 0.98),))
    mem = np.array(m.members)
    winners = []
    for t in np.flatnonzero(m.active):
        row = m.bids[t, mem]
        winners.append(int(mem[np.nanargmin(row)]))
    order = list(m.params["order"])
    # with near-full participation the designated winners follow the fixed order
    expected = [order[i % len(order)] for i in range(len(winners))]
    agree = np.mean(np.array(winners) == np.array(expected))
    assert agree > 0.6


def test_price_fixing_bids_near_level():
    m = make("price_fixing", seed=6)
    mb = _member_bids(m)
    rel = mb / m.estimate[:, None]
    rel = rel[m.active][~np.isnan(mb[m.active])]
    assert np.median(np.abs(np.log(rel / m.params["level"]))) < 0.05


def test_bid_suppression_reduces_member_bids():
    a, b = make("honest", seed=7), make("bid_suppression", seed=7)
    assert (~np.isnan(b.bids)).sum() < (~np.isnan(a.bids)).sum()


def test_intermittent_window():
    m = make("intermittent", seed=8)
    active = np.flatnonzero(m.active)
    assert active.min() >= m.params["start"] and active.max() < m.params["stop"]


def test_partial_fringe_small_cartel():
    for s in range(10):
        assert 2 <= len(make("partial_fringe", seed=s).members) <= 3


def test_uniform_markup_control_is_out_of_scope():
    assert not MECHANISMS["uniform_markup"].in_scope
    assert "uniform_markup" not in IN_SCOPE
    m = make("uniform_markup", seed=9)
    h = make("honest", seed=9)
    assert np.nanmedian(m.bids / h.bids) > 1.2
    assert HONEST.label == 0
