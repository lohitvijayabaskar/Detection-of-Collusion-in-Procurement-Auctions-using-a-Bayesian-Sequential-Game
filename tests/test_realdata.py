"""Loader tests on synthetic frames shaped like the raw CSVs (no real data needed)."""

import numpy as np
import pandas as pd
import pytest

from bidrig.realdata import PTE, REAL_DATA_DIR, load, make_windows, standardise


def raw_frame():
    rows = []
    # tenders listed out of time order; tender 3 has a duplicate firm row
    for tender, date, label in ((3, 300.0, 1), (1, 100.0, 0), (2, 200.0, 1), (4, 400.0, 0)):
        for firm, bid in ((10, 100.0), (11, 110.0), (12, 105.0)):
            rows.append(
                (tender, date, firm, bid + tender, 120.0, int(label and firm != 12), label, 0)
            )
    rows.append((3, 300.0, 10, 50.0, 120.0, 1, 1, 1))  # duplicate (3, 10): lower bid kept
    cols = [
        "Tender",
        "Date",
        "Competitors",
        "Bid_value",
        PTE,
        "Collusive_competitor_original",
        "Collusive_competitor",
        "Winner",
    ]
    return pd.DataFrame(rows, columns=cols)


def test_standardise_orders_by_date_and_dedups():
    df = standardise(raw_frame(), "japan")
    assert list(df.groupby("order").tender.first()) == [1, 2, 3, 4]
    assert len(df) == 12
    assert df[(df.tender == 3) & (df.firm == 10)].bid.item() == 50.0
    assert set(df.columns) >= {
        "tender",
        "order",
        "firm",
        "bid",
        "estimate",
        "member",
        "tender_label",
    }


def test_italy_price_scale():
    raw = raw_frame().drop(columns="Date")
    df = standardise(raw, "italy")
    assert df.bid.max() == pytest.approx((110.0 + 4) * 0.01)
    assert list(df.groupby("order").tender.first()) == [1, 2, 3, 4]  # id order


def test_make_windows_shapes_and_labels():
    df = standardise(raw_frame(), "japan")
    ws = make_windows(df, "japan", window=2)
    assert len(ws) == 2
    w0 = ws[0]
    assert w0.bids.shape == (2, 3) and list(w0.firms) == [10, 11, 12]
    assert list(w0.tenders) == [1, 2]
    assert w0.collusive_share == 0.5
    assert np.allclose(w0.bids[0], [101.0, 111.0, 106.0])
    assert len(make_windows(df, "japan", window=3)) == 1  # trailing partial dropped
    assert len(make_windows(df, "japan", window=2, stride=1)) == 3


@pytest.mark.skipif(not REAL_DATA_DIR.exists(), reason="real data not present")
def test_real_files_load():
    for name in ("japan", "italy", "brazil"):
        df = load(name)
        assert df.tender.nunique() > 50 and (df.bid > 0).all()
        assert (df.groupby("tender").tender_label.nunique() == 1).all()
