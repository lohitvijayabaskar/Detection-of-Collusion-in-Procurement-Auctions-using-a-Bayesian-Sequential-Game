"""Tests for firm-specific identification engine."""

import numpy as np
import pandas as pd
import pytest

from bidrig.firm_identification import FirmCollusionIdentifier, FirmProfile


def test_firm_collusion_identifier_mock_data():
    # Construct a synthetic procurement auction with 2 cartel firms and 3 honest firms
    # Cartel firms (101, 102) always bid together, bid close, and appear in flagged tenders
    records = []
    # 20 tenders
    for t in range(20):
        if t < 10:
            # Collusive tender: firms 101, 102, 103
            records.append({"tender": t, "firm": 101, "bid": 100.0, "member": 1, "winner": 1})
            records.append({"tender": t, "firm": 102, "bid": 101.5, "member": 1, "winner": 0})  # 1.5% close cover
            records.append({"tender": t, "firm": 103, "bid": 125.0, "member": 0, "winner": 0})
        else:
            # Competitive tender: firms 103, 104, 105
            records.append({"tender": t, "firm": 103, "bid": 95.0, "member": 0, "winner": 1})
            records.append({"tender": t, "firm": 104, "bid": 110.0, "member": 0, "winner": 0})
            records.append({"tender": t, "firm": 105, "bid": 120.0, "member": 0, "winner": 0})

    df = pd.DataFrame(records)
    # Tenders 0-9 have high suspicion, 10-19 have low suspicion
    suspicion = np.array([0.9] * 10 + [0.2] * 10)

    identifier = FirmCollusionIdentifier(min_tenders_firm=2)
    res = identifier.fit_predict(df, tender_suspicion_scores=suspicion)

    assert "profiles" in res
    assert "metrics" in res
    profiles = res["profiles"]
    assert len(profiles) == 5

    # Check that cartel firms (101, 102) have higher suspicion than competitive firms (104, 105)
    p_map = {p.firm_id: p for p in profiles}
    assert p_map[101].composite_suspicion > p_map[104].composite_suspicion
    assert p_map[102].composite_suspicion > p_map[105].composite_suspicion
    assert p_map[102].cover_bid_score > 0.0

    # Firm-level AUC should be 1.0 on this separated synthetic case
    m = res["metrics"]
    assert m["firm_auc"] == 1.0
    assert m.get("precision_at_10", 1.0) == 1.0
