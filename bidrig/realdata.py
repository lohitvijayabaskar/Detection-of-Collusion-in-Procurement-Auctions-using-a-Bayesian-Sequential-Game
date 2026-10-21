"""Loader for the García Rodríguez et al. (2022) bid-level datasets and market windowing.

The data is not part of the repository. Download and checksum are recorded in
docs/experiment_log.md (D1); point ``REAL_DATA_DIR`` (or the ``data_dir`` argument) at the
extracted CSV folder.

Only Japan, Italy and Brazil are supported: they have persistent firm IDs and a pre-tender
estimate, which the dependence features need. Facts checked on the files (D1, D2):
  Japan   first-price with a floor (lowest bid wins 84 % of tenders); time order = Date
          (tender id order agrees exactly with date order).
  Italy   average-bid auction (the winner is never the lowest bid); ``Bid_value`` is
          100 x price, price = PTE x (1 - rebate/100); no dates, so tender id order is used.
  Brazil  first-price (lowest bid always wins); dates disagree with id order and two are
          invalid sentinels, so tender id order is used.

Labels: ``member`` is the per-bid cartel-membership flag (``Collusive_competitor_original``);
``tender_label`` is the paper's thresholded tender label (``Collusive_competitor``), which
is constant within each tender.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REAL_DATA_DIR = Path(
    os.environ.get("BIDRIG_REAL_DATA", r"C:/Users/Lohit Vijayabaskar/real_data/rodriguez2022/data")
)

PTE = "Pre-Tender Estimate (PTE)"
DATASETS = {
    "japan": {
        "file": "DB_Collusion_Japan_processed.csv",
        "order": "Date",
        "price_scale": 1.0,
        "format": "first_price_floor",
    },
    "italy": {
        "file": "DB_Collusion_Italy_processed.csv",
        "order": "Tender",
        "price_scale": 0.01,
        "format": "average_bid",
    },
    "brazil": {
        "file": "DB_Collusion_Brazil_processed.csv",
        "order": "Tender",
        "price_scale": 1.0,
        "format": "first_price",
    },
}


def standardise(raw, name: str):
    """Map one raw dataset to columns: tender, order, firm, bid, estimate, member,
    tender_label, winner. Duplicate (tender, firm) rows keep the lowest bid."""
    import pandas as pd

    spec = DATASETS[name]
    df = pd.DataFrame(
        {
            "tender": raw["Tender"].astype(int),
            "order_key": raw[spec["order"]].astype(float),
            "firm": raw["Competitors"].astype(int),
            "bid": raw["Bid_value"].astype(float) * spec["price_scale"],
            "estimate": raw[PTE].astype(float),
            "member": raw["Collusive_competitor_original"].astype(int),
            "tender_label": raw["Collusive_competitor"].astype(int),
            "winner": raw["Winner"].astype(int),
        }
    )
    df = df[(df.bid > 0) & (df.estimate > 0)]
    df = df.sort_values(["tender", "firm", "bid"], kind="stable")
    df = df.drop_duplicates(["tender", "firm"], keep="first")
    first = df.groupby("tender").order_key.first().reset_index()
    first = first.sort_values(["order_key", "tender"], kind="stable")
    first["order"] = np.arange(len(first))
    df = df.merge(first[["tender", "order"]], on="tender").drop(columns="order_key")
    return df.sort_values(["order", "firm"], kind="stable").reset_index(drop=True)


def load(name: str, data_dir: Path | None = None):
    import pandas as pd

    path = Path(data_dir or REAL_DATA_DIR) / DATASETS[name]["file"]
    return standardise(pd.read_csv(path, sep=None, engine="python"), name)


@dataclass
class WindowMarket:
    dataset: str
    index: int
    bids: np.ndarray  # (T, N) NaN where a firm did not bid
    estimate: np.ndarray  # (T,)
    firms: np.ndarray  # (N,) firm ids, sorted
    tenders: np.ndarray  # (T,) tender ids in time order
    collusive_share: float  # share of tenders with tender_label = 1
    member_bid_share: float  # share of bids from cartel members
    meta: dict = field(default_factory=dict)


def make_windows(df, dataset: str, window: int, stride: int | None = None):
    """Cut the time-ordered tender stream into windows of ``window`` consecutive tenders.

    ``stride`` defaults to ``window`` (non-overlapping). A trailing partial window is dropped.
    Firm columns are every firm that bids at least once in the window, in sorted id order.
    """
    stride = stride or window
    orders = np.sort(df["order"].unique())
    out = []
    for k, start in enumerate(range(0, len(orders) - window + 1, stride)):
        keep = orders[start : start + window]
        w = df[df["order"].isin(keep)]
        firms = np.sort(w["firm"].unique())
        col = {f: j for j, f in enumerate(firms)}
        row = {o: i for i, o in enumerate(keep)}
        bids = np.full((window, len(firms)), np.nan)
        bids[w["order"].map(row).to_numpy(), w["firm"].map(col).to_numpy()] = w["bid"].to_numpy()
        est = w.groupby("order").estimate.first().reindex(keep).to_numpy()
        tl = w.groupby("order").tender_label.first().reindex(keep).to_numpy()
        tenders = w.groupby("order").tender.first().reindex(keep).to_numpy()
        out.append(
            WindowMarket(
                dataset=dataset,
                index=k,
                bids=bids,
                estimate=est,
                firms=firms,
                tenders=tenders,
                collusive_share=float(tl.mean()),
                member_bid_share=float(w["member"].mean()),
                meta={"format": DATASETS[dataset]["format"] if dataset in DATASETS else None},
            )
        )
    return out
