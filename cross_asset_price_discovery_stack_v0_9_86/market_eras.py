"""
market_eras.py
==============
The market-structure events inside the 2016-2026 sample span, as CODE: dated
flags every per-day exhibit can carry, and the straddle check that catches a
matched pair whose two legs sit in different microstructure regimes.

Why this exists (the clustering misconception, settled): day clustering fixes
INFERENCE -- it stops within-day dependence from inflating the effective sample
-- but it cannot remove CONFOUNDING. If volatile days cluster in late-sample
years while market structure drifts, the regime contrast partially measures
era, and no standard-error correction touches a bias. The stack's two real era
defenses are structural: per-day / day-FE estimation absorbs anything constant
within a day, and the matched-pair design (control ~364 days before its
volatile day) differences out anything drifting slower than a year. What those
defenses cannot absorb: era variation in the cross-day ESTIMANDS, and any
SHARP rule change that falls INSIDE a pair's one-year gap -- which breaks the
matching exactly where it is relied upon. Hence: flags for the drifts (era
dummies for cross-day regressions), and a straddle check for the breaks.

The events (SPY/ES-relevant; the Tick Size Pilot is deliberately absent -- it
covered small-caps only and never touched SPY):

  mes_launch        2019-05-06  CME Micro E-mini launch: small-trader flow
                                re-sorted out of ES; ES book composition shifts
  memx_miax         2020-09-21  MEMX / MIAX Pearl go live: SPY venue count grows,
                                consolidated-book composition changes
  retail_surge      2020-03-01  zero-commission retail era begins in earnest
                                (window flag through 2021-12-31): off-exchange
                                internalization share steps up; the displayed
                                SPY book becomes a thinner slice of liquidity
  t_plus_1          2024-05-28  T+1 settlement (minor; ETF create/redeem timing)
  tick_regime_2025  2025-11-03  Reg NMS amendments compliance (adopted 2024-09):
                                half-penny quoting for qualifying names -- SPY
                                almost certainly qualifies -- access-fee cap cut,
                                round-lot redefinition (~40 shares at SPY's
                                price under the MDI tiers). THE one SHARP break
                                that changes what the price grid and displayed
                                book MEAN. Date is the scheduled compliance
                                date as of this stack's knowledge -- verify
                                against the actual rollout before quoting.

Only ``tick_regime_2025`` is classified SHARP: the others are drifts or
composition shifts a matched pair absorbs; a pair straddling a sharp break is
mechanically contrasting two grid regimes on top of two volatility regimes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EVENTS = (
    ("mes_launch", "2019-05-06"),
    ("memx_miax", "2020-09-21"),
    ("t_plus_1", "2024-05-28"),
    ("tick_regime_2025", "2025-11-03"),
)
WINDOWS = (
    ("retail_surge", "2020-03-01", "2021-12-31"),
)
SHARP_BREAKS = (
    ("tick_regime_2025", "2025-11-03"),
)


def era_flags(dates) -> pd.DataFrame:
    """Per-date boolean flags: ``post_<event>`` for each dated event, plus one
    in-window flag per windowed era. Index = the input dates (normalized)."""
    idx = pd.DatetimeIndex([pd.Timestamp(d).normalize() for d in dates])
    out = {}
    for name, day in EVENTS:
        out[f"post_{name}"] = idx >= pd.Timestamp(day)
    for name, lo, hi in WINDOWS:
        out[name] = (idx >= pd.Timestamp(lo)) & (idx <= pd.Timestamp(hi))
    df = pd.DataFrame(out, index=idx)
    df.index.name = "date"
    return df


def pair_straddle(volatile, baseline) -> pd.DataFrame:
    """The straddle check: positional pairs whose two legs sit on OPPOSITE sides
    of a sharp break. Those pairs mechanically contrast two grid regimes; exclude
    them from spread/depth-denominated paired contrasts (flow- and share-based
    objects are less exposed) or carry an explicit regime dummy.
    -> DataFrame[volatile, baseline, break, level] -- one row per offense;
    empty means no pair straddles anything sharp."""
    rows = []
    for v, b in zip(list(volatile), list(baseline)):
        try:
            dv, db = pd.Timestamp(v), pd.Timestamp(b)
        except Exception:
            continue
        lo, hi = min(dv, db), max(dv, db)
        for name, day in SHARP_BREAKS:
            brk = pd.Timestamp(day)
            if lo < brk <= hi:
                rows.append({"volatile": str(pd.Timestamp(v).date()),
                             "baseline": str(pd.Timestamp(b).date()),
                             "break": name, "break_date": day, "level": "WARN",
                             "reason": ("pair spans the %s break: the two legs quote on "
                                        "different price grids" % name)})
    return pd.DataFrame(rows)
