#!/usr/bin/env python3
"""es_activity.py -- the offline per-contract activity table (open interest + cleared volume).

The activity contract rule decides from the PRIOR session's totals, head-read off the lake's
pre-open ``mt_product_statistics`` rows. That read failed on six sample sessions ("could not read
volume for both contracts"), leaving the calendar fallback with an UNKNOWN share in the QC report.
This module carries the committed daily table (``sample_inputs/es_activity_oi_volume.csv``:
dt, product, openinterest, clearedvolume for every ES contract, 2017-01-01..2026-08-16) so those
holes can be filled OFFLINE -- no lake read, deterministic, airgap-friendly.

CONFIRM AND REPORT, NEVER OVERRIDE. The table's field is CLEARED volume, and cleared and TRADED
volume demonstrably diverge exactly where the pick matters. The live case is 2020-03-16: the
prior-session table favours ESH0 3:1 by cleared volume (4.37 M vs 1.43 M on 2020-03-13), while
the tape's traded volume on 2020-03-16 itself had already moved to ESM0 (3.03 M vs 1.99 M --
the calendar pick carried 60.4% of that session's trading; see check_roll.measure's docstring).
An offline override on cleared volume would have put an MWCB session on the contract carrying
39.6% of the day's trading. So:

  * when the lake measurement FAILS, the table CONFIRMS an unambiguous calendar pick (rival
    expired, or the calendar contract leads cleared volume too) and REPORTS a contradiction
    (front's cleared share printed for the appendix) -- the extracted contract stays the
    calendar pick either way;
  * the QC report fills "IN A ROLL WINDOW BUT NOT MEASURED" with the table's shares, labelled
    as cleared volume, never conflated with the tape-measured traded shares.

Resolved against the six failed sessions (prior-session cleared volume, calendar share):
2018-03-19 ESM8 79.5% CONFIRMED; 2018-03-22 rival ESH8 expired, CONFIRMED; 2020-09-03 ESU0
98.9% CONFIRMED; 2023-12-20 rival ESZ3 expired, CONFIRMED; 2020-03-16 ESM0 24.7% CONTRADICTED
(see above -- the tape says the calendar pick was right anyway); 2022-06-13 ESU2 46.6%
CONTRADICTED (ESM2 led cleared volume and OI; expiry-Friday week, roll in flight).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

_TABLE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "sample_inputs", "es_activity_oi_volume.csv")
_CACHE: dict = {}


def load_table(path: Optional[str] = None) -> pd.DataFrame:
    """-> DataFrame[dt, product, openinterest, clearedvolume], one row per (dt, product), the
    LAST report of the day winning (the vendor stamps several intraday snapshots; the final one
    carries the settled figures). Cached per path."""
    p = path or _TABLE_PATH
    if p in _CACHE:
        return _CACHE[p]
    df = pd.read_csv(p, index_col=0)
    need = {"dt", "product", "openinterest", "clearedvolume"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"{p}: missing column(s) {sorted(missing)}")
    df["dt"] = pd.to_datetime(df["dt"])
    sort_cols = ["dt", "product"] + (["exchangetimestamp"] if "exchangetimestamp" in df.columns else [])
    df = (df.sort_values(sort_cols).groupby(["dt", "product"], as_index=False).last()
            [["dt", "product", "openinterest", "clearedvolume"]])
    for c in ("openinterest", "clearedvolume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    _CACHE[p] = df
    return df


def prior_session_stats(front: str, rival: str, as_of_date, path: Optional[str] = None) -> dict:
    """Prior-session cleared volume and open interest for the two roll candidates.

    'Prior session' = the last table date strictly before ``as_of_date`` -- the same
    ex-ante criterion as the lake pre-open head-read, on the clearing figures instead of
    the tape. -> {'asof', 'measured', 'volume': {c: v}, 'open_interest': {c: oi},
    'front_share', 'note'}; measured=False when the table cannot supply the date or the
    front contract, and a missing RIVAL is reported as expired/absent (share 1.0)."""
    d = pd.Timestamp(str(as_of_date)[:10])
    t = load_table(path)
    before = t[t["dt"] < d]
    rep = {"asof": None, "measured": False, "front_share": float("nan"),
           "volume": {}, "open_interest": {}, "note": "", "source": "cleared_table"}
    if before.empty:
        rep["note"] = "table starts after this date"
        return rep
    asof = before["dt"].max()
    g = before[before["dt"] == asof].set_index("product")
    rep["asof"] = str(asof.date())
    for c in (front, rival):
        rep["volume"][c] = float(g.loc[c, "clearedvolume"]) if c in g.index else float("nan")
        rep["open_interest"][c] = float(g.loc[c, "openinterest"]) if c in g.index else float("nan")
    vf, vr = rep["volume"][front], rep["volume"][rival]
    if not np.isfinite(vf):
        rep["note"] = f"front {front} absent from the table on {rep['asof']}"
        return rep
    if not np.isfinite(vr):
        rep["measured"] = True
        rep["front_share"] = 1.0
        rep["note"] = f"rival {rival} absent on {rep['asof']} (expired or not yet listed)"
        return rep
    tot = vf + vr
    rep["measured"] = tot > 0
    rep["front_share"] = float(vf / tot) if tot > 0 else float("nan")
    return rep


def _selftest() -> bool:
    ok = True
    r = prior_session_stats("ESM8", "ESH8", "2018-03-19")
    a = r["measured"] and abs(r["front_share"] - 0.795) < 0.01 and r["asof"] == "2018-03-16"
    print("(a) 2018-03-19 ESM8 vs ESH8: measured=%s front_share=%.3f asof=%s : %s"
          % (r["measured"], r["front_share"], r["asof"], a)); ok &= a
    r = prior_session_stats("ESM0", "ESH0", "2020-03-16")
    b = r["measured"] and r["front_share"] < 0.30
    print("(b) 2020-03-16 cleared volume CONTRADICTS the calendar pick (front %.1f%%) : %s"
          % (100 * r["front_share"], b)); ok &= b
    r = prior_session_stats("ESH4", "ESZ3", "2023-12-20")
    c = r["measured"] and r["front_share"] == 1.0 and "absent" in r["note"]
    print("(c) 2023-12-20 rival expired -> trivially confirmed : %s" % c); ok &= c
    return bool(ok)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
