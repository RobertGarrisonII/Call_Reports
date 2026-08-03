"""market_halts.py -- market-wide circuit breaker (MWCB) halt windows, and why the book crosses in them.

During a Level 1 halt the exchanges stop matching but they do NOT cancel resting orders, and new
orders keep arriving. A limit order priced through the last trade therefore sits in the book,
unmatched, for the full fifteen minutes. **A correctly reconstructed book is crossed during a halt.**
That is not a replay fault -- it is what a halted market looks like, and it is the phenomenon the
MWCB event study is about.

This matters because the crossed-book invariant is the stack's primary data-integrity gate, and on
exactly the four sessions the paper cares most about, the invariant is *supposed* to be violated for
900 of 23,401 snapshots. The arithmetic on 2020-03-09 is unambiguous:

    crossed fraction              3.88%  of 23,401 snapshots  =  908 snapshots
    a 15-minute Level 1 halt at 1s                            =  900 snapshots
    crossed rate by session sixth  23.1% 0.0% 0.0% 0.0% 0.1% 0.1%
    23.1% of the first sixth (3,900 snapshots)                =  901 snapshots

and the halt that day ran 09:34:13-09:49:13 ET, inside that first sixth. The residual crossing that
three rounds of diagnosis treated as an open defect is the circuit breaker.

Two consequences the paper has to state, not just the code:

  * A halted market has no valid midpoint. Snapshots inside a halt must be excluded from any
    lead-lag, correlation or information-share estimate -- the "prices" on both legs are stale
    quotes that cannot trade against each other. Including them does not add noise, it adds a
    mechanical cross-asset comovement that has nothing to do with price discovery.
  * The reopening auction is the interesting object (see mwcb_event_study.py, which already keys
    off the 2020-03-09 release at 09:49:13), and it needs the halt boundary to be exact.

Times are ET, from the exchange halt notices. Each MWCB Level 1 halt is exactly 15 minutes.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd

TZ = "America/New_York"

# date -> [(halt_start, halt_end)] in ET. Level 1 (7% S&P 500 decline) halts are 15 minutes.
# 2020-03-16 triggered at the opening bell, so its halt starts at 09:30:00 exactly.
MWCB_HALTS = {
    "2020-03-09": [("09:34:13", "09:49:13")],
    "2020-03-12": [("09:35:37", "09:50:37")],
    "2020-03-16": [("09:30:00", "09:45:00")],
    "2020-03-18": [("12:56:11", "13:11:11")],
}


def halt_windows(date, tz: str = TZ) -> list:
    """-> [(start, end)] tz-aware Timestamps for that date, or [] if the date had no MWCB halt."""
    key = str(date)[:10].replace("/", "-")
    if len(key) == 8 and key.isdigit():                      # accept YYYYMMDD too
        key = f"{key[:4]}-{key[4:6]}-{key[6:]}"
    out = []
    for a, b in MWCB_HALTS.get(key, []):
        out.append((pd.Timestamp(f"{key} {a}", tz=tz), pd.Timestamp(f"{key} {b}", tz=tz)))
    return out


def halt_mask(index: pd.DatetimeIndex, date=None, tz: str = TZ) -> np.ndarray:
    """Boolean mask, True where the timestamp falls inside a halt on that date.

    ``date`` defaults to the index's own first date, so a session frame needs no extra argument.
    Returns all-False for a date with no recorded halt, which is every date but four."""
    if index is None or len(index) == 0:
        return np.zeros(0, bool)
    if not isinstance(index, pd.DatetimeIndex):
        index = pd.DatetimeIndex(index)
    if date is None:
        d0 = index[0]
        date = (d0.tz_convert(tz) if getattr(index, "tz", None) is not None else d0).strftime("%Y-%m-%d")
    wins = halt_windows(date, tz=tz)
    m = np.zeros(len(index), bool)
    if not wins:
        return m
    idx = index.tz_localize(tz) if getattr(index, "tz", None) is None else index.tz_convert(tz)
    for a, b in wins:
        m |= np.asarray((idx >= a) & (idx <= b), dtype=bool)
    return m


def is_halt_date(date) -> bool:
    return bool(halt_windows(date))


def describe(date, tz: str = TZ) -> str:
    w = halt_windows(date, tz=tz)
    if not w:
        return f"{date}: no MWCB halt on record"
    return "%s: MWCB Level 1 halt %s" % (
        date, ", ".join(f"{a.strftime('%H:%M:%S')}-{b.strftime('%H:%M:%S')} ET "
                        f"({int((b - a).total_seconds())}s)" for a, b in w))


def _selftest() -> bool:
    ok = []
    idx = pd.date_range("2020-03-09 09:30:00", "2020-03-09 16:00:00", freq="s", tz=TZ)
    m = halt_mask(idx)
    n = int(m.sum())
    # 09:34:13-09:49:13 inclusive of both endpoints = 901 one-second snapshots
    a = n == 901
    print("(1) 2020-03-09 halt covers %d of %d 1s snapshots (%.2f%% -- the session's crossed rate "
          "was 3.88%%) : %s" % (n, len(idx), 100 * n / len(idx), a))
    ok.append(a)

    b = int(halt_mask(pd.date_range("2020-03-10 09:30", "2020-03-10 16:00", freq="s", tz=TZ)).sum()) == 0
    print("(2) an ordinary date has no halt window : %s" % b)
    ok.append(b)

    c = (len(halt_windows("20200318")) == 1 and len(halt_windows("2020-03-18")) == 1
         and halt_windows("2020-03-18")[0][0].strftime("%H:%M:%S") == "12:56:11")
    print("(3) YYYYMMDD and YYYY-MM-DD both resolve; 2020-03-18 halts at 12:56:11 (afternoon, not "
          "the open) : %s" % c)
    ok.append(c)

    # every MWCB halt is 15 minutes
    d = all(int((y - x).total_seconds()) == 900 for k in MWCB_HALTS for x, y in halt_windows(k))
    print("(4) all four Level 1 halts are exactly 900s : %s" % d)
    ok.append(d)

    # naive index (no tz) must still work -- session frames are not always tz-aware
    e = int(halt_mask(pd.date_range("2020-03-09 09:30", "2020-03-09 16:00", freq="s"),
                      date="2020-03-09").sum()) == 901
    print("(5) a tz-naive index is localized rather than silently mismatched : %s" % e)
    ok.append(e)

    print("\nmarket-halts checks -> %s" % all(ok))
    return all(ok)


if __name__ == "__main__":
    import sys
    sys.exit(0 if _selftest() else 1)
