#!/usr/bin/env python3
"""test_validate_sample.py -- the sample validator must catch the day the run actually wasted.

The 2026-07-30 extraction included 2026-01-19. That is Martin Luther King Jr. Day: NYSE closed, CME
Globex open on an abbreviated session. The log records the result -- `median SPY=nan ES=6913.75` --
a one-legged frame in a cross-asset study, produced after ~10 minutes of vendor I/O, inside a
multi-hour run nobody was watching.

Run: python test_validate_sample.py
"""
from __future__ import annotations

import sys

import pandas as pd

import validate_sample as vs


def _level(tbl, date):
    r = tbl[tbl["date"] == date]
    return (r["level"].iloc[0], r["reason"].iloc[0]) if len(r) else ("missing", "")


def check_closures():
    dates = ["2026-01-19",   # MLK -- the one that cost the run
             "2025-04-18",   # Good Friday
             "2025-01-09",   # Carter state funeral (one-off closure)
             "2022-06-20",   # Juneteenth observed (Sunday the 19th)
             "2024-12-21",   # a Saturday
             "2024-07-24"]   # an ordinary Wednesday
    tbl = vs.check_dates({"volatile": dates}, today=pd.Timestamp("2026-07-30"))
    got = {d: _level(tbl, d)[0] for d in dates}
    want = {"2026-01-19": "ERROR", "2025-04-18": "ERROR", "2025-01-09": "ERROR",
            "2022-06-20": "ERROR", "2024-12-21": "ERROR", "2024-07-24": "ok"}
    ok = got == want
    names = _level(tbl, "2026-01-19")[1]
    named = "Martin Luther King" in names and "one-legged" in names
    print("(1) closures: MLK/Good Friday/state funeral/Juneteenth/Saturday all ERROR, an ordinary "
          "Wednesday ok : %s" % ok)
    if not ok:
        print("    got %s" % got)
    print("    the MLK message names the holiday AND the failure mode it produces : %s" % named)
    return ok and named


def check_half_days():
    dates = ["2024-11-29",   # day after Thanksgiving
             "2024-07-03",   # July 3, a Wednesday
             "2024-12-24",   # Christmas Eve, a Tuesday
             "2021-07-02"]   # Jul 4 was a Sunday: the Friday before is a FULL session
    tbl = vs.check_dates({"baseline": dates}, today=pd.Timestamp("2026-07-30"))
    got = {d: _level(tbl, d)[0] for d in dates}
    want = {"2024-11-29": "WARN", "2024-07-03": "WARN", "2024-12-24": "WARN", "2021-07-02": "ok"}
    ok = got == want
    print("(2) half days WARN (usable but a 09:30-16:00 grid is two-thirds empty), and the Friday "
          "before a Sunday July 4 is NOT one : %s" % ok)
    if not ok:
        print("    got %s" % got)
    return ok


def check_duplicates_and_future():
    tbl = vs.check_dates({"volatile": ["2024-07-24", "2027-01-04"], "baseline": ["2024-07-24"]},
                         today=pd.Timestamp("2026-07-30"))
    dup = tbl[(tbl["group"] == "baseline") & (tbl["date"] == "2024-07-24")]["level"].iloc[0]
    fut = _level(tbl, "2027-01-04")[0]
    ok = dup == "ERROR" and fut == "ERROR"
    print("(3) a date reused across groups=ERROR, a future date=ERROR : %s" % ok)
    return ok


def check_pairing():
    vol = ["2024-12-18", "2025-01-07"]
    base = ["2023-12-20", "2024-01-29"]
    pr = vs.check_pairing(vol, base)
    good = pr.iloc[0]["level"] == "ok"                      # Wed<->Wed, 364d
    bad = pr.iloc[1]["level"] == "WARN" and "Tue vs Mon" in pr.iloc[1]["reason"]
    uneven = vs.check_pairing(["2024-12-18"], ["2023-12-20", "2024-01-29"])
    length = (uneven["level"] == "WARN").any()
    ok = good and bad and length
    print("(4) pairing: same-weekday ~1yr pair passes, the Tue<->Mon 344-day pair warns, unequal "
          "list lengths warn : %s" % ok)
    return ok


def check_exit_codes():
    argv = ["--volatile", "2026-01-19,2024-07-24", "--baseline", "2025-01-13,2023-07-19"]
    rc_bad = vs.main(argv + ["--out", ""])
    rc_forced = vs.main(argv + ["--allow-bad-dates"])
    rc_clean = vs.main(["--volatile", "2024-12-18", "--baseline", "2023-12-20"])
    ok = rc_bad == 1 and rc_forced == 0 and rc_clean == 0
    print("(5) exit codes: holiday in the universe -> 1, --allow-bad-dates -> 0, clean sample -> 0 "
          ": %s  (got %s/%s/%s)" % (ok, rc_bad, rc_forced, rc_clean))
    return ok


def main():
    checks = [check_closures, check_half_days, check_duplicates_and_future, check_pairing,
              check_exit_codes]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception as exc:
            print("%s RAISED: %s: %s" % (fn.__name__, type(exc).__name__, exc))
            res.append(False)
        print()
    ok = all(res)
    print("sample-validation checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
