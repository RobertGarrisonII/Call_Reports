#!/usr/bin/env python3
"""test_halt_aware_qc.py -- a crossed book during a circuit-breaker halt is CORRECT.

The residual crossing on the four March-2020 sessions survived three rounds of diagnosis as an
"open defect". It is the halt. The 2020-03-09 diagnostic settles it arithmetically:

    crossed overall                3.88% of 23,401 snapshots      =  908
    MWCB Level 1 halt 09:34:13-09:49:13 ET at 1s                  =  901
    crossed rate by session sixth  23.1% 0.0% 0.0% 0.0% 0.1% 0.1%
    23.1% of the first sixth (3,900)                              =  901

Exchanges stop matching during a halt but do NOT cancel resting orders, and new orders keep
arriving, so a limit order priced through the last trade sits unmatched for the full fifteen
minutes. The reconstruction is reproducing the market correctly; the invariant is what is wrong,
because it assumes a market that is always matching.

Two things follow, and this pins both:
  * the GATE must judge the replay on snapshots outside the halt, or it flags the paper's own
    subject matter as a data fault
  * the halt snapshots must still be EXCLUDED from every estimate -- a halted market has no valid
    midpoint, and two legs of stale quotes that cannot trade against each other produce mechanical
    comovement, not price discovery

Run: python test_halt_aware_qc.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import market_halts as mh
import mstbook_loader as ml

TZ = "America/New_York"


def _session(day, cross_halt=True, cross_open=0, es=True):
    idx = pd.date_range(f"{day} 09:30:00", f"{day} 16:00:00", freq="s", tz=TZ)
    n = len(idx)
    rng = np.random.default_rng(7)
    mid = 280 + np.cumsum(rng.normal(0, 0.01, n))
    df = pd.DataFrame({"SPY_bidprice_1": mid - 0.005, "SPY_askprice_1": mid + 0.005}, index=idx)
    df["ES_bidprice_1"] = (mid * 10 - 0.25) if es else np.nan
    df["ES_askprice_1"] = (mid * 10 + 0.25) if es else np.nan
    if cross_halt:                                   # what a halted book actually looks like
        h = mh.halt_mask(idx)
        df.loc[h, "SPY_askprice_1"] = df.loc[h, "SPY_bidprice_1"] - 0.02
    if cross_open:                                   # a genuine replay fault, outside any halt
        df.iloc[:cross_open, df.columns.get_loc("SPY_askprice_1")] = \
            df.iloc[:cross_open]["SPY_bidprice_1"] - 0.02
    return df


def check_halt_table():
    ok = []
    n09 = int(mh.halt_mask(pd.date_range("2020-03-09 09:30", "2020-03-09 16:00", freq="s",
                                         tz=TZ)).sum())
    a = n09 == 901 and abs(100 * n09 / 23401 - 3.85) < 0.02
    print("(1) the 2020-03-09 halt is %d of 23,401 snapshots = %.2f%%, against the session's "
          "observed 3.88%% crossed : %s" % (n09, 100 * n09 / 23401, a))
    ok.append(a)
    b = all(mh.is_halt_date(d) for d in ("2020-03-09", "2020-03-12", "2020-03-16", "2020-03-18"))
    c = not mh.is_halt_date("2024-12-18") and not mh.is_halt_date("2020-03-10")
    print("    all four MWCB dates are known=%s; ordinary dates are not=%s" % (b, c))
    ok += [b, c]
    # 2020-03-16 halted AT the open; 2020-03-18 in the afternoon. A fixed "skip the first N minutes"
    # rule would have handled one and not the other -- which is why this is a table, not a heuristic.
    d = (mh.halt_windows("2020-03-16")[0][0].strftime("%H:%M:%S") == "09:30:00"
         and mh.halt_windows("2020-03-18")[0][0].strftime("%H:%M:%S") == "12:56:11")
    print("    2020-03-16 halts at the OPEN and 2020-03-18 in the AFTERNOON, so no "
          "skip-the-first-N-minutes rule covers both : %s" % d)
    ok.append(d)
    return all(ok)


def check_qc_passes_a_halt():
    q = ml.session_qc(_session("2020-03-09"))
    passes = q["ok"]
    reported = (np.isclose(q["SPY"]["crossed_frac"], 901 / 23401, atol=1e-3)
                and np.isclose(q["SPY"]["crossed_frac_ex_halt"], 0.0, atol=1e-9))
    explains = any("IS the halt" in r for r in q["reasons"])
    ok = passes and reported and explains
    print("(2) a session crossed ONLY during its halt passes the gate=%s, with both figures "
          "reported (%.2f%% overall, %.2f%% outside)=%s, and the note explains why=%s : %s"
          % (passes, 100 * q["SPY"]["crossed_frac"], 100 * q["SPY"]["crossed_frac_ex_halt"],
             reported, explains, ok))
    return ok


def check_qc_still_catches_a_fault():
    q = ml.session_qc(_session("2020-03-12", cross_open=4000))
    fails = not q["ok"]
    names = any("OUTSIDE any halt" in r for r in q["reasons"])
    ok = fails and names
    print("(3) a session that crosses OUTSIDE its halt still FAILS=%s and the reason says so=%s "
          "(%.1f%% outside vs %.1f%% overall) : %s"
          % (fails, names, 100 * q["SPY"]["crossed_frac_ex_halt"],
             100 * q["SPY"]["crossed_frac"], ok))
    return ok


def check_non_halt_date_unaffected():
    clean = ml.session_qc(_session("2024-12-18", cross_halt=False))
    faulty = ml.session_qc(_session("2024-12-18", cross_halt=False, cross_open=4000))
    ok = clean["ok"] and not faulty["ok"] and clean["halt_snapshots"] == 0
    print("(4) an ordinary date is unaffected: clean passes=%s, crossed fails=%s, no halt "
          "snapshots=%s : %s" % (clean["ok"], not faulty["ok"], clean["halt_snapshots"] == 0, ok))
    return ok


def check_missing_leg_still_fatal():
    """Halt-awareness must not soften the other failure mode: the four MWCB days also have no ES."""
    q = ml.session_qc(_session("2020-03-09", es=False))
    ok = (not q["ok"]) and any("no usable top of book" in r and r.startswith("ES")
                               for r in q["reasons"])
    print("(5) a missing ES leg still fails on an MWCB date -- the halt explains crossing, not an "
          "absent leg : %s" % ok)
    return ok


def main():
    checks = [check_halt_table, check_qc_passes_a_halt, check_qc_still_catches_a_fault,
              check_non_halt_date_unaffected, check_missing_leg_still_fatal]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception as exc:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("halt-aware-QC checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
