#!/usr/bin/env python3
"""test_run_corrections.py -- the four defects the 2026-08-04 replication run exposed.

That run was the first clean end-to-end extract: 24 sessions, every gate green, the 2020 ES leg
recovered. Reading the log carefully turned up four things the run could not have told anyone, and
each is pinned here.

  (1) THE ES INVARIANT WENT VACUOUS.  `ES_crossed = 0.00%` on all 24 sessions. That reads as a
      clean futures book; it is not. Since v0.9.42 the ES leg is TAKEN from CME's ladder, and a
      venue-published ladder cannot cross by construction. The stack's primary integrity gate
      stopped applying to that leg and nothing said so. `ladder_integrity` supplies the checks that
      CAN fail on such a leg -- monotone depth, level coverage, staleness -- and `session_qc` runs
      them wherever the book came from the ladder.

  (2) THE HALT UNION HID ITS LEGS.  The run printed 900/900/900/901 halt snapshots on the four MWCB
      days: the SPY circuit-breaker windows exactly. On 2020-03-18 the ES leg resumes six seconds
      AFTER the equity reopen, so the union should have been larger -- and there was no column that
      would show whether the ES windows had been derived at all. `session_qc` now reports
      `halt_by_leg` and `qc_frames` prints a per-leg column.

  (3) PEAK MEMORY WAS GUESSED AT HALF ITS MEASURED VALUE.  `autoscale.measure` reported 58.4 GiB
      per worker against a built-in guess of 32, which had sized THIRTEEN workers where the
      measurement implies five. Thirteen at 58 GiB is 758 GiB nominal on a 495 GiB node; the run
      survived only because the peaks did not coincide.

  (4) TABLES 5 AND 7 DID NOT USE THE EXTRACTED DATA.  STAGE 4 re-analysed the paper's PUBLISHED 3x3
      matrices, typed in as literals, on a run that had just extracted 561,624 rows. That is a
      legitimate audit of the published numbers and it is still what runs when no frames exist --
      but nothing in the output distinguished it from an estimate on this sample. The stage now
      builds the panels from the session frames when they carry trade flow, and labels the source.

Run: python test_run_corrections.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import aggregated_book as ab
import autoscale as asc
import mstbook_loader as ml

TZ = "America/New_York"


def _frame(n=300, day="2020-03-12", levels=10, ladder=True):
    idx = pd.date_range(f"{day} 09:30:00", periods=n, freq="s", tz=TZ)
    df = pd.DataFrame(index=idx)
    df["SPY_bidprice_1"] = 250.0 + np.arange(n) * 0.001
    df["SPY_askprice_1"] = df["SPY_bidprice_1"] + 0.01
    for i in range(1, levels + 1):
        df[f"ES_bidprice_{i}"] = 2546.50 - 0.25 * (i - 1) + np.arange(n) * 0.01
        df[f"ES_askprice_{i}"] = 2547.00 + 0.25 * (i - 1) + np.arange(n) * 0.01
    if ladder:
        df.attrs["book_source_ES"] = "aggregated"
    df.attrs["halt_windows_SPY"] = []
    df.attrs["halt_windows_ES"] = []
    return df


def check_ladder_leg_has_an_invariant_again():
    """A mis-mapped ladder level does not cross, so only the new checks can catch it."""
    good = _frame()
    q = ml.session_qc(good, date="2020-03-12")
    healthy = q["ok"] and q["ES"]["ladder"]["monotone_frac"] == 1.0 and q["ES"]["crossed_frac"] == 0.0

    # level 2 published INSIDE level 1 -- a column-mapping or scale error. Note it does NOT cross:
    # the top of book is untouched, so the crossed test still reports 0.00%.
    bad = _frame(); bad.attrs = dict(good.attrs)
    bad["ES_bidprice_2"] = bad["ES_bidprice_1"] + 1.0
    qb = ml.session_qc(bad, date="2020-03-12")
    still_uncrossed = qb["ES"]["crossed_frac"] == 0.0
    caught = (not qb["ok"]) and any("monotone in depth" in r for r in qb["reasons"])

    ok = healthy and still_uncrossed and caught
    print("(1) a healthy ladder passes (%s); a ladder with level 2 inside level 1 STILL reports "
          "crossed=0.00%% (%s) and is caught by the monotonicity check (%s) : %s"
          % (healthy, still_uncrossed, caught, ok))
    return ok


def check_frozen_ladder_is_caught():
    """The other way a taken book fails: carried forward from a handful of rows."""
    frozen = _frame()
    for c in frozen.columns:
        frozen[c] = frozen[c].iloc[0]
    frozen.attrs["book_source_ES"] = "aggregated"
    q = ml.session_qc(frozen, date="2020-03-12")
    stale = q["ES"]["ladder"]["stale_frac"]
    ok = (not q["ok"]) and stale > 0.99 and any("carried forward" in r for r in q["reasons"])
    print("(2) a ladder whose top never changes: stale_frac=%.2f, session FAILED (%s) -- the "
          "aggregated-path analogue of a frozen replay : %s" % (stale, not q["ok"], ok))
    return ok


def check_halt_is_reported_per_leg():
    """The union alone cannot show whether the ES windows were derived."""
    df = _frame()
    df.attrs["halt_windows_SPY"] = [("2020-03-12T09:31:00-04:00", "2020-03-12T09:32:00-04:00")]
    df.attrs["halt_windows_ES"] = [("2020-03-12T09:31:30-04:00", "2020-03-12T09:33:00-04:00")]
    df.attrs["halt_windows"] = df.attrs["halt_windows_SPY"] + df.attrs["halt_windows_ES"]
    q = ml.session_qc(df, date="2020-03-12")
    per = q.get("halt_by_leg", {})
    union_bigger = q["halt_snapshots"] > per.get("SPY", 0)
    both = per.get("SPY", 0) == 61 and per.get("ES", 0) == 91
    # and the ES leg contributing NOTHING must be visible as a zero, not hidden in the union
    df2 = _frame()
    df2.attrs["halt_windows_SPY"] = df.attrs["halt_windows_SPY"]
    df2.attrs["halt_windows_ES"] = []
    df2.attrs["halt_windows"] = df.attrs["halt_windows_SPY"]
    q2 = ml.session_qc(df2, date="2020-03-12")
    visible_zero = q2["halt_by_leg"]["ES"] == 0 and q2["halt_by_leg"]["SPY"] == 61
    ok = union_bigger and both and visible_zero
    print("(3) union=%d vs SPY=%d ES=%d -- the ES contribution is now visible (%s)"
          % (q["halt_snapshots"], per.get("SPY"), per.get("ES"), both))
    print("    and an ES leg that contributes NO halt reads as a 0 rather than vanishing into the "
          "union (%s) : %s" % (visible_zero, ok))
    return ok


def check_memory_guess_is_the_measured_one():
    """13 workers x 58.4 GiB is 758 GiB on a 495 GiB node. The default must not size that."""
    now = asc.extraction_workers(24, ram=495, cores=128)
    then = asc.extraction_workers(24, peak_gb=32, ram=495, cores=128)
    measured = 58.4
    safe = now * measured <= 495
    unsafe_before = then * measured > 495
    ok = safe and unsafe_before and now <= 6
    print("(4) at the built-in guess the driver now sizes %d worker(s) (%.0f GiB nominal, fits in "
          "495) against %d before (%.0f GiB, does not) : %s"
          % (now, now * measured, then, then * measured, ok))
    return ok


def check_table5_can_be_built_from_the_frames():
    """The real-data path exists and produces a panel; that is what STAGE 4 now uses."""
    import tandem_order_flow as tof
    rng = np.random.default_rng(0)
    sess = []
    for d, reg in (("2020-03-12", "volatile"), ("2023-12-20", "benchmark")):
        n = 2000
        idx = pd.date_range(f"{d} 09:30:00", periods=n, freq="s", tz=TZ)
        f = pd.DataFrame(index=idx)
        for a, lam in (("SPY", 505.0), ("ES", 112.0)):
            f[f"{a}_trade_buy"] = rng.poisson(lam / 2, n).astype(float)
            f[f"{a}_trade_sell"] = rng.poisson(lam / 2, n).astype(float)
            f[f"{a}_trade_px"] = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
        sess.append((d, reg, f))
    detected = ml.has_trade_flow(sess)
    arrs = ml.counts_from_frame(sess[0][-1])
    six = arrs is not None and len(arrs) == 6
    by = tof.table5_from_sessions([(d, r, f) for d, r, f in sess], ml.counts_from_frame)
    built = set(by) == {"volatile", "benchmark"} and all("frequency" in v for v in by.values())
    # independent flow -> the corner log odds ratio should sit near zero, which is the sanity check
    d0 = tof.dependence_summary(by["benchmark"]["frequency"], n_bars=23400)
    near_zero = abs(d0["log_OR"]) < 1.0
    ok = detected and six and built and near_zero
    print("(5) frames carrying trade flow are detected (%s), counts_from_frame returns the 6 arrays "
          "(%s), and table5_from_sessions builds both panels (%s)" % (detected, six, built))
    print("    on INDEPENDENT synthetic flow the corner log OR is %.3f, i.e. ~0 as it must be (%s) "
          ": %s" % (d0["log_OR"], near_zero, ok))
    print("    NOTE: these are TRADE counts, not the paper's new-order submissions -- a different "
          "and better-identified statistic, which is why STAGE 4 labels the source.")
    return ok


def check_roll_is_measured_not_extrapolated():
    """Contract CHOICE is per-date; the volume SHARE was measured on four dates in one crisis week.

    The extractor quoted those four March-2020 measurements verbatim on every session in a roll
    window -- 2023-03-09, 2024-12-18 and 2025-06-13 among them -- which reads as a fact about those
    sessions and is an extrapolation across three to five years from a roll conducted under duress.
    The warning now says the share is UNMEASURED and names the tool that measures it."""
    import check_roll as cr
    rng_free = {("20200309", "ESH0"): 2010427, ("20200309", "ESM0"): 135318,
                ("20200312", "ESH0"): 3136474, ("20200312", "ESM0"): 1169026,
                ("20200316", "ESH0"): 1986076, ("20200316", "ESM0"): 3027078,
                ("20200318", "ESH0"): 732903, ("20200318", "ESM0"): 2605122}
    real = cr._fetch_head
    try:
        cr._fetch_head = lambda ymd, p, limit=40, timeout=0: (
            None if (ymd, p) not in rng_free
            else pd.DataFrame({"volume": [float(rng_free[(ymd, p)])], "openinterest": [1.0]}))
        reps = [cr.measure(d) for d in ("2020-03-09", "2020-03-12", "2020-03-16", "2020-03-18")]
    finally:
        cr._fetch_head = real
    shares = [round(100 * r["front_share"], 1) for r in reps]
    exact = shares == [93.7, 72.8, 60.4, 78.0]
    all_agree = all(r["rule_agrees"] for r in reps)
    # the boundary day must pair the front against the DEFERRED month, not the expired one
    boundary = next(r for r in reps if r["roll_offset_days"] == 0)
    right_rival = boundary["front"] == "ESH0" and boundary["rival"] == "ESM0"
    txt = cr.describe(reps)
    flags_split = "SPLIT sessions" in txt and "sample appendix" in txt
    ok = exact and all_agree and right_rival and flags_split
    print("(6) check_roll reproduces the four measured shares exactly: %s (%s), and the calendar "
          "rule picks the leader on all four (%s)"
          % (", ".join("%.1f%%" % s for s in shares), exact, all_agree))
    print("    the boundary day pairs %s against %s -- the deferred month, not the expired one (%s)"
          % (boundary["front"], boundary["rival"], right_rival))
    print("    and split sessions are named with what to do about them (%s) : %s"
          % (flags_split, ok))
    return ok


def main():
    checks = [check_ladder_leg_has_an_invariant_again, check_frozen_ladder_is_caught,
              check_halt_is_reported_per_leg, check_memory_guess_is_the_measured_one,
              check_table5_can_be_built_from_the_frames, check_roll_is_measured_not_extrapolated]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("run-correction checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
