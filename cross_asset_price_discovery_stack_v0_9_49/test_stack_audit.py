#!/usr/bin/env python3
"""test_stack_audit.py -- defects found by auditing the stack after the first clean run.

Five findings, four of them mine. In order of how close each came to costing a result:

  (1) `qc_frames` crashed on the first run where ES price limits appear.  The futures-band notice
      (v0.9.46) appended to `lines`, a name that does not exist in that function -- it accumulates
      into `body`. The branch only executes when `{A}_luld_known > 0`, i.e. the NameError was
      reserved for exactly the moment the feature started working, inside the STAGE 3 gate.
      pyflakes found it; no run had, because no frame has carried ES market state yet.

  (2) The extraction log said `book=reconstruct` while the book came from the ladder.  All 24
      completion lines of the 2026-08-04 run carried a hardcoded string from the pre-v0.9.42 path.
      A reader auditing that log would conclude the ES leg was replayed. The line now reports
      `attrs["book_source_ES"]`.

  (3) The session cache ignored the ES book source.  A frame cached under `--es-book-source replay`
      would satisfy a resume under `aggregated` and vice versa -- identical columns, identical
      shape checks, nothing downstream would notice, and the dataset would silently mix the two
      mechanisms the aggregated default exists to unify. The cache name now carries the source, a
      cached frame's own recorded source is verified on read, and a frame that cannot prove its
      source (the v0.9.42-45 window, where the join dropped the ES attrs) is re-extracted once.

  (4) One flowless session reverted Table 5 to the published matrices.  `table5_from_sessions`
      concatenated `counts_fn` results without handling None, so a single frame without trade
      columns threw, the caller's fallback caught it, and 23 usable sessions were discarded for
      one. The session is now the unit of failure, and the skipped names travel with the result.

  (5) `check_roll` read the cumulative volume across the 18:00 reset.  The head-read took the LAST
      value of the fetched rows; the counter resets at the session open, and on a thin pre-open the
      reset falls inside the head. That compares one contract's fresh counter (hundreds of lots)
      against the other's day-old total -- a front-month ranking produced by row alignment. The max
      is the previous session's total on either side of the reset.

Run: python test_stack_audit.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import mstbook_loader as ml
import qc_frames as qf

TZ = "America/New_York"


def _frame(day="2020-03-12", n=300, es_luld=False):
    idx = pd.date_range(f"{day} 09:30:00", periods=n, freq="s", tz=TZ)
    df = pd.DataFrame(index=idx)
    df["SPY_bidprice_1"] = 250.0 + np.arange(n) * 0.001
    df["SPY_askprice_1"] = df["SPY_bidprice_1"] + 0.01
    for i in range(1, 11):
        df[f"ES_bidprice_{i}"] = 2546.50 - 0.25 * (i - 1) + np.arange(n) * 0.01
        df[f"ES_askprice_{i}"] = 2547.00 + 0.25 * (i - 1) + np.arange(n) * 0.01
    df.attrs["halt_windows_SPY"] = []
    df.attrs["halt_windows_ES"] = []
    if es_luld:
        df.attrs["market_state"] = {"ssr_known": 1.0, "ssr_frac_on": 0.0, "luld_known": 0.0}
        df.attrs["market_state_ES"] = {"luld_known": 0.97, "luld_band_bps_median": 700.0}
    return df


def check_futures_band_notice_does_not_crash():
    """The v0.9.46 NameError: only reachable once ES price limits are populated."""
    sess = [("2020-03-12", "volatile", _frame(es_luld=True))]
    tbl = qf.qc_sessions(sess)
    has_col = "ES_luld_known" in tbl.columns and float(tbl["ES_luld_known"].iloc[0]) > 0.9
    # the crash lived in main()'s report assembly; drive it end to end through the CLI
    import pickle, tempfile, os
    fd, tmp = tempfile.mkstemp(suffix=".pkl"); os.close(fd)
    out = tmp + ".txt"
    try:
        with open(tmp, "wb") as fh:
            pickle.dump(sess, fh)
        rc = qf.main(["--pickle", tmp, "--out", out])
        text = open(out).read() if os.path.exists(out) else ""
    finally:
        for f in (tmp, out):
            try:
                os.unlink(f)
            except OSError:
                pass
    ran = rc == 0
    noticed = "Price limits ARE reported on the futures leg" in text
    ok = has_col and ran and noticed
    print("(1) a session with populated ES price limits: per-leg column present (%s), the full "
          "report runs instead of raising NameError (%s), and the futures-band notice appears "
          "(%s) : %s" % (has_col, ran, noticed, ok))
    return ok


def check_cache_is_source_aware():
    """A replay-built frame must not satisfy an aggregated resume, and vice versa."""
    tagged = ml._cache_path("c", "2020-03-12", "1s", 10, "receipt", "aggregated")
    legacy = ml._cache_path("c", "2020-03-12", "1s", 10, "receipt", "")
    replay = ml._cache_path("c", "2020-03-12", "1s", 10, "receipt", "replay")
    distinct = len({tagged, legacy, replay}) == 3
    ok = distinct and tagged.endswith("_aggregated.pkl") and replay.endswith("_replay.pkl")
    print("(2) cache filenames are distinct per ES book source (%s): %s / %s"
          % (distinct, tagged.split("/")[-1], replay.split("/")[-1]))
    print("    identical columns and shape checks mean nothing downstream would catch a mixed "
          "dataset -- the name is the only place the difference exists : %s" % ok)
    return ok


def check_flowless_session_is_skipped_not_fatal():
    """One frame without trade columns must cost that frame, not the table."""
    import tandem_order_flow as tof
    rng = np.random.default_rng(0)
    sess = []
    for d, reg, flow in (("2020-03-12", "volatile", True), ("2023-12-20", "benchmark", True),
                         ("2024-01-29", "benchmark", False)):
        n = 1500
        idx = pd.date_range(f"{d} 09:30:00", periods=n, freq="s", tz=TZ)
        f = pd.DataFrame(index=idx)
        if flow:
            for a, lam in (("SPY", 505.0), ("ES", 112.0)):
                f[f"{a}_trade_buy"] = rng.poisson(lam / 2, n).astype(float)
                f[f"{a}_trade_sell"] = rng.poisson(lam / 2, n).astype(float)
                f[f"{a}_trade_px"] = 100.0 + np.cumsum(rng.normal(0, 0.01, n))
        sess.append((d, reg, f))
    by = tof.table5_from_sessions(sess, ml.counts_from_frame)
    skipped = by.get("_skipped", [])
    built = {"volatile", "benchmark"} <= set(by)
    named = skipped == ["2024-01-29"]
    counted = by["volatile"].get("n_sessions") == 1 and by["benchmark"].get("n_sessions") == 1
    ok = built and named and counted
    print("(3) 2 sessions with flow + 1 without: both panels build (%s), the flowless one is "
          "skipped BY NAME (%s) rather than reverting the whole table to the published matrices, "
          "and each panel counts its sessions (%s) : %s" % (built, named, counted, ok))
    return ok


def check_roll_read_survives_the_session_reset():
    """A head window that crosses the 18:00 reset must not rank by the post-reset counter."""
    import check_roll as cr
    # front contract: reset falls INSIDE the head -- last value is the fresh counter (tiny).
    # rival: thin stream, no reset in the head -- last value is the day-old total.
    front_rows = pd.DataFrame({"volume": [3136474.0, 3136474.0, np.nan, 104.0, 407.0],
                               "openinterest": [3190717.0] * 5})
    rival_rows = pd.DataFrame({"volume": [1169026.0, 1169026.0],
                               "openinterest": [352905.0] * 2})
    real = cr._fetch_head
    try:
        cr._fetch_head = lambda ymd, p, limit=40, timeout=0: (front_rows if p == "ESH0" else rival_rows)
        r = cr.measure("2020-03-12")
    finally:
        cr._fetch_head = real
    right = r["measured"] and r["rule_agrees"] and abs(r["front_share"] - 0.728) < 0.01
    # the last-value read would have scored the front month at 407 vs 1,169,026 -> 0.03%
    last_would_fail = 407.0 / (407.0 + 1169026.0) < 0.01
    ok = right and last_would_fail
    print("(4) a head window crossing the 18:00 reset: max-read ranks ESH0 at %.1f%% (correct, %s); "
          "a last-value read would have scored it 0.03%% and called the calendar rule WRONG (%s) "
          ": %s" % (100 * r["front_share"], right, last_would_fail, ok))
    return ok


def check_ladder_scope_and_diag_string():
    """The ladder checks stay on the leg that came from the ladder; the log names the source."""
    df = _frame()
    df.attrs["book_source"] = "aggregated"      # frame-level tag only (the v0.9.42-45 shape)
    q = ml.session_qc(df, date="2020-03-12")
    es_checked = "ladder" in q["ES"]
    spy_untouched = "ladder" not in q["SPY"]
    src = ml.__dict__  # the diag string is built inside _extract_one_session; check the template
    import inspect
    body = inspect.getsource(ml._extract_one_session)
    honest = ', book=reconstruct)' not in body and 'ES book=%s' in body
    ok = es_checked and spy_untouched and honest
    print("(5) with only the frame-level tag, the ladder checks run on ES (%s) and NOT on SPY, "
          "whose depth columns do not exist (%s)" % (es_checked, spy_untouched))
    print("    and the extraction completion line reports the recorded source instead of the "
          "hardcoded 'book=reconstruct' that mislabelled all 24 sessions of the 2026-08-04 run "
          "(%s) : %s" % (honest, ok))
    return ok


def main():
    checks = [check_futures_band_notice_does_not_crash, check_cache_is_source_aware,
              check_flowless_session_is_skipped_not_fatal, check_roll_read_survives_the_session_reset,
              check_ladder_scope_and_diag_string]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("stack-audit checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
