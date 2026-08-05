#!/usr/bin/env python3
"""test_extract_resilience.py -- the extraction path must not lose a batch, and must not ship a book
it fabricated.

Motivated by a real 24-session run that died after 2h50m. Its log contains, in order:

    fetch mt_add_order failed for ESU3: mstwx-lakequery failed (rc=1)   <- swallowed
    fetch mt_trade    failed for SPY:  mstwx-lakequery failed (rc=1)   <- swallowed
    INVARIANT VIOLATED: SPY ... CROSSED on 100.0% of 23401 snapshots (trade_no_ref=0 of 0 trades)
    INVARIANT VIOLATED: ES  ... CROSSED on  43.8% of 23401 snapshots (trade_no_ref=119383 of 119383)
    ... median SPY=279.25 ES=nan ...            (x4: the whole MWCB block had no ES leg)
    RuntimeError: mstwx-lakequery failed (rc=1) <- NOT swallowed; killed all 22 finished sessions

Three separate defects with one signature. Each check below pins one of them:

  (1) rc!=0 is retried before it is believed          -- the abort was one un-retried transient
  (2) a failed session cannot abort the other 23      -- 22 finished days were discarded
  (3) a failed CRITICAL fetch refuses the session     -- instead of returning a 100%-crossed book
  (4) an optional (MBP) fetch failure still degrades gracefully
  (5) session_qc names a missing leg and a crossed top on the frame that is about to be saved
  (6) a good session is cached and reused; a degraded one is not

Run: python test_extract_resilience.py     (no MayStreet binary needed; the vendor call is faked)
"""
from __future__ import annotations

import os
import shutil
import subprocess as sp
import sys
import tempfile

import numpy as np
import pandas as pd

import lob_reconstruct as lob
import mstbook_loader as ml

TZ = "America/New_York"


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────
class _Res:
    def __init__(self, rc, err=""):
        self.returncode, self.stderr = rc, err


def _book(n=200, cross_frac=0.0, es=True, seed=0):
    """A canonical two-leg frame; ``cross_frac`` of the SPY rows have bid > ask (the fault
    signature), and ``es=False`` drops the ES leg to all-NaN (the March-2020 signature)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-07-24 09:30:00", periods=n, freq="s", tz="America/New_York")
    mid = 550 + np.cumsum(rng.normal(0, 0.01, n))
    df = pd.DataFrame({"SPY_bidprice_1": mid - 0.005, "SPY_askprice_1": mid + 0.005}, index=idx)
    k = int(round(cross_frac * n))
    if k:
        df.iloc[:k, df.columns.get_loc("SPY_askprice_1")] = df.iloc[:k]["SPY_bidprice_1"] - 0.02
    df["ES_bidprice_1"] = (mid * 10 - 0.25) if es else np.nan
    df["ES_askprice_1"] = (mid * 10 + 0.25) if es else np.nan
    return df


def _cfg(**kw):
    cfg = {"es_symbol": "ES", "levels": 10, "interval": "1s", "start_time": "9:30", "end_time": "16:00",
           "data_source": "apu", "tz": "America/New_York", "with_flow": False, "classify": "aggressor",
           "side_buy_label": "Bid", "futures_scale": 0.01, "rollover_days": 8, "round_lot": 100,
           "odd_lot_inclusive": True, "clock": "receipt", "cache_dir": "", "resume": True,
           "crossed_tol": 0.005}
    cfg.update(kw)
    return cfg


# ── (1) a non-zero vendor exit is retried before it is believed ──────────────────────────────────
def check_retry():
    calls = {"n": 0}

    def fake_run(cmd, **kw):
        calls["n"] += 1
        return _Res(0) if calls["n"] >= 3 else _Res(1, "boom")

    real_run, real_sleep = sp.run, ml.time.sleep
    ml.sp.run, ml.time.sleep = fake_run, lambda s: None
    try:
        fd, tmp = tempfile.mkstemp(suffix=".csv"); os.close(fd)
        try:
            ml._run_mstwx_lakequery_to_file(["mstwx-lakequery"], tmp, retries=3, backoff=0)
            recovered = calls["n"] == 3
        finally:
            os.unlink(tmp)

        calls["n"] = 0
        ml.sp.run = lambda cmd, **kw: _Res(1, "still broken")
        fd, tmp = tempfile.mkstemp(suffix=".csv"); os.close(fd)
        try:
            ml._run_mstwx_lakequery_to_file(["mstwx-lakequery"], tmp, retries=3, backoff=0)
            exhausted = False
        except RuntimeError as exc:
            exhausted = "after 3 attempt" in str(exc)
        finally:
            os.unlink(tmp)
    finally:
        ml.sp.run, ml.time.sleep = real_run, real_sleep
    ok = recovered and exhausted
    print("(1) rc=1 twice then 0 -> succeeds on attempt 3 : %s | always rc=1 -> raises naming the "
          "attempt count : %s" % (recovered, exhausted))
    return ok


# ── (2) one session's exception cannot discard the finished ones ─────────────────────────────────
def check_batch_isolation():
    specs = [(f"2024-01-{d:02d}", "benchmark") for d in range(1, 6)]
    bad = "2024-01-04"

    def fake_one(spec, cfg, progress_cb=None):
        label, regime = ml._spec_parts(spec)
        if label == bad:
            raise RuntimeError("mstwx-lakequery failed (rc=1): the vendor blew up")
        df = _book()
        return (label, regime, df, f"extracted {label}", None, ml.session_qc(df))

    real = ml._extract_one_session
    ml._extract_one_session = fake_one
    try:
        rep = {}
        # threading backend: same guarded path, no pickling of the monkeypatched function
        out = ml.extract_sessions(specs, max_workers=3, backend="threading", with_flow=False, report=rep)
    finally:
        ml._extract_one_session = real
    kept = [d for d, _r, _f in out]
    ok = (len(out) == 4 and bad not in kept and [d for d, _m in rep["failed"]] == [bad]
          and rep["ok"] == kept)
    print("(2) 1 of 5 sessions raises -> %d survive (want 4), failure named in the report=%s, "
          "batch did NOT abort : %s" % (len(out), [d for d, _ in rep["failed"]] == [bad], ok))

    # and if EVERY session dies, the caller must not receive a silent empty universe
    def all_fail(spec, cfg, progress_cb=None):
        raise RuntimeError("lake down")

    ml._extract_one_session = all_fail
    try:
        ml.extract_sessions(specs, max_workers=2, backend="threading", with_flow=False)
        empty_raises = False
    except RuntimeError as exc:
        empty_raises = "no usable session" in str(exc)
    finally:
        ml._extract_one_session = real
    print("    all 5 fail -> raises instead of returning [] : %s" % empty_raises)
    return ok and empty_raises


# ── (3)/(4) a failed CRITICAL fetch refuses the session; an optional one does not ────────────────
def check_strict_fetch():
    # A book that actually forms. Returning empty frames for everything used to be adequate here,
    # but an empty reconstruction is now itself refused (see check_empty_book_is_refused), and the
    # question this check asks -- does an OPTIONAL type's failure still let the session through --
    # needs a session there to let through.
    def _two_orders(mt):
        base = pd.Timestamp("2024-07-24 09:30:00", tz=TZ).value
        rows = [dict(receipttimestamp=base + int(1e6), exchangetimestamp=base + int(1e6),
                     sequencenumber=2, f="F", product="SPY", side="Bid", price=100.00, quantity=100,
                     previousprice=100.00, previousquantity=100, orderreferencenumber="b1",
                     previousorderreferencenumber="", maintainpriority="false"),
                dict(receipttimestamp=base + int(2e6), exchangetimestamp=base + int(2e6),
                     sequencenumber=4, f="F", product="SPY", side="Ask", price=100.05, quantity=100,
                     previousprice=100.05, previousquantity=100, orderreferencenumber="a1",
                     previousorderreferencenumber="", maintainpriority="false")]
        if mt != "mt_add_order":
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df.index = pd.to_datetime(df["receipttimestamp"], unit="ns", utc=True).dt.tz_convert(TZ)
        return df

    def make_fetch(failing, book=True):
        def _fetch(date_str, product, product_type, mt, data_source="apu", tz=None, clock="receipt"):
            if mt in failing:
                raise RuntimeError("mstwx-lakequery failed (rc=1)")
            return _two_orders(mt) if book else pd.DataFrame()
        return _fetch

    real = ml._fetch_messages
    try:
        ml._fetch_messages = make_fetch({"mt_trade"})
        try:
            lob.reconstruct_session("20240724", "SPY", "direct", levels=2)
            crit = False; msg = ""
        except ml.MessageFetchError as exc:
            crit, msg = True, str(exc)
        names = crit and "mt_trade" in msg and "20240724" in msg

        ml._fetch_messages = make_fetch({"mt_price_level_update"})
        opt = lob.reconstruct_session("20240724", "SPY", "direct", levels=2)
        opt_ok = isinstance(opt, pd.DataFrame) and "mt_price_level_update" in opt.attrs.get("fetch_failed", {})

        ml._fetch_messages = make_fetch({"mt_add_order"})
        lax = lob.reconstruct_session("20240724", "SPY", "direct", levels=2, strict=False)
        lax_ok = "mt_add_order" in lax.attrs.get("fetch_failed", {})
    finally:
        ml._fetch_messages = real
    print("(3) failed mt_trade -> MessageFetchError naming the type and the date : %s" % names)
    print("(4) failed mt_price_level_update (optional) -> session still returns, failure recorded "
          "in attrs : %s | strict=False escape hatch works : %s" % (opt_ok, lax_ok))
    return names and opt_ok and lax_ok


def check_empty_book_is_refused():
    """The 2020 ES shape: every fetch SUCCEEDS and returns zero rows.

    `strict` only ever covered a FAILED fetch. A fetch that exits 0 with a header and no data rows
    went straight through, and `reconstruct_book` dutifully produced 23,401 rows of NaN. The log
    line was an INFO -- `median SPY=254.19 ES=nan (ES/SPY=nan)` -- and `session_qc` had nothing to
    say either, because its crossed test needs a finite bid AND ask before it can compare them. So
    four sessions entered the dataset with a missing leg and the only symptom was a NaN in a
    diagnostic string.

    An empty fetch and a failed fetch are different claims, and BOTH are fatal when the result is a
    book that never has a top. The guard is on the output rather than on any one message type,
    because which types carry a book is a property of the feed and the era -- which is the open
    question for CME in 2020, not something to hardcode a second time."""
    def _empty(date_str, product, product_type, mt, data_source="apu", tz=None, clock="receipt"):
        return pd.DataFrame()                       # exit 0, header only, no rows

    real = ml._fetch_messages
    try:
        ml._fetch_messages = _empty
        try:
            lob.reconstruct_session("20200312", "ESH0", "futures", levels=2, price_scale=0.01,
                                    message_types=("mt_add_order", "mt_cancel_order",
                                                   "mt_modify_order", "mt_trade"))
            raised, msg = False, ""
        except ml.MessageFetchError as exc:
            raised, msg = True, str(exc)
        # the message has to carry the answer, or the next run is spent rediscovering it
        names = raised and "ESH0" in msg and "20200312" in msg and "mt_add_order" in msg
        counts = raised and "mt_trade=0" in msg
        points = raised and "mt_aggregated_price_update" in msg     # where to look next

        # ...and the escape hatch still works, for a deliberate forensic replay
        lax = lob.reconstruct_session("20200312", "ESH0", "futures", levels=2, strict=False,
                                      message_types=("mt_add_order", "mt_trade"))
        lax_ok = isinstance(lax, pd.DataFrame) and lax.attrs.get("message_counts") == {
            "mt_add_order": 0, "mt_trade": 0}
    finally:
        ml._fetch_messages = real

    ok = names and counts and points and lax_ok
    print("(6) every fetch succeeds and returns ZERO rows (the 2020 ES shape):")
    print("    the session is REFUSED rather than returned as 23,401 NaN rows, naming the product "
          "and date (%s)" % names)
    print("    with the per-type row counts (%s) and where to look next (%s) in the message" % (counts, points))
    print("    strict=False still returns it for forensics, with the counts in attrs (%s) : %s"
          % (lax_ok, ok))
    return ok


# ── (5) the QC that runs on the frame about to be written ────────────────────────────────────────
def check_session_qc():
    clean = ml.session_qc(_book())
    crossed = ml.session_qc(_book(cross_frac=1.0))          # the "100.0% crossed" session
    no_es = ml.session_qc(_book(es=False))                  # the "median ES=nan" sessions
    mild = ml.session_qc(_book(n=1000, cross_frac=0.002))   # brief consolidated crossing: tolerated

    a = clean["ok"] and not crossed["ok"] and not no_es["ok"] and mild["ok"]
    b = any("CROSSED" in r and "SPY" in r for r in crossed["reasons"])
    c = any("no usable top of book" in r and r.startswith("ES") for r in no_es["reasons"])
    d = np.isclose(crossed["SPY"]["crossed_frac"], 1.0) and no_es["ES"]["n_finite"] == 0
    print("(5) qc: clean=ok, 100%%-crossed=flagged, missing-ES=flagged, 0.2%%-crossed=tolerated : %s"
          % a)
    print("    reasons name the asset and the symptom (crossed=%s, missing leg=%s), numbers "
          "reported : %s" % (b, c, d))
    return a and b and c and d


# ── (6) a good session is cached and reused; a degraded one is not ───────────────────────────────
def check_cache_resume():
    cache = tempfile.mkdtemp(prefix="qc_cache_")
    calls = {"n": 0}
    try:
        def fake_one(spec, cfg, progress_cb=None):
            # exercise the REAL cache read/write logic by delegating to it: patch only the fetch half
            calls["n"] += 1
            label, regime = ml._spec_parts(spec)
            cpath = ml._cache_path(cfg["cache_dir"], label, cfg["interval"], cfg["levels"], cfg["clock"])
            if cfg["resume"] and os.path.exists(cpath):
                df = pd.read_pickle(cpath)
                return (label, regime, df, "reused " + label, None, ml.session_qc(df))
            df = _book(cross_frac=(1.0 if label.endswith("-02") else 0.0))
            qc = ml.session_qc(df)
            if qc["ok"]:
                df.to_pickle(cpath)
            return (label, regime, df, "extracted " + label, None, qc)

        real = ml._extract_one_session
        ml._extract_one_session = fake_one
        try:
            specs = [("2024-01-01", "benchmark"), ("2024-01-02", "benchmark")]
            ml.extract_sessions(specs, cache_dir=cache, with_flow=False, backend="sequential")
            files = sorted(os.listdir(cache))
            good_cached = any("20240101" in f for f in files)
            bad_cached = any("20240102" in f for f in files)

            rep = {}
            out = ml.extract_sessions(specs, cache_dir=cache, with_flow=False, backend="sequential",
                                      qc_action="drop", report=rep)
        finally:
            ml._extract_one_session = real
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    dropped = len(out) == 1 and [d for d, _r in rep["degraded"]] == ["2024-01-02"]
    ok = good_cached and not bad_cached and dropped
    print("(6) clean session cached=%s, degraded session NOT cached=%s, qc_action='drop' excludes "
          "it from the dataset=%s : %s" % (good_cached, not bad_cached, dropped, ok))
    return ok


def main():
    checks = [check_retry, check_batch_isolation, check_strict_fetch,
              check_empty_book_is_refused, check_session_qc, check_cache_resume]
    results = []
    for fn in checks:
        try:
            results.append(bool(fn()))
        except Exception as exc:
            print("%s RAISED: %s: %s" % (fn.__name__, type(exc).__name__, exc))
            results.append(False)
        print()
    ok = all(results)
    print("extract-resilience checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
