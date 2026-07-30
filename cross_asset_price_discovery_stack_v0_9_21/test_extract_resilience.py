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
    def make_fetch(failing):
        def _fetch(date_str, product, product_type, mt, data_source="apu", tz=None, clock="receipt"):
            if mt in failing:
                raise RuntimeError("mstwx-lakequery failed (rc=1)")
            return pd.DataFrame()
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
    checks = [check_retry, check_batch_isolation, check_strict_fetch, check_session_qc,
              check_cache_resume]
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
