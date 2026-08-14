#!/usr/bin/env python3
"""test_pull_once.py -- pull once, process twice: the derivation is exact where it claims to be.

The two-grid runs paid the vendor twice for the same messages. derive_frames.py closes that: the
fine (10ms) extraction runs once and the coarse (1s) frames are DERIVED -- book/state columns by
row selection (the coarse grid points are a subset of the fine grid and a row is the state at its
timestamp), flow counts/volumes by summing left-labeled sub-bins, trade price by last-non-NaN,
VWAP recombined volume-weighted. Pinned here:

  (1) EXACTNESS against an independently-built direct coarse frame from the same synthetic event
      stream: book columns identical, flow sums identical, px last identical, vwap identical when
      every print is signed
  (2) misaligned grids are refused (non-integer multiple; target finer than the fine grid)
  (3) cache semantics mirror the extractor: _cache_path naming, an existing direct cache is NEVER
      overwritten, a qc-failing derived frame is not cached
  (4) --verify-existing catches a real book discrepancy (CLI exit 3) and passes a clean one
  (5) the driver runs the pull-once order -- fine extraction, then derive_frames, then the coarse
      invocation (cache hits) -- and STAGE 2b reports reuse instead of re-extracting; --fine-dates
      (subset) and --no-fine both fall back to the old behaviour

Run: python test_pull_once.py
"""
from __future__ import annotations

import os
import pickle
import subprocess as sp
import sys
import tempfile

import numpy as np
import pandas as pd

import derive_frames as dv
import mstbook_loader as ml

TZ = "America/New_York"
DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_paper_replication.sh")


def _event_stream(day="2024-07-24", n_secs=120, seed=7):
    """A quote step function + a signed trade tape, from which BOTH grids are built directly."""
    rng = np.random.default_rng(seed)
    t0 = pd.Timestamp(f"{day} 09:30:00", tz=TZ)
    # quote updates at random 10ms-aligned times
    n_upd = 900
    upd_ms = np.sort(rng.choice(np.arange(0, n_secs * 1000, 10), size=n_upd, replace=False))
    mid = 500.0 + np.cumsum(rng.normal(0, 0.01, n_upd))
    quotes = pd.DataFrame({"bid": mid - 0.01, "ask": mid + 0.01,
                           "bq": rng.integers(50, 150, n_upd).astype(float),
                           "aq": rng.integers(50, 150, n_upd).astype(float)},
                          index=t0 + pd.to_timedelta(upd_ms, unit="ms"))
    # signed trades at 10ms-aligned times (signed => vwap recombination is exact)
    n_tr = 600
    tr_ms = np.sort(rng.choice(np.arange(0, n_secs * 1000, 10), size=n_tr, replace=False))
    trades = pd.DataFrame({"px": 500.0 + rng.normal(0, 0.05, n_tr),
                           "qty": rng.integers(1, 20, n_tr).astype(float),
                           "buy": rng.integers(0, 2, n_tr).astype(float)},
                          index=t0 + pd.to_timedelta(tr_ms, unit="ms"))
    return t0, n_secs, quotes, trades


def _frame_at(interval, t0, n_secs, quotes, trades):
    """Build the frame the extractor would: state at-or-before each grid point + left-bin flow."""
    grid = pd.date_range(t0, t0 + pd.Timedelta(seconds=n_secs), freq=interval)
    pos = np.searchsorted(quotes.index.values, grid.values, side="right") - 1
    take = np.clip(pos, 0, None); unknown = pos < 0
    df = pd.DataFrame(index=grid)
    for col, src in (("SPY_bidprice_1", "bid"), ("SPY_askprice_1", "ask"),
                     ("SPY_bidquantity_1", "bq"), ("SPY_askquantity_1", "aq")):
        v = quotes[src].to_numpy()[take]
        df[col] = np.where(unknown, np.nan, v)
    for col in ("SPY_bidprice_1", "SPY_askprice_1", "SPY_bidquantity_1", "SPY_askquantity_1"):
        df["ES" + col[3:]] = df[col].to_numpy() * 10.0      # a second leg, same mechanics
    w = pd.DataFrame({"buy": trades["buy"], "sell": 1.0 - trades["buy"],
                      "bv": trades["qty"].where(trades["buy"] > 0),
                      "sv": trades["qty"].where(trades["buy"] == 0),
                      "px": trades["px"], "pxqty": trades["px"] * trades["qty"],
                      "qty": trades["qty"]}, index=trades.index)
    rs = w.resample(interval)
    fl = pd.DataFrame({"SPY_trade_buy": rs["buy"].sum(), "SPY_trade_sell": rs["sell"].sum(),
                       "SPY_trade_buy_volume": rs["bv"].sum(min_count=1),
                       "SPY_trade_sell_volume": rs["sv"].sum(min_count=1),
                       "SPY_trade_px": rs["px"].last(),
                       "SPY_trade_vwap": rs["pxqty"].sum(min_count=1) / rs["qty"].sum(min_count=1)})
    fl = fl.reindex(df.index)
    for c in ("SPY_trade_buy", "SPY_trade_sell", "SPY_trade_buy_volume", "SPY_trade_sell_volume"):
        df[c] = np.nan_to_num(fl[c].to_numpy(float))
    for c in ("SPY_trade_px", "SPY_trade_vwap"):
        df[c] = fl[c].to_numpy(float)
    df.attrs["halt_windows_SPY"] = []; df.attrs["halt_windows_ES"] = []
    df.attrs["book_source_ES"] = "aggregated"; df.attrs["es_contract"] = "ESU4"
    return df


def check_derivation_is_exact():
    t0, n, q, tr = _event_stream()
    fine = _frame_at("10ms", t0, n, q, tr)
    direct = _frame_at("1s", t0, n, q, tr)
    derived = dv.derive_coarse_frame(fine, "1s", "10ms")
    rep = dv._verify(derived, direct)
    # signed tape + interior bars: everything must agree, INCLUDING vwap and the final bar here
    # (the synthetic stream has no post-close prints, so the boundary caveat has nothing to bite)
    vwap_exact = rep["vwap_max_dev"] < 1e-9
    attrs_ok = (derived.attrs.get("es_contract") == "ESU4"
                and derived.attrs.get("derived_from_interval") == "10ms")
    ok = rep["ok"] and vwap_exact and rep["final_bar_flow_diff"] < 1e-9 and attrs_ok
    print("(1) derived 1s vs directly-built 1s from the same events: book/state exact (%s), "
          "flow sums exact (%s), vwap exact on a fully-signed tape (max dev %.2g), attrs carried "
          "(%s) : %s" % (not rep["book_mismatch"], not rep["flow_mismatch"],
                         rep["vwap_max_dev"], attrs_ok, ok))
    return ok


def check_misalignment_refused():
    t0, n, q, tr = _event_stream(n_secs=30)
    fine = _frame_at("30ms", t0, n, q, tr)
    try:
        dv.derive_coarse_frame(fine, "1s", "30ms")          # 1000/30 is not an integer
        bad_ratio = False
    except ValueError:
        bad_ratio = True
    fine1s = _frame_at("1s", t0, n, q, tr)
    try:
        dv.derive_coarse_frame(fine1s, "10ms", "1s")        # target FINER than source
        upsample = False
    except ValueError:
        upsample = True
    ok = bad_ratio and upsample
    print("(2) misalignment refused: non-integer multiple (%s), upsampling (%s) : %s"
          % (bad_ratio, upsample, ok))
    return ok


def check_cache_semantics():
    t0, n, q, tr = _event_stream()
    fine = _frame_at("10ms", t0, n, q, tr)
    with tempfile.TemporaryDirectory(prefix="po_cache_") as td:
        cpath = ml._cache_path(td, "2024-07-24", "1s", 10, "receipt", "aggregated")
        _, reports = dv.derive_sessions([("2024-07-24", "benchmark", fine)], "1s", "10ms",
                                        td, 10, "receipt", "aggregated", verify_existing=False)
        wrote = reports["2024-07-24"]["cached"] == cpath and os.path.exists(cpath)
        # a pre-existing DIRECT cache must never be overwritten
        sentinel = pd.DataFrame({"x": [1.0]})
        sentinel.to_pickle(cpath)
        _, rep2 = dv.derive_sessions([("2024-07-24", "benchmark", fine)], "1s", "10ms",
                                     td, 10, "receipt", "aggregated", verify_existing=False)
        kept = (rep2["2024-07-24"]["cached"] == "existing direct cache kept"
                and len(pd.read_pickle(cpath)) == 1)
        # a qc-failing frame (no usable book) is not cached
        broken = fine.copy(); broken.attrs = dict(fine.attrs)
        for c in broken.columns:
            if "price" in c:
                broken[c] = np.nan
        _, rep3 = dv.derive_sessions([("2024-08-05", "volatile", broken)], "1s", "10ms",
                                     td, 10, "receipt", "aggregated", verify_existing=False)
        c3 = ml._cache_path(td, "2024-08-05", "1s", 10, "receipt", "aggregated")
        unqc = (rep3["2024-08-05"]["qc_ok"] is False) and not os.path.exists(c3)
    ok = wrote and kept and unqc
    print("(3) cache: _cache_path naming written (%s); existing direct cache untouched (%s); "
          "qc-failing frame not cached (%s) : %s" % (wrote, kept, unqc, ok))
    return ok


def check_verify_cli_catches_mismatch():
    t0, n, q, tr = _event_stream()
    fine = _frame_at("10ms", t0, n, q, tr)
    direct = _frame_at("1s", t0, n, q, tr)
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory(prefix="po_verify_") as td:
        fpkl = os.path.join(td, "frames_10ms.pkl")
        with open(fpkl, "wb") as fh:
            pickle.dump([("2024-07-24", "benchmark", fine)], fh)
        cpath = ml._cache_path(td, "2024-07-24", "1s", 10, "receipt", "aggregated")
        direct.to_pickle(cpath)
        r_ok = sp.run([sys.executable, "derive_frames.py", "--pickle", fpkl, "--target", "1s",
                       "--fine-interval", "10ms", "--cache-dir", td, "--verify-existing"],
                      capture_output=True, text=True, timeout=300, cwd=here)
        clean = r_ok.returncode == 0 and "VERIFIED against direct cache" in r_ok.stdout
        bad = direct.copy(); bad.attrs = dict(direct.attrs)
        bad.iloc[50, bad.columns.get_loc("SPY_bidprice_1")] += 1.0
        bad.to_pickle(cpath)
        r_bad = sp.run([sys.executable, "derive_frames.py", "--pickle", fpkl, "--target", "1s",
                        "--fine-interval", "10ms", "--cache-dir", td, "--verify-existing"],
                       capture_output=True, text=True, timeout=300, cwd=here)
        caught = r_bad.returncode == 3 and "MISMATCH" in r_bad.stdout
    ok = clean and caught
    print("(4) --verify-existing: clean direct cache verifies with exit 0 (%s); a corrupted book "
          "cell is caught with exit 3 (%s) : %s" % (clean, caught, ok))
    return ok


def _dry(*args):
    with tempfile.TemporaryDirectory(prefix="po_dry_") as td:
        r = sp.run(["bash", DRIVER, "--dry-run", "--out-dir", td, *args],
                   capture_output=True, text=True, timeout=300)
        return r.returncode, r.stdout + r.stderr


def check_driver_pull_once_order():
    rc, out = _dry("--source", "extract", "--stages", "2")
    i_pull = out.find("PULL ONCE")
    i_fine = out.find("--interval 10ms")
    i_der = out.find("derive_frames.py")
    i_coarse = out.find("--interval 1s ")
    i_reuse = out.find("already extracted in STAGE 2 (pull once")
    ordered = -1 < i_pull < i_fine < i_der < i_coarse < i_reuse
    rc2, out2 = _dry("--source", "extract", "--fine-dates", "2020-03-09,2020-03-12", "--stages", "2")
    subset_fallback = "PULL ONCE" not in out2 and "STAGE 2b" in out2
    rc3, out3 = _dry("--source", "extract", "--no-fine", "--stages", "2")
    nofine = "PULL ONCE" not in out3 and "derive_frames" not in out3
    ok = rc == 0 and ordered and rc2 == 0 and subset_fallback and rc3 == 0 and nofine
    print("(5) driver: fine pull -> derive -> coarse (cache hits) -> 2b reuse, in that order (%s); "
          "--fine-dates subset falls back to two pulls (%s); --no-fine has no pull-once (%s) : %s"
          % (ordered, subset_fallback, nofine, ok))
    return ok


def main():
    checks = [check_derivation_is_exact,
              check_misalignment_refused,
              check_cache_semantics,
              check_verify_cli_catches_mismatch,
              check_driver_pull_once_order]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("pull-once checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
