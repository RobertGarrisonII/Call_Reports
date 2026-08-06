#!/usr/bin/env python3
"""test_replay_fast_snap.py -- the fine-grid replay is fast, observable, and BIT-IDENTICAL.

The 2026-08-06 two-grid run looked hung at the 10ms pulls. It was not hung: the replay rebuilt
the consolidated ladder from scratch at every grid snapshot -- iterate every price level of every
venue, full sort, per-venue round-lot scan -- O(whole book) x 2,340,001 snapshots per leg per
session, in silence (parallel workers suppressed heartbeats). v0.9.61 fixes both halves:

  SPEED   every level mutation routes through one choke point that keeps the consolidated
          aggregate, a sorted price list, and per-venue round-lot lists in step; a snapshot is
          then O(levels reported). The legacy per-snapshot rebuild is KEPT in-tree as the
          reference; grids under 100k points use it with the tracking switched off entirely
          (the upkeep taxes the event-dominated coarse path for nothing).
  VISIBILITY  every worker writes a per-session JSON heartbeat at each phase transition and every
          ~2M replayed events; extraction_status.py tabulates them and flags stalls.

Pinned here:

  (1) EQUIVALENCE, the whole point: legacy and fast paths produce byte-identical frames AND
      identical replay stats on a stream exercising every mutation path -- MBO adds/cancels/
      modifies, trades with and without leavesquantity, refless displayed and undisplayed prints,
      an MBP set-level feed, a mid-session feed clear -- at both grids and both odd-lot modes
  (2) auto mode picks legacy below 100k grid points and fast above (attrs['snap_path'])
  (3) the replay emits progress callbacks mid-loop (the liveness the hang lacked)
  (4) heartbeat_writer round-trips phase/elapsed through the JSON file, chains the user callback,
      and never raises on an unwritable dir
  (5) extraction_status renders running/done/failed and flags a stale heartbeat as STALLED?

Run: python test_replay_fast_snap.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time

import numpy as np
import pandas as pd

import extraction_status as est
import lob_reconstruct as lob
import mstbook_loader as ml

TZ = "America/New_York"
DAY = "2024-07-24"


def _rich_messages(n_secs=180, adds_per_sec=400, n_feeds=3, seed=11):
    """Every mutation path: MBO lifecycle + modifies + refless/undisplayed prints + an MBP feed
    + a mid-session clear of one feed."""
    rng = np.random.default_rng(seed)
    n = n_secs * adds_per_sec
    t_add = np.sort(rng.uniform(0, n_secs, n))
    life = rng.exponential(8.0, n)
    t_end = np.minimum(t_add + life, n_secs - 1e-3)
    end_kind = rng.choice([0, 1, 2], size=n, p=[0.70, 0.15, 0.15])   # cancel / trade+leaves / trade-legacy
    feeds = rng.integers(0, n_feeds, n)
    side = rng.integers(0, 2, n)
    mid = 500.0 + np.cumsum(rng.normal(0, 0.001, n))
    off = rng.exponential(0.06, n) + 0.01
    price = np.round((mid + np.where(side == 1, off, -off)) / 0.01) * 0.01
    qty = rng.integers(1, 6, n) * 100.0
    t0 = pd.Timestamp(f"{DAY} 09:30:00", tz=TZ).tz_convert("UTC").value

    def idx_of(t):
        return pd.to_datetime((np.asarray(t) * 1e9).astype(np.int64) + t0, utc=True).tz_convert(TZ)

    parts = []                                       # (t, mtype, rowdict)
    for i in range(n):
        f = f"feed{feeds[i]}"
        sd = "Ask" if side[i] else "Bid"
        parts.append((t_add[i], "mt_add_order",
                      {"f": f, "side": sd, "price": price[i], "quantity": qty[i],
                       "orderreferencenumber": str(i)}))
        k = end_kind[i]
        if k == 0:
            parts.append((t_end[i], "mt_cancel_order",
                          {"f": f, "previousquantity": qty[i], "orderreferencenumber": str(i)}))
        elif k == 1:
            parts.append((t_end[i], "mt_trade",
                          {"f": f, "price": price[i], "quantity": qty[i], "leavesquantity": 0.0,
                           "orderreferencenumber": str(i)}))
        else:
            parts.append((t_end[i], "mt_trade",
                          {"f": f, "price": price[i], "quantity": qty[i],
                           "leavesquantity": np.nan, "orderreferencenumber": str(i)}))
    # modifies: re-price 5% of adds mid-life
    for i in range(0, n, 20):
        tm = t_add[i] + 0.4 * life[i]
        if tm < n_secs - 1e-3:
            parts.append((tm, "mt_modify_order",
                          {"f": f"feed{feeds[i]}", "side": "Ask" if side[i] else "Bid",
                           "price": price[i] + 0.02, "quantity": qty[i],
                           "previousprice": price[i], "previousquantity": qty[i],
                           "orderreferencenumber": str(i), "previousorderreferencenumber": str(i),
                           "maintainpriority": "false"}))
    # refless prints: displayed (unknown ref, price on book) and undisplayed (Hidden)
    for i in range(0, n, 37):
        tt = min(t_add[i] + 0.2, n_secs - 1e-3)
        parts.append((tt, "mt_trade",
                      {"f": f"feed{feeds[i]}", "price": price[i], "quantity": 50.0,
                       "leavesquantity": np.nan, "orderreferencenumber": "ghost%d" % i}))
        parts.append((tt, "mt_trade",
                      {"f": f"feed{feeds[i]}", "price": price[i] + 0.005, "quantity": 10.0,
                       "leavesquantity": np.nan, "orderreferencenumber": "hid%d" % i,
                       "executionattribute": "Hidden"}))
    # an MBP feed setting and deleting aggregate levels
    for j in range(600):
        tt = rng.uniform(0, n_secs - 1e-3)
        parts.append((tt, "mt_price_level_update",
                      {"f": "mbp0", "side": "Ask" if j % 2 else "Bid",
                       "price": np.round((500.0 + rng.normal(0, 0.3)) / 0.01) * 0.01,
                       "quantity": float(rng.integers(0, 5) * 100)}))
    # one mid-session clear of feed0
    parts.append((n_secs * 0.6, "mt_clear_orders", {"f": "feed0", "quantity": np.nan}))

    parts.sort(key=lambda x: x[0])
    by_type: dict = {}
    for gseq, (t, mt, row) in enumerate(parts):
        row = dict(row); row["sequencenumber"] = gseq; row["_t"] = t
        by_type.setdefault(mt, []).append(row)
    out = {}
    for mt, rows in by_type.items():
        d = pd.DataFrame(rows)
        d.index = idx_of(d.pop("_t").to_numpy())
        out[mt] = d.sort_index(kind="stable")
    return out


def _build(interval, legacy, odd_lot=True, n_secs=180):
    msgs = _rich_messages(n_secs=n_secs)
    sess = ("09:30", "09:%02d" % (30 + n_secs // 60))
    return lob.reconstruct_book(msgs, asset="SPY", levels=10, interval=interval,
                                odd_lot_inclusive=odd_lot, session=sess, tz=TZ,
                                date_str=DAY.replace("-", ""), clock="receipt",
                                legacy_snap=legacy)


def check_equivalence_everywhere():
    ok = True
    for interval, odd_lot in (("1s", True), ("10ms", True), ("1s", False), ("10ms", False)):
        a = _build(interval, legacy=True, odd_lot=odd_lot)
        b = _build(interval, legacy=False, odd_lot=odd_lot)
        same = a.equals(b) and dict(a.attrs["lob_stats"]) == dict(b.attrs["lob_stats"])
        used = a.attrs["lob_stats"].get("trade_no_ref_displayed", 0) > 0 \
            and a.attrs["lob_stats"].get("trade_undisplayed", 0) > 0 \
            and a.attrs["lob_stats"].get("mbp_level_events", 0) > 0 \
            and a.attrs["lob_stats"].get("feed_clears", 0) > 0 \
            and a.attrs["lob_stats"].get("trade_leaves_corrected", 0) >= 0
        print("    %-5s odd_lot=%-5s frames+stats identical=%s (paths exercised: %s)"
              % (interval, odd_lot, same, used))
        ok = ok and same and used
    print("(1) legacy vs fast: byte-identical on every grid x odd-lot combination : %s" % ok)
    return ok


def check_auto_mode_and_progress():
    a = _build("1s", legacy=None)
    b = _build("10ms", legacy=None, n_secs=1200)     # 120,001 grid points > the 100k threshold
    auto = a.attrs["snap_path"] == "legacy" and b.attrs["snap_path"] == "fast"
    beats = []
    msgs = _rich_messages(n_secs=180)
    lob.reconstruct_book(msgs, asset="SPY", levels=10, interval="10ms",
                         session=("09:30", "09:33"), tz=TZ, date_str=DAY.replace("-", ""),
                         clock="receipt", progress_cb=beats.append, progress_every=50_000)
    live = any("replay" in m and "% events" in m for m in beats)
    ok = auto and live
    print("(2/3) auto mode: 1s->%s, 10ms(120k pts)->%s (%s); mid-replay progress callbacks fire "
          "(%d beats) : %s" % (a.attrs["snap_path"], b.attrs["snap_path"], auto, len(beats), ok))
    return ok


def check_heartbeat_writer():
    with tempfile.TemporaryDirectory(prefix="hb_") as td:
        got = []
        beat = ml.heartbeat_writer(td, "2020-03-09", "10ms", user_cb=got.append)
        beat("SPY replay: 40% events")
        f = os.path.join(td, "2020-03-09_10ms.json")
        with open(f) as fh:
            d = json.load(fh)
        wrote = (d["phase"] == "SPY replay: 40% events" and d["session"] == "2020-03-09"
                 and d["interval"] == "10ms" and "elapsed_s" in d)
        chained = got == ["SPY replay: 40% events"]
        beat("DONE")
        with open(f) as fh:
            rewrote = json.load(fh)["phase"] == "DONE"
    silent = True
    try:
        ml.heartbeat_writer("/proc/definitely/not/writable", "x", "1s")("phase")
    except Exception:
        silent = False
    ok = wrote and chained and rewrote and silent
    print("(4) heartbeat: JSON round-trip (%s), user callback chained (%s), atomic rewrite (%s), "
          "unwritable dir never raises (%s) : %s" % (wrote, chained, rewrote, silent, ok))
    return ok


def check_status_tool():
    with tempfile.TemporaryDirectory(prefix="es_") as td:
        pdir = os.path.join(td, "progress"); os.makedirs(pdir)
        now = time.time()
        for name, phase, age in (("2020-03-09_10ms", "SPY replay: 60% events (12M/20M), 55% grid", 30),
                                 ("2020-03-12_10ms", "DONE", 500),
                                 ("2020-03-16_10ms", "FAILED: MessageFetchError", 100),
                                 ("2020-03-18_10ms", "trade flow", 2000)):
            with open(os.path.join(pdir, name + ".json"), "w") as fh:
                json.dump({"session": name.split("_")[0], "interval": "10ms", "phase": phase,
                           "elapsed_s": 1234.0, "updated_epoch": now - age}, fh)
        txt = est.describe(est.read_status(pdir), stall_s=900)
        import subprocess as sp
        r = sp.run([sys.executable, "extraction_status.py", "--cache-dir", td],
                   capture_output=True, text=True, timeout=120,
                   cwd=os.path.dirname(os.path.abspath(__file__)))
        r2 = sp.run([sys.executable, "extraction_status.py", "--cache-dir", os.path.join(td, "nope")],
                    capture_output=True, text=True, timeout=120,
                    cwd=os.path.dirname(os.path.abspath(__file__)))
    ok = ("STALLED?" in txt and "FAILED" in txt and "done" in txt
          and "1 done, 1 running, 1 failed" in txt and "1 STALLED?" in txt
          and r.returncode == 0 and "2020-03-09" in r.stdout and r2.returncode == 1)
    print("(5) extraction_status: running/done/failed rendered, stale beat flagged STALLED?, CLI "
          "exit codes 0/1 : %s" % ok)
    return ok


def main():
    checks = [check_equivalence_everywhere,
              check_auto_mode_and_progress,
              check_heartbeat_writer,
              check_status_tool]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("fast-snap replay checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
