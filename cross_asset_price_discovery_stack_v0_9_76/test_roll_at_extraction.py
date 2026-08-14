#!/usr/bin/env python3
"""test_roll_at_extraction.py -- the contract pick verifies itself, with the split in the log.

Until v0.9.54 the roll machinery was three disconnected pieces: the calendar rule picked the
contract on every session, `roll_window_days` flagged roll-week sessions on every session, and the
actual volume share had been measured on exactly four dates -- manually, via `check_roll.py`, and
only because the March-2020 statistics files were supplied by hand. Three sessions of the current
sample sit in a roll window with their share never measured (2023-03-09, 2024-12-18, 2025-06-13);
if any of them rolled unusually slowly, the calendar pick could be the minority contract and
nothing would ever say so.

`measure_roll_at_extraction` closes that loop: any session within (-3, +7) days of the boundary
gets the same cheap `mt_product_statistics` head-read AT EXTRACTION, the log carries the measured
volume AND open-interest split (plus turnover = vol/OI per contract), the report lands in
`df.attrs["roll_measurement"]`, and `session_qc` / the QC table surface it. The pick itself stays
the calendar rule -- an auto-switch on a same-day volume read would make the series definition
data-dependent -- but a minority pick is a log.error and a QC reason, not a post-hoc discovery.

Pinned here:

  (1) in-window sessions measure and log the full split (volume, OI, turnover -- all in the line);
      out-of-window sessions do not touch the lake at all
  (2) a MINORITY calendar pick logs at ERROR, lands in session_qc's reasons, and shows as
      roll_rule_ok=NO in the QC table -- without flipping qc.ok (the book itself is sound, and a
      hard fail would loop: re-extraction under the same rule repeats the same pick)
  (3) a split session (<90%) logs at WARNING with the do-not-splice instruction
  (4) volume-vs-OI leader disagreement gets its own log line (stock vs flow)
  (5) a lake failure NEVER kills extraction: fallback warning, (offset, None) returned
  (6) `check_roll.measure` now reports per-contract turnover, reset-proof like the rest
  (7) the QC table distinguishes measured / n-m, and flags in-window-but-unmeasured frames

Run: python test_roll_at_extraction.py
"""
from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd

import mstbook_loader as ml

TZ = "America/New_York"


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append((record.levelno, record.getMessage()))

    def text(self, level=None):
        return "\n".join(m for lv, m in self.records if level is None or lv == level)


def _capture():
    h = _Capture()
    ml.log.addHandler(h)
    ml.log.setLevel(logging.DEBUG)
    return h


def _rep(front="ESH5", rival="ESZ4", fv=3_000_000, rv=200_000, fo=1_800_000, ro=500_000,
         measured=True):
    tot, otot = fv + rv, fo + ro
    return {"date": "2024-12-18", "front": front, "rival": rival, "roll_offset_days": 6,
            "measured": measured, "front_share": fv / tot if measured else float("nan"),
            "volume": {front: float(fv), rival: float(rv)},
            "open_interest": {front: float(fo), rival: float(ro)},
            "oi_share": fo / otot, "turnover": {front: fv / fo, rival: rv / ro},
            "rule_agrees": fv >= rv, "note": "synthetic"}


def check_in_window_measures_and_logs_split():
    h = _capture()
    try:
        off, rep = ml.measure_roll_at_extraction("2024-12-18", _measure=lambda *a: _rep())
    finally:
        ml.log.removeHandler(h)
    t = h.text()
    in_win = off == 6 and rep is not None and rep["measured"]
    logged = all(s in t for s in ("volume ESH5=3,000,000 (93.8%)", "ESZ4=200,000 (6.2%)",
                                  "open interest ESH5=1,800,000", "turnover (vol/OI)"))
    info_level = any(lv == logging.INFO and "roll split measured" in m for lv, m in h.records)
    ok = in_win and logged and info_level
    print("(1a) in-window (+%d): measured, split logged with volume/OI/turnover (%s) at INFO "
          "for a concentrated pick (%s) : %s" % (off, logged, info_level, ok))
    return ok


def check_out_of_window_touches_nothing():
    called = []

    def _boom(*a):
        called.append(a)
        raise AssertionError("measure must not be called out of window")

    off, rep = ml.measure_roll_at_extraction("2024-07-24", _measure=_boom)
    ok = rep is None and not called and abs(off) > 7
    print("(1b) out-of-window (offset %+d): no lake read attempted (%s) : %s"
          % (off, not called, ok))
    return ok


def check_minority_pick_is_loud_but_not_fatal():
    rep_in = _rep(fv=800_000, rv=2_600_000, fo=900_000, ro=2_000_000)   # rival carries the market
    h = _capture()
    try:
        _off, rep = ml.measure_roll_at_extraction("2024-12-18", _measure=lambda *a: rep_in)
    finally:
        ml.log.removeHandler(h)
    err = h.text(logging.ERROR)
    loud = "MINORITY CONTRACT" in err and "23.5%" in err
    # session_qc surfaces it as a reason without flipping ok
    idx = pd.date_range("2024-12-18 09:30:00", periods=200, freq="s", tz=TZ)
    df = pd.DataFrame({"SPY_bidprice_1": 600.0, "SPY_askprice_1": 600.01,
                       "ES_bidprice_1": 6000.0, "ES_askprice_1": 6000.25}, index=idx)
    df.attrs["halt_windows_SPY"] = []; df.attrs["halt_windows_ES"] = []
    df.attrs["roll_offset_days"] = 6; df.attrs["roll_measurement"] = rep
    qc = ml.session_qc(df)
    reason = any("MINORITY" in r for r in qc["reasons"])
    not_fatal = qc["ok"]
    surfaced = qc.get("roll", {}).get("rule_agrees") is False
    ok = loud and reason and not_fatal and surfaced
    print("(2) minority pick: log.error fired (%s); session_qc reason recorded (%s) with ok "
          "preserved (%s) and roll block surfaced (%s) : %s"
          % (loud, reason, not_fatal, surfaced, ok))
    return ok


def check_split_session_warns_do_not_splice():
    rep_in = _rep(fv=1_600_000, rv=1_000_000)                            # 61.5%: leads, but split
    h = _capture()
    try:
        ml.measure_roll_at_extraction("2024-12-18", _measure=lambda *a: rep_in)
    finally:
        ml.log.removeHandler(h)
    w = h.text(logging.WARNING)
    ok = "ROLL SPLIT MEASURED" in w and "do NOT splice" in w and "61.5%" in w
    print("(3) split session (61.5%%): WARNING carries the share and the do-not-splice "
          "instruction : %s" % ok)
    return ok


def check_vol_oi_disagreement_gets_its_own_line():
    # volume says the front leads, OI still parked in the rival (the 2020-03-16 shape)
    rep_in = _rep(fv=3_000_000, rv=1_900_000, fo=1_400_000, ro=2_700_000)
    h = _capture()
    try:
        ml.measure_roll_at_extraction("2024-12-18", _measure=lambda *a: rep_in)
    finally:
        ml.log.removeHandler(h)
    t = h.text(logging.INFO)
    ok = "DISAGREE" in t and "day-stale STOCK" in t
    print("(4) vol-vs-OI leader disagreement: dedicated line explaining stock vs flow : %s" % ok)
    return ok


def check_lake_failure_never_kills_extraction():
    def _die(*a):
        raise RuntimeError("mstwx-lakequery not on PATH -- run this on the workbench")

    h = _capture()
    try:
        off, rep = ml.measure_roll_at_extraction("2024-12-18", _measure=_die)
    finally:
        ml.log.removeHandler(h)
    w = h.text(logging.WARNING)
    ok = rep is None and off == 6 and "could NOT be measured" in w and "check_roll.py" in w
    print("(5) lake failure: (offset, None) returned with the fallback warning, extraction "
          "survives : %s" % ok)
    return ok


def check_measure_reports_turnover():
    import check_roll as cr
    tabs = {"ESH5": (3_000_000, 1_800_000), "ESZ4": (1_000_000, 600_000)}

    def _fake_fetch(ymd, product, limit=40, timeout=1800):
        v, o = tabs[product]
        return pd.DataFrame({"volume": [v * 0.4, v], "openinterest": [o, o * 0.9]})

    orig = cr._fetch_head
    cr._fetch_head = _fake_fetch
    try:
        rep = cr.measure("2024-12-18")
    finally:
        cr._fetch_head = orig
    tr = rep.get("turnover", {})
    ok = (rep["measured"] and abs(tr.get("ESH5", 0) - 3_000_000 / 1_800_000) < 1e-9
          and abs(tr.get("ESZ4", 0) - 1_000_000 / 600_000) < 1e-9)
    print("(6) check_roll.measure turnover: ESH5=%.3f ESZ4=%.3f (max-read, reset-proof) : %s"
          % (tr.get("ESH5", float("nan")), tr.get("ESZ4", float("nan")), ok))
    return ok


def check_qc_table_carries_the_split():
    import qc_frames as qf

    def _frame(day, attrs):
        idx = pd.date_range(f"{day} 09:30:00", periods=100, freq="s", tz=TZ)
        df = pd.DataFrame({"SPY_bidprice_1": 600.0, "SPY_askprice_1": 600.01,
                           "ES_bidprice_1": 6000.0, "ES_askprice_1": 6000.25}, index=idx)
        df.attrs["halt_windows_SPY"] = []; df.attrs["halt_windows_ES"] = []
        df.attrs.update(attrs)
        return df

    sessions = [
        ("2024-12-18", "volatile", _frame("2024-12-18",
                                          {"roll_offset_days": 6, "roll_measurement": _rep()})),
        ("2025-06-13", "baseline", _frame("2025-06-13", {"roll_offset_days": 1})),  # in-window, n/m
        ("2024-07-24", "baseline", _frame("2024-07-24", {})),                       # not in window
    ]
    tbl = qf.qc_sessions(sessions)
    r0, r1, r2 = tbl.iloc[0], tbl.iloc[1], tbl.iloc[2]
    measured = (abs(r0["ES_front_vol"] - 0.9375) < 1e-9 and r0["roll_rule_ok"] == "yes"
                and np.isfinite(r0["ES_front_oi"]))
    nm = r1["roll_rule_ok"] == "n/m" and not np.isfinite(r1["ES_front_vol"]) and r1["roll_off"] == 1
    out = r2["roll_rule_ok"] == "n/m" and not np.isfinite(r2["roll_off"])
    ok = measured and nm and out
    print("(7) QC table: measured row carries vol%%/OI%%/rule (%s); in-window unmeasured is n/m "
          "with its offset (%s); out-of-window has no offset (%s) : %s"
          % (measured, nm, out, ok))
    return ok


def main():
    checks = [check_in_window_measures_and_logs_split,
              check_out_of_window_touches_nothing,
              check_minority_pick_is_loud_but_not_fatal,
              check_split_session_warns_do_not_splice,
              check_vol_oi_disagreement_gets_its_own_line,
              check_lake_failure_never_kills_extraction,
              check_measure_reports_turnover,
              check_qc_table_carries_the_split]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("roll-at-extraction checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
