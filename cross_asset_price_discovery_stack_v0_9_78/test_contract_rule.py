#!/usr/bin/env python3
"""Gate: v0.9.73 activity-based contract selection (select_contract).

The calendar front-month rule pinned roll-window sessions to whichever contract
the calendar named -- measured as low as 60.4% of two-contract volume in March
2020, i.e. the extracted ES leg missed the MAJORITY of futures price discovery
on those days. The activity rule measures first (prior-session closing volume/OI
via the cheap mt_product_statistics head-read -- deterministic per date) and
extracts the leader. This pins:

  1. rule='calendar' is untouched: same pick as get_front_month_contract, no
     lake read, no report
  2. rule='activity' far from the roll (|offset| > window): calendar pick, and
     the measurement is NEVER attempted -- distant neighbours are dead, so cache
     hits and quiet dates stay free
  3. rule='activity' in-window with the rival leading: the rival is extracted,
     the report says overrode=True, and the log announces the override loudly
  4. rule='activity' in-window with the front leading: calendar pick confirmed
  5. a lake failure falls back to the calendar pick with a warning -- selection
     can never kill extraction
  6. measure_roll_at_extraction(extracted=<leader>) demotes the MINORITY error
     to an informational confirmation when the activity rule already extracted
     the leader; with extracted unset the loud error survives
  7. the surface: extract_sessions takes contract_rule, run_analysis defaults
     --contract-rule activity, and the replication driver passes the rule to
     every extraction call
"""
import inspect
import logging
import sys

import mstbook_loader as ml


class _Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append((record.levelno, record.getMessage()))

    def text(self):
        return "\n".join(m for _l, m in self.records)


def _cap():
    h = _Capture()
    ml.log.addHandler(h)
    ml.log.setLevel(logging.DEBUG)
    return h


def _rep_rival_leads():
    return {"front": "ESH5", "rival": "ESZ4", "measured": True, "rule_agrees": False,
            "volume": {"ESH5": 200_000.0, "ESZ4": 3_000_000.0},
            "open_interest": {"ESH5": 300_000.0, "ESZ4": 1_800_000.0},
            "turnover": {"ESH5": 0.67, "ESZ4": 1.67},
            "front_share": 200_000.0 / 3_200_000.0, "oi_share": 300_000.0 / 2_100_000.0}


def _rep_front_leads():
    return {"front": "ESH5", "rival": "ESZ4", "measured": True, "rule_agrees": True,
            "volume": {"ESH5": 3_000_000.0, "ESZ4": 200_000.0},
            "open_interest": {"ESH5": 1_800_000.0, "ESZ4": 300_000.0},
            "turnover": {"ESH5": 1.67, "ESZ4": 0.67},
            "front_share": 3_000_000.0 / 3_200_000.0, "oi_share": 1_800_000.0 / 2_100_000.0}


def check_calendar_untouched():
    d = ml._parse_yyyymmdd("20241218")
    calls = []
    con, rep = ml.select_contract("ES", d, rule="calendar",
                                  _measure=lambda *a: calls.append(a) or _rep_front_leads())
    want = ml.get_front_month_contract("ES", as_of_date=d)
    ok = con == want and rep is None and not calls
    print("(1) calendar rule: pick %s == %s, no measurement, no report : %s" % (con, want, ok))
    return bool(ok)


def check_out_of_window_free():
    d = ml._parse_yyyymmdd("20240724")
    calls = []
    con, rep = ml.select_contract("ES", d, rule="activity",
                                  _measure=lambda *a: calls.append(a) or _rep_front_leads())
    want = ml.get_front_month_contract("ES", as_of_date=d)
    ok = con == want and rep is None and not calls
    print("(2) activity rule far from the roll: calendar pick %s, measurement never "
          "attempted : %s" % (con, ok))
    return bool(ok)


def check_override():
    d = ml._parse_yyyymmdd("20241218")                    # +6 days from the boundary
    h = _cap()
    try:
        con, rep = ml.select_contract("ES", d, rule="activity", label="2024-12-18",
                                      _measure=lambda *a: _rep_rival_leads())
    finally:
        ml.log.removeHandler(h)
    ok_pick = con == "ESZ4"
    ok_rep = rep is not None and rep.get("overrode") is True and rep.get("calendar_pick") == "ESH5"
    ok_log = "ACTIVITY RULE OVERRODE THE CALENDAR PICK" in h.text()
    print("(3) rival carries 93.8%% of volume: extracted %s (overrode=%s), loud log (%s)"
          % (con, rep.get("overrode") if rep else None, ok_log))
    return bool(ok_pick and ok_rep and ok_log)


def check_confirmation():
    d = ml._parse_yyyymmdd("20241218")
    con, rep = ml.select_contract("ES", d, rule="activity",
                                  _measure=lambda *a: _rep_front_leads())
    ok = con == "ESH5" and rep is not None and rep.get("overrode") is False
    print("(4) front leads: calendar pick confirmed by measurement : %s" % ok)
    return bool(ok)


def check_failure_falls_back():
    d = ml._parse_yyyymmdd("20241218")

    def _boom(*a):
        raise RuntimeError("lakequery: connection refused")

    h = _cap()
    try:
        con, rep = ml.select_contract("ES", d, rule="activity", _measure=_boom)
    finally:
        ml.log.removeHandler(h)
    want = ml.get_front_month_contract("ES", as_of_date=d)
    ok = con == want and rep is None and "FALLING BACK" in h.text()
    print("(5) lake failure: calendar fallback %s, warning logged, no exception : %s"
          % (con, ok))
    return bool(ok)


def check_minority_demotion():
    h = _cap()
    try:
        ml.measure_roll_at_extraction("2024-12-18", "ES", 8,
                                      _measure=lambda *a: _rep_rival_leads(),
                                      extracted="ESZ4")
    finally:
        ml.log.removeHandler(h)
    errs = [m for l, m in h.records if l >= logging.ERROR]
    ok_demoted = not errs and "already extracted the leader" in h.text()
    h2 = _cap()
    try:
        ml.measure_roll_at_extraction("2024-12-18", "ES", 8,
                                      _measure=lambda *a: _rep_rival_leads())
    finally:
        ml.log.removeHandler(h2)
    errs2 = [m for l, m in h2.records if l >= logging.ERROR]
    ok_kept = any("MINORITY CONTRACT" in m for m in errs2)
    print("(6) minority error demoted to info when the leader was extracted (%s); "
          "loud error preserved otherwise (%s)" % (ok_demoted, ok_kept))
    return bool(ok_demoted and ok_kept)


def check_surface():
    sig = inspect.signature(ml.extract_sessions)
    ok_lib = ("contract_rule" in sig.parameters
              and sig.parameters["contract_rule"].default == "calendar")
    import run_analysis as ra
    a = ra.parse_args(["--source", "demo"])
    ok_cli = getattr(a, "contract_rule", None) == "activity"
    src = open("run_paper_replication.sh").read()
    ok_sh = (src.count('--contract-rule "$CONTRACT_RULE"') == 3
             and 'CONTRACT_RULE="activity"' in src
             and "select_contract('ES'" in src)
    print("(7) extract_sessions(contract_rule='calendar' lib default) (%s); run_analysis "
          "--contract-rule DEFAULTS to activity (%s); driver passes the rule to all 3 "
          "extraction calls and the QC contract derivation (%s)" % (ok_lib, ok_cli, ok_sh))
    return bool(ok_lib and ok_cli and ok_sh)


def main():
    checks = [check_calendar_untouched, check_out_of_window_free, check_override,
              check_confirmation, check_failure_falls_back, check_minority_demotion,
              check_surface]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("contract-rule checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    sys.exit(main())
