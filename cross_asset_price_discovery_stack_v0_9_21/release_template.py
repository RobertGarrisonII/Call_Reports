#!/usr/bin/env python3
"""release_template.py -- generate a fill-in template for authoritative release dates, and VALIDATE a
filled one before it is ingested.

Why a validator and not just a template. A wrong release date does not raise. It produces an onset
that splits the session in the wrong place -- or, if it lands on a non-trading day or outside the
09:30-16:00 extraction window, an empty pre-window -- and `event_onset_estimate` returns
transmission=NaN, so the event silently vanishes from the surface. A mistyped year would be worse: a
valid session, a valid-looking number, and an observation centred on nothing. Every check below exists
to convert one of those silent failures into a loud one BEFORE the dates become load-bearing.

Two row kinds in the template:
  NEEDS_DATE -- 10:00 releases with no publisher rule (JOLTS, NEW_HOME, UMICH). Date column blank;
                fill it from the agency calendar.
  VERIFY     -- the rule-generated cluster (ISM_MFG, ISM_SVC, CONF_BOARD), pre-filled with the
                generated date so it can be checked against the source and corrected in place. A row
                left unchanged is a no-op (ingest of the same date re-states what generation produced);
                a corrected row overrides generation for that category-month.

The 08:30 block (CPI, NFP, PPI, RETAIL, GDP, PCE, CLAIMS, DURABLE, HOUSING, TRADE) and OPEX (09:30)
are deliberately ABSENT: under an RTH extraction window their pre-window is empty and every such event
is dropped. Adding them here would let a large, useless ingest look successful.

CLI:
    python release_template.py make --start 2020-01-01 --end 2026-07-23 --out template.csv
    python release_template.py validate filled.csv
    python release_template.py validate filled.csv --pre-steps 600 --post-steps 600
"""
import argparse
import datetime as dt
import sys

import pandas as pd

import market_shocks as mk

# Template columns. `date`, `category`, `time_et` are read by load_release_schedule; the rest are
# entry aids and are ignored by the loader (_coerce_rows keeps only the keys it needs).
COLUMNS = ["date", "category", "time_et", "ref_month", "status", "notes"]

# 10:00 releases with no clean publisher rule -> (releases per month, note)
NEEDS_DATE = {
    "JOLTS":    (1, "Job Openings and Labor Turnover Survey"),
    "NEW_HOME": (1, "New Residential Sales (Census/HUD)"),
    "UMICH":    (2, "U. Michigan Consumer Sentiment (preliminary, then final)"),
}
UMICH_PARTS = ["preliminary", "final"]

RTH_OPEN_S = 9 * 3600 + 30 * 60
RTH_CLOSE_S = 16 * 3600
EARLY_CLOSE_S = 13 * 3600
MIN_BARS = 20                     # event_onset_estimate's hard floor: fewer -> transmission = NaN


# ── NYSE early closes (1pm). A 10:00 onset is unaffected, but the validator must know for any other
# ── time, or it would pass an onset that has no post-window on a half day.
def early_close(d):
    """True when the NYSE closes at 13:00 ET on date `d`."""
    d = d if isinstance(d, dt.date) else pd.Timestamp(d).date()
    if not mk.is_trading_day(d):
        return False
    thanksgiving = mk._nth_weekday(d.year, 11, 3, 4)
    if d == thanksgiving + dt.timedelta(days=1):                 # Friday after Thanksgiving
        return True
    if d == dt.date(d.year, 12, 24) and d.weekday() < 5:         # Christmas Eve on a weekday
        return True
    jul4 = dt.date(d.year, 7, 4)                                 # July 3, when the 4th is a weekday
    if jul4.weekday() < 5 and d == jul4 - dt.timedelta(days=1) and d.weekday() < 5:
        return True
    return False


def session_close_s(d):
    return EARLY_CLOSE_S if early_close(d) else RTH_CLOSE_S


def _months(start, end):
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    out, y, m = [], lo.year, lo.month
    while (y, m) <= (hi.year, hi.month):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def make_template(start="2020-01-01", end=None, include_verify=True):
    """Build the fill-in template as a DataFrame, one row per expected release."""
    end = end or dt.date.today().isoformat()
    rows = []
    for (y, m) in _months(start, end):
        ref = "%04d-%02d" % (y, m)
        for cat, (per_month, note) in NEEDS_DATE.items():
            for k in range(per_month):
                part = (" -- %s" % UMICH_PARTS[k]) if cat == "UMICH" else ""
                rows.append(dict(date="", category=cat, time_et="10:00", ref_month=ref,
                                 status="NEEDS_DATE", notes=note + part))
    if include_verify:
        lo, hi = pd.Timestamp(start), pd.Timestamp(end)
        for cat in mk.RULE_RELEASE_CATEGORIES:
            _rule, annot = mk._RULE_RELEASES[cat]
            for d in mk.rule_release_days(cat, lo.year, hi.year):
                if not (lo.date() <= d <= hi.date()):
                    continue
                rows.append(dict(date=d.isoformat(), category=cat, time_et="10:00",
                                 ref_month="%04d-%02d" % (d.year, d.month), status="VERIFY",
                                 notes="rule-generated: %s -- confirm against the publisher calendar" % annot))
    df = pd.DataFrame(rows, columns=COLUMNS)
    return df.sort_values(["ref_month", "category", "date"], kind="stable").reset_index(drop=True)


def _s(v):
    """String-coerce a cell, treating NaN/NaT/None as empty. pandas reads a blank CSV cell as float
    NaN, which is TRUTHY -- so `str(v or "")` yields the string "nan" and a blank date silently reaches
    the parser. This is the guard for that."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v).strip()


def _parse_time_s(t):
    t = _s(t)
    if not t:
        return None
    for fmt in ("%H:%M", "%H:%M:%S"):
        try:
            x = dt.datetime.strptime(t, fmt).time()
            return x.hour * 3600 + x.minute * 60 + x.second
        except ValueError:
            continue
    return None


def validate(source, pre_steps=600, post_steps=600):
    """Validate a filled template. Returns a report dict; `errors` are rows that would silently
    produce a NaN/mis-centred observation, `warnings` are rows that load but with a truncated window."""
    rows = mk._coerce_rows(source)
    errors, warns, ok_rows = [], [], []
    seen, by_date = {}, {}
    known = set(mk.DEFAULT_RELEASE_TIME_ET)

    for i, r in enumerate(rows, start=2):                        # line 2 = first data row under a header
        date_raw = _s(r.get("date"))
        cat = _s(r.get("category"))
        status = _s(r.get("status"))
        tm = _s(r.get("time_et")) or _s(r.get("time"))
        if not date_raw:
            if status == "NEEDS_DATE":
                continue                                         # an unfilled template row: skip silently
            errors.append((i, cat, "", "missing date"))
            continue
        if not cat:
            errors.append((i, "", date_raw, "missing category"))
            continue
        if cat not in known:
            errors.append((i, cat, date_raw, "unknown category (not in DEFAULT_RELEASE_TIME_ET)"))
            continue
        try:
            ts_d = pd.Timestamp(date_raw)
            if pd.isna(ts_d):
                raise ValueError("NaT")
            d = ts_d.date()
        except Exception:
            errors.append((i, cat, date_raw, "unparseable date (want YYYY-MM-DD)"))
            continue
        if d.isoformat() != date_raw:
            warns.append((i, cat, date_raw, "date normalised to %s -- confirm the intended day" % d.isoformat()))
        if not mk.is_trading_day(d):
            errors.append((i, cat, date_raw, "not an NYSE trading day -> no session to extract"))
            continue
        ts = _parse_time_s(tm)
        if ts is None:
            errors.append((i, cat, date_raw, "missing/unparseable time_et (want HH:MM ET)"))
            continue
        close_s = session_close_s(d)
        pre_avail, post_avail = ts - RTH_OPEN_S, close_s - ts
        if pre_avail < MIN_BARS or post_avail < MIN_BARS:
            why = ("onset %s is at/before the 09:30 open" % tm) if pre_avail < MIN_BARS else \
                  ("onset %s leaves <%ds before the %s close" % (tm, MIN_BARS, "13:00" if close_s == EARLY_CLOSE_S else "16:00"))
            errors.append((i, cat, date_raw, why + " -> transmission=NaN, event dropped"))
            continue
        if pre_avail < pre_steps or post_avail < post_steps:
            warns.append((i, cat, date_raw, "window truncated (pre=%ds post=%ds vs requested %d/%d%s)"
                          % (pre_avail, post_avail, pre_steps, post_steps,
                             "; 13:00 early close" if close_s == EARLY_CLOSE_S else "")))
        key = (d.isoformat(), cat)
        if key in seen:
            errors.append((i, cat, date_raw, "duplicate (date, category) -- collapses with line %d" % seen[key]))
            continue
        seen[key] = i
        by_date.setdefault(d.isoformat(), set()).add(tm)
        ok_rows.append(dict(date=d.isoformat(), category=cat, time_et=tm))

    collisions = {d: sorted(t) for d, t in by_date.items() if len(t) > 1}
    return dict(n_rows=len(rows), n_valid=len(ok_rows), errors=errors, warnings=warns,
                collisions=collisions, rows=ok_rows)


def report_lines(rep):
    L = ["release-schedule validation: %d rows read, %d valid, %d errors, %d warnings"
         % (rep["n_rows"], rep["n_valid"], len(rep["errors"]), len(rep["warnings"]))]
    for tag, items in (("ERROR", rep["errors"]), ("warn", rep["warnings"])):
        for (ln, cat, date_raw, why) in items[:40]:
            L.append("  %-5s line %-5d %-11s %-12s %s" % (tag, ln, cat, date_raw, why))
        if len(items) > 40:
            L.append("  %-5s ... and %d more" % (tag, len(items) - 40))
    if rep["collisions"]:
        L.append("  note: %d date(s) carry more than one release TIME -- one session per date resolves"
                 % len(rep["collisions"]))
        L.append("        to a single onset, so choose deliberately (see market_shocks.release_collisions):")
        for d, ts in sorted(rep["collisions"].items())[:10]:
            L.append("        %s  %s" % (d, ", ".join(ts)))
    L.append("  -> %s" % ("READY TO INGEST" if not rep["errors"] else "FIX ERRORS BEFORE INGEST"))
    return L


def _main(argv=None):
    ap = argparse.ArgumentParser(description="make / validate an authoritative release-date template")
    sub = ap.add_subparsers(dest="cmd", required=True)
    mk_ = sub.add_parser("make")
    mk_.add_argument("--start", default="2020-01-01")
    mk_.add_argument("--end", default=None)
    mk_.add_argument("--out", default="release_schedule_template.csv")
    mk_.add_argument("--no-verify-rows", action="store_true",
                     help="omit the pre-filled rule-generated rows; emit only the blank NEEDS_DATE rows")
    va = sub.add_parser("validate")
    va.add_argument("path")
    va.add_argument("--pre-steps", type=int, default=600)
    va.add_argument("--post-steps", type=int, default=600)
    a = ap.parse_args(argv)

    if a.cmd == "make":
        df = make_template(a.start, a.end, include_verify=not a.no_verify_rows)
        df.to_csv(a.out, index=False)
        n_fill = int((df["status"] == "NEEDS_DATE").sum())
        print("wrote %s: %d rows (%d to fill, %d pre-filled to verify)"
              % (a.out, len(df), n_fill, len(df) - n_fill))
        return 0
    rep = validate(a.path, a.pre_steps, a.post_steps)
    print("\n".join(report_lines(rep)))
    return 0 if not rep["errors"] else 1


if __name__ == "__main__":
    sys.exit(_main())
