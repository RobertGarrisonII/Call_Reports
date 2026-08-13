#!/usr/bin/env python3
"""validate_release_csv.py -- check an uploaded release schedule BEFORE it enters the calendar.

Every failure this catches is SILENT downstream, which is why the check exists as a separate gate
rather than a warning inside the loader:

  * an UNMAPPED release name is skipped by load_release_schedule without an error, so a typo in one
    row quietly shrinks the sample and nothing in the run reports it;
  * a date on a NON-TRADING day has no session to extract, so the event yields an empty frame;
  * a release time OUTSIDE the extraction window (the 08:30 macro block, or OPEX at 09:30) produces an
    empty pre-window -> transmission = NaN -> the event vanishes from the surface;
  * a DUPLICATE (date, category) collapses in the loader's own dedup, so the row count you uploaded is
    not the row count you get;
  * a date that already carries a DIFFERENT release time collides: cross_event_onset builds one session
    per DATE and takes the first matching row, so the onset actually used is arbitrary.

Usage:
    python validate_release_csv.py my_dates.csv
    python validate_release_csv.py my_dates.csv --pre-steps 600 --post-steps 600
    python validate_release_csv.py my_dates.csv --quiet     # exit code only (0 clean, 1 problems)

Accepted columns (case-insensitive; extra columns such as `note` are ignored):
    date        required, any pandas-parseable date (YYYY-MM-DD recommended)
    category    the category code directly (JOLTS, NEW_HOME, UMICH, ISM_MFG, ...)
      -- or --
    release     a release NAME that maps via market_shocks.RELEASE_NAME_TO_CATEGORY
    time_et     release time, ET ('10:00', '8:30 AM', '2:00 PM'). Optional; falls back to
                DEFAULT_RELEASE_TIME_ET for the category.
"""
import argparse
import sys

import pandas as pd

import market_shocks as mk

SESSION_OPEN, SESSION_CLOSE = "09:30", "16:00"
_MIN_BARS = 20                       # event_onset_estimate returns NaN below this on either side


def _resolve_category(row):
    """Category code for a row, or None. Mirrors load_release_schedule's own resolution order."""
    cat = row.get("category")
    cat = str(cat).strip() if cat is not None and str(cat).strip().lower() != "nan" else ""
    if cat:
        return cat.upper() if cat.upper() in mk.DEFAULT_RELEASE_TIME_ET or cat.isupper() else cat
    name = row.get("release") or row.get("name") or ""
    return mk._name_to_category(str(name).strip()) or None


def _minutes(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def validate(path, pre_steps=600, post_steps=600):
    """Returns (ok, rows, issues). `issues` are (severity, row_number, message) with severity in
    {'ERROR', 'WARN'}; ERROR means the row will not survive ingest or will silently produce NaN."""
    issues, rows = [], []
    try:
        raw = pd.read_csv(path)
    except Exception as ex:
        return False, [], [("ERROR", 0, "could not read CSV: %r" % ex)]
    recs = [{str(k).strip().lower(): v for k, v in r.items()} for r in raw.to_dict("records")]
    if not recs:
        return False, [], [("ERROR", 0, "file has no data rows")]
    have = set().union(*(set(r) for r in recs))
    if "date" not in have:
        issues.append(("ERROR", 0, "missing required column 'date' (found: %s)" % ", ".join(sorted(have))))
    if not ({"category", "release", "name"} & have):
        issues.append(("ERROR", 0, "need a 'category' column (or 'release'/'name' to map from)"))
    if any(s == "ERROR" for s, _, _ in issues):
        return False, [], issues

    open_m, close_m = _minutes(SESSION_OPEN), _minutes(SESSION_CLOSE)
    seen = {}
    for i, r in enumerate(recs, start=2):                      # +2: header is line 1
        d_raw = r.get("date")
        try:
            d = pd.Timestamp(d_raw).date()
        except Exception:
            issues.append(("ERROR", i, "date %r does not parse (placeholder row left in the file?)" % d_raw))
            continue
        cat = _resolve_category(r)
        if not cat:
            issues.append(("ERROR", i, "release/category %r maps to nothing -- load_release_schedule "
                                       "SKIPS this row silently. See the category list below."
                           % (r.get("category") or r.get("release") or r.get("name"))))
            continue
        if cat not in mk.DEFAULT_RELEASE_TIME_ET:
            issues.append(("WARN", i, "category %r is not in DEFAULT_RELEASE_TIME_ET; it will load but "
                                      "has no fallback time" % cat))
        t = mk._norm_time(r.get("time_et") or r.get("time")) or mk.DEFAULT_RELEASE_TIME_ET.get(cat, "")
        fatal = False
        if not t:
            issues.append(("WARN", i, "%s %s has no time -> treated as an overnight/open-to-open event"
                           % (d, cat)))
        else:
            om = _minutes(t)
            pre_bars, post_bars = om * 60 - open_m * 60, close_m * 60 - om * 60
            if pre_bars < _MIN_BARS or post_bars < _MIN_BARS:
                issues.append(("ERROR", i, "%s %s at %s ET is outside the %s-%s extraction window "
                                           "(pre_bars=%d, post_bars=%d) -> transmission=NaN, event "
                                           "silently dropped" % (d, cat, t, SESSION_OPEN, SESSION_CLOSE,
                                                                 max(pre_bars, 0), max(post_bars, 0))))
                fatal = True
            elif pre_bars < pre_steps or post_bars < post_steps:
                issues.append(("WARN", i, "%s %s at %s ET gives a SHORT window (pre_bars=%d/%d, "
                                          "post_bars=%d/%d) -- it will estimate on fewer bars"
                               % (d, cat, t, pre_bars, pre_steps, post_bars, post_steps)))
        if not mk.is_trading_day(d):
            issues.append(("ERROR", i, "%s is not an NYSE trading day (weekend/holiday) -- no session "
                                       "exists to extract" % d))
            continue
        if fatal:                                   # time is unusable -> the row would not survive
            continue
        key = (d.isoformat(), cat)
        if key in seen:
            issues.append(("ERROR", i, "duplicate %s %s (also line %d) -- the loader dedups on "
                                       "(date, category), so one is discarded" % (d, cat, seen[key])))
            continue
        seen[key] = i
        rows.append({"date": d.isoformat(), "category": cat, "time_et": t, "line": i})

    # cross-check against the calendar already in place: same date, different onset time
    if rows:
        lo = min(r["date"] for r in rows)
        hi = max(r["date"] for r in rows)
        try:
            existing = mk.shocks_frame(include_scheduled=True, scheduled_categories=None, start=lo, end=hi)
            existing = existing[existing["impact_blind"]] if "impact_blind" in existing else existing
            by_date = {}
            for _, e in existing.iterrows():
                ds = pd.Timestamp(e["date"]).strftime("%Y-%m-%d")
                et = str(e.get("event_et") or "").strip()
                if et:
                    by_date.setdefault(ds, set()).add((str(e["category"]), et))
            for r in rows:
                other = {(c, t) for c, t in by_date.get(r["date"], set())
                         if c != r["category"] and t and t != r["time_et"]}
                if other:
                    issues.append(("WARN", r["line"], "%s %s@%s shares a session with %s -- one session "
                                                      "per DATE, so the onset used is whichever row sorts "
                                                      "first" % (r["date"], r["category"], r["time_et"],
                                                                 ", ".join("%s@%s" % x for x in sorted(other)))))
        except Exception:
            pass

    ok = not any(s == "ERROR" for s, _, _ in issues)
    return ok, rows, issues


def _print_report(path, ok, rows, issues):
    errs = [x for x in issues if x[0] == "ERROR"]
    warns = [x for x in issues if x[0] == "WARN"]
    print("release-schedule validation -- %s" % path)
    print("  rows that would load : %d" % len(rows))
    print("  errors               : %d" % len(errs))
    print("  warnings             : %d" % len(warns))
    if errs or warns:
        print()
        for sev, line, msg in sorted(issues, key=lambda x: (x[1], x[0])):
            print("  [%-5s] line %-4s %s" % (sev, line or "-", msg))
    if errs:
        print("\n  valid category codes:")
        cats = sorted(mk.DEFAULT_RELEASE_TIME_ET)
        for i in range(0, len(cats), 6):
            print("    " + "  ".join("%-12s" % c for c in cats[i:i + 6]))
        print("\n  RTH-valid release times (09:30-16:00 extraction): 10:00 and 14:00 load cleanly;")
        print("  08:15 / 08:30 / 09:30 do NOT -- widen the extraction window first.")
    if rows:
        by_cat = {}
        for r in rows:
            by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
        print("\n  by category: " + ", ".join("%s=%d" % kv for kv in sorted(by_cat.items())))
        print("  span       : %s .. %s" % (min(r["date"] for r in rows), max(r["date"] for r in rows)))
    print("\n  %s" % ("CLEAN -- safe to ingest." if ok else "NOT CLEAN -- fix the errors above first."))
    if ok and rows:
        print("\n  to use it:")
        print("      import market_shocks as mk")
        print("      mk.load_release_schedule(%r)" % path)
        print("    then add the categories to the run, e.g.")
        print("      --categories FOMC,ISM_MFG,ISM_SVC,CONF_BOARD,%s"
              % ",".join(sorted({r["category"] for r in rows})))


def main(argv=None):
    ap = argparse.ArgumentParser(description="validate a release-schedule CSV before ingest")
    ap.add_argument("path")
    ap.add_argument("--pre-steps", type=int, default=600)
    ap.add_argument("--post-steps", type=int, default=600)
    ap.add_argument("--quiet", action="store_true", help="exit code only")
    a = ap.parse_args(argv)
    ok, rows, issues = validate(a.path, a.pre_steps, a.post_steps)
    if not a.quiet:
        _print_report(a.path, ok, rows, issues)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
