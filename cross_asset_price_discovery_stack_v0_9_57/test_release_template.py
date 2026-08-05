"""test_release_template.py -- guard for the release-date template and its pre-ingest validator.

The validator's whole purpose is to convert SILENT failures into loud ones: a date on a holiday, an
onset outside the RTH extraction window, or a duplicate row does not raise anywhere downstream -- it
produces transmission=NaN and the event just disappears from the surface. So this test does not check
that the validator runs; it feeds a CSV containing ONE INSTANCE OF EACH failure mode and asserts every
one is caught, then asserts the clean rows survive and actually ingest.
"""
import os
import sys
import tempfile
import warnings

import pandas as pd

import market_shocks as mk
import release_template as rt

BROKEN = """date,category,time_et,ref_month,status,notes
2024-03-29,JOLTS,10:00,2024-03,NEEDS_DATE,Good Friday -- market closed
2024-11-30,JOLTS,10:00,2024-11,NEEDS_DATE,a Saturday
2025-01-09,NEW_HOME,10:00,2025-01,NEEDS_DATE,Carter day of mourning
2025-03-11,JOLTS,08:30,2025-03,NEEDS_DATE,pre-open onset -- empty pre-window
2025-03-12,UMICH,16:30,2025-03,NEEDS_DATE,after the close
2025-13-45,JOLTS,10:00,2025-04,NEEDS_DATE,unparseable date
2025-04-15,NOT_A_RELEASE,10:00,2025-04,NEEDS_DATE,unknown category
2025-05-06,JOLTS,10:00,2025-05,NEEDS_DATE,valid
2025-05-06,JOLTS,10:00,2025-05,NEEDS_DATE,duplicate of the line above
2025-06-10,NEW_HOME,,2025-06,NEEDS_DATE,missing time
2024-11-29,UMICH,14:30,2024-11,NEEDS_DATE,half day (1pm close) -- onset is AFTER the close
2025-07-08,UMICH,10:00,2025-07,NEEDS_DATE,valid
2025-07-08,JOLTS,14:00,2025-07,NEEDS_DATE,valid but a second TIME on the same date -> collision
2025-05-07,UMICH,09:35,2025-05,NEEDS_DATE,valid but only 300s of pre-window -> truncation warning
,JOLTS,10:00,2025-08,NEEDS_DATE,unfilled row -- must be skipped silently
"""

# what each line number must be flagged as: (line, kind, substring of the reason)
EXPECT = [
    (2,  "error", "trading day"),
    (3,  "error", "trading day"),
    (4,  "error", "trading day"),
    (5,  "error", "before the 09:30 open"),
    (6,  "error", "close"),
    (7,  "error", "unparseable"),
    (8,  "error", "unknown category"),
    (10, "error", "duplicate"),
    (11, "error", "time_et"),
    (12, "error", "13:00 close"),          # half day: 14:30 is past the early close
    (15, "warn",  "truncated"),            # 09:35 leaves only 300s of pre-window
]


def main():
    warnings.simplefilter("ignore")
    ok = True
    tmp = tempfile.mkdtemp()

    # (A) template structure: the loader-visible columns exist, NEEDS_DATE rows are blank, VERIFY rows
    #     are pre-filled with real trading days, and no pre-open release sneaks in.
    df = rt.make_template("2024-01-01", "2024-12-31")
    need = df[df["status"] == "NEEDS_DATE"]
    ver = df[df["status"] == "VERIFY"]
    cols_ok = list(df.columns) == rt.COLUMNS
    blank_ok = bool((need["date"] == "").all()) and len(need) == (1 + 1 + 2) * 12      # JOLTS+NEW_HOME+UMICHx2
    ver_ok = len(ver) == 3 * 12 and all(mk.is_trading_day(pd.Timestamp(d).date()) for d in ver["date"])
    times_ok = set(df["time_et"]) == {"10:00"}
    a_ok = cols_ok and blank_ok and ver_ok and times_ok
    ok &= a_ok
    print("(A) template: cols=%s blank NEEDS_DATE=%d verify=%d all-trading-days=%s all-10:00=%s -> %s"
          % (cols_ok, len(need), len(ver), ver_ok, times_ok, a_ok))

    # (B) THE detection: every planted failure mode must be caught, on the right line
    p = os.path.join(tmp, "broken.csv")
    open(p, "w").write(BROKEN)
    rep = rt.validate(p)
    err_by_line = {ln: why for (ln, _c, _d, why) in rep["errors"]}
    warn_by_line = {ln: why for (ln, _c, _d, why) in rep["warnings"]}
    b_ok = True
    for ln, kind, frag in EXPECT:
        pool = err_by_line if kind == "error" else warn_by_line
        hit = ln in pool and frag.lower() in pool[ln].lower()
        if not hit:
            b_ok = False
            print("    MISSED line %d (%s, want %r) -- got %r" % (ln, kind, frag, pool.get(ln)))
    ok &= b_ok
    print("(B) all %d planted failure modes caught on the right line -> %s" % (len(EXPECT), b_ok))

    # (C) the unfilled NEEDS_DATE row is skipped silently (not an error), and valid rows survive
    valid_dates = {r["date"] for r in rep["rows"]}
    c_ok = (valid_dates == {"2025-05-06", "2025-07-08", "2025-05-07"}
            and not any(ln == 16 for (ln, _c, _d, _w) in rep["errors"]))
    ok &= c_ok
    print("(C) unfilled row skipped, valid rows kept (%s) -> %s" % (sorted(valid_dates), c_ok))

    # (D) collision surfaced: 2025-07-08 carries a 10:00 and a 14:00
    d_ok = rep["collisions"].get("2025-07-08") == ["10:00", "14:00"]
    ok &= d_ok
    print("(D) multi-time date surfaced: %s -> %s" % (rep["collisions"], d_ok))

    # (E) a CLEAN file passes and actually ingests through load_release_schedule
    clean = os.path.join(tmp, "clean.csv")
    pd.DataFrame([{"date": "2025-03-11", "category": "JOLTS", "time_et": "10:00"},
                  {"date": "2025-03-25", "category": "NEW_HOME", "time_et": "10:00"}]).to_csv(clean, index=False)
    rc = rt.validate(clean)
    mk.clear_loaded()
    loaded = mk.load_release_schedule(clean)
    cats = {e["category"] for e in loaded}
    ets = {e["event_et"] for e in loaded}
    e_ok = (not rc["errors"] and rc["n_valid"] == 2 and len(loaded) == 2
            and cats == {"JOLTS", "NEW_HOME"} and ets == {"10:00"})
    mk.clear_loaded()
    ok &= e_ok
    print("(E) clean file validates (%d errors) and ingests (%d events, %s) -> %s"
          % (len(rc["errors"]), len(loaded), sorted(cats), e_ok))

    # (F) report renders and states the verdict
    lines = "\n".join(rt.report_lines(rep))
    f_ok = "FIX ERRORS BEFORE INGEST" in lines and "READY TO INGEST" in "\n".join(rt.report_lines(rc))
    ok &= f_ok
    print("(F) verdict lines render for both broken and clean -> %s" % f_ok)

    print("\nrelease-template checks ->", ok)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
