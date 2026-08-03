"""test_release_csv_validation.py -- guard for the pre-ingest release-schedule validator.

Each check below plants ONE specific defect that is silent downstream and asserts the validator
catches THAT defect -- not merely that validation runs and returns something. The defects are the ones
this project actually hit: a release time outside the RTH extraction window (empty pre-window ->
transmission NaN -> event vanishes), a date with no trading session, a release name the loader skips
without erroring, and a duplicate that the loader's own dedup silently absorbs.

A clean file must come back clean, and the shipped template must come back DIRTY -- the placeholder
rows are meant to fail loudly rather than be ingested as-is.
"""
import os
import sys
import tempfile
import warnings

import pandas as pd

import market_shocks as mk
import validate_release_csv as V


def write_csv(rows, cols=("date", "category", "time_et")):
    fd, path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    pd.DataFrame(rows, columns=list(cols)).to_csv(path, index=False)
    return path


def errors_for(issues, needle):
    return [m for s, _, m in issues if s == "ERROR" and needle.lower() in m.lower()]


def warns_for(issues, needle):
    return [m for s, _, m in issues if s == "WARN" and needle.lower() in m.lower()]


def main():
    warnings.simplefilter("ignore")
    ok = True
    good = mk.nth_trading_day(2025, 5, 8).isoformat()          # a known-good trading day
    good2 = mk.nth_trading_day(2025, 6, 8).isoformat()

    # (A) a clean file validates and yields exactly its rows
    p = write_csv([{"date": good, "category": "JOLTS", "time_et": "10:00"},
                   {"date": good2, "category": "NEW_HOME", "time_et": "10:00"}])
    a_ok, rows, iss = V.validate(p)
    a = a_ok and len(rows) == 2 and {r["category"] for r in rows} == {"JOLTS", "NEW_HOME"}
    ok &= a
    print("(A) clean file: ok=%s rows=%d -> %s" % (a_ok, len(rows), a))
    os.unlink(p)

    # (B) the 08:30 / 09:30 block -- the failure that silently empties the pre-window
    p = write_csv([{"date": good, "category": "CPI", "time_et": "08:30"},
                   {"date": good2, "category": "OPEX", "time_et": "09:30"}])
    b_ok, rows, iss = V.validate(p)
    b = (not b_ok) and len(errors_for(iss, "outside the 09:30-16:00 extraction window")) == 2 and not rows
    ok &= b
    print("(B) 08:30 and 09:30 flagged as outside the window (2 errors, 0 rows loaded) -> %s" % b)
    os.unlink(p)

    # (C) non-trading days: a weekend and Good Friday 2025 (2025-04-18)
    p = write_csv([{"date": "2025-05-10", "category": "JOLTS", "time_et": "10:00"},      # Saturday
                   {"date": "2025-04-18", "category": "JOLTS", "time_et": "10:00"}])     # Good Friday
    c_ok, rows, iss = V.validate(p)
    c = (not c_ok) and len(errors_for(iss, "not an NYSE trading day")) == 2
    ok &= c
    print("(C) weekend + Good Friday flagged as non-sessions -> %s" % c)
    os.unlink(p)

    # (D) an unmapped release NAME -- load_release_schedule skips it with no error at all
    p = write_csv([{"date": good, "release": "Totally Made Up Index", "time_et": "10:00"}],
                  cols=("date", "release", "time_et"))
    d_ok, rows, iss = V.validate(p)
    d = (not d_ok) and len(errors_for(iss, "maps to nothing")) == 1 and not rows
    ok &= d
    print("(D) unmapped release name caught (loader would skip it silently) -> %s" % d)
    os.unlink(p)

    # (D2) ...while a MAPPED name resolves through the alias table
    p = write_csv([{"date": good, "release": "Job Openings", "time_et": "10:00"}],
                  cols=("date", "release", "time_et"))
    d2_ok, rows, iss = V.validate(p)
    d2 = d2_ok and len(rows) == 1 and rows[0]["category"] == "JOLTS"
    ok &= d2
    print("(D2) mapped alias 'Job Openings' -> JOLTS -> %s" % d2)
    os.unlink(p)

    # (E) duplicate (date, category) -- the loader dedups, so the uploaded count != the loaded count
    p = write_csv([{"date": good, "category": "JOLTS", "time_et": "10:00"},
                   {"date": good, "category": "JOLTS", "time_et": "10:00"}])
    e_ok, rows, iss = V.validate(p)
    e = (not e_ok) and len(errors_for(iss, "duplicate")) == 1 and len(rows) == 1
    ok &= e
    print("(E) duplicate (date, category) caught -> %s" % e)
    os.unlink(p)

    # (F) a short-but-usable window warns rather than errors (it estimates on fewer bars)
    p = write_csv([{"date": good, "category": "JOLTS", "time_et": "09:35"}])
    f_ok, rows, iss = V.validate(p)
    f = f_ok and len(warns_for(iss, "SHORT window")) == 1 and len(rows) == 1
    ok &= f
    print("(F) 09:35 warns (short window) but still loads -> %s" % f)
    os.unlink(p)

    # (G) collision: a date already carrying an FOMC 14:00 gets a WARN when a 10:00 row lands on it
    fomc = [d for d in mk.FOMC_DECISION_DAYS.get(2025, [])]
    coll = "2025-%s" % fomc[2] if len(fomc) > 2 else None
    g = True
    if coll and mk.is_trading_day(pd.Timestamp(coll).date()):
        p = write_csv([{"date": coll, "category": "JOLTS", "time_et": "10:00"}])
        g_ok, rows, iss = V.validate(p)
        g = g_ok and len(warns_for(iss, "shares a session")) == 1
        os.unlink(p)
    print("(G) same-date/different-time collision warns (onset would be arbitrary) -> %s" % g)
    ok &= g

    # (H) the shipped template must NOT validate -- its placeholders are meant to fail loudly
    tmpl = os.path.join(os.path.dirname(os.path.abspath(__file__)), "release_schedule_template.csv")
    h = True
    if os.path.exists(tmpl):
        h_ok, rows, iss = V.validate(tmpl)
        h = (not h_ok) and not rows and len(errors_for(iss, "does not parse")) >= 1
    print("(H) shipped template fails validation as intended (placeholders) -> %s" % h)
    ok &= h

    print("\nrelease-CSV validation checks ->", ok)
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
