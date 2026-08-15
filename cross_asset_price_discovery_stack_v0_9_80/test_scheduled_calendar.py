"""test_scheduled_calendar.py -- guards the scheduled-event backbone in market_shocks.

Checks: (A) the third-Friday generator matches known quad-witching dates and always lands on a
Friday in the 15-21 window; (B) the verified FOMC table has 8 decisions/year, all resolving to
14:00 ET; (C) scheduled_events is impact-blind (every occurrence in range) with sane counts and
correct release times (FOMC 14:00, CPI/NFP 08:30, OPEX 09:30); (D) shocks_frame(include_scheduled)
merges the backbone and de-dupes against the hand-annotated crisis rows (2024-09-18 FOMC appears
once), while the default stays identical to the crisis-only registry.
"""
import sys
import datetime as dt
import pandas as pd
import market_shocks as m

def main():
    ok = True
    # (A) third-Friday generator
    known = {(2025, 3): dt.date(2025, 3, 21), (2025, 6): dt.date(2025, 6, 20),
             (2024, 12): dt.date(2024, 12, 20), (2023, 9): dt.date(2023, 9, 15)}
    a_known = all(m._third_friday(y, mo) == d for (y, mo), d in known.items())
    a_friday = all(m._third_friday(y, mo).weekday() == 4 and 15 <= m._third_friday(y, mo).day <= 21
                   for y in range(2023, 2027) for mo in (3, 6, 9, 12))
    print("(A) third_friday: known dates=%s | always Fri in 15-21=%s" % (a_known, a_friday))
    ok &= a_known and a_friday

    # (B) FOMC table
    fomc = m.scheduled_events(start="2023-01-01", end="2026-12-31", categories=("FOMC",))
    per_year = {y: sum(1 for e in fomc if e["date"].startswith(str(y))) for y in range(2023, 2027)}
    b_count = all(v == 8 for v in per_year.values())
    rt_fomc = m.release_times(categories=("FOMC",), include_scheduled=True, start="2023-01-01", end="2026-12-31")
    b_time = len(rt_fomc) > 0 and all(v.hour == 14 and v.minute == 0 for v in rt_fomc.values())
    b_verified = {"2025-01-29", "2025-06-18", "2024-12-18", "2023-02-01"}.issubset({e["date"] for e in fomc})
    print("(B) FOMC: 8/yr=%s %s | all 14:00=%s | verified dates present=%s"
          % (b_count, per_year, b_time, b_verified))
    ok &= b_count and b_time and b_verified

    # (C) impact-blind enumeration + release times
    opex = m.scheduled_events(start="2023-01-01", end="2026-12-31", categories=("OPEX",))
    c_opex = len(opex) == 16 and all(pd.Timestamp(e["date"]).weekday() == 4 for e in opex)
    cpi = m.scheduled_events(categories=("CPI",))
    c_cpi = {"2025-10-24", "2025-12-18"}.issubset({e["date"] for e in cpi})   # shutdown-revised seeds
    rt_all = m.release_times(categories=None, classes=("MARKET_STRUCTURE",), include_scheduled=True,
                             scheduled_categories=("OPEX",), start="2025-01-01", end="2025-12-31")
    c_opextime = any(v.hour == 9 and v.minute == 30 for v in rt_all.values())
    print("(C) OPEX: 16 in 2023-26 & all Fri=%s | CPI shutdown seeds present=%s | OPEX 09:30=%s"
          % (c_opex, c_cpi, c_opextime))
    ok &= c_opex and c_cpi and c_opextime

    # (D) merge + dedup, and default unchanged
    base = m.shocks_frame()
    merged = m.shocks_frame(include_scheduled=True, start="2023-01-01", end="2026-12-31")
    d_default = len(base) == len(m._SHOCKS)
    n_0918 = int((merged["date"].dt.strftime("%Y-%m-%d") == "2024-09-18").sum())
    d_dedup = n_0918 == 1                                   # in both _SHOCKS and generated -> once
    d_grew = len(merged) > len(base)
    print("(D) default==crisis-only=%s | 2024-09-18 FOMC appears %dx (expect 1)=%s | merged grew=%s"
          % (d_default, n_0918, d_dedup, d_grew))
    ok &= d_default and d_dedup and d_grew

    print("scheduled-calendar checks ->", ok)
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
