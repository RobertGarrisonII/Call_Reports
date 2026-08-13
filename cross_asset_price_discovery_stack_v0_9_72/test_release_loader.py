"""test_release_loader.py -- guards the agency-general release-schedule loader.

(A) load_release_schedule maps agency release NAMES to category codes (GDP/BEA, Retail/Census, ISM,
    FOMC) and a `category` column is used as-is; release times come from a time column when present,
    else the per-category default. (B) loaded releases flow through scheduled_events / release_times
    once registered, and clear_loaded() reverts the calendar. (C) the iCal parser extracts
    DTSTART/SUMMARY and reuses the same mapping. (D) duplicates collapse on (date, category).
"""
import sys
import pandas as pd
import market_shocks as m

def main():
    ok = True
    m.clear_loaded()

    # (A) name + category mapping, time handling
    df = pd.DataFrame([
        {"date": "2025-01-30", "release": "Gross Domestic Product, 4th quarter (Advance)"},   # BEA -> GDP, 08:30 default
        {"date": "2025-01-16", "release": "Advance Monthly Retail Trade Survey"},              # Census -> RETAIL
        {"date": "2025-02-03", "release": "ISM Manufacturing PMI", "time": "10:00 AM"},         # ISM, explicit time
        {"date": "2025-01-15", "category": "CPI", "time_et": "8:30"},                           # category as-is
        {"date": "2025-03-19", "release": "FOMC statement"},                                    # -> FOMC, 14:00 default
        {"date": "2025-01-09", "release": "Some Obscure Index Nobody Trades"},                  # unmapped -> skipped
    ])
    ev = m.load_release_schedule(df)
    by = {e["category"]: e for e in ev}
    a_map = set(by) == {"GDP", "RETAIL", "ISM_MFG", "CPI", "FOMC"}                              # obscure one skipped
    a_time = (by["GDP"]["event_et"] == "08:30" and by["ISM_MFG"]["event_et"] == "10:00"
              and by["CPI"]["event_et"] == "08:30" and by["FOMC"]["event_et"] == "14:00")
    a_class = by["FOMC"]["event_class"] == "MONETARY" and by["GDP"]["event_class"] == "MACRO"
    print("(A) mapped=%s (obscure skipped) | times GDP/ISM/CPI/FOMC=%s | classes=%s"
          % (sorted(by), a_time, a_class))
    ok &= a_map and a_time and a_class

    # (B) flows through the accessors once registered; clear reverts
    got = m.scheduled_events(start="2025-01-01", end="2025-12-31", categories=("GDP", "RETAIL"))
    b_flow = {e["category"] for e in got} == {"GDP", "RETAIL"} and len(got) == 2
    rt = m.release_times(categories=("ISM_MFG",), include_scheduled=True,
                         scheduled_categories=("ISM_MFG",), start="2025-01-01", end="2025-12-31")
    b_rt = any(v.hour == 10 and v.minute == 0 for v in rt.values())
    none_all = {e["category"] for e in m.scheduled_events(start="2025-01-01", end="2025-12-31", categories=None)}
    b_none = {"FOMC", "OPEX", "GDP", "RETAIL", "ISM_MFG", "CPI"}.issubset(none_all)             # None => all incl loaded
    m.clear_loaded()
    # clear_loaded() reverts LOADER-ONLY categories. ISM_MFG is now also RULE-GENERATED (the 10:00
    # backbone), so it correctly persists after a clear -- only GDP/RETAIL, which exist solely by
    # ingest, disappear.
    after = list(m.scheduled_events(start="2025-01-01", end="2025-12-31", categories=None))
    b_revert = (all(e["category"] not in {"GDP", "RETAIL"} for e in after)
                and any(e["category"] == "ISM_MFG" for e in after))
    print("(B) loaded flow=%s | ISM release 10:00=%s | None=>all=%s | clear reverts loader-only=%s"
          % (b_flow, b_rt, b_none, b_revert))
    ok &= b_flow and b_rt and b_none and b_revert

    # (B2) an INGESTED date must SUPPRESS the generated one for that category-month, otherwise a
    # rescheduled release double-counts (both dates survive the (date, category) dedup).
    m.clear_loaded()
    gen_may = [d.isoformat() for d in m.rule_release_days("ISM_MFG", 2025, 2025) if d.month == 5]
    m.load_release_schedule([{"date": "2025-05-15", "release": "ISM Manufacturing PMI", "time": "10:00"}])
    may = [e["date"] for e in m.scheduled_events(start="2025-05-01", end="2025-05-31",
                                                 categories=("ISM_MFG",))]
    b2 = (len(may) == 1 and may[0] == "2025-05-15" and gen_may == ["2025-05-01"])
    print("(B2) ingest overrides generation for that month: generated=%s loaded=2025-05-15 -> kept %s = %s"
          % (gen_may, may, b2))
    ok &= b2
    m.clear_loaded()

    # (C) iCal parsing
    ics = ("BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:Consumer Price Index - May 2025\n"
           "DTSTART:20250611\nEND:VEVENT\nBEGIN:VEVENT\nSUMMARY:The Employment Situation - May 2025\n"
           "DTSTART:20250606\nEND:VEVENT\nEND:VCALENDAR\n")
    m.clear_loaded()
    iev = m.load_ics(ics)
    c_ics = {e["category"] for e in iev} == {"CPI", "NFP"} and {e["date"] for e in iev} == {"2025-06-11", "2025-06-06"}
    c_time = all(e["event_et"] == "08:30" for e in iev)
    print("(C) iCal -> CPI+NFP on right dates=%s | default 08:30=%s" % (c_ics, c_time))
    ok &= c_ics and c_time

    # (D) dedup on (date, category)
    m.clear_loaded()
    dd = m.load_release_schedule(pd.DataFrame([
        {"date": "2025-07-30", "release": "FOMC"}, {"date": "2025-07-30", "category": "FOMC"}]))
    d_dedup = len(dd) == 1
    print("(D) duplicate (date,category) collapses to one=%s" % d_dedup)
    ok &= d_dedup

    m.clear_loaded()
    print("release-loader checks ->", ok)
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
