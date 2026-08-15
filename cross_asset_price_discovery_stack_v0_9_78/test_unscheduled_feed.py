"""test_unscheduled_feed.py -- guards the impact-blind unscheduled feed: selection provenance, the
scheduled-uncertainty stratum (US elections generated), and impact-blind tagging through the frame.

(A) the election generator matches known general-election days and is tagged scheduled_uncertainty /
overnight; (B) shocks_frame exposes selection + impact_blind, with generated scheduled / elections
impact-blind and the hand-curated crisis registry salience-curated; (C) the loader stamps provenance
(news_classified) so a categorized news/event feed enters impact-blind, defaulting to overnight when
no intraday time is given.
"""
import sys
import datetime as dt
import pandas as pd
import market_shocks as m

def main():
    ok = True

    # (A) election generator + provenance
    ed = set(m._election_days(2022, 2026))
    a_gen = ed == {dt.date(2022, 11, 8), dt.date(2024, 11, 5), dt.date(2026, 11, 3)}
    ev = m.scheduled_events(start="2022-01-01", end="2026-12-31", categories=("ELECTION",))
    a_tag = bool(ev) and all(e["selection"] == "scheduled_uncertainty" and e["category"] == "ELECTION"
                             and e["timing"] == "overnight" and e["event_et"] == "" for e in ev)
    a_dates = {"2024-11-05", "2022-11-08", "2026-11-03"}.issubset({e["date"] for e in ev})
    print("(A) election days match=%s | tagged scheduled_uncertainty/overnight=%s | dates present=%s"
          % (a_gen, a_tag, a_dates))
    ok &= a_gen and a_tag and a_dates

    # (B) impact_blind partition in shocks_frame
    fr = m.shocks_frame(include_scheduled=True, scheduled_categories=("FOMC", "ELECTION"),
                        start="2022-01-01", end="2026-12-31")
    has_cols = "selection" in fr.columns and "impact_blind" in fr.columns
    sel = dict(zip(fr["date"].dt.strftime("%Y-%m-%d"), fr["selection"]))
    ib = dict(zip(fr["date"].dt.strftime("%Y-%m-%d"), fr["impact_blind"]))
    b_fomc = sel.get("2025-01-29") == "scheduled" and ib.get("2025-01-29") == True        # generated FOMC
    b_elec = sel.get("2024-11-05") == "scheduled_uncertainty" and ib.get("2024-11-05") == True
    b_crisis = sel.get("2010-05-06") == "salience_curated" and ib.get("2010-05-06") == False  # registry
    print("(B) cols=%s | FOMC impact-blind=%s | election impact-blind=%s | crisis salience-curated=%s"
          % (has_cols, b_fomc, b_elec, b_crisis))
    ok &= has_cols and b_fomc and b_elec and b_crisis

    # (C) loader provenance for a categorized news feed
    m.clear_loaded()
    news = pd.DataFrame([
        {"date": "2025-02-14", "category": "GEOPOLITICAL", "name": "Hypothetical overnight geopolitical headline"},
        {"date": "2024-11-06", "release": "US election result"},        # name -> ELECTION
    ])
    lev = m.load_release_schedule(news, selection="news_classified")
    c_sel = bool(lev) and all(e["selection"] == "news_classified" for e in lev)
    c_cat = {e["category"] for e in lev} == {"GEOPOLITICAL", "ELECTION"}
    c_overnight = all(e["event_et"] == "" and e["timing"] == "overnight" for e in lev)   # no time given
    fr2 = m.shocks_frame(include_scheduled=True, scheduled_categories=("GEOPOLITICAL", "ELECTION"),
                         start="2024-01-01", end="2025-12-31")
    ib2 = dict(zip(fr2["date"].dt.strftime("%Y-%m-%d"), fr2["impact_blind"]))
    c_ib = ib2.get("2025-02-14") == True                                  # news_classified is impact-blind
    print("(C) news selection=%s | categories %s | overnight default=%s | impact-blind in frame=%s"
          % (c_sel, sorted({e["category"] for e in lev}), c_overnight, c_ib))
    ok &= c_sel and c_cat and c_overnight and c_ib
    m.clear_loaded()

    print("unscheduled-feed checks ->", ok)
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
