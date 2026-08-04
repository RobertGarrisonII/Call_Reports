"""test_event_study_2025.py -- proves the 2025 NON-MWCB events flow through the event-clock path
end-to-end (catalog -> release_times -> treatment/control -> panel -> event-study/DiD/RD/spillover).
The driver self-test only exercises MWCB; this drives the TARIFF category (Liberation-Day 2025-04-03
overnight + the 2025-04-09 intraday pause) through run_event_study on synthetic event-dated sessions
with a post-release depth withdrawal on treated days and flat controls."""
import sys
import numpy as np
import pandas as pd
import market_shocks as mk
import event_study_driver as esd

tz = "America/New_York"

if __name__ == "__main__":
    print("event-study path on a 2025 non-MWCB category (TARIFF)")
    # window covering the 2025 tariff events + clean controls on either side
    days = pd.bdate_range("2025-03-10", "2025-04-25", tz=tz)
    rel_map = mk.release_times(categories=("TARIFF",), tz=tz)        # intraday tariff boundaries (04-09 13:18)
    rby = {k.date(): v for k, v in rel_map.items()}
    ev_dates = set(mk.event_dates(categories=("TARIFF",)).normalize().date)   # {2025-04-03, 2025-04-09}
    in_win = sorted(d for d in days if d.date() in ev_dates)

    sessions = {}
    for i, d in enumerate(days):
        treated = d.date() in ev_dates
        reopen = rby.get(d.date()) or (pd.Timestamp(f"{d.date()} 09:30", tz=tz) if treated else None)
        sessions[d] = esd._day_book(d.date().isoformat(), reopen, treated=treated, seed=200 + i)
    rng = np.random.default_rng(0)
    state = pd.Series(np.cumsum(rng.normal(0, 0.1, len(days))) + 5.0, index=days)  # smooth ex-ante level

    # (1) release resolution: the intraday tariff event resolves to its registry boundary
    et_0409 = rel_map.get(pd.Timestamp("2025-04-09", tz=tz))
    has_intraday = et_0409 is not None and (et_0409.hour, et_0409.minute) == (13, 18)
    print("  (1) release_times(TARIFF): 2025-04-09 -> %s  (intraday boundary resolved=%s)"
          % (et_0409.time() if et_0409 is not None else None, has_intraday))

    # (2) assembly for the TARIFF regime
    a = esd.build_event_study_panel(state, sessions, categories=("TARIFF",), n_controls=2,
                                    buffer_days=3, window_days=30, default_release_et="09:30", tz=tz)
    keys = [k for k, _ in a["treated"]] + [k for k, _ in a["control"]]
    cov = all(k in a["release_by_date"] for k in keys)
    ctrl_clean = all(pd.Timestamp(k).date() not in ev_dates for k, _ in a["control"])
    print("  (2) assembly: treated=%d (expected %d), control=%d, regimes=%s, mixed=%s, dropped=%d"
          % (a["n_treated"], len(in_win), a["n_control"], a["regimes"], a["mixed_regime"], len(a["dropped"])))
    print("      release map covers all keys=%s | controls non-event=%s" % (cov, ctrl_clean))

    # (3) end-to-end estimators
    res = esd.run_event_study(state, sessions, categories=("TARIFF",), n_controls=2, buffer_days=3,
                              window_days=30, asset="SPY", outcome="cost_to_fill", source="ES",
                              target="SPY", pre_min=5, post_min=5, target_qty=100, n_levels=6)
    did, rd, sp = res["did"], res["rd"], res["spillover"]
    spc = sp.get("spill_post_treated", {}) if isinstance(sp, dict) else {}
    print("  (3) estimators: DiD=%.3f bps (t=%.1f) | RD jump=%.3f bps (t=%.1f) | ES->SPY spill=%.3f"
          % (did["did_coef"], did["did_t"], rd["jump"], rd["jump_t"], spc.get("coef", float("nan"))))

    ok = (has_intraday
          and a["n_treated"] == len(in_win) and len(in_win) == 2          # both tariff days resolved
          and a["n_control"] >= 2 and cov and ctrl_clean                  # clean, covered controls
          and res["panel"] is not None and len(res["panel"]) > 0          # panel built
          and np.isfinite(did["did_coef"]) and did["did_coef"] > 0        # treated cost jumps post-release
          and np.isfinite(rd["jump"]) and np.isfinite(spc.get("coef", np.nan)))  # RD + spillover finite
    print("checks -> %s" % ok)
    sys.exit(0 if ok else 1)
