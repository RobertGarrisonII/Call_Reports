"""
event_study_driver.py
=====================
End-to-end glue from the shock registry to the event-study estimators, so a contagion event
study is *one call per identification regime*. It:

  1. selects an event subset (one `timing` regime) from `market_shocks` by `classes` (theme)
     and/or `categories` (mechanism);
  2. matches each event's first-reaction session to control sessions on an EX-ANTE stress
     level via `market_shocks.treatment_control_assignment` (never a vol outcome);
  3. resolves each treated session's release boundary -- the halt reopen / scheduled 14:00 /
     unscheduled release time from the registry, or, for overnight/weekend events with no
     intraday boundary, the session open (`default_release_et`) for the open-to-open design --
     and gives every control the SAME wall-clock release time on its own date (the
     same-time-of-day counterfactual);
  4. assembles the (date, df) treated/control lists and the release map that
     `mwcb_event_study.build_panel` consumes, and (in `run_event_study`) runs the DiD, the
     RD-in-time, and the cross-market spillover.

Identification is not uniform across `timing`, so a panel should mix only one regime. The
driver reports the regime(s) it assembled and flags a mix. Pure numpy / pandas.
"""
import numpy as np
import pandas as pd

import market_shocks as mk
import mwcb_event_study as mw

EPS = 1e-12


def _resolve_release(date_ts, rel_map, default_et, tz):
    """Release Timestamp for a session date: the registry intraday boundary if the event has
    one (`event_et` populated -> in `rel_map`), else the session open `default_et` (the
    overnight/weekend open-to-open anchor)."""
    for k, v in rel_map.items():
        if k.date() == date_ts.date():
            return pd.Timestamp(v)
    hh, mm = (int(x) for x in str(default_et).split(":"))
    return pd.Timestamp(date_ts.date(), tz=tz) + pd.Timedelta(hours=hh, minutes=mm)


def build_event_study_panel(daily_state, sessions_by_date, categories=None, classes=None,
                            spy_es_only=False, n_controls=2, buffer_days=5, window_days=60,
                            caliper=None, exact_weekday=False, default_release_et="09:30",
                            tz="America/New_York"):
    """Assemble the treated/control sessions and release map for one identification regime.

    daily_state      : Series indexed by trading date of an EX-ANTE stress level (prior-close
                       VIX, a vol forecast, the long-run vol level) -- drives control matching.
    sessions_by_date : {date-like -> intraday book DataFrame} providing the actual session
                       data; keyed by calendar date (matched on .date()).
    categories/classes/spy_es_only : select the event subset (one regime).

    Returns a dict: `assignment` (the treatment->control table), `treated` and `control`
    ((date_key, df) lists for build_panel), `release_by_date` ({date_key -> Timestamp}, every
    key covered), `regimes` (the set of `timing` tags present -- ideally one), `mixed_regime`
    (bool), `dropped` (events/controls with no session), and counts."""
    rel_map = mk.release_times(categories=categories, classes=classes, spy_es_only=spy_es_only, tz=tz)
    assign = mk.treatment_control_assignment(daily_state, categories=categories, classes=classes,
                                             n_controls=n_controls, buffer_days=buffer_days,
                                             window_days=window_days, exact_weekday=exact_weekday,
                                             caliper=caliper, spy_es_only=spy_es_only)
    sess = {pd.Timestamp(k).date(): df for k, df in sessions_by_date.items()}
    shocks = mk.shocks_frame(tz)
    timing_by_date = {r["date"].date(): r["timing"] for _, r in shocks.iterrows()}

    treated, control, release_by_date, dropped, regimes = [], [], {}, [], set()
    seen_treat = set()
    for _, row in assign.iterrows():
        td = row["treatment_date"]; cd = row["control_date"]
        if sess.get(td.date()) is None:
            dropped.append(("treatment_no_session", str(td.date()))); continue
        tr_rel = _resolve_release(td, rel_map, default_release_et, tz)
        tkey = td.date().isoformat()
        if tkey not in seen_treat:
            treated.append((tkey, sess[td.date()])); release_by_date[tkey] = tr_rel
            seen_treat.add(tkey); regimes.add(timing_by_date.get(td.date(), "unknown"))
        if sess.get(cd.date()) is None:
            dropped.append(("control_no_session", str(cd.date()))); continue
        ckey = cd.date().isoformat()
        ctrl_rel = pd.Timestamp(cd.date(), tz=tz) + (tr_rel - tr_rel.normalize())   # same time-of-day
        control.append((ckey, sess[cd.date()])); release_by_date[ckey] = ctrl_rel

    return {"assignment": assign, "treated": treated, "control": control,
            "release_by_date": release_by_date, "regimes": sorted(regimes),
            "mixed_regime": len(regimes) > 1, "dropped": dropped,
            "n_treated": len(treated), "n_control": len(control)}


def run_event_study(daily_state, sessions_by_date, categories=None, classes=None,
                    spy_es_only=False, n_controls=2, buffer_days=5, window_days=60, caliper=None,
                    exact_weekday=False, default_release_et="09:30", tz="America/New_York",
                    asset="SPY", outcome="cost_to_fill", source="ES", target="SPY",
                    pre_min=5, post_min=5, target_qty=None, n_levels=10, rd_bandwidth_min=None):
    """One call: assemble the regime's panel (build_event_study_panel) and run the estimators
    (event-study profile, DiD, RD-in-time, cross-market spillover). Returns the assembly dict
    plus `panel`, `profile`, `did`, `rd`, `spillover`. RD bandwidth defaults to pre_min."""
    a = build_event_study_panel(daily_state, sessions_by_date, categories=categories, classes=classes,
                                spy_es_only=spy_es_only, n_controls=n_controls, buffer_days=buffer_days,
                                window_days=window_days, caliper=caliper, exact_weekday=exact_weekday,
                                default_release_et=default_release_et, tz=tz)
    out = dict(a)
    if a["n_treated"] == 0 or a["n_control"] == 0:
        out.update({"panel": None, "profile": None, "did": None, "rd": None, "spillover": None,
                    "warning": "no treated and/or control sessions resolved"})
        return out
    panel = mw.build_panel(a["treated"], a["control"], a["release_by_date"], asset, outcome,
                           pre_min, post_min, target_qty=target_qty, n_levels=n_levels)
    out["panel"] = panel
    out["profile"] = mw.event_study(panel, bin_sec=30)
    out["did"] = mw.did(panel)
    out["rd"] = mw.rd_in_time(panel, bandwidth_min=(rd_bandwidth_min or pre_min))
    try:
        out["spillover"] = mw.cross_market_spillover(a["treated"], a["control"], a["release_by_date"],
                                                     source, target, outcome, pre_min, post_min,
                                                     target_qty=target_qty, n_levels=n_levels)
    except Exception as e:
        out["spillover"] = {"error": repr(e)}
    return out


# ── self-test ─────────────────────────────────────────────────────────────────
def _day_book(date, reopen, treated, seed, n_levels=6, tz="America/New_York"):
    """Compact 09:30-14:00 fifteen-second two-asset book. On treated days both books drop
    depth from `reopen` onward (a persistent step; futures more), so the cost-to-fill jumps
    post-reopen; control days are flat."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(pd.Timestamp(f"{date} 09:30", tz=tz), pd.Timestamp(f"{date} 14:00", tz=tz),
                        freq="15s")
    cols = {}
    for asset, base_px, tick, base_depth, wd in (("SPY", 500.0, 0.01, 320.0, 0.30),
                                                 ("ES", 5000.0, 0.25, 220.0, 0.20)):
        mid = base_px * np.exp(rng.normal(0, 1, len(idx)).cumsum() / 8000.0)
        factor = np.ones(len(idx))
        if treated and reopen is not None:
            secs = (idx - pd.Timestamp(reopen)).total_seconds().to_numpy()
            factor = np.where(secs > 0, wd, 1.0)                 # persistent step withdrawal
        depth = base_depth * factor * (1 + rng.normal(0, 0.02, len(idx)))
        for l in range(1, n_levels + 1):
            q = np.maximum(depth * np.exp(-0.3 * (l - 1)), 1.0)
            cols[f"{asset}_bidprice_{l}"] = mid - l * tick; cols[f"{asset}_askprice_{l}"] = mid + l * tick
            cols[f"{asset}_bidquantity_{l}"] = q; cols[f"{asset}_askquantity_{l}"] = q
    return pd.DataFrame(cols, index=idx)


def _selftest():
    print("event_study_driver self-test")
    tz = "America/New_York"

    # (1) release resolution: registry boundary when present, else session open
    rel_map = mk.release_times(categories=("MWCB",), tz=tz)
    d318 = pd.Timestamp("2020-03-18", tz=tz)                      # has a 13:11 reopen
    dnone = pd.Timestamp("2023-08-02", tz=tz)                     # SOVEREIGN_CREDIT, overnight -> open
    r1 = _resolve_release(d318, rel_map, "09:30", tz)
    r2 = _resolve_release(dnone, rel_map, "09:30", tz)
    print("\n(1) release resolution")
    print("    2020-03-18 (halt) -> %s ; 2023-08-02 (overnight) -> %s" % (r1.time(), r2.time()))
    print("    checks:", (r1.hour, r1.minute) == (13, 11) and (r2.hour, r2.minute) == (9, 30))

    # build compact sessions over a window containing the four 2020 MWCB days + clean controls
    days = pd.bdate_range("2020-02-18", "2020-04-09", tz=tz)
    rby_dates = {k.date(): v for k, v in rel_map.items()}        # reopen ts by date (2020 only)
    sessions = {}
    for i, d in enumerate(days):
        reopen = rby_dates.get(d.date())
        sessions[d] = _day_book(d.date().isoformat(), reopen, treated=(reopen is not None), seed=100 + i)
    rng = np.random.default_rng(0)
    state = pd.Series(np.cumsum(rng.normal(0, 0.1, len(days))) + 5.0, index=days)   # smooth ex-ante level

    # (2) assembly: treated = the in-window MWCB days; controls clean, level-matched; release map total
    a = build_event_study_panel(state, sessions, categories=("MWCB",), n_controls=2,
                                buffer_days=3, window_days=30, default_release_et="09:30", tz=tz)
    keys = [k for k, _ in a["treated"]] + [k for k, _ in a["control"]]
    cov = all(k in a["release_by_date"] for k in keys)
    evset = set(mk.event_dates(categories=("MWCB",)).normalize().date)
    ctrl_clean = all(pd.Timestamp(k).date() not in evset for k, _ in a["control"])
    print("\n(2) panel assembly (MWCB regime)")
    print("    treated=%d, control=%d, regimes=%s, mixed=%s, dropped=%d"
          % (a["n_treated"], a["n_control"], a["regimes"], a["mixed_regime"], len(a["dropped"])))
    print("    release map covers all keys:", cov, "| controls non-event:", ctrl_clean)
    print("    checks:", a["n_treated"] == 4 and a["n_control"] >= 4 and cov and ctrl_clean
          and a["regimes"] == ["halt_reopen"] and not a["mixed_regime"])

    # (3) end-to-end estimators via run_event_study: reopening asymmetry shows up
    res = run_event_study(state, sessions, categories=("MWCB",), n_controls=2, buffer_days=3,
                          window_days=30, asset="SPY", outcome="cost_to_fill", source="ES",
                          target="SPY", pre_min=5, post_min=5, target_qty=100, n_levels=6)
    did, rd, sp = res["did"], res["rd"], res["spillover"]
    print("\n(3) end-to-end estimators (cost_to_fill)")
    print("    DiD = %.3f bps (t=%.1f) ; RD jump = %.3f bps (t=%.1f)"
          % (did["did_coef"], did["did_t"], rd["jump"], rd["jump_t"]))
    print("    ES->SPY spillover (post*treated) = %.3f (t=%.1f)"
          % (sp["spill_post_treated"]["coef"], sp["spill_post_treated"]["t"]))
    print("    checks:", did["did_coef"] > 0 and did["did_t"] > 2 and rd["jump"] > 0
          and np.isfinite(sp["spill_post_treated"]["coef"]) and res["panel"] is not None
          and len(res["panel"]) > 0)


if __name__ == "__main__":
    _selftest()
