#!/usr/bin/env python3
"""Gate: design_sample.py -- the sample design as a derivable rule (v0.9.76).

The manual workflow (top-10 log-diff days from a 3-4y V-Lab download, hand-picked
controls) becomes code. Pins:

  1. threshold selection: planted spikes recovered at the fixed quantile; adjacent
     spike days share an episode; the per-episode cap keeps a 3-day crisis from
     eating three sample slots
  2. control derivation: same weekday, 350-371 calendar days earlier, trading day,
     vol screen -- a planted HIGH-vol same-weekday candidate at exactly 364 days is
     REJECTED for a calmer neighbour; the graduated fallback records its screen_q;
     controls are unique across pairs
  3. power arithmetic: the closed form reproduces the review's numbers (diff 0.20,
     SD 0.235 -> 22 per regime); achieved power is monotone in n; the day-level SD
     can be estimated from a prior run's per-day CSV
  4. CLI end-to-end: report + --emit-args; mid-band days are NOT in the --volatile
     list (binary-contrast contamination); emitted pairs clear
     validate_sample.check_pairing
"""
import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd

import design_sample as ds
import validate_sample as vs


def _flat_series(spikes=(), seed=0, lo="2018-01-02", hi="2026-08-14"):
    """Mean-reverting calm series (no drift, so the median screen is meaningful)
    with planted multiplicative spikes."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(lo, hi)
    idx = idx[[str(d.date()) not in vs._ONE_OFF for d in idx]]
    v = 16.0 + 0.5 * np.sin(np.arange(len(idx)) / 37.0) + rng.normal(0, 0.3, len(idx))
    s = pd.Series(v, index=idx)
    for d, mult in spikes:
        t = pd.Timestamp(d)
        if t in s.index:
            s.loc[t] = s.loc[t] * mult
    return s


def check_selection_and_episodes():
    # STEPPED multipliers: the selection statistic is the log-DIFF, so a flat-topped
    # spike only fires on its first day -- each escalation day must itself be a jump
    spikes = [("2020-03-16", 3.0), ("2020-03-17", 9.0), ("2020-03-18", 27.0),
              ("2022-06-13", 2.5), ("2024-08-05", 2.7)]
    s = _flat_series(spikes)
    # a PADDED holiday row with a huge spurious spike (the V-Lab 2026-01-19 MLK case):
    # loaders drop it; the selection path must never see it. _flat_series uses
    # bdate_range, which contains NYSE holidays, so plant on New Year's Day 2024.
    s.loc[pd.Timestamp("2024-01-01")] = 500.0
    st = ds.selection_stat(s, "logdiff")
    sel = ds.select_volatile(s, st, threshold_q=0.997, episode_gap=5, max_per_episode=2)
    kept = {str(d.date()) for d in sel.index[sel["kept"]]}
    # the three March-2020 spike days are one episode; the cap keeps only two
    march = {"2020-03-16", "2020-03-17", "2020-03-18"}
    n_march_kept = len(kept & march)
    ep_march = sel.loc[[pd.Timestamp(d) for d in sorted(march) if pd.Timestamp(d) in sel.index],
                       "episode"].nunique()
    ok_sel = {"2022-06-13", "2024-08-05"} <= kept
    ok_ep = ep_march == 1 and n_march_kept == 2
    # the padded holiday row: simulate the loader's calendar filter, then select --
    # neither the holiday nor its poisoned next-day log-diff may survive
    import validate_sample as _vs
    hols = set(_vs.NYSECalendar().holidays(start=s.index[0], end=s.index[-1]))
    s2 = s.drop([d for d in s.index if d in hols or str(d.date()) in _vs._ONE_OFF])
    sel2 = ds.select_volatile(s2, ds.selection_stat(s2, "logdiff"), 0.997, 5, 2)
    kept2 = {str(d.date()) for d in sel2.index[sel2["kept"]]}
    ok_hol = "2024-01-01" not in kept2 and "2024-01-02" not in kept2
    print("    padded NYSE-holiday spike excluded after the calendar filter (%s)" % ok_hol)
    print("(1) planted spikes recovered (%s); March 2020 = ONE episode, capped at 2 of 3 "
          "days kept (%s)" % (ok_sel, ok_ep))
    return bool(ok_sel and ok_ep and ok_hol)


def check_control_rules():
    # volatile day Monday 2024-08-05; exactly 364d earlier is Monday 2023-08-07.
    # Plant 2023-08-07 HOT (above every screen): the rule must skip the tempting
    # 364-day candidate for a calmer same-weekday neighbour.
    s = _flat_series([("2024-08-05", 3.0), ("2023-08-07", 3.5)])
    s.loc[pd.Timestamp("2023-08-14")] = 14.0     # calm same-weekday neighbour, 357d out
    v = pd.Timestamp("2024-08-05")
    ctl = ds.assign_controls([v], s, control_max_q=0.5, qualifying={v})
    c = ctl.loc[v]
    ok_pick = pd.notna(c["control"]) and str(c["control"].date()) != "2023-08-07"
    ok_rule = (pd.notna(c["control"]) and c["control"].dayofweek == v.dayofweek
               and 350 <= c["gap_days"] <= 371)
    ok_screen = np.isfinite(c["control_level"]) and c["control_level"] <= float(s.quantile(0.75))
    # uniqueness: two volatile days a week apart cannot share the one calm Monday
    v2 = pd.Timestamp("2024-08-12")
    both = ds.assign_controls([v, v2], s, control_max_q=0.5, qualifying={v, v2})
    cc = [x for x in both["control"] if pd.notna(x)]
    ok_uniq = len(cc) == len(set(cc))
    print("(2) hot 364-day candidate rejected (%s); weekday+window respected (%s); "
          "screen honoured (%s); controls unique (%s)"
          % (ok_pick, ok_rule, ok_screen, ok_uniq))
    return bool(ok_pick and ok_rule and ok_screen and ok_uniq)


def check_power_math():
    n = ds.required_days_per_group(0.20, 0.235, alpha=0.05, power=0.80)
    ok_n = n == 22
    p_small = ds.achieved_power(0.20, 0.235, 10, 10)
    p_big = ds.achieved_power(0.20, 0.235, 25, 25)
    ok_mono = 0.2 < p_small < p_big < 0.99 and p_big >= 0.80
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "per_day.csv")
        pd.DataFrame({"CS_ES": np.r_[np.full(12, 0.2), np.full(12, 0.6)]}).to_csv(path)
        sd = ds.day_sd_from_csv(path, "CS_ES")
        want = float(pd.Series(np.r_[np.full(12, 0.2), np.full(12, 0.6)]).std(ddof=1))
        ok_csv = abs(sd - want) < 1e-12
    print("(3) required per group at (diff .20, sd .235) = %d (== 22: %s); power monotone "
          "10->25 days: %.2f -> %.2f (%s); SD from per-day CSV (%s)"
          % (n, ok_n, p_small, p_big, ok_mono, ok_csv))
    return bool(ok_n and ok_mono and ok_csv)


def check_screen_col():
    """Selection on one series, control screen on another: a candidate that is calm
    in the SELECTION series but hot in the SCREEN series must be rejected -- the
    two-component MF2-GARCH split (spike vs secular state) is the point."""
    sel_series = _flat_series([("2024-08-05", 3.0)])
    screen = sel_series * 0 + 10.0                       # flat, calm everywhere...
    screen.loc[pd.Timestamp("2023-08-07")] = 60.0        # ...except the 364d candidate
    v = pd.Timestamp("2024-08-05")
    d = ds.build_design(sel_series, "logdiff", ("2018-01-01", "2026-08-14"),
                        0.999, 5, 3, 0.5, (0.55, 0.85), 0, 0.20, 0.235, 0.05, 0.80,
                        screen_series=screen)
    ctl = d["controls"]
    ok_sel = v in ctl.index
    c = ctl.loc[v] if ok_sel else None
    ok_skip = ok_sel and pd.notna(c["control"]) and str(c["control"].date()) != "2023-08-07"
    ok_lvl = ok_sel and abs(c["control_level"] - 10.0) < 1e-9   # level reported from SCREEN series
    print("(5) screen-col: hot-in-screen 364d candidate rejected (%s); control level "
          "reported from the screen series (%s)" % (ok_skip, ok_lvl))
    return bool(ok_sel and ok_skip and ok_lvl)


def check_data_floor():
    """v0.9.83: MIDAS carries no futures before 2017-06-26. The floor must (a) drop
    a selected pre-floor day into dropped_below_floor -- reported, never silent;
    (b) leave a just-post-floor volatile day UNPAIRABLE because every control
    candidate (350-371d earlier) predates the tape; (c) not touch a later pair;
    (d) surface the floor line in the rendered report. The series itself stays
    un-truncated: only extractability is constrained."""
    spikes = [("2018-06-25", 3.0),   # pre-floor: selected by the statistic, dropped by the floor
              ("2019-08-05", 3.0),   # 35d post-floor: control window is entirely pre-floor
              ("2024-08-05", 3.0)]   # far post-floor: must pair normally
    s = _flat_series(spikes)
    d = ds.build_design(s, "logdiff", ("2018-01-01", "2026-08-14"),
                        0.999, 5, 3, 0.5, (0.55, 0.85), 0, 0.20, 0.235, 0.05, 0.80,
                        data_floor="2019-07-01")
    vol = {str(x.date()) for x in d["volatile"]}
    ok_drop = "2018-06-25" in d["dropped_below_floor"] and "2018-06-25" not in vol
    v_near, v_far = pd.Timestamp("2019-08-05"), pd.Timestamp("2024-08-05")
    ctl = d["controls"]
    ok_unpair = (v_near in ctl.index and pd.isna(ctl.loc[v_near, "control"]))
    c_far = ctl.loc[v_far, "control"] if v_far in ctl.index else pd.NaT
    ok_far = pd.notna(c_far) and c_far >= pd.Timestamp("2019-07-01")
    txt = ds.render(d)
    ok_line = "data floor" in txt and "2019-07-01" in txt and "2018-06-25" in txt
    # without the floor the same design pairs BOTH near days -- the floor, not the
    # rule, is what unpaired 2019-08-05
    d0 = ds.build_design(s, "logdiff", ("2018-01-01", "2026-08-14"),
                         0.999, 5, 3, 0.5, (0.55, 0.85), 0, 0.20, 0.235, 0.05, 0.80)
    ok_ctrl = (v_near in d0["controls"].index
               and pd.notna(d0["controls"].loc[v_near, "control"]))
    print("(6) data floor: pre-floor day reported dropped (%s); just-post-floor day "
          "unpairable under the floor (%s) but pairable without it (%s); later pair "
          "untouched and control >= floor (%s); floor line rendered (%s)"
          % (ok_drop, ok_unpair, ok_ctrl, ok_far, ok_line))
    return bool(ok_drop and ok_unpair and ok_ctrl and ok_far and ok_line)


def check_forbid_control():
    """v0.9.85: a date whose TAPE is known bad (2017-12-05: lossy SPY capture, CHECK 10
    verdict DATA) may never serve as a control even though its volatility record is calm.
    The picker must route around it to the next-best same-weekday candidate, the forbidden
    date must never appear anywhere in the controls, and the report must say so."""
    s = _flat_series([("2024-08-05", 3.0)])
    v = pd.Timestamp("2024-08-05")                   # Monday
    s.loc[pd.Timestamp("2023-08-07")] = 14.0         # natural pick: calm Monday, exactly 364d
    s.loc[pd.Timestamp("2023-08-14")] = 14.2         # the alternate the picker must reroute to
    d0 = ds.build_design(s, "logdiff", ("2018-01-01", "2026-08-14"),
                         0.999, 5, 3, 0.5, (0.55, 0.85), 0, 0.20, 0.235, 0.05, 0.80)
    nat = d0["controls"].loc[v, "control"]
    ok_nat = pd.notna(nat)
    d1 = ds.build_design(s, "logdiff", ("2018-01-01", "2026-08-14"),
                         0.999, 5, 3, 0.5, (0.55, 0.85), 0, 0.20, 0.235, 0.05, 0.80,
                         forbid_controls=[str(nat.date())] if ok_nat else [])
    c1 = d1["controls"].loc[v, "control"]
    ok_rerouted = (ok_nat and pd.notna(c1) and c1 != nat
                   and c1.dayofweek == v.dayofweek
                   and 350 <= int((v - c1).days) <= 371)
    all_ctl = [x for x in d1["controls"]["control"] if pd.notna(x)]
    ok_absent = nat not in all_ctl
    txt = ds.render(d1)
    ok_line = "FORBIDDEN" in txt and (str(nat.date()) in txt if ok_nat else False)
    print("(7) forbid-control: natural pick exists (%s); forbidding it reroutes to another "
          "same-weekday in-window candidate (%s); the forbidden date appears in no pair (%s); "
          "report names it (%s)" % (ok_nat, ok_rerouted, ok_absent, ok_line))
    return bool(ok_nat and ok_rerouted and ok_absent and ok_line)


def check_cli():
    s = _flat_series([("2020-03-16", 3.0), ("2022-06-13", 2.5), ("2024-08-05", 2.7)])
    with tempfile.TemporaryDirectory() as td:
        csv = os.path.join(td, "vol.csv")
        pd.DataFrame({"date": s.index.strftime("%Y-%m-%d"), "vol": s.to_numpy()}).to_csv(
            csv, index=False)
        out = os.path.join(td, "design.txt")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = ds.main(["--series", csv, "--window", "2018-01-01:2026-08-14",
                          "--threshold-q", "0.997", "--n-mid", "3",
                          "--emit-args", "--out", out])
        txt = buf.getvalue()
        ok_rc = rc == 0 and os.path.exists(out)
        ok_blocks = ("POWER:" in txt and "VOLATILE" in txt and "MID-BAND" in txt
                     and "--volatile" in txt)
        vol_line = [l for l in txt.splitlines() if l.startswith("--volatile")]
        mid_line = [l for l in txt.splitlines() if l.startswith("# mid-band (dose-response")]
        ok_sep = bool(vol_line) and bool(mid_line)
        mid_dates = []
        for l in txt.splitlines():
            if l.startswith("# binary regime contrasts):"):
                mid_dates = l.split(":", 1)[1].strip().split(",")
        ok_no_mix = all(d not in vol_line[0] for d in mid_dates if d)
        vols, bases = [], []
        for l in txt.splitlines():
            if l.startswith("--volatile"):
                vols = l.replace("--volatile", "").replace("\\", "").strip().split(",")
            if l.startswith("--baseline"):
                bases = l.replace("--baseline", "").strip().split(",")
        pair = vs.check_pairing(vols[:len(bases)], bases)
        ok_pair = "level" not in pair.columns or not (pair["level"] == "ERROR").any()
    print("(4) CLI: rc=0 + report written (%s); all blocks present (%s); mid-band emitted "
          "separately (%s), never inside --volatile (%s); pairs clear check_pairing (%s)"
          % (ok_rc, ok_blocks, ok_sep, ok_no_mix, ok_pair))
    return bool(ok_rc and ok_blocks and ok_sep and ok_no_mix and ok_pair)


def main():
    checks = [check_selection_and_episodes, check_control_rules, check_power_math,
              check_screen_col, check_data_floor, check_forbid_control, check_cli]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("sample-design checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    sys.exit(main())
