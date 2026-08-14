#!/usr/bin/env python3
"""design_sample.py -- the sample design as CODE: threshold selection, episode
grouping, power-based sufficiency, and computed control dates.

Replaces the manual workflow (download 3-4 years from V-Lab, eyeball the top-10
log-diff days, hand-pick controls) with a derivable rule, per the sample-size
review:

  * SELECTION BY FIXED THRESHOLD, not top-N: a day qualifies when the selection
    statistic (default: the daily log-difference of the supplied volatility
    series -- the existing workflow's statistic; --stat level uses the level)
    exceeds a fixed quantile of the WHOLE window. Top-N makes the treatment
    definition sample-relative -- the same day flips label depending on which
    window was downloaded; a threshold is portable and lets n grow with the
    window.
  * EPISODES, not just days: qualifying days closer than --episode-gap trading
    days belong to one episode, and each episode contributes at most
    --max-per-episode days (highest statistic first). Adjacent crisis days
    share an event; the unit that buys statistical power is the episode.
  * POWER-BASED SUFFICIENCY: with the day-level SD of the target metric (given
    directly, or estimated from a prior run's per-day CSV) and the effect size
    to detect, the script reports the required days per regime
    n = ceil(2 ((z_{1-a/2}+z_pow) sd / diff)^2), the power the SELECTED sample
    achieves, and whether the window can supply the deficit.
  * COMPUTED CONTROLS, not date ranges: each selected day gets an explicit
    control -- same weekday, 350-371 calendar days earlier, present in the
    series (i.e. a trading day), not a one-off closure, not itself a qualifying
    day, UNUSED by any other pair, and with its volatility LEVEL below the
    --control-max-q quantile (the screen that keeps a 2022 bear-market day from
    serving as "calm"). Among survivors the candidate closest to 364 days wins.
    Ranges cannot express this rule; the extraction pipeline consumes explicit
    date lists anyway.
  * Optional MID-BAND days (--n-mid > 0): days whose statistic falls in the
    --mid-band quantile band, evenly spaced across the window, each with its
    own control -- the dose-response design that turns the binary contrast into
    a regression on the continuous volatility state.

Usage
    python design_sample.py --series vlab_mf2garch.csv --window 2018-01-01:2026-08-14
    python design_sample.py --series vix.csv --col vix --threshold-q 0.95 \
        --per-day-csv output/.../information_shares__per_day.csv --emit-args
    python design_sample.py --selftest

CSV format: a date column (auto-detected) plus the volatility column (--col, else
the last numeric column). Exit codes: 0 ok; 1 bad input.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

import validate_sample as vs

PAIR_MIN_D, PAIR_MAX_D, PAIR_TARGET_D = 350, 371, 364


# ── input ─────────────────────────────────────────────────────────────────────
def load_series(path: str, date_col: str = "", col: str = "") -> pd.Series:
    """-> float Series indexed by normalized date, ascending, NaNs dropped."""
    df = pd.read_csv(path)
    cols = list(df.columns)
    if date_col:
        dc = date_col
    else:
        dc = None
        for c in cols:
            parsed = pd.to_datetime(df[c], errors="coerce")
            if parsed.notna().mean() > 0.9:
                dc = c
                break
        if dc is None:
            raise SystemExit(f"{path}: no parseable date column found (have {cols})")
    if col:
        vc = col
        if vc not in cols:
            lower = {c.lower(): c for c in cols}
            if vc.lower() not in lower:
                raise SystemExit(f"{path}: no column {col!r} (have {cols})")
            vc = lower[vc.lower()]
    else:
        numeric = [c for c in cols if c != dc
                   and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9]
        if not numeric:
            raise SystemExit(f"{path}: no numeric volatility column found")
        vc = numeric[-1]
    s = pd.Series(pd.to_numeric(df[vc], errors="coerce").to_numpy(),
                  index=pd.to_datetime(df[dc], errors="coerce").dt.normalize())
    s = s[s.index.notna() & s.notna()].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if len(s) < 300:
        raise SystemExit(f"{path}: only {len(s)} usable daily rows -- need a longer history")
    return s


# ── selection ─────────────────────────────────────────────────────────────────
def selection_stat(series: pd.Series, stat: str = "logdiff") -> pd.Series:
    if stat == "level":
        return series.astype(float)
    if stat == "logdiff":
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.log(series.astype(float)).diff()
    raise SystemExit(f"unknown --stat {stat!r}")


def group_episodes(dates, index, gap: int):
    """Episode id per qualifying date: consecutive qualifying dates fewer than
    `gap` TRADING days apart (positions in the series index) share an episode."""
    pos = {d: i for i, d in enumerate(index)}
    eps, cur = [], 0
    prev = None
    for d in sorted(dates):
        if prev is not None and pos[d] - pos[prev] > gap:
            cur += 1
        eps.append(cur)
        prev = d
    return dict(zip(sorted(dates), eps))


def select_volatile(series: pd.Series, stat: pd.Series, threshold_q: float,
                    episode_gap: int, max_per_episode: int) -> pd.DataFrame:
    """-> DataFrame[date-indexed: stat, level, episode, kept] over qualifying days."""
    thr = float(stat.quantile(threshold_q))
    qual = stat[stat >= thr].dropna()
    if qual.empty:
        return pd.DataFrame(columns=["stat", "level", "episode", "kept"])
    epi = group_episodes(list(qual.index), list(series.index), episode_gap)
    df = pd.DataFrame({"stat": qual, "level": series.reindex(qual.index),
                       "episode": [epi[d] for d in qual.index]})
    df["kept"] = False
    for _e, g in df.groupby("episode"):
        keep = g.sort_values("stat", ascending=False).index[:max_per_episode]
        df.loc[keep, "kept"] = True
    df.attrs["threshold"] = thr
    return df.sort_index()


def select_mid_band(series: pd.Series, stat: pd.Series, band, n_mid: int,
                    exclude) -> list:
    """Evenly spaced (deterministic) dates whose stat sits in the quantile band."""
    if n_mid <= 0:
        return []
    lo, hi = (float(stat.quantile(band[0])), float(stat.quantile(band[1])))
    cand = [d for d in stat[(stat >= lo) & (stat < hi)].dropna().index
            if d not in exclude]
    cand.sort()
    if len(cand) <= n_mid:
        return cand
    idx = np.linspace(0, len(cand) - 1, n_mid).round().astype(int)
    return [cand[i] for i in sorted(set(idx))]


# ── controls ──────────────────────────────────────────────────────────────────
def pick_control(day, series: pd.Series, level_caps, forbidden: set):
    """Same weekday, 350-371 calendar days earlier, a trading day in the series,
    not a one-off closure, volatility LEVEL below a cap, not in `forbidden`
    (qualifying days + already-used controls). `level_caps` is a list of
    (quantile, cap) tried in order -- the GRADUATED screen: the strict cap
    first, then the fallback, so a spike year whose whole prior year sits above
    the median (volatility regimes are persistent; 2021 before 2022 is the live
    case) yields a relaxed-but-screened pair with the relaxation RECORDED,
    rather than an unpairable day. Closest to 364 wins within a cap.
    -> (control date or None, gap_days, screen_q used, reason when None)."""
    v = pd.Timestamp(day).normalize()
    window = series.loc[v - pd.Timedelta(days=PAIR_MAX_D): v - pd.Timedelta(days=PAIR_MIN_D)]
    base = []
    for b, lvl in window.items():
        if b.dayofweek != v.dayofweek or b in forbidden:
            continue
        if str(b.date()) in vs._ONE_OFF:
            continue
        if not np.isfinite(lvl):
            continue
        base.append((abs((v - b).days - PAIR_TARGET_D), b, lvl))
    for q, cap in level_caps:
        cands = [(k, b) for k, b, lvl in base if lvl <= cap]
        if cands:
            _, best = min(cands)
            return best, int((v - best).days), float(q), ""
    return None, None, float("nan"), ("no same-weekday candidate below any vol screen "
                                      "in 350-371d")


def assign_controls(days, series: pd.Series, control_max_q: float, qualifying: set,
                    fallback_q: float = 0.75):
    """Controls for `days` in order (highest-priority first). -> DataFrame with
    the screen quantile each pair needed (screen_q > control_max_q marks the
    relaxed pairs -- report them, they are controls with a caveat)."""
    caps = [(control_max_q, float(series.quantile(control_max_q)))]
    if fallback_q > control_max_q:
        caps.append((fallback_q, float(series.quantile(fallback_q))))
    used = set()
    rows = []
    for d in days:
        c, gap, sq, why = pick_control(d, series, caps, qualifying | used)
        if c is not None:
            used.add(c)
        rows.append({"date": pd.Timestamp(d), "control": c, "gap_days": gap,
                     "screen_q": sq,
                     "control_level": float(series.get(c, np.nan)) if c is not None else np.nan,
                     "unpairable_reason": why})
    return pd.DataFrame(rows).set_index("date")


# ── power ─────────────────────────────────────────────────────────────────────
def _z(p: float) -> float:
    from scipy.stats import norm
    return float(norm.ppf(p))


def required_days_per_group(diff: float, day_sd: float, alpha: float = 0.05,
                            power: float = 0.80) -> int:
    """Two-sample comparison of day-level means: n per group so a true difference
    `diff` is detected with `power` at two-sided `alpha`, given cross-day SD."""
    z = _z(1.0 - alpha / 2.0) + _z(power)
    return int(np.ceil(2.0 * (z * day_sd / diff) ** 2))


def achieved_power(diff: float, day_sd: float, n1: int, n2: int,
                   alpha: float = 0.05) -> float:
    from scipy.stats import norm
    if min(n1, n2) < 2:
        return float("nan")
    se = day_sd * np.sqrt(1.0 / n1 + 1.0 / n2)
    return float(norm.cdf(abs(diff) / se - _z(1.0 - alpha / 2.0)))


def day_sd_from_csv(path: str, metric: str = "CS_ES") -> float:
    df = pd.read_csv(path)
    if metric not in df.columns:
        raise SystemExit(f"{path}: no column {metric!r} (have {list(df.columns)})")
    v = pd.to_numeric(df[metric], errors="coerce").dropna()
    if len(v) < 5:
        raise SystemExit(f"{path}: only {len(v)} finite {metric} rows")
    return float(v.std(ddof=1))


# ── report ────────────────────────────────────────────────────────────────────
def build_design(series, stat_kind, window, threshold_q, episode_gap, max_per_episode,
                 control_max_q, mid_band, n_mid, diff, day_sd, alpha, power):
    lo, hi = window
    s = series.loc[lo:hi]
    if len(s) < 300:
        raise SystemExit(f"window {lo}..{hi} holds only {len(s)} rows")
    st = selection_stat(s, stat_kind)
    sel = select_volatile(s, st, threshold_q, episode_gap, max_per_episode)
    vol_days = list(sel.index[sel["kept"]])
    qualifying = set(sel.index)
    ctl = assign_controls(vol_days, s, control_max_q, qualifying)
    mid_days = select_mid_band(s, st, mid_band, n_mid, qualifying)
    mid_ctl = assign_controls(mid_days, s, control_max_q, qualifying) if mid_days else pd.DataFrame()
    n_vol = len(vol_days)
    paired = ctl[ctl["control"].notna()]
    n_req = required_days_per_group(diff, day_sd, alpha, power)
    pw = achieved_power(diff, day_sd, len(paired), len(paired), alpha)
    return {"series_window": (str(s.index[0].date()), str(s.index[-1].date())),
            "threshold": float(sel.attrs.get("threshold", np.nan)),
            "selected": sel, "volatile": vol_days, "controls": ctl,
            "mid": mid_days, "mid_controls": mid_ctl,
            "n_episodes": int(sel.loc[sel["kept"], "episode"].nunique()) if n_vol else 0,
            "n_volatile": n_vol, "n_paired": int(len(paired)),
            "n_required_per_group": n_req, "achieved_power": pw,
            "params": {"stat": stat_kind, "threshold_q": threshold_q,
                       "episode_gap": episode_gap, "max_per_episode": max_per_episode,
                       "control_max_q": control_max_q, "diff": diff, "day_sd": day_sd,
                       "alpha": alpha, "power": power}}


def render(design) -> str:
    p_ = design["params"]
    p = p_
    L = []
    L.append("SAMPLE DESIGN (derivable rule; see design_sample.py)")
    L.append("series window: %s .. %s" % design["series_window"])
    L.append("selection: %s >= q%.2f (threshold %.4f); episodes joined at <=%d trading days;"
             % (p["stat"], p["threshold_q"], design["threshold"], p["episode_gap"]))
    L.append("           max %d day(s)/episode; control screen: level <= q%.2f"
             % (p["max_per_episode"], p["control_max_q"]))
    L.append("")
    L.append("POWER: detect diff=%.3f at alpha=%.2f with day-level SD=%.3f" %
             (p["diff"], p["alpha"], p["day_sd"]))
    L.append("  required per regime (%.0f%% power): %d days   selected pairs: %d   "
             "power at selected n: %.0f%%"
             % (100 * p["power"], design["n_required_per_group"], design["n_paired"],
                100 * design["achieved_power"]))
    verdict = ("SUFFICIENT" if design["n_paired"] >= design["n_required_per_group"]
               else "INSUFFICIENT -- widen the window or lower --threshold-q")
    L.append("  verdict: %s" % verdict)
    L.append("")
    L.append("VOLATILE (%d days, %d episodes):" % (design["n_volatile"], design["n_episodes"]))
    sel, ctl = design["selected"], design["controls"]
    for d in design["volatile"]:
        r, c = sel.loc[d], ctl.loc[pd.Timestamp(d)]
        if pd.notna(c["control"]):
            relax = "" if c["screen_q"] <= p_["control_max_q"] else " RELAXED q%.2f" % c["screen_q"]
            cc = str(c["control"].date()) + " (%dd, level %.2f%s)" % (c["gap_days"], c["control_level"], relax)
        else:
            cc = "UNPAIRABLE: " + c["unpairable_reason"]
        L.append("  %s  ep%02d  stat=%+.4f  level=%.2f  control=%s"
                 % (d.date(), r["episode"], r["stat"], r["level"], cc))
    if design["mid"]:
        L.append("")
        L.append("MID-BAND (dose-response, %d days):" % len(design["mid"]))
        for d in design["mid"]:
            c = design["mid_controls"].loc[pd.Timestamp(d)]
            cc = str(c["control"].date()) if pd.notna(c["control"]) else "UNPAIRABLE"
            L.append("  %s  control=%s" % (d.date(), cc))
    return "\n".join(L)


def emit_args(design) -> str:
    """Ready-to-paste driver arguments. Mid-band days are emitted SEPARATELY: the
    driver's taxonomy is binary (--volatile list vs everything-else=benchmark), so
    folding mid days into either list would contaminate the two-group contrast --
    they exist for the dose-response regression on the continuous state. To extract
    them, append the mid list to --baseline and EXCLUDE those dates from any binary
    regime table; the comment line carries them."""
    vol = [str(d.date()) for d in design["volatile"]]
    base = [str(c.date()) for c in design["controls"]["control"] if pd.notna(c)]
    out = "--volatile %s \\\n--baseline %s" % (",".join(sorted(vol)), ",".join(sorted(base)))
    mid = [str(d.date()) for d in design["mid"]]
    if mid:
        mctl = [str(c.date()) for c in design["mid_controls"]["control"] if pd.notna(c)] \
            if len(design["mid_controls"]) else []
        out += ("\n# mid-band (dose-response; append to --baseline for extraction, EXCLUDE from"
                "\n# binary regime contrasts): %s"
                "\n# mid-band controls: %s" % (",".join(mid), ",".join(mctl)))
    return out


# ── selftest & CLI ────────────────────────────────────────────────────────────
def _synthetic_series(seed=0):
    """Business-day vol series 2018-2026 with planted spike episodes."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2018-01-02", "2026-08-14")
    idx = idx[[str(d.date()) not in vs._ONE_OFF for d in idx]]
    v = 15.0 * np.exp(0.2 * rng.standard_normal(len(idx)).cumsum() * 0.02)
    s = pd.Series(v, index=idx)
    return s


def _selftest() -> int:
    s = _synthetic_series()
    spikes = ["2020-03-16", "2020-03-17", "2020-03-18", "2022-06-13", "2024-08-05"]
    for i, d in enumerate(spikes):
        t = pd.Timestamp(d)
        if t in s.index:
            s.loc[t] = s.loc[t] * (3.0 + i * 0.1)
    d = build_design(s, "logdiff", ("2018-01-01", "2026-08-14"), 0.995, 5, 2,
                     0.5, (0.55, 0.85), 4, 0.20, 0.235, 0.05, 0.80)
    print(render(d))
    print()
    print(emit_args(d))
    got = {str(x.date()) for x in d["volatile"]}
    ok = {"2020-03-16", "2022-06-13", "2024-08-05"} <= got
    print("\nselftest: planted spikes recovered:", ok)
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="sample design: threshold selection, power, controls")
    ap.add_argument("--series", default="", help="CSV with a date column + a volatility column")
    ap.add_argument("--date-col", default="")
    ap.add_argument("--col", default="", help="volatility column (default: last numeric)")
    ap.add_argument("--stat", choices=["logdiff", "level"], default="logdiff")
    ap.add_argument("--window", default="2018-01-01:2026-12-31", help="lo:hi dates")
    ap.add_argument("--threshold-q", type=float, default=0.95)
    ap.add_argument("--episode-gap", type=int, default=5, help="trading days joining an episode")
    ap.add_argument("--max-per-episode", type=int, default=3)
    ap.add_argument("--control-max-q", type=float, default=0.50,
                    help="control days must sit below this vol-LEVEL quantile")
    ap.add_argument("--mid-band", default="0.55:0.85", help="quantile band for dose-response days")
    ap.add_argument("--n-mid", type=int, default=0, help="mid-band day count (0 = off)")
    ap.add_argument("--diff", type=float, default=0.20, help="effect size to detect")
    ap.add_argument("--day-sd", type=float, default=0.235,
                    help="cross-day SD of the target metric (2026-08-14 run: 0.235 for CS_ES)")
    ap.add_argument("--per-day-csv", default="",
                    help="estimate --day-sd from a prior run's per-day CSV instead")
    ap.add_argument("--metric", default="CS_ES")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--emit-args", action="store_true",
                    help="print ready-to-paste --volatile/--baseline lists")
    ap.add_argument("--out", default="", help="write the report here as well")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return _selftest()
    if not a.series:
        ap.error("--series is required (or --selftest)")
    s = load_series(a.series, a.date_col, a.col)
    lo, _, hi = a.window.partition(":")
    day_sd = day_sd_from_csv(a.per_day_csv, a.metric) if a.per_day_csv else a.day_sd
    b0, _, b1 = a.mid_band.partition(":")
    design = build_design(s, a.stat, (lo, hi or str(s.index[-1].date())), a.threshold_q,
                          a.episode_gap, a.max_per_episode, a.control_max_q,
                          (float(b0), float(b1)), a.n_mid, a.diff, day_sd,
                          a.alpha, a.power)
    txt = render(design)
    print(txt)
    if a.emit_args:
        print()
        print(emit_args(design))
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(txt + "\n\n" + emit_args(design) + "\n")
        print("\nwrote %s" % a.out)
    # the shipped pairing validator gets the last word
    vol = [str(d.date()) for d in design["volatile"]]
    base = [str(c.date()) for c in design["controls"]["control"] if pd.notna(c)]
    if vol and base:
        pair = vs.check_pairing(vol[:len(base)], base)
        bad = pair[pair["level"] == "ERROR"] if "level" in pair.columns else pd.DataFrame()
        if len(bad):
            print("\nWARNING: %d pair(s) fail validate_sample.check_pairing -- inspect" % len(bad))
    return 0


if __name__ == "__main__":
    sys.exit(main())
