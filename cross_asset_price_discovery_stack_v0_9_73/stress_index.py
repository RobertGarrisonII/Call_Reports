"""
stress_index.py
==============
Turn a daily conditional-volatility series (e.g. NYU V-Lab MF2-GARCH on SPY) into the
stress objects the liquidity-contagion redo needs, fixing the day-selection issues:

  * continuous standardized stress STATE  -> the conditioning input S for the spread /
    information-share machinery (ecm_sde, correlation_svar.spread_conditioned_is). The
    contagion regressions should use this, not binned days.
  * long-run / short-run split (tau_t, g_t): tau_t is the persistent-stress regime
    signal (MF2's long-run component); g_t is the transient piece. Pass V-Lab's long-run
    series directly if you have it; otherwise tau is approximated by a long smoother.
  * ONSET (acceleration) flag = ln(sigma_t / sigma_{t-1}) > thresh -- a sharp-shock
    detector, kept SEPARATE from the level (it misses sustained-stress plateaus).
  * percentile REGIMES (calm / elevated / stress) from the EX-ANTE level -- for Rigobon
    heteroskedasticity ID (#2) and the Markov-switching VECM (#4).
  * vol SURPRISE = ln(realized / forecast) -- a look-ahead-free stress innovation.
  * a broadcaster mapping the daily state onto an intraday session index.

NOTE on the design this replaces: selecting "volatile" by a one-day log-vol change > 0.5
captures acceleration but excludes crisis plateaus; selecting "treatment" by log change
< 0 is endogenous (vol falls because the episode is resolving) and is not a treatment.
The MWCB treatment (#1) is the reopening asymmetry, dated at the release boundary, not a
vol threshold. Pure numpy / pandas.
"""
import numpy as np
import pandas as pd

EPS = 1e-12


def log_change(vol):
    """Daily log-change of conditional vol, ln(sigma_t / sigma_{t-1}) (volatility
    acceleration). Leading NaN. This is an ONSET signal, not a level."""
    v = pd.Series(np.asarray(vol, float), index=getattr(vol, "index", None))
    return np.log(v / v.shift(1))


def long_short_decomposition(vol, long_run=None, window=63, method="rolling", halflife=21):
    """Split conditional vol into a long-run component tau_t (persistent stress regime)
    and a short-run ratio g_t = vol / tau (mean ~1). If `long_run` (V-Lab's MF2 long-run
    series) is supplied it is used directly -- preferred; otherwise tau is APPROXIMATED by
    a long smoother (rolling mean over `window` days, or EWMA with `halflife`). The
    approximation is a stand-in for MF2's MIDAS long-run filter, not a re-estimation."""
    v = pd.Series(np.asarray(vol, float), index=getattr(vol, "index", None))
    if long_run is not None:
        tau = pd.Series(np.asarray(long_run, float), index=v.index)
    elif method == "ewma":
        tau = v.ewm(halflife=halflife, min_periods=max(5, halflife // 2)).mean()
    else:
        tau = v.rolling(window, min_periods=max(5, window // 3)).mean()
    g = v / tau.replace(0.0, np.nan)
    return tau, g


def stress_state(vol, long_run=None, use="long_run", robust=True, **kw):
    """Continuous standardized stress STATE for conditioning (the recommended contagion
    regressor / ecm_sde state). use='long_run' (default) standardizes log tau_t (the
    persistent component); use='total' standardizes log vol; use='surprise' expects a
    (realized, forecast) split via stress_surprise instead. Standardization is robust
    (median / MAD) by default so a single crisis does not dominate the scale."""
    if use == "total":
        x = np.log(np.maximum(pd.Series(np.asarray(vol, float)), EPS))
    else:
        tau, _ = long_short_decomposition(vol, long_run, **kw)
        x = np.log(np.maximum(tau, EPS))
    x = np.asarray(x, float)
    if robust:
        med = np.nanmedian(x); mad = np.nanmedian(np.abs(x - med))
        s = 1.4826 * mad
        if not (s > EPS):
            s = np.nanstd(x); s = s if s > EPS else 1.0
        z = (x - med) / s
    else:
        mu = np.nanmean(x); sd = np.nanstd(x); z = (x - mu) / (sd if sd > EPS else 1.0)
    return pd.Series(z, index=getattr(vol, "index", None))


def stress_surprise(realized_vol, forecast_vol):
    """Look-ahead-free stress innovation, ln(realized / forecast). `forecast_vol` is the
    ex-ante one-step MF2 forecast sigma_{t|t-1}; `realized_vol` the realized vol for t."""
    r = np.asarray(realized_vol, float); f = np.asarray(forecast_vol, float)
    return pd.Series(np.log(np.maximum(r, EPS) / np.maximum(f, EPS)),
                     index=getattr(realized_vol, "index", None))


def classify_regimes(level_series, q_low=0.33, q_high=0.90):
    """Percentile REGIMES from an EX-ANTE level series (vol or its long-run component):
    'calm' below q_low, 'stress' at/above q_high, 'elevated' between. Returns a
    string Series. For Rigobon ID (#2) the three regimes give the distinct shock
    covariances that pin down the contemporaneous structural matrix with no ordering."""
    s = pd.Series(np.asarray(level_series, float), index=getattr(level_series, "index", None))
    pct = s.rank(pct=True)
    out = pd.Series(np.where(pct <= q_low, "calm",
                    np.where(pct >= q_high, "stress", "elevated")), index=s.index)
    out[s.isna()] = np.nan
    return out


def select_days(vol, long_run=None, level_q=0.90, onset_thresh=0.5, calm_q=0.33,
                calm_change=0.10, **kw):
    """Recommended day labeling for the contagion design, replacing the
    'change > 0.5 / change < 0' rule. Returns a DataFrame with vol, tau, g, log_change,
    level_pct, onset (bool), and `label`:
        'stress'   : sustained high level (level_pct >= level_q) OR sharp onset
                     (log_change > onset_thresh)        <- captures the plateaus the
                                                            pure-change rule misses
        'calm'     : low level (level_pct <= calm_q) AND stable (|log_change| < calm_change)
        'elevated' : otherwise
    The MWCB treatment is NOT a vol label -- it is the reopening-asymmetry event (#1)."""
    v = pd.Series(np.asarray(vol, float), index=getattr(vol, "index", None))
    tau, g = long_short_decomposition(v, long_run, **kw)
    lc = log_change(v)
    level = tau if (long_run is not None or True) else v       # regime on the persistent level
    pct = level.rank(pct=True)
    onset = lc > onset_thresh
    stress = (pct >= level_q) | onset
    calm = (pct <= calm_q) & (lc.abs() < calm_change)
    label = pd.Series(np.where(stress, "stress", np.where(calm, "calm", "elevated")), index=v.index)
    return pd.DataFrame({"vol": v, "tau": tau, "g": g, "log_change": lc,
                         "level_pct": pct, "onset": onset, "label": label})


def match_controls(panel, label_col="label", level_col="tau", stress="stress", control="calm"):
    """Nearest-vol-level control for each stress day (a synthetic-control-lite matcher for
    the DiD, #1). Returns a list of (stress_date, control_date) pairs matching on the
    persistent level. Prefer this to the paper's same-weekday-prior-year match when a
    close vol match exists."""
    s = panel[panel[label_col] == stress]; c = panel[panel[label_col] == control]
    if len(c) == 0:
        return []
    cv = c[level_col].to_numpy(); cidx = c.index.to_numpy()
    pairs = []
    for d, lv in s[level_col].items():
        if not np.isfinite(lv):
            continue
        j = int(np.nanargmin(np.abs(cv - lv)))
        pairs.append((d, cidx[j]))
    return pairs


def broadcast_to_session(daily_state, session_index):
    """Map a per-day state (Series indexed by date) onto an intraday DatetimeIndex, so the
    daily stress state becomes the per-bar conditioning state S for the spread / IS
    machinery. Matches on calendar date regardless of timezone. Returns a numpy array
    aligned to session_index."""
    def _dates(ix):
        ix = pd.DatetimeIndex(pd.to_datetime(ix))
        if ix.tz is not None:
            ix = ix.tz_localize(None)
        return ix.normalize()
    key = pd.Series(np.asarray(daily_state, float), index=_dates(daily_state.index))
    key = key[~key.index.duplicated()]
    return key.reindex(_dates(session_index)).to_numpy()


# ── self-test ─────────────────────────────────────────────────────────────────
def _synthetic_vol(seed=0):
    """A daily conditional-vol path: calm ~12, sharp ramp to ~55 (onset), a multi-day
    plateau, then decay -- the plateau is exactly what a one-day-change rule misses."""
    rng = np.random.default_rng(seed)
    base = np.r_[np.full(40, 12.0),
                 np.linspace(12, 55, 4),            # onset ramp
                 np.full(12, 55.0),                 # crisis plateau
                 np.linspace(55, 14, 20),           # decay
                 np.full(40, 13.0)]
    vol = base * np.exp(rng.normal(0, 0.04, len(base)))
    idx = pd.bdate_range("2014-01-02", periods=len(vol))
    return pd.Series(vol, index=idx)


def _selftest():
    print("stress_index self-test")
    vol = _synthetic_vol()

    # (1) long/short split + continuous state
    tau, g = long_short_decomposition(vol, window=21)
    st = stress_state(vol, use="long_run", window=21)
    print("\n(1) decomposition + continuous state")
    print("    std(log tau)=%.3f < std(log vol)=%.3f (tau smoother);  mean g=%.3f (~1)"
          % (np.nanstd(np.log(tau)), np.nanstd(np.log(vol)), np.nanmean(g)))
    print("    stress state: min=%.2f  max=%.2f  (peaks in the crisis window)" % (st.min(), st.max()))
    print("    checks:", np.nanstd(np.log(tau)) < np.nanstd(np.log(vol))
          and 0.8 < np.nanmean(g) < 1.25 and st.max() > 1.5 and np.isfinite(st).sum() > 100)

    # (2) the improvement: recommended labels catch the plateau the pure-change rule misses
    panel = select_days(vol, window=21, level_q=0.90, onset_thresh=0.5)
    onset_only = int(panel["onset"].sum())
    stress_total = int((panel["label"] == "stress").sum())
    print("\n(2) day selection vs the pure one-day-change rule")
    print("    onset (ln change > 0.5) days = %d   |   recommended 'stress' days = %d (adds the plateau)"
          % (onset_only, stress_total))
    print("    regime counts:", panel["label"].value_counts().to_dict())
    print("    checks:", stress_total > onset_only and onset_only >= 1
          and set(panel["label"].unique()) <= {"calm", "elevated", "stress"}
          and (panel["label"] == "calm").sum() > 0)

    # (3) percentile regimes + ex-ante surprise
    reg = classify_regimes(tau, q_low=0.33, q_high=0.90)
    rng = np.random.default_rng(1)
    surprise = stress_surprise(vol * np.exp(rng.normal(0, 0.1, len(vol))), tau)
    print("\n(3) percentile regimes + ex-ante surprise")
    print("    regime counts:", reg.value_counts().to_dict())
    print("    surprise mean=%.3f std=%.3f (look-ahead-free innovation)"
          % (np.nanmean(surprise), np.nanstd(surprise)))
    print("    checks:", set(reg.dropna().unique()) <= {"calm", "elevated", "stress"}
          and (reg == "stress").sum() > 0 and (reg == "calm").sum() > 0 and np.isfinite(surprise).sum() > 100)

    # (4) control matching + intraday broadcast
    pairs = match_controls(panel)
    sess = pd.date_range("2014-02-27 09:30", periods=3 * 390, freq="min", tz="America/New_York")
    bc = broadcast_to_session(st, sess)
    days_covered = len(np.unique(pd.DatetimeIndex(sess).normalize()))
    print("\n(4) control matching + intraday broadcast")
    print("    matched stress->calm pairs: %d   |   broadcast covers %d intraday days, len=%d"
          % (len(pairs), days_covered, len(bc)))
    print("    checks:", len(pairs) >= 1 and len(bc) == len(sess)
          and np.isfinite(bc).any() and len(np.unique(bc[np.isfinite(bc)])) <= days_covered)


if __name__ == "__main__":
    _selftest()
