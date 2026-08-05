"""
price_discovery_shares.py
=========================
Cross-asset price-discovery metrics for the SPY-ES study, consuming the aligned
per-session mid-price series from the liquidity pipeline (mid = (bidprice_1 +
askprice_1)/2 per asset on a common 1s grid).

Reports, per market, the verified and non-contested measures:
  * Hasbrouck (1995) Information Share        -> lower / upper / midpoint
  * Lien-Shrestha (2009) generalized IS       -> unique, order-invariant
  * Gonzalo-Granger (1995) Component Share    -> permanent-transitory leadership

Putnins' (2013) Information Leadership Share is intentionally NOT reported by
default: the formula is orientation-subtle and the measure is actively contested
(Shrestha & Lee 2023 find serious flaws; Shen & Zivot 2024 defend; Shen, Zhang &
Zivot 2025 JEF propose a corrected version). On a known-leader simulation the
vanilla ILS could contradict both IS and CS. Treat leadership-net-of-noise
qualitatively, or implement the Shen-Zhang-Zivot (2025) variant after validating
the exact formula against source. See information_leadership_share() stub.

Estimation is PER SESSION (stacking non-contiguous days into one VECM is invalid:
overnight gaps, contract rolls). estimate_sample() loops days; panel_vecm() pools
with day FE + a regime interaction on alpha; compare_regimes() does permutation
inference. Cointegrating vector fixed at the no-arbitrage relation beta=(1,-1) on
log mids; the SPY scale and intraday basis are within-day constants absorbed by a
per-day mean (a day FE on the error-correction term).

For a bivariate cointegrated system with beta=(1,-1), the common-factor row
vector is psi proportional to alpha_perp = (alpha_2, -alpha_1); beta_perp=(1,1)'
makes the long-run rows identical and the scalar short-run term cancels, so
IS/MIS/CS depend only on (alpha, Omega). Dependencies: numpy, pandas.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

EPS = 1e-15


# ── linear algebra ───────────────────────────────────────────────────────────
def _ols(Y, X):
    B, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return B, Y - X @ B

def _sym_sqrt(M):
    w, V = np.linalg.eigh(M)
    return (V * np.sqrt(np.clip(w, 0.0, None))) @ V.T


# ── VECM with fixed cointegrating vector (1,-1) on log prices ─────────────────
def _design_within_day(p1, p2, n_lags, start=None):
    """Δp_t = c + α·z_{t-1} + Σ_{L=1..n_lags} Γ_L Δp_{t-L} + ε_t, with z=(p1-p2)-mean (β=(1,-1)).
    ``start`` is the first usable row (defaults to n_lags). Pass start=pmax when comparing candidate
    lag orders so every candidate is fit on the SAME sample (else an information criterion favours
    larger p spuriously, by scoring fewer effective rows)."""
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    z = (p1 - p2); z = z - np.nanmean(z)                  # beta=(1,-1) + day const
    dp1 = np.diff(p1); dp2 = np.diff(p2); T = dp1.shape[0]
    s = n_lags if start is None else int(start)
    cols = [np.ones(T - s), z[s:T]]                       # const + EC term z_{t-1}
    for L in range(1, n_lags + 1):
        cols.append(dp1[s - L:T - L]); cols.append(dp2[s - L:T - L])
    dY = np.column_stack([dp1[s:T], dp2[s:T]])
    X = np.column_stack(cols)
    # Finite rows only. Halt snapshots arrive as NaN (market_halts.mask_frame), and NaN propagates
    # through the diffs and every lag column, so this single mask drops the halt, the halt/reopen
    # seam, and every observation whose lag window touches either -- for the VECM, both information
    # shares, Gonzalo-Granger, lag selection, the windowed panel and the jump split, all of which
    # come through this function. Before it existed a single NaN poisoned the whole OLS instead.
    ok = np.all(np.isfinite(dY), axis=1) & np.all(np.isfinite(X), axis=1)
    return dY[ok], X[ok]

def _fit_vecm_fixed(p1, p2, n_lags):
    dY, X = _design_within_day(p1, p2, n_lags)
    B, resid = _ols(dY, X)
    return B[1, :].copy(), np.cov(resid, rowvar=False, bias=True), resid


# ── information-criterion lag-order selection (BIC default; AIC / HQ available) ─
_CRIT_KEY = {"bic": "bic", "sbc": "bic", "schwarz": "bic", "schwartz": "bic",
             "aic": "aic", "hqic": "hqic", "hq": "hqic", "hannan-quinn": "hqic"}


def _ic_row(dY, X, p, n_ec=1):
    """(logdet, AIC, BIC, HQ) for one candidate. Penalty counts free conditional-mean parameters
    m(p)=K·(1 + n_ec + K·p) [intercept + EC loadings + Γ_1..Γ_p]; β is fixed so it is not counted.
    The intercept/EC terms are p-invariant, so they shift every criterion equally and do not affect
    the argmin -- they are included only so the reported values are the standard Lütkepohl form."""
    T = dY.shape[0]; K = dY.shape[1]
    _B, resid = _ols(dY, X)
    Sigma = np.atleast_2d(np.cov(resid, rowvar=False, bias=True))
    sign, ld = np.linalg.slogdet(Sigma)
    if sign <= 0 or not np.isfinite(ld) or T <= 1:
        return (np.nan, np.nan, np.nan, np.nan)
    m = K * (1 + n_ec + K * p)
    aic = ld + (2.0 / T) * m
    bic = ld + (np.log(T) / T) * m
    hq = ld + (2.0 * np.log(np.log(T)) / T) * m if T > np.e else np.nan
    return (float(ld), float(aic), float(bic), float(hq))


def select_lag(p1, p2, pmax=10, criterion="bic", pmin=0):
    """Pick the VECM lagged-difference order p∈[pmin,pmax] minimising the chosen criterion, with every
    candidate fit on the common sample of the largest model (start=pmax). Returns (p*, table) where
    ``table`` is a DataFrame indexed by p with logdet/aic/bic/hqic. p=0 means a pure error-correction
    model (no lagged differences); n_lags here is the VECM lag, one less than the levels-VAR order."""
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    ok = np.isfinite(p1) & np.isfinite(p2); p1, p2 = p1[ok], p2[ok]
    rows = []
    for p in range(max(0, pmin), pmax + 1):
        dY, X = _design_within_day(p1, p2, p, start=pmax)
        if dY.shape[0] < X.shape[1] + 5:                  # too few obs to score this candidate
            rows.append((p, np.nan, np.nan, np.nan, np.nan)); continue
        ld, aic, bic, hq = _ic_row(dY, X, p)
        rows.append((p, ld, aic, bic, hq))
    tab = pd.DataFrame(rows, columns=["p", "logdet", "aic", "bic", "hqic"]).set_index("p")
    col = tab[_CRIT_KEY.get(str(criterion).lower(), "bic")]
    p_star = int(col.idxmin()) if col.notna().any() else max(0, pmin)
    return p_star, tab


def select_lag_pooled(price_pairs, pmax=10, criterion="bic", pmin=0):
    """One lag order for a set of sessions: minimise the SUMMED criterion Σ_d IC_d(p) (the pooled
    objective for cross-session comparability). ``price_pairs`` is an iterable of (p1, p2) log-price
    series. Returns (p*, pooled_table) where pooled_table has one column per session plus 'pooled'."""
    key = _CRIT_KEY.get(str(criterion).lower(), "bic")
    cols = {}
    for i, (p1, p2) in enumerate(price_pairs):
        _, tab = select_lag(p1, p2, pmax=pmax, criterion=criterion, pmin=pmin)
        cols[f"s{i}"] = tab[key]
    pooled = pd.DataFrame(cols)
    pooled["pooled"] = pooled.sum(axis=1, min_count=1)
    p_star = int(pooled["pooled"].idxmin()) if pooled["pooled"].notna().any() else max(0, pmin)
    return p_star, pooled


def select_lag_var(X, pmax=10, criterion="bic", pmin=0):
    """IC lag selection for a reduced-form VAR(p) on a T×K matrix X (intercept + p lags), every
    candidate fit on the common sample of the largest model. Generic in K, so it serves the SVAR
    blocks (and the n-asset path later). Returns (p*, table). For the bivariate fixed-β price VECM use
    ``select_lag`` instead (it scores the error-correction model rather than a plain VAR)."""
    X = np.asarray(X, float)
    X = X[np.isfinite(X).all(axis=1)]
    T0, K = X.shape
    rows = []
    for p in range(max(0, pmin), pmax + 1):
        Y = X[pmax:]
        Z = np.column_stack([np.ones(Y.shape[0])] + [X[pmax - i:T0 - i] for i in range(1, p + 1)])
        if Y.shape[0] < Z.shape[1] + 5:
            rows.append((p, np.nan, np.nan, np.nan, np.nan)); continue
        ld, aic, bic, hq = _ic_row(Y, Z, p, n_ec=0)       # plain VAR: only the intercept is fixed
        rows.append((p, ld, aic, bic, hq))
    tab = pd.DataFrame(rows, columns=["p", "logdet", "aic", "bic", "hqic"]).set_index("p")
    col = tab[_CRIT_KEY.get(str(criterion).lower(), "bic")]
    p_star = int(col.idxmin()) if col.notna().any() else max(1, pmin)
    return p_star, tab


# ── shares from (alpha, Omega) ───────────────────────────────────────────────
def _psi(alpha):
    """Common-factor weights for beta=(1,-1): psi = (alpha_2, -alpha_1)."""
    a1, a2 = alpha
    return np.array([a2, -a1], dtype=float)

def gonzalo_granger(alpha):
    """Component shares CS_j = psi_j / sum(psi). Leadership = not adjusting."""
    psi = _psi(alpha); s = psi.sum()
    cs = psi / s if abs(s) > EPS else np.array([0.5, 0.5])
    return np.abs(cs) / np.abs(cs).sum()

def _is_for_factor(psi, F, denom):
    return (psi @ F) ** 2 / denom

def hasbrouck_is(alpha, Omega):
    """IS lower/upper bounds (two Cholesky orderings) + midpoint, per market."""
    psi = _psi(alpha); denom = float(psi @ Omega @ psi.T)
    if denom <= EPS:
        z = np.array([0.5, 0.5]); return z, z, z
    F1 = np.linalg.cholesky(Omega)
    is1 = _is_for_factor(psi, F1, denom)
    P = np.array([[0.0, 1.0], [1.0, 0.0]])
    F2 = P @ np.linalg.cholesky(P @ Omega @ P) @ P
    is2 = _is_for_factor(psi, F2, denom)
    lo = np.minimum(is1, is2); hi = np.maximum(is1, is2)
    return lo, hi, (lo + hi) / 2.0

def lien_shrestha_is(alpha, Omega):
    """Unique, order-invariant IS via symmetric correlation factorization:
    F = V * corr^{1/2}_sym, FF' = Omega."""
    psi = _psi(alpha); denom = float(psi @ Omega @ psi.T)
    if denom <= EPS:
        return np.array([0.5, 0.5])
    v = np.sqrt(np.diag(Omega))
    Vinv = np.diag(1.0 / np.where(v > EPS, v, np.nan))
    F = np.diag(v) @ _sym_sqrt(Vinv @ Omega @ Vinv)
    return _is_for_factor(psi, F, denom)

def information_leadership_share(*_a, **_k):
    """DEFERRED / contested. The Putnins (2013) ILS is orientation-subtle and
    academically disputed (Shrestha & Lee 2023; Shen & Zivot 2024; corrected by
    Shen, Zhang & Zivot 2025 JEF). On a known-leader sim the vanilla ILS could
    contradict both IS and CS. Use IS / MIS / CS as primary; implement the
    Shen-Zhang-Zivot (2025) variant here only after validating against source."""
    raise NotImplementedError(information_leadership_share.__doc__)


# ── per-session estimation ───────────────────────────────────────────────────
def estimate_day(mid_spy, mid_es, n_lags=5, names=("SPY", "ES"), use_log=True,
                 criterion=None, pmax=10):
    """One session's shares. If ``criterion`` is given ('bic'|'aic'|'hqic'), the VECM lag order is
    chosen by that criterion over [0, pmax] (BIC default elsewhere); otherwise the passed ``n_lags``
    is used (criterion=None reproduces the fixed-lag behaviour exactly). The lag used is returned in
    the ``n_lags`` field."""
    f = np.log if use_log else (lambda x: np.asarray(x, float))
    p1, p2 = f(np.asarray(mid_spy, float)), f(np.asarray(mid_es, float))
    ok = np.isfinite(p1) & np.isfinite(p2); p1, p2 = p1[ok], p2[ok]
    if criterion is not None:
        n_lags, _ = select_lag(p1, p2, pmax=pmax, criterion=criterion)
    alpha, Omega, _ = _fit_vecm_fixed(p1, p2, n_lags)
    lo, hi, mid = hasbrouck_is(alpha, Omega)
    cs = gonzalo_granger(alpha); mis = lien_shrestha_is(alpha, Omega)
    out = {"alpha_spy": alpha[0], "alpha_es": alpha[1], "n_obs": p1.shape[0] - n_lags,
           "n_lags": int(n_lags)}
    for j, nm in enumerate(names):
        out.update({f"IS_lo_{nm}": lo[j], f"IS_hi_{nm}": hi[j], f"IS_mid_{nm}": mid[j],
                    f"MIS_{nm}": mis[j], f"CS_{nm}": cs[j]})
    return out

def estimate_sample(sessions, n_lags=5, names=("SPY", "ES"), criterion=None, pmax=10,
                    lag_selection="pooled", use_log=True):
    """sessions: iterable of (label, regime, mid_spy, mid_es). Returns a per-day DataFrame of all
    shares. Lag order: with ``criterion`` set, ``lag_selection`` controls how p is chosen --
    'pooled' (default; ONE p minimising the summed criterion across all sessions, for cross-session
    comparability), 'per_day' (each session picks its own p), or 'fixed' (use ``n_lags``). With
    criterion=None the fixed ``n_lags`` is used (unchanged behaviour). When pooled, the chosen lag and
    the pooled IC table are attached to ``df.attrs``."""
    f = np.log if use_log else (lambda x: np.asarray(x, float))
    sessions = list(sessions)
    chosen, pooled_tab = n_lags, None
    if criterion is not None and lag_selection == "pooled":
        pairs = []
        for _label, _regime, m_spy, m_es in sessions:
            a = f(np.asarray(m_spy, float)); b = f(np.asarray(m_es, float))
            ok = np.isfinite(a) & np.isfinite(b); pairs.append((a[ok], b[ok]))
        chosen, pooled_tab = select_lag_pooled(pairs, pmax=pmax, criterion=criterion)
    rows = []
    for label, regime, m_spy, m_es in sessions:
        if criterion is not None and lag_selection == "per_day":
            r = estimate_day(m_spy, m_es, n_lags=n_lags, names=names, use_log=use_log,
                             criterion=criterion, pmax=pmax)
        else:
            use_lag = chosen if (criterion is not None and lag_selection == "pooled") else n_lags
            r = estimate_day(m_spy, m_es, n_lags=use_lag, names=names, use_log=use_log, criterion=None)
        r.update({"date": label, "regime": regime}); rows.append(r)
    df = pd.DataFrame(rows).set_index("date")
    if pooled_tab is not None:
        df.attrs["pooled_lag"] = int(chosen)
        df.attrs["pooled_ic_table"] = pooled_tab
        df.attrs["lag_criterion"] = str(criterion)
    return df


# ── regime inference (small-N -> permutation, not asymptotics) ────────────────
def compare_regimes(per_day, metric="CS_ES", n_perm=20000, seed=0):
    """Permutation test of mean(metric) across volatile vs benchmark sessions."""
    vol = per_day.loc[per_day.regime == "volatile", metric].to_numpy()
    ben = per_day.loc[per_day.regime == "benchmark", metric].to_numpy()
    obs = vol.mean() - ben.mean()
    pool = np.concatenate([vol, ben]); n_v = len(vol)
    rng = np.random.default_rng(seed); cnt = 0
    for _ in range(n_perm):
        rng.shuffle(pool)
        if abs(pool[:n_v].mean() - pool[n_v:].mean()) >= abs(obs) - EPS:
            cnt += 1
    return {"metric": metric, "vol_mean": float(vol.mean()), "ben_mean": float(ben.mean()),
            "diff": float(obs), "p_perm": (cnt + 1) / (n_perm + 1)}


# ── panel VECM: day FE + regime interaction on the adjustment term ────────────
def _cluster_se(X, u, groups, XtXinv=None):
    """One-way cluster-robust (Liang-Zeger) SEs, clustering by `groups` (session id per row), with the
    same small-cluster correction cross_asset_pd_liquidity.panel_regression / ecm_sde use:
    c = G/(G-1)*(N-1)/(N-K). Permits arbitrary within-day correlation, treats days as independent.
    Returns (se_vector, G)."""
    X = np.asarray(X, float); u = np.asarray(u, float)
    if XtXinv is None:
        XtXinv = np.linalg.pinv(X.T @ X)
    g = np.asarray(groups); uniq = np.unique(g); G = len(uniq); N, K = X.shape
    meat = np.zeros((K, K))
    for gi in uniq:
        m = g == gi; s = X[m].T @ u[m]; meat += np.outer(s, s)
    c = (G / (G - 1.0)) * ((N - 1.0) / (N - K)) if (G > 1 and N > K) else np.nan
    V = (XtXinv @ meat @ XtXinv) * c
    return np.sqrt(np.clip(np.diag(V), 0.0, None)), G


def panel_vecm(sessions, n_lags=5, names=("SPY", "ES"), criterion=None, pmax=10):
    """Pooled across sessions, lags within-day, common beta=(1,-1) with a per-day
    mean on z (day FE on the EC level), and a volatile-dummy interaction on the EC
    term so alpha_volatile = alpha + alpha_int. Returns regime-specific alpha,
    Omega, share family, and the day-clustered interaction t-stat (cluster = session).
    With ``criterion`` set, ONE pooled lag order (summed criterion across sessions) is chosen and
    applied to every day, so the pooled fit and the per-session shares share a specification."""
    sessions = list(sessions)
    if criterion is not None:
        pairs = []
        for _l, _r, m_spy, m_es in sessions:
            a = np.log(np.asarray(m_spy, float)); b = np.log(np.asarray(m_es, float))
            ok = np.isfinite(a) & np.isfinite(b); pairs.append((a[ok], b[ok]))
        n_lags, _ = select_lag_pooled(pairs, pmax=pmax, criterion=criterion)
    dYs, ECs, Ds, Ls, Gs = [], [], [], [], []
    for _gi, (label, regime, m_spy, m_es) in enumerate(sessions):
        p1 = np.log(np.asarray(m_spy, float)); p2 = np.log(np.asarray(m_es, float))
        ok = np.isfinite(p1) & np.isfinite(p2); p1, p2 = p1[ok], p2[ok]
        z = (p1 - p2); z = z - z.mean()                   # day FE on EC level
        dp1 = np.diff(p1); dp2 = np.diff(p2); T = len(dp1)
        dYs.append(np.column_stack([dp1[n_lags:T], dp2[n_lags:T]]))
        ECs.append(z[n_lags:T])
        Ds.append(np.full(T - n_lags, 1.0 if regime == "volatile" else 0.0))
        lag = [np.ones(T - n_lags)]
        for L in range(1, n_lags + 1):
            lag.append(dp1[n_lags - L:T - L]); lag.append(dp2[n_lags - L:T - L])
        Ls.append(np.column_stack(lag))
        Gs.append(np.full(T - n_lags, _gi))               # session id for day-clustered SEs
    dY = np.vstack(dYs); ec = np.concatenate(ECs); D = np.concatenate(Ds); Lall = np.vstack(Ls)
    groups = np.concatenate(Gs)
    X = np.column_stack([ec, ec * D, Lall])               # [ z , z*D , const+lags ]
    B, resid = _ols(dY, X)
    alpha_base = B[0, :].copy(); alpha_int = B[1, :].copy()
    alpha_vol = alpha_base + alpha_int
    Om_b = np.cov(resid[D == 0], rowvar=False, bias=True)
    Om_v = np.cov(resid[D == 1], rowvar=False, bias=True)

    def _shares(alpha, Om):
        lo, hi, mid = hasbrouck_is(alpha, Om); cs = gonzalo_granger(alpha)
        mis = lien_shrestha_is(alpha, Om); d = {}
        for j, nm in enumerate(names):
            d.update({f"IS_mid_{nm}": mid[j], f"MIS_{nm}": mis[j], f"CS_{nm}": cs[j]})
        return d

    XtXinv = np.linalg.inv(X.T @ X)                       # day-clustered SE on the z*D interaction (col 1)
    se = []; G = None
    for k in range(2):
        se_k, G = _cluster_se(X, resid[:, k], groups, XtXinv)
        se.append(float(se_k[1]))
    from scipy import stats as _stats
    pv = (lambda t: float(2.0 * _stats.t.sf(abs(t), df=G - 1))) if (G and G > 1) else (lambda t: float("nan"))
    ti_spy = alpha_int[0] / (se[0] + EPS); ti_es = alpha_int[1] / (se[1] + EPS)
    return {"alpha_benchmark": alpha_base, "alpha_volatile": alpha_vol,
            "alpha_interaction": alpha_int, "n_lags": int(n_lags),
            "t_interaction_spy": ti_spy, "t_interaction_es": ti_es,
            "se_interaction_spy": se[0], "se_interaction_es": se[1],
            "p_interaction_spy": pv(ti_spy), "p_interaction_es": pv(ti_es),
            "n_clusters": int(G) if G else 0, "se_kind": "day-cluster",
            "benchmark": _shares(alpha_base, Om_b), "volatile": _shares(alpha_vol, Om_v)}


# ── self-test (identified DGP: the FAST-adjusting market is the FOLLOWER) ──────
def _selftest():
    rng = np.random.default_rng(1)

    def sim(n=25000, lead="ES", a_fast=0.10, a_slow=0.004, noise=0.02, mu=0.004, corr=0.3):
        f = np.cumsum(rng.normal(0, mu, n))               # small common efficient trend
        L = np.linalg.cholesky([[1, corr], [corr, 1]])
        e = rng.normal(0, noise, (n, 2)) @ L.T             # transient pricing errors
        p1 = np.empty(n); p2 = np.empty(n); p1[0] = f[0]; p2[0] = f[0]
        a1, a2 = (a_slow, a_fast) if lead == "SPY" else (a_fast, a_slow)
        for t in range(1, n):
            z = p1[t - 1] - p2[t - 1]
            p1[t] = p1[t - 1] + (f[t] - f[t - 1]) - a1 * z + e[t, 0]
            p2[t] = p2[t - 1] + (f[t] - f[t - 1]) + a2 * z + e[t, 1]
        return np.exp(p1), np.exp(p2)

    sessions = []
    for d in range(7):
        s, e = sim(lead="ES", a_fast=0.12, a_slow=0.004, noise=0.025, corr=0.4)
        sessions.append((f"vol_{d}", "volatile", s, e))
    for d in range(7):
        s, e = sim(lead="ES", a_fast=0.05, a_slow=0.035, noise=0.015, corr=0.2)
        sessions.append((f"ben_{d}", "benchmark", s, e))

    df = estimate_sample(sessions, n_lags=5)
    print(df[["regime", "IS_mid_ES", "MIS_ES", "CS_ES"]].round(3).to_string())
    print("\nchecks")
    print("  IS sums to 1 (mean):", round(float((df.IS_mid_SPY + df.IS_mid_ES).mean()), 4))
    print("  bounds ordered (lo<=mid<=hi all):",
          bool(((df.IS_lo_ES <= df.IS_mid_ES + 1e-9) & (df.IS_mid_ES <= df.IS_hi_ES + 1e-9)).all()))
    print("  ES recovered as leader by IS, MIS, CS (>0.5 every session):",
          bool((df.IS_mid_ES > .5).all() and (df.MIS_ES > .5).all() and (df.CS_ES > .5).all()))
    print("  volatile leadership > benchmark (CS_ES):",
          df.loc[df.regime == 'volatile', 'CS_ES'].mean() > df.loc[df.regime == 'benchmark', 'CS_ES'].mean())
    print("\nregime test:", compare_regimes(df, "CS_ES", n_perm=5000))
    pv = panel_vecm(sessions, n_lags=5)
    print("\npanel alpha bench:", np.round(pv['alpha_benchmark'], 4),
          "vol:", np.round(pv['alpha_volatile'], 4))
    print("panel CS_ES bench=%.3f vol=%.3f | t(interaction ES)=%.2f"
          % (pv['benchmark']['CS_ES'], pv['volatile']['CS_ES'], pv['t_interaction_es']))

    # ── (A) information-criterion lag selection ───────────────────────────────
    def sim_lag(n=30000, phis=(0.30, 0.20), a=(0.03, 0.05), noise=0.02, mu=0.003, corr=0.3, seed=11):
        """Common trend + EC + a KNOWN number of lagged-difference terms (true VECM lag = len(phis))."""
        rg = np.random.default_rng(seed)
        fc = np.cumsum(rg.normal(0, mu, n))
        Lc = np.linalg.cholesky([[1, corr], [corr, 1]]); e = rg.normal(0, noise, (n, 2)) @ Lc.T
        p1 = np.zeros(n); p2 = np.zeros(n); d1 = np.zeros(n); d2 = np.zeros(n)
        p1[0] = p2[0] = fc[0]; K = len(phis)
        for t in range(1, n):
            z = p1[t - 1] - p2[t - 1]
            ar1 = sum(phis[j - 1] * d1[t - j] for j in range(1, K + 1) if t - j >= 0)
            ar2 = sum(phis[j - 1] * d2[t - j] for j in range(1, K + 1) if t - j >= 0)
            d1[t] = (fc[t] - fc[t - 1]) - a[0] * z + ar1 + e[t, 0]
            d2[t] = (fc[t] - fc[t - 1]) + a[1] * z + ar2 + e[t, 1]
            p1[t] = p1[t - 1] + d1[t]; p2[t] = p2[t - 1] + d2[t]
        return np.exp(p1), np.exp(p2)

    s1, s2 = sim_lag(phis=(0.30, 0.20))                   # true VECM lag = 2
    p_bic, tab_bic = select_lag(np.log(s1), np.log(s2), pmax=8, criterion="bic")
    p_aic, _ = select_lag(np.log(s1), np.log(s2), pmax=8, criterion="aic")
    lag_ok = (p_bic == 2) and (p_bic <= p_aic)
    print("\n(A) lag selection: BIC p*=%d (true=2), AIC p*=%d, BIC<=AIC: %s" % (p_bic, p_aic, lag_ok))

    pairs = [(np.log(np.asarray(s, float)), np.log(np.asarray(e_, float))) for _l, _r, s, e_ in sessions]
    p_pool, _pooltab = select_lag_pooled(pairs, pmax=8, criterion="bic")
    df_sel = estimate_sample(sessions, criterion="bic", pmax=8, lag_selection="pooled")
    pooled_ok = (df_sel.attrs.get("pooled_lag") == p_pool and "n_lags" in df_sel.columns
                 and (df_sel["n_lags"] == p_pool).all())
    pv_sel = panel_vecm(sessions, criterion="bic", pmax=8)
    print("    pooled BIC lag=%d applied to all %d sessions: %s | panel pooled lag=%d"
          % (p_pool, len(df_sel), pooled_ok, pv_sel["n_lags"]))

    df_fixed = estimate_sample(sessions, n_lags=5)        # criterion=None reproduces fixed-lag path
    compat_ok = (df_fixed.attrs.get("pooled_lag") is None
                 and bool((df_fixed["IS_mid_ES"] > 0.5).all()))

    is_sum_ok = abs(float((df.IS_mid_SPY + df.IS_mid_ES).mean()) - 1.0) < 1e-6
    bounds_ok = bool(((df.IS_lo_ES <= df.IS_mid_ES + 1e-9) & (df.IS_mid_ES <= df.IS_hi_ES + 1e-9)).all())
    leader_ok = bool((df.IS_mid_ES > .5).all() and (df.MIS_ES > .5).all() and (df.CS_ES > .5).all())
    ok = all([is_sum_ok, bounds_ok, leader_ok, lag_ok, pooled_ok, compat_ok])
    print("\nchecks:", ok)
    return ok

if __name__ == "__main__":
    _selftest()
