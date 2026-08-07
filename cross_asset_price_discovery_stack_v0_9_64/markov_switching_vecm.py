"""
markov_switching_vecm.py
=======================
A two-state Markov-switching VECM for the SPY/ES price system -- item #4. H1/H3 are about
nonlinear regime dependence, which the paper handles with hand-picked high-vol days and
medium-return dummies. Here the calm / stress regimes are a LATENT state estimated
endogenously (Hamilton filter + EM), with regime-specific:

  * error-correction speeds alpha(S_t) -- how fast the SPY-ES basis reverts (arbitrage),
  * innovation covariance Sigma(S_t)   -- the stress regime is the high-variance one,
  * information shares (Gonzalo-Granger) -- who leads price discovery, per regime.

The contagion reading: in the stress regime the error correction that closes the basis
weakens and the lead shifts -- the H3 breakdown, dated by the model rather than by hand.
This is the latent-discrete-state complement to the continuous-observable-state ECM-SDE.

Model (per regime k):
    dY_t = c(k) + alpha(k) (beta' Y_{t-1}) + sum_{i=1}^p Gamma_i(k) dY_{t-i} + eps_t,
    eps_t ~ N(0, Sigma(k)),   S_t ~ Markov(P).
beta (cointegrating vector) is estimated by a first-stage cointegrating regression (or
supplied). Estimation is exact EM: Hamilton filter for the likelihood and filtered probs,
Kim smoother for the smoothed and joint probabilities, weighted-LS M-step. numpy/pandas.
"""
import numpy as np
import pandas as pd

EPS = 1e-12


# ── cointegration + VECM design ───────────────────────────────────────────────
def _cointegrate(y, beta=None):
    """Return (hedge ratio b, const c, ECT series) with ECT = y0 - b*y1 - c. beta=(1,-b)
    estimated by OLS of y0 on y1 unless supplied (e.g. b=1 for log prices)."""
    if beta is not None:
        b = float(beta); c = float(np.mean(y[:, 0] - b * y[:, 1]))
    else:
        X = np.column_stack([np.ones(len(y)), y[:, 1]])
        coef, *_ = np.linalg.lstsq(X, y[:, 0], rcond=None)
        c, b = float(coef[0]), float(coef[1])
    return b, c, y[:, 0] - b * y[:, 1] - c


def _build_design(y, ect, p):
    """Response dY_t and regressors [ECT_{t-1}, const, dY_{t-1..t-p}]."""
    dy = np.diff(y, axis=0); n = len(dy); rows = np.arange(p, n)
    Xc = [ect[rows][:, None], np.ones((len(rows), 1))]
    for L in range(1, p + 1):
        Xc.append(dy[rows - L])
    return dy[rows], np.hstack(Xc)


def _select_p_single_regime(y, ect, pmax=8, criterion="bic"):
    """Pick the VECM lag by information criterion on the SINGLE-regime (linear) model, on a common
    sample across candidates; the switching model is then fit at that p (the standard order: select
    the lag on the linear VECM, then add the regimes). Returns p*."""
    import price_discovery_shares as pds
    dy = np.diff(y, axis=0); n = len(dy)
    key = pds._CRIT_KEY.get(str(criterion).lower(), "bic")
    best_p, best_v = max(0, 0), np.inf
    for p in range(0, pmax + 1):
        rows = np.arange(pmax, n)                          # common sample for every candidate
        Xc = [ect[rows][:, None], np.ones((len(rows), 1))]
        for L in range(1, p + 1):
            Xc.append(dy[rows - L])
        X = np.hstack(Xc); Yr = dy[rows]
        if Yr.shape[0] < X.shape[1] + 5:
            continue
        ld, aic, bic, hq = pds._ic_row(Yr, X, p, n_ec=1)   # const + EC are the fixed regressors
        v = {"aic": aic, "bic": bic, "hqic": hq}[key]
        if np.isfinite(v) and v < best_v:
            best_v, best_p = v, p
    return best_p


# ── weighted LS, densities, filter, smoother ──────────────────────────────────
def _wls(Y, X, w):
    WX = w[:, None] * X
    B = np.linalg.solve(X.T @ WX + np.eye(X.shape[1]) * 1e-10, X.T @ (w[:, None] * Y))
    E = Y - X @ B
    sw = max(w.sum(), EPS)
    return B, (E.T @ (w[:, None] * E)) / sw


def _log_densities(Y, X, B, Sig):
    R, d = Y.shape; K = len(B); out = np.empty((R, K))
    for k in range(K):
        E = Y - X @ B[k]; S = Sig[k] + np.eye(d) * 1e-12
        Si = np.linalg.inv(S); ld = np.linalg.slogdet(S)[1]
        maha = np.einsum("ij,jk,ik->i", E, Si, E)
        out[:, k] = -0.5 * d * np.log(2 * np.pi) - 0.5 * ld - 0.5 * maha
    return out


def _hamilton(logdens, P, pi0):
    """Forward filter. Returns filtered probs (R,K), predicted probs (R,K), log-likelihood."""
    R, K = logdens.shape
    filt = np.empty((R, K)); pred = np.empty((R, K)); ll = 0.0; prev = pi0.copy()
    for t in range(R):
        pr = prev if t == 0 else (P.T @ prev)
        lw = np.log(np.maximum(pr, EPS)) + logdens[t]
        mx = lw.max(); s = np.exp(lw - mx); tot = s.sum()
        ll += mx + np.log(tot); ft = s / tot
        filt[t] = ft; pred[t] = pr; prev = ft
    return filt, pred, ll


def _kim(filt, pred, P):
    """Backward smoother. Returns smoothed probs (R,K), transition numerator (K,K) =
    sum_t P(S_t=j,S_{t+1}=k|T), and sum_t smoothed[t] over t=0..R-2."""
    R, K = filt.shape
    smooth = np.empty((R, K)); smooth[-1] = filt[-1]
    trans_num = np.zeros((K, K)); smooth_sum = np.zeros(K)
    for t in range(R - 2, -1, -1):
        ratio = smooth[t + 1] / np.maximum(pred[t + 1], EPS)
        st = filt[t] * (P @ ratio); st /= max(st.sum(), EPS)
        smooth[t] = st
        trans_num += filt[t][:, None] * P * ratio[None, :]
        smooth_sum += st
    return smooth, trans_num, smooth_sum


def _ergodic(P):
    w, v = np.linalg.eig(P.T)
    pi = np.real(v[:, np.argmin(np.abs(w - 1.0))])
    return np.abs(pi) / np.abs(pi).sum()


def _init_regimes(Y, X, K):
    B0, _ = _wls(Y, X, np.ones(len(Y)))
    score = ((Y - X @ B0) ** 2).sum(1)
    qs = np.quantile(score, np.linspace(0, 1, K + 1)); B, Sig = [], []
    for k in range(K):
        w = ((score >= qs[k]) & (score <= qs[k + 1])).astype(float)
        if w.sum() < X.shape[1] + 2:
            w = np.ones(len(Y))
        Bk, Sk = _wls(Y, X, w); B.append(Bk); Sig.append(Sk)
    return B, Sig


# ── information shares from the adjustment vector ─────────────────────────────
def gonzalo_granger_shares(alpha):
    """Gonzalo-Granger permanent-component shares from the adjustment vector alpha=(a0,a1):
    the orthogonal complement alpha_perp=(a1,-a0), normalized. The market that adjusts LESS
    carries more of the common trend (leads). Returns shares summing to 1."""
    a = np.asarray(alpha, float); perp = np.abs(np.array([a[1], -a[0]])); s = perp.sum()
    return perp / s if s > EPS else np.array([0.5, 0.5])


def hasbrouck_is_from_alpha(alpha, Sigma):
    """Hasbrouck information-share bounds from (alpha_perp, Sigma): IS_i under each Cholesky
    ordering; returns (lower, upper, mid) per asset (2,).

    psi keeps its COMPONENT SIGNS (matching price_discovery_shares._psi): taking abs() first
    flipped the cross-covariance term whenever the two alphas share a sign -- days real runs
    demonstrably contain (2023-08-07: +0.052/+0.111) -- shifting the bounds by tens of points
    while agreeing on well-behaved opposite-signed days, which is why the selftests never saw
    it. The IS ratio itself is invariant to the OVERALL sign of psi, so no orientation is
    lost. Degeneracy and jitter are RELATIVE to Sigma's own scale: the old additive EPS in
    the denominator deflated every IS ~10% at raw 1s log-return scale (diag ~4e-9) and
    collapsed the 10ms shares to ~1e-5; the old absolute 1e-12 Cholesky jitter inflated 10ms
    variances ~2.5%."""
    a = np.asarray(alpha, float); psi = np.array([a[1], -a[0]])
    S = np.asarray(Sigma, float); tot = float(psi @ S @ psi)
    scale = float(psi @ psi) * float(np.trace(S))
    if not np.isfinite(tot) or scale <= 0.0 or tot <= 1e-12 * scale:
        z = np.array([0.5, 0.5]); return z, z, z
    jit = 1e-12 * float(np.trace(S))
    vals = []
    for order in ([0, 1], [1, 0]):
        Sp = S[np.ix_(order, order)]; L = np.linalg.cholesky(Sp + np.eye(2) * jit)
        pl = psi[order]; contrib = (pl @ L) ** 2 / tot
        is_ord = np.empty(2); is_ord[order] = contrib; vals.append(is_ord)
    lo = np.minimum(vals[0], vals[1]); hi = np.maximum(vals[0], vals[1])
    return lo, hi, 0.5 * (lo + hi)


# ── EM fit ────────────────────────────────────────────────────────────────────
def fit_ms_vecm(Y, K=2, p=1, beta=None, max_iter=200, tol=1e-6, criterion=None, pmax=8):
    """Fit the K-regime Markov-switching VECM by EM. Returns the transition matrix, the
    regime-specific adjustment vectors alpha(k) (error-correction speeds), full coefficient
    matrices B(k), innovation covariances Sigma(k), smoothed regime probabilities, the
    ergodic distribution, the log-likelihood, and the calm/stress regime labels (stress =
    largest innovation variance). With ``criterion`` set ('bic'|'aic'|'hqic') the lag ``p`` is chosen
    by that criterion on the single-regime VECM over [0, pmax]; otherwise the passed ``p`` is used."""
    y = np.asarray(Y, float)
    b, c, ect = _cointegrate(y, beta)
    if criterion is not None:
        p = _select_p_single_regime(y, ect, pmax=pmax, criterion=criterion)
    Yr, X = _build_design(y, ect, p)
    K = int(K)
    B, Sig = _init_regimes(Yr, X, K)
    P = np.full((K, K), 0.1 / max(K - 1, 1)); np.fill_diagonal(P, 0.9)
    pi0 = np.full(K, 1.0 / K); ll_old = -np.inf; ll = -np.inf
    for it in range(max_iter):
        logdens = _log_densities(Yr, X, B, Sig)
        filt, pred, ll = _hamilton(logdens, P, pi0)
        smooth, trans_num, smooth_sum = _kim(filt, pred, P)
        P = trans_num / np.maximum(smooth_sum[:, None], EPS)
        P = P / P.sum(axis=1, keepdims=True)
        pi0 = smooth[0].copy()
        for k in range(K):
            B[k], Sig[k] = _wls(Yr, X, smooth[:, k])
        if it > 3 and ll - ll_old < tol * (abs(ll_old) + 1.0):
            break
        ll_old = ll
    alpha = [B[k][0, :].copy() for k in range(K)]                 # ECT coefficient row per regime
    traces = np.array([np.trace(Sig[k]) for k in range(K)])
    stress = int(np.argmax(traces)); calm = int(np.argmin(traces))
    return {"K": K, "p": p, "hedge_ratio": b, "const": c, "P": P,
            "alpha": alpha, "B": B, "Sigma": Sig, "smoothed": smooth, "filtered": filt,
            "ergodic": _ergodic(P), "loglik": ll, "stress_regime": stress, "calm_regime": calm,
            "ect": ect, "n_obs": Yr.shape[0]}


def single_regime_loglik(Y, p=1, beta=None):
    """Gaussian log-likelihood of the no-switching VECM (one regime) -- the benchmark the
    switching model should beat when regimes are real."""
    y = np.asarray(Y, float); b, c, ect = _cointegrate(y, beta)
    Yr, X = _build_design(y, ect, p)
    B0, S0 = _wls(Yr, X, np.ones(len(Yr)))
    return float(_log_densities(Yr, X, [B0], [S0]).sum())


def regime_summary(fit, names=("SPY", "ES")):
    """Per-regime table: error-correction speeds, total EC strength, innovation vols,
    Gonzalo-Granger shares, ergodic probability, with calm/stress labels."""
    rows = []
    for k in range(fit["K"]):
        a = fit["alpha"][k]; S = fit["Sigma"][k]; cs = gonzalo_granger_shares(a)
        rows.append({
            "regime": k,
            "label": "stress" if k == fit["stress_regime"] else ("calm" if k == fit["calm_regime"] else f"r{k}"),
            f"ec_{names[0]}": a[0], f"ec_{names[1]}": a[1], "ec_strength": float(np.abs(a).sum()),
            f"vol_{names[0]}": float(np.sqrt(max(S[0, 0], 0))), f"vol_{names[1]}": float(np.sqrt(max(S[1, 1], 0))),
            f"GG_{names[0]}": cs[0], f"GG_{names[1]}": cs[1], "ergodic": float(fit["ergodic"][k]),
        })
    return pd.DataFrame(rows)


# ── self-test ─────────────────────────────────────────────────────────────────
def _sim(T=4000, seed=0):
    """Two-regime VECM: calm = strong, roughly balanced error correction; stress = weak and
    asymmetric error correction with 3x innovation vol. Shared trend keeps them cointegrated."""
    rng = np.random.default_rng(seed)
    P = np.array([[0.985, 0.015], [0.08, 0.92]])
    alpha = [np.array([-0.22, 0.18]), np.array([-0.01, 0.05])]   # stress: SPY barely adjusts -> leads
    idio = [1e-4, 9e-4]; common_sd = 0.01
    S = np.zeros(T, int)
    for t in range(1, T):
        S[t] = rng.choice(2, p=P[S[t - 1]])
    y = np.zeros((T, 2)); y[0] = [10.0, 10.0]
    for t in range(1, T):
        ect = y[t - 1, 0] - y[t - 1, 1]; k = S[t]
        common = rng.normal(0, common_sd); e = rng.normal(0, np.sqrt(idio[k]), 2)
        y[t] = y[t - 1] + alpha[k] * ect + common + e
    return y, S


def _selftest():
    print("markov_switching_vecm self-test")
    y, S_true = _sim(4000, seed=1)
    fit = fit_ms_vecm(y, K=2, p=1, beta=1.0)
    summ = regime_summary(fit)
    st, cm = fit["stress_regime"], fit["calm_regime"]

    # (1) two regimes recovered; stress = high vol, weaker error correction
    print("\n(1) regime summary")
    print(summ.round(4).to_string(index=False))
    ec_stress = float(np.abs(fit["alpha"][st]).sum()); ec_calm = float(np.abs(fit["alpha"][cm]).sum())
    print("    EC strength: calm=%.3f  stress=%.3f  (arbitrage weaker in stress)" % (ec_calm, ec_stress))
    print("    checks:", ec_calm > ec_stress
          and np.trace(fit["Sigma"][st]) > np.trace(fit["Sigma"][cm]))

    # (2) smoothed stress probability tracks the true regime path
    p_stress = fit["smoothed"][:, st]
    S_aligned = S_true[1 + fit["p"]:]                         # align to the design rows (diff + p lags)
    corr = float(np.corrcoef(p_stress, S_aligned)[0, 1])
    print("\n(2) latent-regime recovery")
    print("    corr(smoothed P[stress], true regime) = %.3f" % corr)
    print("    transition diag = %.3f, %.3f (persistent regimes); ergodic = %s"
          % (fit["P"][0, 0], fit["P"][1, 1], np.round(fit["ergodic"], 3)))
    print("    checks:", corr > 0.4 and fit["P"][0, 0] > 0.8 and fit["P"][1, 1] > 0.8)

    # (3) switching beats the single-regime VECM
    ll1 = single_regime_loglik(y, p=1, beta=1.0)
    print("\n(3) switching vs single-regime VECM")
    print("    loglik switching = %.1f  >  single-regime = %.1f  (delta = %.1f)"
          % (fit["loglik"], ll1, fit["loglik"] - ll1))
    print("    checks:", fit["loglik"] > ll1 + 20)

    # (4) information share is regime-dependent (the price-discovery story)
    cs_calm = gonzalo_granger_shares(fit["alpha"][cm]); cs_stress = gonzalo_granger_shares(fit["alpha"][st])
    print("\n(4) regime-dependent information share (Gonzalo-Granger, SPY share)")
    print("    calm SPY share = %.3f  ->  stress SPY share = %.3f" % (cs_calm[0], cs_stress[0]))
    print("    checks:", abs(cs_calm[0] - cs_stress[0]) > 0.1 and 0 <= cs_stress[0] <= 1)


if __name__ == "__main__":
    _selftest()
