"""
jump_robust.py
==============
Jump-robust realized measures for the SPY-ES study, and a decomposition of price
discovery into a CONTINUOUS (diffusive) part and a JUMP (news) part.

Why: transaction/quote prices are jump-diffusion semimartingales, and the standard
Hasbrouck / Gonzalo-Granger information share — which attributes the variance of the
common efficient-price innovation — silently BLENDS continuous and jump quadratic
variation. Pulling them apart answers two distinct questions the liquidity thesis
cares about: which venue leads *continuous* price discovery, and which reacts first
to *jumps*; and the deeper book should absorb jumps with less impact.

Estimators (input = a 1-D return series r, e.g. diff(log mid)):
  bipower_variation / med_rv / min_rv   jump-robust integrated variance (IV)
  truncated_rv (Mancini)                threshold IV + a jump mask
  jump_variation                        RV, IV, JV = RV-IV, and the jump fraction
  realized_quarticity / tripower_quarticity
  bns_jump_test (Barndorff-Nielsen-Shephard / Huang-Tauchen log-ratio test)
  lee_mykland                           intraday jump TIMES (Gumbel threshold)

Bivariate (rx, ry on a common grid):
  realized_covariance, bipower_covariation (polarization), threshold_covariance
  continuous_jump_cov                   the 2x2 continuous & jump (co)variation matrices
  cojump_alignment                      who jumps first across the two assets

Integration (built on price_discovery_shares — the VECM, hasbrouck_is, GG):
  continuous_jump_information_shares     total / continuous / jump information shares
  estimate_sample_jump_split            per-session DataFrame (parallels pds.estimate_sample)

The split applies the jump-robust estimators to the VECM INNOVATIONS, so it composes
with the existing fixed-(1,-1) VECM pipeline. This is the residual-innovation analogue
of the high-frequency QV split (separating the diffusive bulk of the news from the
heavy-tailed jump news), not a formal Delta->0 result; the raw-return estimators above
are the textbook high-frequency versions to apply to mid returns directly.
numpy / pandas / scipy.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd
from scipy.stats import norm

import price_discovery_shares as pds

EPS = 1e-15
MU1     = math.sqrt(2.0 / math.pi)                       # E|Z|
BV_C    = math.pi / 2.0                                  # mu1^{-2}
MEDRV_C = math.pi / (6.0 - 4.0 * math.sqrt(3.0) + math.pi)
MINRV_C = math.pi / (math.pi - 2.0)
MU43    = 2.0 ** (2.0 / 3.0) * math.gamma(7.0 / 6.0) / math.gamma(0.5)
TP_C    = MU43 ** (-3.0)
BNS_THETA = (math.pi ** 2) / 4.0 + math.pi - 5.0         # ~0.6090


def log_returns(mid):
    return np.diff(np.log(np.asarray(mid, float)))


# ── univariate realized measures ──────────────────────────────────────────────
def realized_variance(r):
    r = np.asarray(r, float); return float(np.sum(r * r))

def bipower_variation(r, finite_sample=True):
    """mu1^{-2} * sum |r_{i-1}||r_i| -> integrated variance, robust to jumps."""
    r = np.asarray(r, float); n = r.shape[0]
    if n < 2: return realized_variance(r)
    bv = BV_C * float(np.sum(np.abs(r[:-1]) * np.abs(r[1:])))
    if finite_sample and n > 1: bv *= n / (n - 1.0)
    return bv

def med_rv(r):
    """Andersen-Dobrev-Schaumburg MedRV — IV robust to jumps and occasional zeros."""
    r = np.asarray(r, float); n = r.shape[0]
    if n < 3: return realized_variance(r)
    w = np.abs(r); med = np.median(np.column_stack([w[:-2], w[1:-1], w[2:]]), axis=1)
    return MEDRV_C * (n / (n - 2.0)) * float(np.sum(med * med))

def min_rv(r):
    r = np.asarray(r, float); n = r.shape[0]
    if n < 2: return realized_variance(r)
    w = np.abs(r); mn = np.minimum(w[:-1], w[1:])
    return MINRV_C * (n / (n - 1.0)) * float(np.sum(mn * mn))

def realized_quarticity(r):
    r = np.asarray(r, float); n = r.shape[0]
    return (n / 3.0) * float(np.sum(r ** 4))

def tripower_quarticity(r):
    """Estimates integrated quarticity (IQ); used in the BNS test denominator."""
    r = np.asarray(r, float); n = r.shape[0]
    if n < 3: return realized_quarticity(r)
    w = np.abs(r)
    s = float(np.sum((w[:-2] * w[1:-1] * w[2:]) ** (4.0 / 3.0)))
    return n * TP_C * (n / (n - 2.0)) * s

def truncated_rv(r, c=3.0, scale=None):
    """Mancini threshold RV: keep r_i^2 only where |r_i| <= c * (robust per-step sd).
    Returns the truncated (continuous) RV, the threshold, and the jump mask."""
    r = np.asarray(r, float); n = r.shape[0]
    if scale is None: scale = math.sqrt(max(med_rv(r), EPS) / max(n, 1))
    u = c * scale
    jmask = np.abs(r) > u
    return {"trv": float(np.sum(r[~jmask] ** 2)), "threshold": float(u),
            "jump_mask": jmask, "n_jumps": int(jmask.sum())}

def jump_variation(r, iv="bipower", K=None, alpha=0.01):
    """Realized jump variation and two jump-fraction estimates. 'jump_fraction' is the
    integrated (RV-IV)/RV measure (Huang-Tauchen) -- simple but it loads fat tails,
    microstructure noise, and tick discreteness into the jump bucket, biasing UP at high
    frequency. 'jump_fraction_lm' counts only returns the Lee-Mykland test flags against
    a LOCAL bipower volatility, which deflates that bias; it is the one to trust on real
    (100ms) data."""
    r = np.asarray(r, float)
    rv = realized_variance(r)
    iv_est = {"bipower": bipower_variation, "medrv": med_rv, "minrv": min_rv}[iv](r)
    jv = max(rv - iv_est, 0.0)
    lm = lee_mykland(r, K=K, alpha=alpha)
    jv_lm = float(np.sum(r[lm["jump_idx"]] ** 2)) if lm["jump_idx"].size else 0.0
    return {"rv": rv, "iv": iv_est, "jv": jv,
            "jump_fraction": jv / rv if rv > EPS else 0.0,
            "jump_fraction_lm": jv_lm / rv if rv > EPS else 0.0,
            "jv_lm": jv_lm, "n_jumps_lm": int(lm["n_jumps"])}

def bns_jump_test(r):
    """Barndorff-Nielsen-Shephard / Huang-Tauchen adjusted log-ratio jump test.
    H0: no jumps -> z ~ N(0,1); large positive z (RV >> BV) => a jump component."""
    r = np.asarray(r, float); n = r.shape[0]
    if n < 4: return {"rv": realized_variance(r), "bv": realized_variance(r),
                      "z": 0.0, "p_value": 1.0, "has_jump": False}
    rv = realized_variance(r); bv = bipower_variation(r)
    tq = tripower_quarticity(r)
    ratio = max(1.0, tq / max(bv * bv, EPS))
    denom = math.sqrt(BNS_THETA * (1.0 / n) * ratio)
    z = (math.log(max(rv, EPS)) - math.log(max(bv, EPS))) / max(denom, EPS)
    return {"rv": rv, "bv": bv, "z": float(z), "p_value": float(norm.sf(z)),
            "has_jump": bool(norm.sf(z) < 0.05)}

def lee_mykland(r, K=None, alpha=0.01):
    """Lee-Mykland (2008) intraday jump test. L_i = r_i / sigma_hat_i with sigma_hat
    from a trailing bipower window of K; jumps flagged where |L_i| exceeds the Gumbel
    critical value. Returns the L series, jump indices, signs, and the threshold."""
    r = np.asarray(r, float); n = r.shape[0]
    if K is None: K = max(20, int(round(math.sqrt(n))))
    if n < K + 2:
        return {"L": np.full(n, np.nan), "jump_idx": np.array([], int),
                "jump_sign": np.array([], int), "threshold": np.inf, "n_jumps": 0, "K": K}
    prod = np.abs(r[:-1]) * np.abs(r[1:])                  # |r_{j}||r_{j+1}|
    csum = np.concatenate([[0.0], np.cumsum(prod)])        # len n
    hi = np.arange(n)                                      # use products strictly before i
    lo = np.maximum(0, hi - K)
    s = csum[np.minimum(hi, n - 1)] - csum[lo]
    cnt = np.minimum(hi, n - 1) - lo
    sig2 = s / np.maximum(cnt - 2, 1)                      # ~ (1/(K-2)) sum |r||r|  ~ mu1^2 sigma^2 dt
    with np.errstate(invalid="ignore", divide="ignore"):
        L = r / np.sqrt(sig2)
    L[:K] = np.nan
    ln = math.log(n); a2 = math.sqrt(2.0 * ln)
    Cn = a2 / MU1 - (math.log(math.pi) + math.log(ln)) / (2.0 * MU1 * a2)
    Sn = 1.0 / (MU1 * a2)
    thr = Cn + Sn * (-math.log(-math.log(1.0 - alpha)))    # Gumbel quantile
    jmask = np.abs(L) > thr
    idx = np.where(jmask)[0]
    return {"L": L, "jump_idx": idx, "jump_sign": np.sign(r[idx]).astype(int),
            "threshold": float(thr), "n_jumps": int(idx.size), "K": K}

def decompose(r, iv="bipower"):
    """Convenience: RV, IV, JV/fraction, BNS test, and Lee-Mykland jump count."""
    jv = jump_variation(r, iv=iv); bns = bns_jump_test(r); lm = lee_mykland(r)
    return {**jv, "bns_z": bns["z"], "bns_p": bns["p_value"],
            "n_jumps_LM": lm["n_jumps"]}


# ── bivariate (co)variation ───────────────────────────────────────────────────
def realized_covariance(rx, ry):
    rx = np.asarray(rx, float); ry = np.asarray(ry, float)
    return np.array([[np.sum(rx * rx), np.sum(rx * ry)],
                     [np.sum(rx * ry), np.sum(ry * ry)]], float)

def bipower_covariation(rx, ry):
    """Jump-robust continuous covariation via polarization:
    (BV(x+y) - BV(x-y)) / 4 -> integrated covariance."""
    rx = np.asarray(rx, float); ry = np.asarray(ry, float)
    return 0.25 * (bipower_variation(rx + ry) - bipower_variation(rx - ry))

def threshold_covariance(rx, ry, c=3.0):
    rx = np.asarray(rx, float); ry = np.asarray(ry, float)
    ux = truncated_rv(rx, c=c)["threshold"]; uy = truncated_rv(ry, c=c)["threshold"]
    keep = (np.abs(rx) <= ux) & (np.abs(ry) <= uy)
    return float(np.sum(rx[keep] * ry[keep]))

def continuous_jump_cov(rx, ry, method="truncation", c=3.0, K=None, alpha=0.01):
    """2x2 continuous (jump-robust) and jump (co)variation matrices, summed scale.

    method='truncation' (default): classify a time step as a jump step if either leg
      exceeds its Mancini threshold; Sigma_j = sum of outer products over jump steps,
      Sigma_c = RC - Sigma_j. BOTH PSD by construction (Gram matrices) and RC = Sc+Sj
      exactly -- the safe input for Hasbrouck, and it captures the full jump return.
    method='lee_mykland': identical Gram split but jump steps are flagged by the
      Lee-Mykland test (LOCAL bipower volatility, FWER-controlled at alpha) on either
      leg -- adapts to the local scale, so it does not over-flag in high-vol stretches
      the way a global threshold does. Also PSD and exact.
    method='bipower' / 'threshold': diagonal IV via bipower / truncated RV, off-diagonal
      via bipower covariation (polarization) / threshold covariance. Standard realized
      estimators for REPORTING, but Sigma_j = RC - Sigma_c can be indefinite (a market
      with no jumps has RV-BV ~ small +/- noise), so do not feed these to hasbrouck_is."""
    rx = np.asarray(rx, float); ry = np.asarray(ry, float)
    RC = realized_covariance(rx, ry)
    if method in ("truncation", "lee_mykland"):
        if method == "truncation":
            ux = truncated_rv(rx, c=c)["threshold"]; uy = truncated_rv(ry, c=c)["threshold"]
            jstep = (np.abs(rx) > ux) | (np.abs(ry) > uy)
        else:
            jstep = np.zeros(rx.shape[0], bool)
            jstep[lee_mykland(rx, K=K, alpha=alpha)["jump_idx"]] = True
            jstep[lee_mykland(ry, K=K, alpha=alpha)["jump_idx"]] = True
        Rj = np.column_stack([rx[jstep], ry[jstep]])
        Sj = Rj.T @ Rj if jstep.any() else np.zeros((2, 2))
        return {"RC": RC, "Sigma_c": RC - Sj, "Sigma_j": Sj, "method": method,
                "n_jump_steps": int(jstep.sum())}
    if method == "threshold":
        cx = truncated_rv(rx, c=c)["trv"]; cy = truncated_rv(ry, c=c)["trv"]
        cxy = threshold_covariance(rx, ry, c=c)
    else:
        cx = bipower_variation(rx); cy = bipower_variation(ry)
        cxy = bipower_covariation(rx, ry)
    Sc = np.array([[cx, cxy], [cxy, cy]], float)
    return {"RC": RC, "Sigma_c": Sc, "Sigma_j": RC - Sc, "method": method}

def cojump_alignment(rx, ry, K=None, alpha=0.01, max_lag=1, names=("SPY", "ES")):
    """Align Lee-Mykland jump times across the two assets to see who reacts first.
    A co-jump = a jump in each within +/- max_lag steps; lead = whoever is earlier."""
    jx = lee_mykland(rx, K=K, alpha=alpha); jy = lee_mykland(ry, K=K, alpha=alpha)
    ix = list(map(int, jx["jump_idx"])); iy = set(map(int, jy["jump_idx"]))
    used, pairs, lead_x, lead_y, simul, agree = set(), [], 0, 0, 0, 0
    for i in sorted(ix):
        cand = [j for j in iy if abs(j - i) <= max_lag and j not in used]
        if not cand: continue
        j = min(cand, key=lambda j: abs(j - i)); used.add(j); pairs.append((i, j))
        if   i < j: lead_x += 1
        elif j < i: lead_y += 1
        else:       simul += 1
        if np.sign(rx[i]) == np.sign(ry[j]): agree += 1
    nc = len(pairs)
    return {f"n_jumps_{names[0]}": len(ix), f"n_jumps_{names[1]}": len(iy),
            "n_cojump": nc, f"lead_{names[0]}": lead_x, f"lead_{names[1]}": lead_y,
            "simultaneous": simul,
            "sign_agreement": (agree / nc) if nc else float("nan")}

def cojump_from_mids(mid_spy, mid_es, **kw):
    return cojump_alignment(log_returns(mid_spy), log_returns(mid_es), **kw)


# ── continuous / jump information-share split (built on price_discovery_shares) ─
def _psd(M, floor_frac=1e-10):
    M = np.asarray(M, float); M = 0.5 * (M + M.T)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, floor_frac * max(float(np.max(w)), 1.0), None)
    return (V * w) @ V.T

def _is_triplet(alpha, Omega):
    lo, hi, mid = pds.hasbrouck_is(alpha, _psd(Omega))
    return lo, hi, mid

def continuous_jump_information_shares(mid_spy, mid_es, n_lags=5, method="truncation",
                                       c=4.0, K=None, alpha=0.01, names=("SPY", "ES"), use_log=True):
    """Decompose the Hasbrouck information share into continuous and jump components.

    Fits the fixed-(1,-1) VECM (via price_discovery_shares), splits the realized
    innovation covariance RC into a continuous part (jump-robust) and a jump part,
    and applies hasbrouck_is to each. The common-factor direction psi ∝ alpha_perp is
    shared, so CS (loadings-only) is identical across components; what differs is the
    diffusive-vs-jump composition of each venue's news. Total IS == pds.estimate_day IS
    (the IS ratio is scale-invariant, so independent of the split `method`).

    Jump fraction: `jump_frac_cf` is the `method` estimate (truncation with multiple c).
    `jump_frac_cf_lm` is the Lee-Mykland classification of the common-factor return -- it
    scales jumps by LOCAL volatility, so unlike the global threshold (or the integrated
    RV-BV measure) it does not absorb fat tails, microstructure noise, and tick
    discreteness into the jump bucket. jump_frac_cf_lm is the recommended jump fraction
    for real data, and it deflates the inflated global-threshold estimate; run it on
    100ms returns so the trailing bipower window (K~sqrt(n)) has enough points to
    estimate local vol. c (truncation) and alpha (Lee-Mykland FWER) are the size knobs."""
    f = np.log if use_log else (lambda x: np.asarray(x, float))
    p1, p2 = f(np.asarray(mid_spy, float)), f(np.asarray(mid_es, float))
    ok = np.isfinite(p1) & np.isfinite(p2); p1, p2 = p1[ok], p2[ok]
    alpha_v, _, resid = pds._fit_vecm_fixed(p1, p2, n_lags)
    rx, ry = resid[:, 0], resid[:, 1]

    cm = continuous_jump_cov(rx, ry, method=method, c=c, K=K, alpha=alpha)
    RC, Sc, Sj = cm["RC"], cm["Sigma_c"], cm["Sigma_j"]
    psi = pds._psi(alpha_v); cs = pds.gonzalo_granger(alpha_v)
    lo_t, hi_t, mid_t = _is_triplet(alpha_v, RC)
    lo_c, hi_c, mid_c = _is_triplet(alpha_v, Sc)

    cf_var = float(psi @ RC @ psi); cf_jmp = float(psi @ _psd(Sj) @ psi)
    jump_frac_cf = max(min(cf_jmp / cf_var, 1.0), 0.0) if cf_var > EPS else 0.0
    # jump IS only meaningful if there is a non-trivial jump covariance
    if np.trace(Sj) <= 1e-8 * max(np.trace(RC), EPS) or cf_jmp <= EPS:
        z = np.array([0.5, 0.5]); lo_j, hi_j, mid_j = z, z, z
    else:
        lo_j, hi_j, mid_j = _is_triplet(alpha_v, Sj)

    cf_ret = resid @ psi
    jf_cf = jump_variation(cf_ret, K=K, alpha=alpha)         # Lee-Mykland on the common factor
    jump_frac_cf_lm = jf_cf["jump_fraction_lm"]
    bns = bns_jump_test(cf_ret)
    rvd = np.diag(RC); jvd = np.clip(np.diag(Sj), 0.0, None)
    jvx, jvy = jump_variation(rx, K=K, alpha=alpha), jump_variation(ry, K=K, alpha=alpha)
    jf_lm_leg = {names[0]: jvx["jump_fraction_lm"], names[1]: jvy["jump_fraction_lm"]}

    out = {"alpha_spy": float(alpha_v[0]), "alpha_es": float(alpha_v[1]),
           "n_obs": int(resid.shape[0]), "method": method,
           "jump_frac_cf": jump_frac_cf, "jump_frac_cf_lm": jump_frac_cf_lm,
           "n_jumps_cf_lm": jf_cf["n_jumps_lm"], "bns_cf_z": bns["z"], "bns_cf_p": bns["p_value"]}
    for j, nm in enumerate(names):
        out.update({
            f"IS_{nm}": mid_t[j], f"IS_lo_{nm}": lo_t[j], f"IS_hi_{nm}": hi_t[j],
            f"ISc_{nm}": mid_c[j], f"ISj_{nm}": mid_j[j], f"CS_{nm}": cs[j],
            f"jump_frac_{nm}": float(jvd[j] / rvd[j]) if rvd[j] > EPS else 0.0,
            f"jump_frac_lm_{nm}": jf_lm_leg[nm]})
    return out

def estimate_sample_jump_split(sessions, n_lags=5, method="truncation", c=4.0, K=None,
                               alpha=0.01, names=("SPY", "ES")):
    """sessions: iterable of (label, regime, mid_spy, mid_es). Per-day DataFrame of the
    continuous/jump information-share split (parallels pds.estimate_sample). The reported
    jump_frac_cf_lm column is the Lee-Mykland (local-vol) common-factor jump fraction;
    jump_frac_cf is the truncation estimate kept alongside for comparison."""
    rows = []
    for label, regime, m_spy, m_es in sessions:
        r = continuous_jump_information_shares(m_spy, m_es, n_lags=n_lags, method=method,
                                               c=c, K=K, alpha=alpha, names=names)
        r.update({"date": label, "regime": regime}); rows.append(r)
    return pd.DataFrame(rows).set_index("date")


# ── self-test ─────────────────────────────────────────────────────────────────
def _selftest():
    rng = np.random.default_rng(0)
    n, sig = 23400, 6e-4

    # (1) univariate: jumps inflate RV but not the jump-robust IV; tests/detectors fire
    cont = rng.normal(0, sig, n)
    jr_ = cont.copy(); jidx = sorted(rng.choice(np.arange(300, n - 300), 12, replace=False))
    for k in jidx: jr_[int(k)] += 12 * sig * (1 if rng.random() < .5 else -1)
    rv_j, bv_j, bv_c = realized_variance(jr_), bipower_variation(jr_), bipower_variation(cont)
    jf_j, jf_c = jump_variation(jr_)["jump_fraction"], jump_variation(cont)["jump_fraction"]
    jf_j_lm, jf_c_lm = jump_variation(jr_)["jump_fraction_lm"], jump_variation(cont)["jump_fraction_lm"]
    bns_j, bns_c = bns_jump_test(jr_), bns_jump_test(cont)
    lm_j, lm_c = lee_mykland(jr_), lee_mykland(cont)
    rec = set(map(int, jidx)).issubset(set(lm_j["jump_idx"].tolist()))
    near = abs(med_rv(jr_) - bv_j) / bv_j < 0.15 and abs(min_rv(jr_) - bv_j) / bv_j < 0.15
    print("(1) univariate jump-robustness")
    print("    RV(jumps)=%.2e  BV(jumps)=%.2e  BV(cont)=%.2e  MedRV=%.2e MinRV=%.2e"
          % (rv_j, bv_j, bv_c, med_rv(jr_), min_rv(jr_)))
    print("    jump_fraction (RV-BV): jumps=%.3f cont=%.4f | Lee-Mykland: jumps=%.3f cont=%.4f"
          % (jf_j, jf_c, jf_j_lm, jf_c_lm))
    print("    BNS has_jump: jumps=%s cont=%s" % (bns_j["has_jump"], bns_c["has_jump"]))
    print("    Lee-Mykland: planted recovered=%s (found %d) | false +ve on cont=%d"
          % (rec, lm_j["n_jumps"], lm_c["n_jumps"]))
    print("    checks:", bv_j < rv_j and jf_j > 0.04 and jf_j > 3 * jf_c and near
          and bns_j["has_jump"] and not bns_c["has_jump"] and rec and lm_c["n_jumps"] <= 5
          and jf_j_lm > 0.04 and jf_c_lm < 0.01)

    # (2a) bivariate: SIMULTANEOUS co-jumps inflate contemporaneous cov; robust cov ~ continuous
    rho = 0.5; Lc = np.linalg.cholesky([[1, rho], [rho, 1]])
    e = rng.normal(0, sig, (n, 2)) @ Lc.T
    rx, ry = e[:, 0].copy(), e[:, 1].copy(); cj = sorted(rng.choice(np.arange(300, n - 300), 8, replace=False))
    for k in cj:
        s = 1 if rng.random() < .5 else -1; rx[int(k)] += 10 * sig * s; ry[int(k)] += 10 * sig * s
    RC = realized_covariance(rx, ry); bpc = bipower_covariation(rx, ry); cm = continuous_jump_cov(rx, ry)
    print("\n(2a) co-jump covariation")
    print("    RC offdiag=%.2e  bipower-cov=%.2e  threshold-cov=%.2e  -> Sigma_j offdiag=%.2e"
          % (RC[0, 1], bpc, threshold_covariance(rx, ry), cm["Sigma_j"][0, 1]))
    print("    checks:", RC[0, 1] > bpc and cm["Sigma_j"][0, 1] > 0)

    # (2b) LAGGED co-jumps: who reacts first (SPY leads, ES one step behind)
    rx2, ry2 = e[:, 0].copy(), e[:, 1].copy()
    for k in cj:
        s = 1 if rng.random() < .5 else -1; rx2[int(k)] += 10 * sig * s; ry2[int(k) + 1] += 10 * sig * s
    ca = cojump_alignment(rx2, ry2, max_lag=2)
    print("(2b) co-jump lead-lag")
    print("    co-jumps=%d  lead_SPY=%d lead_ES=%d  sign-agree=%.2f"
          % (ca["n_cojump"], ca["lead_SPY"], ca["lead_ES"], ca["sign_agreement"]))
    print("    checks:", ca["n_cojump"] >= 4 and ca["lead_SPY"] > ca["lead_ES"])

    # (3) the IS split on a VECM DGP (ES continuous+jump leader); + no-jump control
    def sim(n=20000, a_fast=0.10, a_slow=0.004, noise=8e-3, mu=4e-3, corr=0.3,
            n_jump=25, jsize=0.06, with_jumps=True):
        ftrend = np.cumsum(rng.normal(0, mu, n))
        L = np.linalg.cholesky([[1, corr], [corr, 1]]); ee = rng.normal(0, noise, (n, 2)) @ L.T
        jt = set(map(int, rng.choice(np.arange(200, n - 200), n_jump, replace=False))) if with_jumps else set()
        jv = {t: jsize * (1 if rng.random() < .5 else -1) for t in jt}
        p1 = np.zeros(n); p2 = np.zeros(n); p1[0] = p2[0] = ftrend[0]
        a1, a2 = a_fast, a_slow                            # SPY fast (follower), ES slow (leader)
        for t in range(1, n):
            z = p1[t - 1] - p2[t - 1]; d = ftrend[t] - ftrend[t - 1]
            jp2 = jv.get(t, 0.0)                           # jump enters ES first
            p1[t] = p1[t - 1] + d - a1 * z + ee[t, 0]
            p2[t] = p2[t - 1] + d + a2 * z + ee[t, 1] + jp2
        return np.exp(p1), np.exp(p2)

    sJ, eJ = sim(with_jumps=True); s0, e0 = sim(with_jumps=False)
    rJ = continuous_jump_information_shares(sJ, eJ, n_lags=5)
    r0 = continuous_jump_information_shares(s0, e0, n_lags=5)
    rL = continuous_jump_information_shares(sJ, eJ, n_lags=5, method="lee_mykland")
    ref = pds.estimate_day(sJ, eJ, n_lags=5)               # consistency: total IS == pds IS
    print("\n(3) continuous/jump information-share split")
    print("    with jumps : IS_ES=%.3f  ISc_ES=%.3f  ISj_ES=%.3f  jump_frac_cf=%.3f (trunc) %.3f (LM)  BNS p=%.1e"
          % (rJ["IS_ES"], rJ["ISc_ES"], rJ["ISj_ES"], rJ["jump_frac_cf"], rJ["jump_frac_cf_lm"], rJ["bns_cf_p"]))
    print("    no jumps   : IS_ES=%.3f  ISc_ES=%.3f  ISj_ES=%.3f  jump_frac_cf=%.3f (trunc) %.3f (LM)"
          % (r0["IS_ES"], r0["ISc_ES"], r0["ISj_ES"], r0["jump_frac_cf"], r0["jump_frac_cf_lm"]))
    print("    consistency: split total IS_ES=%.4f vs pds IS_mid_ES=%.4f | LM-method IS_ES=%.4f (method-invariant)"
          % (rJ["IS_ES"], ref["IS_mid_ES"], rL["IS_ES"]))
    print("    checks:",
          abs(rJ["IS_ES"] - ref["IS_mid_ES"]) < 1e-3            # total == pds
          and abs(rL["IS_ES"] - rJ["IS_ES"]) < 1e-9             # total IS independent of split method
          and rJ["ISc_ES"] > 0.5 and r0["ISc_ES"] > 0.5         # continuous leader = ES
          and rJ["jump_frac_cf"] > 0.03 and r0["jump_frac_cf"] < 0.02  # jumps only when present
          and rJ["jump_frac_cf_lm"] > r0["jump_frac_cf_lm"]     # LM fraction tracks presence of jumps
          and rJ["ISj_ES"] > 0.5)                               # jump leader = ES

    # (4) Lee-Mykland deflates the realized jump fraction. Heavy-tailed (Student-t)
    #     diffusion with NO jumps: (RV-BV)/RV loads the fat tails into "jumps"; the
    #     local-vol LM classifier does not. Genuine jumps are still caught.
    rng4 = np.random.default_rng(7)
    tt = rng4.standard_t(5, 40000); tt = tt / np.std(tt) * 8e-4   # fat tails, zero jumps
    vt = jump_variation(tt)
    tj = tt.copy(); jx4 = rng4.choice(40000, 20, replace=False)
    tj[jx4] += rng4.choice([-1.0, 1.0], 20) * 0.05               # 20 genuine ~60-sigma jumps
    vtj = jump_variation(tj)
    print("\n(4) Lee-Mykland deflation (heavy-tailed diffusion, no jumps)")
    print("    no jumps : (RV-BV)/RV=%.3f   Lee-Mykland=%.3f   (LM deflates the tail-driven fraction)"
          % (vt["jump_fraction"], vt["jump_fraction_lm"]))
    print("    w/ jumps : (RV-BV)/RV=%.3f   Lee-Mykland=%.3f   (LM still catches genuine jumps)"
          % (vtj["jump_fraction"], vtj["jump_fraction_lm"]))
    print("    checks:", vt["jump_fraction_lm"] < vt["jump_fraction"]      # deflation on fat-tailed diffusion
          and vt["jump_fraction_lm"] < 0.5 * vt["jump_fraction"]          # materially deflated
          and vtj["jump_fraction_lm"] > vt["jump_fraction_lm"])           # genuine jumps detected

if __name__ == "__main__":
    _selftest()
