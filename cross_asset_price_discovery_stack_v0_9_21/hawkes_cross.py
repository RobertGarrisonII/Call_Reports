"""
hawkes_cross.py
==============
Bivariate (multivariate) Hawkes process for cross-market liquidity-event **contagion** --
item #3, and the most structural of the redo. The paper's clustering facts (Table 5 corner
cells, Table 6 state continuation, "orders arrive in clusters") are the signature of self-
and cross-excitation, yet everything is modeled in linear discrete-time SVARs. A Hawkes
process is the continuous-time generative model:

    lambda_i(t) = mu_i + sum_j sum_{t_k^j < t} alpha_ij exp(-beta_j (t - t_k^j))

with marks = **liquidity events** (depth-withdrawal or spread-jump times) in each market.
The cross-excitation kernel alpha_{SPY,ES} is the continuous-time spillover (futures
illiquidity triggering ETF illiquidity); the **branching matrix** G_ij = alpha_ij / beta_j
(expected type-i events directly triggered by one type-j event) and its spectral radius --
the **branching ratio** -- measure reflexive fragility / endogeneity that should rise under
stress, a dynamic H3 test that the static `cross_impact` matrix only shadows.

Exact recursive log-likelihood (exponential kernels, O(N*D)); MLE via L-BFGS-B; Ogata-
thinning simulator (for the self-test and for parametric bootstrap of the branching ratio);
stress-conditional fit (calm vs stress, paired with `stress_index`). numpy / pandas / scipy.
"""
import numpy as np
import pandas as pd

EPS = 1e-12


# ── liquidity-event marks from a book frame (no message tape needed) ──────────
def liquidity_events(df, asset, kind="depth_withdrawal", q=0.95, n_inside=1, n_levels=10):
    """Event times (seconds from the start of the frame) of liquidity shocks in one market.
    kind='depth_withdrawal' -> times of large relative drops in inside depth (top 1-q of
    drops); kind='spread_jump' -> times of large widenings of the quoted spread. These are
    the marks of the point process; the bivariate (SPY, ES) pair drives the contagion fit."""
    t = (df.index - df.index[0]).total_seconds()
    t = np.asarray(t.total_seconds() if hasattr(t, "total_seconds") else t, float)
    if kind == "depth_withdrawal":
        dep = np.zeros(len(df))
        for i in range(1, n_inside + 1):
            dep += df[f"{asset}_bidquantity_{i}"].to_numpy() + df[f"{asset}_askquantity_{i}"].to_numpy()
        rel = np.diff(dep) / np.maximum(dep[:-1], EPS)
        drops = -rel                                            # positive = withdrawal
        cand = drops[drops > 0]
        thr = np.quantile(cand, q) if cand.size else np.inf
        mask = drops > thr
    elif kind == "spread_jump":
        a1 = df[f"{asset}_askprice_1"].to_numpy(); b1 = df[f"{asset}_bidprice_1"].to_numpy()
        spr = (a1 - b1) / np.maximum((a1 + b1) / 2.0, EPS) * 1e4
        chg = np.diff(spr)
        cand = chg[chg > 0]
        thr = np.quantile(cand, q) if cand.size else np.inf
        mask = chg > thr
    else:
        raise ValueError("kind must be 'depth_withdrawal' or 'spread_jump'")
    return np.sort(t[1:][mask])


# ── parameter packing ─────────────────────────────────────────────────────────
def _pack(mu, alpha, beta):
    return np.concatenate([np.ravel(mu), np.ravel(alpha), np.ravel(beta)])


def _unpack(theta, D):
    mu = theta[:D]; alpha = theta[D:D + D * D].reshape(D, D); beta = theta[D + D * D:]
    return mu, alpha, beta


# ── exact recursive negative log-likelihood (exponential kernels) ─────────────
def _negloglik(theta, merged_t, merged_c, counts, last_exp, T, D):
    """O(N*D) recursive log-likelihood. merged_t/_c are time-sorted event times and types;
    counts[j] = N_j; last_exp[j] = sum over type-j events of exp(-beta_j (T - t)) is passed
    as a closure that recomputes per beta (cheap)."""
    mu, alpha, beta = _unpack(theta, D)
    if np.any(mu <= 0) or np.any(beta <= 0) or np.any(alpha < 0):
        return 1e12
    A = np.zeros(D); last_t = 0.0; ll = 0.0
    for n in range(merged_t.shape[0]):
        t = merged_t[n]; c = merged_c[n]
        A *= np.exp(-beta * (t - last_t))
        lam_c = mu[c] + alpha[c] @ A
        if lam_c <= 0:
            return 1e12
        ll += np.log(lam_c)
        A[c] += 1.0
        last_t = t
    # compensator_i = mu_i T + sum_j (alpha_ij/beta_j)(N_j - E_j)
    E = last_exp(beta)
    comp = mu * T + (alpha / beta[None, :]) @ (counts - E)
    return -(ll - np.sum(comp))


def _prep(events, T):
    D = len(events)
    times = np.concatenate(events) if D else np.array([])
    types = np.concatenate([np.full(len(events[j]), j, dtype=int) for j in range(D)])
    order = np.argsort(times, kind="mergesort")
    merged_t = times[order]; merged_c = types[order]
    counts = np.array([len(e) for e in events], float)

    def last_exp(beta):
        return np.array([np.sum(np.exp(-beta[j] * (T - events[j]))) if len(events[j]) else 0.0
                         for j in range(D)])
    return D, merged_t, merged_c, counts, last_exp


def branching_matrix(alpha, beta):
    """G_ij = alpha_ij / beta_j: expected type-i events directly triggered per type-j event."""
    return np.asarray(alpha, float) / np.asarray(beta, float)[None, :]


def branching_ratio(G):
    """Spectral radius of the branching matrix (endogeneity / reflexivity; < 1 = stationary;
    -> 1 = near-critical self-sustaining cascades). Equals the fraction of events that are
    endogenously triggered."""
    return float(np.max(np.abs(np.linalg.eigvals(np.asarray(G, float)))))


# ── MLE ───────────────────────────────────────────────────────────────────────
def fit_hawkes(events, T, init=None, maxiter=300):
    """Fit a multivariate Hawkes (exponential kernels, per-source decay beta_j) by MLE.
    `events` is a list of D arrays of event times on [0, T]. Returns mu, alpha, beta, the
    branching matrix G, the branching ratio, and the log-likelihood."""
    from scipy.optimize import minimize
    D, merged_t, merged_c, counts, last_exp = _prep(events, T)
    if init is None:
        mu0 = np.maximum(counts / max(T, EPS) * 0.5, 1e-3)
        alpha0 = np.full((D, D), 0.1); beta0 = np.full(D, 1.0)
    else:
        mu0, alpha0, beta0 = init
    theta0 = _pack(mu0, alpha0, beta0)
    bounds = [(1e-8, 100.0)] * D + [(0.0, 100.0)] * (D * D) + [(1e-3, 100.0)] * D
    res = minimize(_negloglik, theta0, args=(merged_t, merged_c, counts, last_exp, T, D),
                   method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})
    mu, alpha, beta = _unpack(res.x, D)
    G = branching_matrix(alpha, beta)
    return {"mu": mu, "alpha": alpha, "beta": beta, "branching_matrix": G,
            "branching_ratio": branching_ratio(G), "loglik": -res.fun, "success": bool(res.success),
            "n_events": counts}


def poisson_loglik(events, T):
    """Log-likelihood of the no-excitation (homogeneous Poisson) benchmark, mu_i = N_i/T.
    Hawkes should beat this whenever events cluster."""
    ll = 0.0
    for e in events:
        N = len(e); rate = N / max(T, EPS)
        ll += N * np.log(max(rate, EPS)) - rate * T
    return ll


# ── Ogata-thinning simulator ──────────────────────────────────────────────────
def simulate_hawkes(mu, alpha, beta, T, seed=0, max_events=300000):
    """Simulate a multivariate Hawkes by Ogata's thinning (intensity is non-increasing
    between events, so its value at the interval start bounds it). Returns a list of D
    event-time arrays."""
    rng = np.random.default_rng(seed)
    mu = np.asarray(mu, float); alpha = np.asarray(alpha, float); beta = np.asarray(beta, float)
    D = len(mu); events = [[] for _ in range(D)]
    A = np.zeros(D); last_t = 0.0; t = 0.0; total = 0
    while t < T:
        A_t = A * np.exp(-beta * (t - last_t))
        lam = mu + alpha @ A_t
        lam_sum = lam.sum()
        if lam_sum <= 0:
            break
        t = t + rng.exponential(1.0 / lam_sum)
        if t > T:
            break
        A_t = A * np.exp(-beta * (t - last_t)); last_t = t
        lam = mu + alpha @ A_t; lam_sum_now = lam.sum()
        A = A_t
        if rng.random() <= lam_sum_now / lam_sum:
            c = rng.choice(D, p=lam / lam_sum_now)
            events[c].append(t); A[c] += 1.0; total += 1
            if total >= max_events:
                break
    return [np.array(e) for e in events]


# ── stress-conditional fit (the dynamic H3 test) ──────────────────────────────
def stress_conditional_fit(events_calm, T_calm, events_stress, T_stress, names=("SPY", "ES")):
    """Fit the Hawkes separately on calm and stress samples and compare. H3 prediction:
    the branching ratio (reflexivity) and the cross-excitation rise under stress -- markets
    become more endogenously cascade-prone and more contagious. Pair the samples with
    `stress_index` regimes / windows."""
    fc = fit_hawkes(events_calm, T_calm); fs = fit_hawkes(events_stress, T_stress)
    nm = list(names)
    return {"calm": fc, "stress": fs,
            "branching_ratio_calm": fc["branching_ratio"], "branching_ratio_stress": fs["branching_ratio"],
            "cross_excitation_calm": pd.DataFrame(fc["branching_matrix"], index=nm, columns=nm),
            "cross_excitation_stress": pd.DataFrame(fs["branching_matrix"], index=nm, columns=nm)}


def intensity_path(fit, events, grid):
    """Evaluate each lambda_i on a time grid given a fit and the event list (for diagnostics
    / plotting). Returns an array [len(grid), D]."""
    mu = fit["mu"]; alpha = fit["alpha"]; beta = fit["beta"]; D = len(mu)
    grid = np.asarray(grid, float); out = np.empty((len(grid), D))
    for gi, t in enumerate(grid):
        A = np.array([np.sum(np.exp(-beta[j] * (t - events[j][events[j] < t]))) if len(events[j]) else 0.0
                      for j in range(D)])
        out[gi] = mu + alpha @ A
    return out


# ── self-test ─────────────────────────────────────────────────────────────────
def _selftest():
    print("hawkes_cross self-test")

    # (1) recover a known bivariate Hawkes (asymmetric cross-excitation), check stationarity
    mu = np.array([0.6, 0.6])
    alpha = np.array([[0.30, 0.20], [0.10, 0.30]])     # SPY<-ES 0.20 > ES<-SPY 0.10 (ES leads)
    beta = np.array([1.0, 1.2])
    G_true = branching_matrix(alpha, beta); br_true = branching_ratio(G_true)
    ev = simulate_hawkes(mu, alpha, beta, T=1200.0, seed=1)
    T = 1200.0
    fit = fit_hawkes(ev, T)
    G = fit["branching_matrix"]
    print("\n(1) recovery from simulated data (N = %s events)" % fit["n_events"].astype(int).tolist())
    print("    true branching ratio = %.3f   estimated = %.3f" % (br_true, fit["branching_ratio"]))
    print("    true G:\n" + np.array2string(G_true, precision=3))
    print("    est  G:\n" + np.array2string(G, precision=3))
    print("    checks:",
          fit["success"] and abs(fit["branching_ratio"] - br_true) < 0.12
          and fit["branching_ratio"] < 1.0
          and (G[0, 1] > G[1, 0]) == (G_true[0, 1] > G_true[1, 0]))     # cross-excitation asymmetry preserved

    # (2) Hawkes beats the Poisson benchmark when events cluster
    ll_h = fit["loglik"]; ll_p = poisson_loglik(ev, T)
    print("\n(2) excitation is real: Hawkes vs homogeneous Poisson")
    print("    loglik Hawkes = %.1f  >  Poisson = %.1f   (delta = %.1f)" % (ll_h, ll_p, ll_h - ll_p))
    print("    checks:", ll_h > ll_p + 10)

    # (3) stress-conditional: cross-excitation and branching ratio rise under stress (H3)
    alpha_stress = np.array([[0.33, 0.42], [0.20, 0.33]])               # cross terms up
    ev_calm = simulate_hawkes(mu, alpha, beta, T=1200.0, seed=2)
    ev_str = simulate_hawkes(mu, alpha_stress, beta, T=1200.0, seed=3)
    sc = stress_conditional_fit(ev_calm, 1200.0, ev_str, 1200.0)
    print("\n(3) stress-conditional fit (dynamic H3 test)")
    print("    branching ratio: calm = %.3f -> stress = %.3f" % (sc["branching_ratio_calm"], sc["branching_ratio_stress"]))
    print("    cross-excitation SPY<-ES: calm = %.3f -> stress = %.3f"
          % (sc["cross_excitation_calm"].loc["SPY", "ES"], sc["cross_excitation_stress"].loc["SPY", "ES"]))
    print("    checks:",
          sc["branching_ratio_stress"] > sc["branching_ratio_calm"]
          and sc["cross_excitation_stress"].loc["SPY", "ES"] > sc["cross_excitation_calm"].loc["SPY", "ES"])

    # (4) liquidity-event extraction from a book frame + end-to-end fit
    rng = np.random.default_rng(7); n = 6000
    idx = pd.date_range("2020-03-09 09:30", periods=n, freq="s", tz="America/New_York")
    cols = {}
    for a, base, tick in (("SPY", 500.0, 0.01), ("ES", 5000.0, 0.25)):
        dep = np.maximum(300 + 50 * rng.standard_normal(n).cumsum() * 0 + rng.gamma(3, 60, n), 5.0)
        # inject clustered withdrawals
        shock = rng.random(n) < 0.03
        dep = dep * np.where(shock, 0.4, 1.0)
        mid = base * np.exp(rng.normal(0, 1, n).cumsum() / 4000.0)
        for l in range(1, 11):
            cols[f"{a}_bidprice_{l}"] = mid - l * tick; cols[f"{a}_askprice_{l}"] = mid + l * tick
            q = np.maximum(dep * np.exp(-0.3 * (l - 1)), 1.0)
            cols[f"{a}_bidquantity_{l}"] = q; cols[f"{a}_askquantity_{l}"] = q
    book = pd.DataFrame(cols, index=idx)
    e_spy = liquidity_events(book, "SPY", "depth_withdrawal", q=0.95)
    e_es = liquidity_events(book, "ES", "depth_withdrawal", q=0.95)
    Tb = (book.index[-1] - book.index[0]).total_seconds()
    fit_book = fit_hawkes([e_spy, e_es], Tb)
    print("\n(4) liquidity-event marks from a book + fit")
    print("    extracted events: SPY=%d  ES=%d  over %.0f s" % (len(e_spy), len(e_es), Tb))
    print("    branching ratio = %.3f (success=%s)" % (fit_book["branching_ratio"], fit_book["success"]))
    print("    checks:", len(e_spy) > 30 and len(e_es) > 30 and 0 <= fit_book["branching_ratio"] < 1.0
          and fit_book["success"])


if __name__ == "__main__":
    _selftest()
