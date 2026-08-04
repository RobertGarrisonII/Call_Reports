"""
inference.py
===========
Inference layer for the redo -- the rest of item #7. Three corrections the paper omits:

  1. Generated-regressor SEs. Delta-rho is an ESTIMATED rolling/DCC correlation, used as the
     SVAR dependent variable (Eq. 5) and as a Table-14 regressor. Naive OLS SEs ignore both
     its first-stage estimation error and its heavy serial correlation (overlapping windows).
     The fix is a dependent (moving-block) bootstrap that re-generates Delta-rho inside each
     replication, propagating both.
  2. Bootstrap inference for the SVAR / IRF coefficients (the IRFs report stars with no
     sampling distribution): wrap the IRF estimator as the bootstrap statistic.
  3. Multiple testing across the dozens of IRF / DiD coefficients: Romano-Wolf step-down
     (bootstrap-based, robust to cross-correlation and far more powerful than Bonferroni),
     plus Holm (FWER) and Benjamini-Hochberg (FDR).

Pure numpy/pandas. The moving-block bootstrap engine takes any statistic(resampled_data);
romano_wolf_from_boot takes the point estimates and their bootstrap draws.
"""
import numpy as np
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor

EPS = 1e-12


# ── moving-block (dependent) bootstrap ────────────────────────────────────────
def auto_block_length(x):
    """A simple data-driven block length ~ n^{1/3} (the MBB optimal rate), inflated by
    persistence: doubled when lag-1 autocorrelation is high. Override when you know the
    dependence horizon (e.g. the rolling-correlation window)."""
    x = np.asarray(x, float)
    x = x[:, 0] if x.ndim > 1 else x
    n = len(x); base = max(2, int(round(n ** (1.0 / 3.0))))
    xc = x - np.nanmean(x); denom = np.nansum(xc * xc)
    rho1 = float(np.nansum(xc[1:] * xc[:-1]) / denom) if denom > EPS else 0.0
    return int(base * (2 if abs(rho1) > 0.3 else 1))


def parallel_bootstrap(draw_one, n_boot, n_jobs=None, backend="threading", seed=0):
    """Run ``n_boot`` bootstrap replicates of ``draw_one(child_seed)`` and collect the
    non-``None`` results into a list. Order is **not** significant downstream (SE, percentile
    CI and Romano-Wolf are all order-invariant), so completion order is fine.

    Reproducibility: each replicate is handed its OWN ``numpy.random.SeedSequence`` (a spawned
    child of ``SeedSequence(seed)``), so the *set* of draws is deterministic and identical
    across backends and worker counts — serial == threaded == process for a given ``seed``.
    That is what lets the self-test assert serial/parallel equivalence. (Note this seeding
    differs from a single shared ``default_rng(seed)`` stream, so exact draws — hence the third
    decimal of a bootstrap SE — will differ from the pre-parallel code by Monte-Carlo noise;
    nothing systematic.)

    backend:
      ``"threading"`` (default) — a ``ThreadPoolExecutor``. The right choice for the SVAR/IRF
        day-cluster bootstrap and other in-memory refits: the data (session frames / design
        matrices) is shared **read-only** with no pickling, and each replicate's refit is
        numpy/pandas C code that releases the GIL, so it scales across cores. Because the
        bootstrap runs *after* extraction with the data already resident, it is **not**
        memory-bound (unlike extraction) — so ``n_jobs=None`` uses ALL cores (``os.cpu_count()``).
        Keep BLAS pinned to one thread per process (the shell driver does) so the threads don't
        oversubscribe inside each linear-algebra call.
      ``"sequential"`` — a plain loop (also taken automatically when ``n_jobs == 1`` or ``n_boot < 2``).
      ``"process"`` — joblib loky if importable, for CPU-bound *pure-Python* replicates where the
        GIL would bind; requires ``draw_one`` to be picklable (a module-level function, NOT a
        closure), so it is opt-in and not used by the closure-based SVAR path.

    ``draw_one(child_seed)`` takes a single ``SeedSequence`` and returns an array-like draw, or
    ``None`` to drop the replicate (e.g. a degenerate resample that could not be fit).
    """
    if not n_boot or int(n_boot) < 1:
        return []
    n_boot = int(n_boot)
    children = np.random.SeedSequence(seed).spawn(n_boot)
    if n_jobs is None:
        # autoscale, not os.cpu_count(): cpu_count() ignores CPU affinity, so inside a container
        # pinned to a subset of a large node it reports the NODE's cores and this pool
        # oversubscribes every one of them. autoscale.cpu_jobs() takes the max across cpu_count,
        # /proc/cpuinfo and sched_getaffinity, honours the BOOT_WORKERS override, and is the same
        # number the shell drivers size themselves from -- so bash and Python cannot disagree.
        try:
            import autoscale as _autoscale
            n_jobs = _autoscale.cpu_jobs()
        except Exception:
            n_jobs = os.cpu_count() or 1
    else:
        n_jobs = int(n_jobs)
    n_jobs = max(1, min(n_jobs, n_boot))                       # never more workers than replicates
    if backend == "sequential" or n_jobs == 1 or n_boot < 2:
        return [d for d in (draw_one(c) for c in children) if d is not None]
    if backend == "process":
        try:
            from joblib import Parallel, delayed
            res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(draw_one)(c) for c in children)
            return [d for d in res if d is not None]
        except Exception:
            backend = "threading"                              # fall back if joblib/pickling unavailable
    with ThreadPoolExecutor(max_workers=n_jobs) as ex:         # threading (default)
        res = list(ex.map(draw_one, children))
    return [d for d in res if d is not None]


def moving_block_bootstrap(data, statistic, n_boot=499, block_len=None, seed=0, n_jobs=1):
    """Moving-block bootstrap for time series. Resamples overlapping blocks of rows (with
    replacement) to preserve short-range dependence, applies `statistic` to each resample,
    and returns a (n_boot, k) array of statistic values. `data` is (T,) or (T, m); when the
    statistic re-generates a quantity (e.g. a rolling correlation) inside it, the first-stage
    uncertainty is propagated automatically. ``n_jobs`` parallelizes the replicates via
    :func:`parallel_bootstrap` (threading; default ``1`` = serial); ``n_jobs=None`` uses all
    cores — appropriate when ``statistic`` is a heavy numpy/pandas refit."""
    data = np.asarray(data); n = len(data)
    if block_len is None:
        block_len = auto_block_length(data)
    block_len = max(1, min(block_len, n))
    n_blocks = int(np.ceil(n / block_len)); starts_max = n - block_len
    offs = np.arange(block_len)

    def _one(child_seed):
        r = np.random.default_rng(child_seed)
        starts = r.integers(0, starts_max + 1, n_blocks)
        idx = (starts[:, None] + offs[None, :]).ravel()[:n]
        return np.atleast_1d(np.asarray(statistic(data[idx]), float))

    out = parallel_bootstrap(_one, n_boot, n_jobs=n_jobs, backend="threading", seed=seed)
    return np.array(out)


def bootstrap_summary(boot, point, alpha=0.05):
    """SE, percentile CI, and basic (reverse-percentile) CI from bootstrap draws (B,k)
    and the point estimate (k,). The bootstrap t = point/SE."""
    boot = np.asarray(boot, float); point = np.atleast_1d(np.asarray(point, float))
    se = boot.std(axis=0, ddof=1)
    lo = np.percentile(boot, 100 * alpha / 2, axis=0); hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)
    return {"se": se, "ci_percentile": (lo, hi), "ci_basic": (2 * point - hi, 2 * point - lo),
            "t": point / np.where(se > EPS, se, np.nan)}


# ── generated-regressor convenience ───────────────────────────────────────────
def _ols(y, X):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    return b, resid


def ols_naive_se(y, X):
    """Classical (iid-homoskedastic) OLS coefficients and SEs -- the (wrong) benchmark for
    a serially-correlated generated regressor."""
    b, resid = _ols(np.asarray(y, float), np.asarray(X, float))
    n, k = X.shape
    s2 = float(resid @ resid) / max(n - k, 1)
    cov = s2 * np.linalg.pinv(X.T @ X)
    return b, np.sqrt(np.clip(np.diag(cov), 0, None))


def generated_regressor_bootstrap(data, generate_and_fit, n_boot=399, block_len=None, seed=0):
    """Two-stage moving-block bootstrap for an estimand built on a generated quantity.
    `generate_and_fit(resampled_data) -> coefficient vector` must recompute the generated
    regressor (e.g. the rolling correlation) AND the second-stage fit, so the returned
    bootstrap SEs include both the first-stage estimation error and the serial dependence.
    Returns the bootstrap draws and a summary."""
    boot = moving_block_bootstrap(data, generate_and_fit, n_boot, block_len, seed)
    point = np.atleast_1d(np.asarray(generate_and_fit(data), float))
    return {"boot": boot, "point": point, **bootstrap_summary(boot, point)}


# ── cluster-robust inference for FEW clusters / FEW events ─────────────────────
# The paper pools ~23.4M intraday observations and reports essentially every coefficient at
# p<0.01, but the effective number of independent units is the number of DAYS (~20 in the
# 2014-2017 panel) or the number of MWCB EVENTS (4). Under strong within-day dependence the
# naive/iid t is overstated by roughly sqrt(n_within * intra-cluster-corr) -- one to two
# orders of magnitude here -- and even the cluster-robust CRVE t with a normal reference
# over-rejects when clusters are few. The defensible procedures are the wild CLUSTER bootstrap
# (Cameron, Gelbach & Miller 2008; Webb 2014 six-point weights when clusters <= ~12;
# MacKinnon & Webb 2017, 2018) and the Ibragimov-Muller (2010, 2016) group t-statistic.

def _cluster_cov(X, u, g, XtXinv):
    """Liang-Zeger meat with the stack's finite-sample factor c=G/(G-1)*(N-1)/(N-K)."""
    uniq = np.unique(g); G = len(uniq); N, K = X.shape
    meat = np.zeros((K, K))
    for gi in uniq:
        m = g == gi; s = X[m].T @ u[m]; meat += np.outer(s, s)
    c = (G / (G - 1.0)) * ((N - 1.0) / (N - K)) if (G > 1 and N > K) else 1.0
    return (XtXinv @ meat @ XtXinv) * c, G


def cluster_robust_ols(y, X, clusters):
    """One-way cluster-robust (Liang-Zeger 1986) OLS. Returns dict with b, se, t, n_clusters.
    The t uses a normal reference; for FEW clusters prefer wild_cluster_bootstrap / ibragimov_muller."""
    y = np.asarray(y, float); X = np.asarray(X, float); g = np.asarray(clusters)
    XtXinv = np.linalg.pinv(X.T @ X); b = XtXinv @ (X.T @ y); u = y - X @ b
    V, G = _cluster_cov(X, u, g, XtXinv); se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    return {"b": b, "se": se, "t": b / (se + EPS), "n_clusters": int(G), "vcov": V}


def _rademacher(rng, G):
    return rng.choice(np.array([-1.0, 1.0]), size=G)


def _webb6(rng, G):
    # Webb (2014) six-point distribution: ±sqrt(1.5), ±1, ±sqrt(0.5) each w.p. 1/6. With few
    # clusters Rademacher gives only 2^G distinct sign patterns (G=6 -> 64 -> coarse p-values);
    # the six-point lattice gives 6^G and a far finer bootstrap distribution.
    v = np.array([-np.sqrt(1.5), -1.0, -np.sqrt(0.5), np.sqrt(0.5), 1.0, np.sqrt(1.5)])
    return rng.choice(v, size=G)


def wild_cluster_bootstrap(y, X, clusters, test_idx=None, r=0.0, n_boot=999,
                           weights="auto", seed=0, return_dist=False):
    """Wild cluster RESTRICTED bootstrap (WCR) of Cameron, Gelbach & Miller (2008) for the
    hypothesis beta_j = r. Imposes the null, resamples cluster-level wild weights on the
    RESTRICTED residuals, refits UNRESTRICTED each replicate, and compares the cluster-robust
    t to its bootstrap distribution. This is the recommended few-cluster procedure (it has
    correct size where the CRVE t does not). ``weights='auto'`` uses Webb six-point when the
    number of clusters G <= 12, else Rademacher.

    test_idx=None tests every coefficient (a separate restricted fit per coefficient).
    Returns (per coefficient) a dict with the cluster-robust t, the bootstrap p-value, the
    cluster SE, the point estimate, G, and the weight scheme."""
    y = np.asarray(y, float); X = np.asarray(X, float); g = np.asarray(clusters)
    N, K = X.shape; uniq = np.unique(g); G = len(uniq)
    gi_of = np.searchsorted(uniq, g)                                   # cluster index per row
    XtXinv = np.linalg.pinv(X.T @ X)
    bhat = XtXinv @ (X.T @ y); uhat = y - X @ bhat
    V, _ = _cluster_cov(X, uhat, g, XtXinv); se = np.sqrt(np.clip(np.diag(V), 0.0, None))
    use_webb = (G <= 12) if weights == "auto" else (str(weights).lower() == "webb6")
    master = np.random.default_rng(seed)
    idxs = range(K) if test_idx is None else [int(test_idx)]
    out = {}
    for j in idxs:
        cols = [c for c in range(K) if c != j]
        Xr = X[:, cols]; yr = y - r * X[:, j]
        br = np.linalg.pinv(Xr.T @ Xr) @ (Xr.T @ yr)
        btil = np.zeros(K); btil[j] = r
        for c, bc in zip(cols, br):
            btil[c] = bc
        util = y - X @ btil                                           # restricted residuals
        fitted = X @ btil
        t_j = (bhat[j] - r) / (se[j] + EPS)
        tboot = np.empty(n_boot); ge = 0
        for b in range(n_boot):
            rng = np.random.default_rng(master.integers(0, 2 ** 63 - 1))
            w = _webb6(rng, G) if use_webb else _rademacher(rng, G)
            ystar = fitted + w[gi_of] * util                          # same weight within a cluster
            bst = XtXinv @ (X.T @ ystar); ust = ystar - X @ bst
            Vs, _ = _cluster_cov(X, ust, g, XtXinv); ses = np.sqrt(np.clip(np.diag(Vs), 0.0, None))
            tb = (bst[j] - r) / (ses[j] + EPS); tboot[b] = tb
            ge += abs(tb) >= abs(t_j)
        p = (ge + 1.0) / (n_boot + 1.0)                               # never exactly 0
        d = {"t": float(t_j), "p": float(p), "se_cluster": float(se[j]), "beta": float(bhat[j]),
             "n_clusters": int(G), "weights": "webb6" if use_webb else "rademacher"}
        if return_dist:
            d["t_dist"] = tboot
        out[j] = d
    return out if test_idx is None else out[int(test_idx)]


def ibragimov_muller(group_estimates, r=0.0):
    """Ibragimov-Muller (2010, 2016) few-clusters t-test. Given a coefficient estimated
    SEPARATELY in each of G groups (e.g. per day or per MWCB event), beta_g, the group mean
    theta=mean(beta_g) is tested with se=sd(beta_g)/sqrt(G) against a t_{G-1} reference. Valid
    (conservative) with few, heterogeneous clusters where the CRVE/normal approximation fails;
    this is the right tool for the 4-event MWCB analysis. Returns theta, se, t, two-sided p,
    df, and the t_{G-1} 95% CI."""
    from scipy import stats
    b = np.asarray(group_estimates, float); b = b[np.isfinite(b)]; G = len(b)
    if G < 2:
        return {"theta": float(b[0]) if G else np.nan, "se": np.nan, "t": np.nan,
                "p": np.nan, "df": max(G - 1, 0), "ci": (np.nan, np.nan), "n_groups": G}
    theta = float(b.mean()); se = float(b.std(ddof=1) / np.sqrt(G)); t = (theta - r) / (se + EPS)
    p = float(2.0 * stats.t.sf(abs(t), df=G - 1)); crit = float(stats.t.ppf(0.975, df=G - 1))
    return {"theta": theta, "se": se, "t": float(t), "p": p, "df": G - 1,
            "ci": (theta - crit * se, theta + crit * se), "n_groups": G}


# ── multiple testing ──────────────────────────────────────────────────────────
def holm(pvals, alpha=0.05):
    """Holm step-down FWER control. Returns adjusted p-values and rejections."""
    p = np.asarray(pvals, float); k = len(p); order = np.argsort(p)
    adj = np.empty(k); running = 0.0
    for i, j in enumerate(order):
        running = max(running, (k - i) * p[j]); adj[j] = min(running, 1.0)
    return {"adj_p": adj, "reject": adj <= alpha}


def benjamini_hochberg(pvals, q=0.05):
    """Benjamini-Hochberg FDR control. Returns adjusted p-values and rejections at level q."""
    p = np.asarray(pvals, float); k = len(p); order = np.argsort(p); adj = np.empty(k); prev = 1.0
    for i in range(k - 1, -1, -1):
        j = order[i]; prev = min(prev, p[j] * k / (i + 1)); adj[j] = min(prev, 1.0)
    return {"adj_p": adj, "reject": adj <= q}


def romano_wolf_from_boot(point, boot, alpha=0.05):
    """Romano-Wolf step-down FWER control using the bootstrap distribution of the
    studentized statistics -- robust to cross-correlation among the coefficients and more
    powerful than Bonferroni/Holm. `point` (k,) are the estimates, `boot` (B,k) their
    bootstrap draws. Returns studentized t, adjusted p-values, rejections, and SEs."""
    point = np.asarray(point, float); boot = np.asarray(boot, float)
    se = boot.std(axis=0, ddof=1); se_safe = np.where(se > EPS, se, np.nan)
    t = np.abs(point / se_safe)
    boot_t = np.abs((boot - point) / se_safe[None, :])              # recentered -> null distribution
    order = np.argsort(t)[::-1]; adj = np.ones(len(point)); prev = 0.0
    for step in range(len(order)):
        active = order[step:]
        maxdist = np.nanmax(boot_t[:, active], axis=1)
        p = max(float(np.mean(maxdist >= t[order[step]])), prev)
        adj[order[step]] = p; prev = p
        if p > alpha:
            adj[order[step:]] = np.maximum(adj[order[step:]], p)     # remaining all fail (monotone)
            break
    return {"t": t, "adj_p": adj, "reject": adj <= alpha, "se": se}


# ── self-test ─────────────────────────────────────────────────────────────────
def _rolling_corr(a, b, w):
    return pd.Series(a).rolling(w).corr(pd.Series(b)).to_numpy()


def _selftest():
    print("inference self-test")

    # (1) moving-block bootstrap captures serial-correlation SE inflation (AR(1) mean)
    rng = np.random.default_rng(1); n = 4000; phi = 0.6
    e = rng.normal(0, 1, n); x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    boot = moving_block_bootstrap(x, lambda d: d.mean(), n_boot=500, block_len=40)
    se_block = float(boot.std(ddof=1)); se_iid = float(x.std(ddof=1) / np.sqrt(n))
    se_theory = (1.0 / (1 - phi)) / np.sqrt(n)                       # LRV of AR(1) = sigma_e^2/(1-phi)^2
    print("\n(1) block bootstrap vs iid SE for the mean of an AR(1)")
    print("    iid SE=%.4f  block-bootstrap SE=%.4f  theoretical SE=%.4f" % (se_iid, se_block, se_theory))
    print("    checks:", se_block > 1.5 * se_iid and 0.5 * se_theory < se_block < 2.0 * se_theory)

    # (2) generated regressor: naive OLS SE understates; two-stage block bootstrap fixes it
    n = 6000; rng = np.random.default_rng(2)
    state = np.zeros(n)
    for t in range(1, n):
        state[t] = 0.99 * state[t - 1] + rng.normal(0, 0.1)         # slow state drives correlation
    rho_t = np.tanh(state)                                          # true time-varying correlation
    z1 = rng.normal(0, 1, n); z2 = rng.normal(0, 1, n)
    r1 = z1
    r2 = rho_t * z1 + np.sqrt(np.clip(1 - rho_t ** 2, 0, 1)) * z2   # corr(r1,r2)=rho_t
    data = np.column_stack([r1, r2, state])
    w = 60

    def gen_and_fit(d):                                            # regress rolling corr on the state
        rc = _rolling_corr(d[:, 0], d[:, 1], w)
        m = np.isfinite(rc)
        X = np.column_stack([np.ones(m.sum()), d[m, 2]])
        b, _ = _ols(rc[m], X)
        return b[1]                                                # slope on the state

    rc0 = _rolling_corr(r1, r2, w); m0 = np.isfinite(rc0)
    X0 = np.column_stack([np.ones(m0.sum()), state[m0]])
    _, se_naive = ols_naive_se(rc0[m0], X0)
    boot_fix = moving_block_bootstrap(np.column_stack([rc0, state]),  # block bootstrap, rho FIXED
                                      lambda d: _ols(d[np.isfinite(d[:, 0]), 0],
                                                     np.column_stack([np.ones(np.isfinite(d[:, 0]).sum()),
                                                                      d[np.isfinite(d[:, 0]), 1]]))[0][1],
                                      n_boot=300, block_len=300)
    gr = generated_regressor_bootstrap(data, gen_and_fit, n_boot=300, block_len=300)
    se_block_fixed = float(boot_fix.std(ddof=1)); se_two_stage = float(gr["se"][0])
    print("\n(2) generated regressor (rolling corr on a state): naive vs bootstrap SE")
    print("    naive OLS SE=%.4f  block-bootstrap SE (rho fixed)=%.4f  two-stage SE (rho regenerated)=%.4f"
          % (se_naive[1], se_block_fixed, se_two_stage))
    print("    checks:", se_block_fixed > 2 * se_naive[1] and se_two_stage > se_naive[1] and np.isfinite(se_two_stage))

    # (3) Romano-Wolf step-down vs Bonferroni / Holm on correlated coefficients
    k = 20; B = 800; rng = np.random.default_rng(3)
    point = np.array([6.0, 5.0, 4.5, 4.0] + [0.0] * (k - 4))        # 4 clear signals, 16 nulls (se ~ 1 -> t ~ point)
    R = 0.5 * np.ones((k, k)) + 0.5 * np.eye(k); L = np.linalg.cholesky(R)
    boot = point[None, :] + (rng.standard_normal((B, k)) @ L.T)
    rw = romano_wolf_from_boot(point, boot)
    pvals = 2 * (1 - _norm_cdf(np.abs(point) / boot.std(0, ddof=1)))
    bonf_adj = np.minimum(k * pvals, 1.0); bonf = bonf_adj <= 0.05
    hl = holm(pvals); bh = benjamini_hochberg(pvals)
    # power comparison via critical values: bootstrap max-|t| 95th quantile vs Bonferroni z
    null_t = np.abs((boot - point) / boot.std(0, ddof=1)[None, :])
    rw_crit = float(np.quantile(null_t.max(axis=1), 0.95)); bonf_crit = _norm_ppf(1 - 0.025 / k)
    print("\n(3) multiple testing across %d correlated coefficients (4 true signals)" % k)
    print("    rejections: Romano-Wolf=%d  Holm=%d  Bonferroni=%d  BH=%d"
          % (rw["reject"].sum(), hl["reject"].sum(), bonf.sum(), bh["reject"].sum()))
    print("    RW rejects signals:", rw["reject"][:4].tolist(), " false positives:", int(rw["reject"][4:].sum()))
    print("    critical value: Romano-Wolf=%.3f < Bonferroni=%.3f (RW more powerful under dependence)"
          % (rw_crit, bonf_crit))
    rw_null = romano_wolf_from_boot(np.zeros(k), rng.standard_normal((B, k)) @ L.T)   # degenerate null
    print("    checks:", rw["reject"][:4].all() and rw["reject"][4:].sum() == 0
          and rw["reject"].sum() >= bonf.sum()
          and rw_crit < bonf_crit
          and rw_null["reject"].sum() == 0)

    # (4) parallel bootstrap: per-replicate spawned seeds make the draw SET deterministic and
    #     identical across backends and worker counts -> serial == threaded (and order-free).
    def _draw(child_seed):
        r = np.random.default_rng(child_seed)
        return float(r.normal(3.0, 1.0, 256).mean())                # stand-in replicate statistic
    ser = np.array(parallel_bootstrap(_draw, 300, n_jobs=1, seed=7))
    thr = np.array(parallel_bootstrap(_draw, 300, n_jobs=8, backend="threading", seed=7))
    mb1 = moving_block_bootstrap(x, lambda d: d.mean(), n_boot=200, block_len=40, seed=11, n_jobs=1)
    mb8 = moving_block_bootstrap(x, lambda d: d.mean(), n_boot=200, block_len=40, seed=11, n_jobs=8)
    print("\n(4) parallel bootstrap reproducibility (serial == threaded, order-free)")
    print("    parallel_bootstrap: n=%d  serial mean=%.5f  threaded mean=%.5f  max|diff|=%.2e"
          % (len(ser), ser.mean(), thr.mean(), float(np.max(np.abs(np.sort(ser) - np.sort(thr))))))
    print("    moving_block n_jobs 1 vs 8 identical:", bool(np.allclose(mb1, mb8)))
    print("    checks:", len(ser) == 300 and np.allclose(np.sort(ser), np.sort(thr))
          and np.allclose(mb1, mb8) and bool(np.isfinite(ser).all()))


def _norm_cdf(z):
    from math import erf, sqrt
    z = np.asarray(z, float)
    return 0.5 * (1.0 + np.vectorize(lambda v: erf(v / sqrt(2.0)))(z))


def _norm_ppf(p):
    lo, hi = -12.0, 12.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if float(_norm_cdf(mid)) < p:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


if __name__ == "__main__":
    _selftest()
