"""test_leadlag.py -- guards noise_robust_cov.lead_lag (Hoffmann-Rosenbaum-Yoshida 2013).

A common efficient factor f drives both assets; one observes f in real time, the other observes
f delayed by a known lag delta (asynchronous Poisson sampling for each). The estimator must
recover delta and its SIGN: positive lead_lag => the FIRST series leads. Checks forward lead,
the no-lead null, and a reversed lead.
"""
import sys
import numpy as np
import noise_robust_cov as nrc

def make_factor(T, dt_f, sigma, seed):
    rng = np.random.default_rng(seed)
    g = np.arange(0.0, T, dt_f)
    f = np.concatenate([[0.0], np.cumsum(rng.normal(0, sigma, len(g) - 1))])
    return g, f

def poisson_times(T, rate, rng):
    t = np.cumsum(rng.exponential(1.0 / rate, int(T * rate * 1.3)))
    return t[t < T]

def sample(g, f, times, lag):
    idx = np.clip(np.searchsorted(g, times - lag, side="right") - 1, 0, len(f) - 1)
    return f[idx]

def experiment(delta, T=60.0, rate=40.0, sigma=2e-4, seed=0):
    # asset 1 observes f in real time; asset 2 observes f delayed by delta (so 1 leads 2 by delta)
    g, f = make_factor(T, 1e-3, sigma, seed)
    rng = np.random.default_rng(seed + 1)
    t1 = poisson_times(T, rate, rng); t2 = poisson_times(T, rate, rng)
    lp1 = sample(g, f, t1, 0.0)
    lp2 = sample(g, f, t2, delta)               # asset 2 is delayed by delta
    return nrc.lead_lag(t1, lp1, t2, lp2, max_lag=0.2, n_grid=81)

if __name__ == "__main__":
    step = 0.4 / 80                              # grid step = 0.005 s
    print("HRY lead-lag estimator (grid step = %.0f ms; positive => FIRST series leads)" % (1000 * step))
    fwd = experiment(delta=+0.05, seed=10)       # asset 1 leads asset 2 by 50 ms
    nul = experiment(delta=0.0,   seed=20)       # no lead
    rev = experiment(delta=-0.05, seed=30)       # asset 2 leads asset 1 by 50 ms
    print("  injected +50ms : est=%+.0f ms  leader=%-6s peak=%.2f" % (1000 * fwd["lead_lag"], fwd["leader"], fwd["peak_contrast"]))
    print("  injected   0ms : est=%+.0f ms  leader=%-6s peak=%.2f" % (1000 * nul["lead_lag"], nul["leader"], nul["peak_contrast"]))
    print("  injected -50ms : est=%+.0f ms  leader=%-6s peak=%.2f" % (1000 * rev["lead_lag"], rev["leader"], rev["peak_contrast"]))
    tol = 0.011                                  # within ~2 grid steps
    ok = (abs(fwd["lead_lag"] - 0.05) <= tol and fwd["leader"] == "first" and fwd["peak_contrast"] > 0.3
          and abs(nul["lead_lag"]) <= tol
          and abs(rev["lead_lag"] + 0.05) <= tol and rev["leader"] == "second")
    print("checks: forward=%s null=%s reverse=%s -> %s"
          % (abs(fwd["lead_lag"] - 0.05) <= tol and fwd["leader"] == "first",
             abs(nul["lead_lag"]) <= tol, abs(rev["lead_lag"] + 0.05) <= tol and rev["leader"] == "second", ok))
    sys.exit(0 if ok else 1)
