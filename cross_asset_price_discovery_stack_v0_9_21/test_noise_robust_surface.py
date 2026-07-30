"""test_noise_robust_surface.py -- the referee's objection, run and defused.

NO spiral exists: the fundamental ES->SPY transmission is a constant 0.8 across the whole (basis,holl)
plane, so the TRUE interaction b3 is exactly 0. But observed mids carry bid-ask bounce whose POST-onset
variance scales with basis*holl -- the empirically natural confound (a hollow touch shatters harder
under a bigger shock, flips faster, noisier mid). Errors-in-variables then attenuates the naive Rigobon
transmission most in the hi-basis/hi-holl corner, manufacturing a spurious NEGATIVE b3 that mimics the
liquidity spiral. The noise-robust transmission (TSRV variances + HY cross) must recover b3 ~ 0; the
omega noise-level control must attenuate the naive b3 toward 0.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import onset_response as onr

NY = "America/New_York"

def make_bounce_event(basis_i, holl_i, beta=0.8, T=2200, base_var=9e-10, k=2e-9, date="2025-01-01", seed=0):
    rng = np.random.default_rng(seed); o = T // 2
    es = np.empty(T - 1); es[:o - 1] = rng.normal(0, 2e-5, o - 1); es[o - 1:] = rng.normal(0, 1.6e-4, (T - 1) - (o - 1))
    spy = beta * es + rng.normal(0, 2e-5, T - 1)                       # CONSTANT beta -> no spiral
    lpe = np.r_[np.log(5000.0), np.log(5000.0) + np.cumsum(es)]
    lps = np.r_[np.log(500.0), np.log(500.0) + np.cumsum(spy)]
    # additive bid-ask bounce in OBSERVED log-mids: iid per point, variance jumps post-onset, the jump
    # scaled by basis*holl so attenuation is ~linear in the interaction (the spurious-b3 channel).
    v_post = base_var + k * basis_i * holl_i
    sd = np.full(T, np.sqrt(base_var)); sd[o:] = np.sqrt(v_post)
    obs_s = lps + rng.normal(0, 1, T) * sd
    obs_e = lpe + rng.normal(0, 1, T) * sd
    ms, me = np.exp(obs_s), np.exp(obs_e)
    idx = pd.date_range(f"{date} 09:30", periods=T, freq="s", tz=NY)
    cols = {}
    for a, mid, tk in (("SPY", ms, 0.01), ("ES", me, 0.25)):
        for i in range(1, 3):
            cols[f"{a}_bidprice_{i}"] = mid - (i - 0.5) * tk
            cols[f"{a}_askprice_{i}"] = mid + (i - 0.5) * tk
            cols[f"{a}_bidquantity_{i}"] = 100.0
            cols[f"{a}_askquantity_{i}"] = 100.0
    return pd.DataFrame(cols, index=idx), idx[o]

def main():
    warnings.simplefilter("ignore")
    g = np.linspace(0.6, 4.0, 8); Bm, Hm = np.meshgrid(g, g)
    basis, holl = Bm.ravel(), Hm.ravel()
    rows = []
    for k, (bi, hi) in enumerate(zip(basis, holl)):
        df, ots = make_bounce_event(bi, hi, seed=100 + k)
        e = onr.event_onset_estimate(df, ots, pre_steps=850, post_steps=850)
        rows.append(dict(state=bi, state_holl=hi, transmission=e["transmission"],
                         transmission_robust=e["transmission_robust"], pre_noise_bps=e["pre_noise_bps"],
                         impact_blind=True, id_strength=1.0))
    panel = pd.DataFrame(rows)

    # corner contrast (diagnostic): naive should sag in the hi/hi corner, robust should not
    lo = panel[(panel.state == g[0]) & (panel.state_holl == g[0])].iloc[0]
    hi = panel[(panel.state == g[-1]) & (panel.state_holl == g[-1])].iloc[0]
    print("   corner naive: lo=%.3f hi=%.3f (sag=%.3f) | robust: lo=%.3f hi=%.3f"
          % (lo.transmission, hi.transmission, lo.transmission - hi.transmission,
             lo.transmission_robust, hi.transmission_robust))

    fit_n = onr.fit_stress_surface(panel, y_col="transmission")
    fit_r = onr.fit_stress_surface(panel, y_col="transmission_robust")
    fit_c = onr.fit_stress_surface(panel, y_col="transmission", covar_cols=["pre_noise_bps"])
    b3n, pn = fit_n["interaction"], fit_n["interaction_p"]
    b3r, pr = fit_r["interaction"], fit_r["interaction_p"]
    b3c = fit_c["interaction"]

    noise_sd = float(np.std(panel["pre_noise_bps"]))
    a_ok = b3n < 0 and pn < 0.05                              # naive finds a spurious, significant negative spiral
    b_ok = abs(b3r) < abs(b3n) and pr > 0.05                  # robust shrinks it to insignificance
    c_ok = (b3c < 0) and (abs(b3c) > 0.5 * abs(b3n))          # a PRE-window control cannot undo a POST-onset shift
    print("(A) naive   b3=%+.4f p=%.3f  spurious-spiral=%s" % (b3n, pn, a_ok))
    print("(B) robust  b3=%+.4f p=%.3f  neutralized=%s" % (b3r, pr, b_ok))
    print("(C) pre-noise control b3=%+.4f  (pre-omega sd across events=%.4f bps; cannot rescue)=%s"
          % (b3c, noise_sd, c_ok))
    ok = a_ok and b_ok and c_ok
    print("noise-robust-surface checks ->", ok)
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
