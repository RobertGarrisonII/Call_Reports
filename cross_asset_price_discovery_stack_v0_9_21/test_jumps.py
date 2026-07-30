"""test_jumps.py -- guards the jump-robust layer in noise_robust_cov.

A continuous diffusion with known integrated variance IV plus a handful of large jumps. Naive RV
absorbs the jumps (biased up); bipower variation and threshold RV recover IV; the BNS/Huang-Tauchen
ratio test flags the discontinuities on the jump path and not on a no-jump control.
"""
import sys
import numpy as np
import noise_robust_cov as nrc

def path(n, sd, n_jumps, jump, seed):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0, sd, n)                       # diffusive returns
    if n_jumps:
        loc = rng.choice(n, size=n_jumps, replace=False)
        r[loc] += rng.choice([-1.0, 1.0], n_jumps) * jump
    return np.concatenate([[0.0], np.cumsum(r)])     # log-price

if __name__ == "__main__":
    n, sd, njmp, jump = 2000, 5e-4, 5, 1e-2
    IV = n * sd ** 2                                  # true integrated variance
    print("Jump-robust variation  (IV_true=%.3e; %d jumps of %.0fx the per-step sd)" % (IV, njmp, jump / sd))

    lp = path(n, sd, njmp, jump, seed=7)
    rv = nrc.realized_variance(lp, "rv"); bv = nrc.bipower_variation(lp)
    trv = nrc.threshold_variance(lp); jt = nrc.jump_test(lp)
    print("  WITH jumps : RV=%.3e  BV=%.3e  TRV=%.3e   jump_test z=%.2f p=%.4f" % (rv, bv, trv, jt["z"], jt["p_value"]))
    print("              BV/IV=%.2f  TRV/IV=%.2f  RV/IV=%.2f (RV inflated by jumps)" % (bv / IV, trv / IV, rv / IV))

    lp0 = path(n, sd, 0, 0.0, seed=7)
    rv0 = nrc.realized_variance(lp0, "rv"); bv0 = nrc.bipower_variation(lp0); jt0 = nrc.jump_test(lp0)
    print("  NO jumps   : RV=%.3e  BV=%.3e   RV/BV=%.2f   jump_test z=%.2f p=%.4f" % (rv0, bv0, rv0 / bv0, jt0["z"], jt0["p_value"]))

    ok = (rv > 1.5 * bv                                          # RV inflated by jumps vs robust BV
          and 0.80 <= bv / IV <= 1.20 and 0.80 <= trv / IV <= 1.20   # BV, TRV recover IV
          and jt["p_value"] < 0.01                               # jumps detected
          and abs(rv0 / bv0 - 1.0) < 0.10)                       # no-jump control: RV ~ BV (no false inflation)
    print("checks: RV_inflated=%s BV~IV=%s TRV~IV=%s jumps_detected=%s control_clean=%s -> %s"
          % (rv > 1.5 * bv, 0.80 <= bv / IV <= 1.20, 0.80 <= trv / IV <= 1.20,
             jt["p_value"] < 0.01, abs(rv0 / bv0 - 1.0) < 0.10, ok))
    sys.exit(0 if ok else 1)
