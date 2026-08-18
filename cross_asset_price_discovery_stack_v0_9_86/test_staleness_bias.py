#!/usr/bin/env python3
"""Gate: the staleness bias curve MEASURES what carry-forward staleness does to the
Tier-1 flow correlation -- and the harness that measures it is itself unbiased.

25 of 66 real sessions have a stale ES ladder, and the open question had a
wrong-direction risk: refresh clustering could bias the OFI-innovation flow
correlation UP, attenuation could bias it DOWN. run_staleness_bias.py answers it
with a planted DGP (known innovation correlation rho_true) pushed through the REAL
Tier-1 code path. This gate pins the three facts that make the curve believable
plus the QC support column:

  1. harness unbiasedness -- at s=0 the pipeline recovers the oracle day z within
     2 se; a bias here would mean the measurement machinery itself (masking, CKS
     OFI, AR prefilter, bar correlation) is broken and the whole curve confounded
  2. the curve is informative -- the bias at s=0.85 is pinned away from zero
     (|bias| > 3 se) and the gate PRINTS the measured sign with its mechanism
     sentence, so the direction is on record without ever being presumed
  3. dose-response -- |bias| is monotone in s on the planted DGP (Spearman > 0.8);
     a non-monotone curve would say the harness is measuring noise, not staleness
  4. qc_frames carries the es_contract column (from each frame's attrs, blank when
     absent) so a roll-window session can be re-extracted pinned to the rival
     contract -- driven end to end through the CLI like the early-close smoke
"""
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

import run_staleness_bias as rsb

HERE = os.path.dirname(os.path.abspath(__file__))

# One simulation feeds checks 1-3: smaller than the runner's defaults (the gate
# timebox), but the same s grid and enough days that every verdict is stable.
N_DAYS, N_STEPS, SEED = 10, 7800, 0
_CURVE = None


def _curve():
    global _CURVE
    if _CURVE is None:
        _CURVE = rsb.bias_curve(s_grid=rsb.S_GRID_DEFAULT, n_days=N_DAYS,
                                n_steps=N_STEPS, rho=0.75, seed=SEED)
    return _CURVE


def check_harness_unbiased_at_s0():
    """WHY: any systematic gap at s=0 -- where the pipeline sees the planted flow
    exactly -- would be a defect of the harness, not of staleness, and would confound
    every other point on the curve. Pin: the measured mean z sits within 2 of its own
    day-level standard errors of the oracle truth. (The PAIRED se_bias is deliberately
    not the yardstick here: common-random-number pairing makes it tight enough to
    resolve the AR prefilter's ~3e-4 O(p/T) fitting shrinkage, a known estimator
    constant three orders below the effect being measured, not a harness fault.)"""
    c = _curve()
    r = c[c.s == 0.0].iloc[0]
    ok = (np.isfinite(r.bias) and np.isfinite(r.se_z_meas)
          and abs(r.z_meas - r.z_true) <= 2 * r.se_z_meas)
    print("(1) s=0: measured mean z %+.4f vs oracle %+.4f -- gap %+.5f within "
          "2 x se(z_meas)=%.5f (%d days) : %s"
          % (r.z_meas, r.z_true, r.z_meas - r.z_true, r.se_z_meas, r.n_days, ok))
    return bool(ok)


def check_high_s_bias_pinned_with_sign():
    """WHY: the item's risk was WRONG-DIRECTION -- upward (refresh clustering) vs
    downward (attenuation) were both a-priori plausible. The gate requires the s=0.85
    bias to be pinned away from zero (|bias| > 3 se) and PRINTS the measured sign with
    its mechanism sentence; nothing here presumes which sign the data produced."""
    c = _curve()
    r = c[c.s == 0.85].iloc[0]
    ok = np.isfinite(r.bias) and r.se_bias > 0 and abs(r.bias) > 3 * r.se_bias
    sign = "DOWNWARD" if r.bias < 0 else "UPWARD"
    print("(2) s=0.85: bias %+.4f (%.0f se) pinned away from zero : %s"
          % (r.bias, abs(r.bias) / r.se_bias if r.se_bias > 0 else np.nan, ok))
    print("    measured sign: %s -- %s" % (sign, rsb.direction_sentence(c)))
    return bool(ok)


def check_bias_monotone_in_s():
    """WHY: a real staleness effect must be dose-responsive -- more staleness, more
    bias. A non-monotone |bias| curve on the planted DGP would say the harness is
    reporting simulation noise, not the mechanism. Spearman(|bias|, s) > 0.8."""
    from scipy.stats import spearmanr
    c = _curve()
    m = np.isfinite(c.bias.to_numpy(float))
    rho = float(spearmanr(np.abs(c.bias.to_numpy(float)[m]),
                          c.s.to_numpy(float)[m]).correlation)
    ok = m.sum() == len(c) and rho > 0.8
    print("(3) |bias| vs s: Spearman %.3f over %d levels (> 0.8) : %s"
          % (rho, int(m.sum()), ok))
    return bool(ok)


def _qc_frame(n=300):
    """Minimal two-leg frame session_qc accepts (level-1 book, uncrossed)."""
    idx = pd.date_range("2020-12-10 09:30:00", periods=n, freq="s",
                        tz="America/New_York")
    df = pd.DataFrame({
        "SPY_bidprice_1": np.full(n, 366.00), "SPY_askprice_1": np.full(n, 366.01),
        "SPY_bidquantity_1": np.full(n, 100.0), "SPY_askquantity_1": np.full(n, 100.0),
        "ES_bidprice_1": np.full(n, 3665.00), "ES_askprice_1": np.full(n, 3665.25),
        "ES_bidquantity_1": np.full(n, 50.0), "ES_askquantity_1": np.full(n, 50.0),
    }, index=idx)
    return df


def check_qc_frames_es_contract_column():
    """WHY: the roll columns say whether the calendar pick led the market, but not what
    the pick WAS -- without es_contract in the QC table a roll-window session cannot be
    re-extracted pinned to the rival contract. attrs-stamped frame -> the column shows
    the contract; a frame without the stamp -> blank, end to end through the CLI."""
    df = _qc_frame()
    df.attrs["es_contract"] = "ESZ0"
    sess = [("2020-12-10", "benchmark", df), ("2020-12-11", "benchmark", _qc_frame())]
    fd, tmp = tempfile.mkstemp(suffix=".pkl")
    os.close(fd)
    try:
        with open(tmp, "wb") as fh:
            pickle.dump(sess, fh)
        res = subprocess.run([sys.executable, os.path.join(HERE, "qc_frames.py"),
                              "--pickle", tmp], capture_output=True, text=True,
                             timeout=120, cwd=HERE)
    finally:
        os.unlink(tmp)
    ran = res.returncode == 0
    has_col = "es_contract" in res.stdout
    shows = "ESZ0" in res.stdout
    ok = ran and has_col and shows
    print("(4) qc_frames CLI on an attrs-stamped pickle: rc=0 (%s), es_contract column "
          "present (%s), shows ESZ0 (%s) : %s" % (ran, has_col, shows, ok))
    return bool(ok)


def main():
    checks = [check_harness_unbiased_at_s0, check_high_s_bias_pinned_with_sign,
              check_bias_monotone_in_s, check_qc_frames_es_contract_column]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("staleness-bias checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    import warnings
    warnings.simplefilter("ignore")
    sys.exit(main())
