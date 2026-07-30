"""test_rigobon_id.py -- guards rigobon_id.identification_diagnostic.

Identification through heteroskedasticity needs RELATIVE heteroskedasticity: the structural shock
variances must move by DIFFERENT factors across assets. When they move by a COMMON factor the model
is not identified (M proportional to I, coincident eigenvalues) even though the eigen-decomposition
still returns a (meaningless) answer. The diagnostic must pass the first case and flag the second.
"""
import sys
import numpy as np
import rigobon_id as rig

if __name__ == "__main__":
    g12, g21 = 0.4, 0.2                                          # true ETF<-ES, ES<-ETF
    print("Rigobon identifying-variation diagnostic (true C: ETF<-ES=%.2f, ES<-ETF=%.2f)" % (g12, g21))

    # (A) relative heteroskedasticity: asset variances scale by DIFFERENT factors -> identified
    U, reg = rig._sim_two_regime(g12, g21, sig_calm=[1.0, 1.0], sig_stress=[6.0, 1.5], seed=1)
    cov = rig.regime_residual_cov(U, reg)
    dA = rig.identification_diagnostic(cov, names=["ETF", "ES"])
    res = rig.rigobon_identify(U, reg, names=["ETF", "ES"]); C = res["contemp_rigobon"]
    print("  (A) relative het  : var_ratios=%s spread=%.2f rel_eig_gap=%.2f  identified=%s"
          % (np.round(dA["variance_ratios"], 2), dA["var_ratio_spread"], dA["rel_eig_gap"], dA["identified"]))
    print("       recovered C: ETF<-ES=%.3f  ES<-ETF=%.3f" % (C.loc["ETF", "ES"], C.loc["ES", "ETF"]))

    # (B) common-scale regimes: both variances x3 -> NOT identified
    U2, reg2 = rig._sim_two_regime(g12, g21, sig_calm=[1.0, 1.0], sig_stress=[3.0, 3.0], seed=1)
    cov2 = rig.regime_residual_cov(U2, reg2)
    dB = rig.identification_diagnostic(cov2, names=["ETF", "ES"])
    print("  (B) common scale  : var_ratios=%s spread=%.2f rel_eig_gap=%.2f  identified=%s"
          % (np.round(dB["variance_ratios"], 2), dB["var_ratio_spread"], dB["rel_eig_gap"], dB["identified"]))

    ok = (dA["identified"] is True and dA["var_ratio_spread"] > 1.0
          and abs(C.loc["ETF", "ES"] - g12) < 0.06 and abs(C.loc["ES", "ETF"] - g21) < 0.06
          and dB["identified"] is False and dB["var_ratio_spread"] < 0.15 and dB["rel_eig_gap"] < 0.10)
    print("checks: A_identified=%s A_recovers_C=%s B_flagged_unidentified=%s -> %s"
          % (dA["identified"] and dA["var_ratio_spread"] > 1.0,
             abs(C.loc["ETF", "ES"] - g12) < 0.06 and abs(C.loc["ES", "ETF"] - g21) < 0.06,
             (not dB["identified"]) and dB["var_ratio_spread"] < 0.15, ok))
    sys.exit(0 if ok else 1)
