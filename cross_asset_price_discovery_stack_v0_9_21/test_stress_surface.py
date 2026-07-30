"""test_stress_surface.py -- guards the 2-D liquidity-spiral interaction layer.

(A) interaction_identified FIRES on a diagonal-only event scatter (basis and hollowness move together
    -> states collinear, off-diagonal corners empty): identified=False.
(B) interaction_identified PASSES on a plane-spanning scatter (independent states, all corners
    populated): identified=True, corr~0, VIF~1.
(C) fit_stress_surface recovers a known interaction b3 (and the main effects) on the plane-spanning
    panel; centering leaves the slopes invariant.
(D) the leave-one-event-out jackknife band on b3 is finite and brackets the estimate (the few-events
    robustness companion that replaces IM for a regression coefficient).
(E) wiring: cross_event_onset emits a finite state_holl on a deep (multi-level) book.
"""
import sys
import numpy as np
import pandas as pd
import onset_response as onr

NY = "America/New_York"

def make_deep_session(beta, T=2500, nlev=6, date="2025-03-15", seed=0):
    rng = np.random.default_rng(seed); o = T // 2
    es = np.empty(T - 1); es[:o - 1] = rng.normal(0, 2e-5, o - 1); es[o - 1:] = rng.normal(0, 1.2e-4, (T - 1) - (o - 1))
    spy = beta * es + rng.normal(0, 2e-5, T - 1)
    me = np.exp(np.r_[np.log(5000.0), np.log(5000.0) + np.cumsum(es)])
    ms = np.exp(np.r_[np.log(500.0), np.log(500.0) + np.cumsum(spy)])
    idx = pd.date_range(f"{date} 09:30", periods=T, freq="s", tz=NY)
    cols = {}
    for a, mid, tick in (("SPY", ms, 0.01), ("ES", me, 0.25)):
        base = rng.uniform(50, 150, (T, nlev))                    # varying per-level depth -> real hollowness
        for i in range(1, nlev + 1):
            cols[f"{a}_bidprice_{i}"] = mid - (i - 0.5) * tick
            cols[f"{a}_askprice_{i}"] = mid + (i - 0.5) * tick
            cols[f"{a}_bidquantity_{i}"] = base[:, i - 1]
            cols[f"{a}_askquantity_{i}"] = base[:, i - 1]
    return pd.DataFrame(cols, index=idx), idx[o]

def _panel(basis, holl, b0=1.0, b1=0.5, b2=0.3, b3=0.4, noise=0.08, seed=0):
    rng = np.random.default_rng(seed)
    bc, hc = basis - basis.mean(), holl - holl.mean()
    trans = b0 + b1 * basis + b2 * holl + b3 * bc * hc + rng.normal(0, noise, len(basis))
    return pd.DataFrame({"date": pd.date_range("2025-01-01", periods=len(basis), freq="3D"),
                         "state": basis, "state_holl": holl, "transmission": trans,
                         "impact_blind": True, "id_strength": 1.0})

def main():
    ok = True
    B0, B1, B2, B3 = 1.0, 0.5, 0.3, 0.4

    # (A) diagonal-only -> NOT identified
    rng = np.random.default_rng(1); n = 30
    basis_d = rng.uniform(0, 5, n); holl_d = 0.8 * basis_d + rng.normal(0, 0.2, n)
    pan_d = _panel(basis_d, holl_d, B0, B1, B2, B3, seed=2)
    diag = onr.interaction_identified(pan_d)
    a_ok = (not diag["identified"]) and abs(diag["corr"]) > 0.85 and diag["min_offdiag_quadrant"] < 3
    print("(A) diagonal: corr=%.2f vif=%.1f min_offdiag=%d identified=%s reasons=%s -> %s"
          % (diag["corr"], diag["vif_interaction"], diag["min_offdiag_quadrant"], diag["identified"],
             diag["reasons"], a_ok))
    ok &= a_ok

    # (B) plane-spanning -> identified
    g = np.linspace(0.5, 5, 6); Bm, Hm = np.meshgrid(g, g)
    basis_s, holl_s = Bm.ravel(), Hm.ravel()
    pan_s = _panel(basis_s, holl_s, B0, B1, B2, B3, seed=3)
    span = onr.interaction_identified(pan_s)
    b_ok = span["identified"] and abs(span["corr"]) < 0.2 and span["vif_interaction"] < 2.0 and span["min_offdiag_quadrant"] >= 3
    print("(B) spanning: corr=%.2f vif=%.2f quads=%s identified=%s -> %s"
          % (span["corr"], span["vif_interaction"], span["quadrant_counts"], span["identified"], b_ok))
    ok &= b_ok

    # (C) fit recovers b1,b2,b3
    fit = onr.fit_stress_surface(pan_s)
    c = fit["coef"]
    c_ok = (abs(c["b1"] - B1) < 0.15 and abs(c["b2"] - B2) < 0.15 and abs(fit["interaction"] - B3) < 0.15)
    print("(C) recovered b1=%.3f(%.1f) b2=%.3f(%.1f) b3=%.3f(%.1f) p3=%.4f r2=%.3f -> %s"
          % (c["b1"], B1, c["b2"], B2, fit["interaction"], B3, fit["interaction_p"], fit["r2"], c_ok))
    ok &= c_ok

    # (D) jackknife band finite + brackets the estimate
    jk = fit["jackknife"]
    d_ok = (jk is not None and np.isfinite(jk["b3_loo_range"]) and jk["b3_loo_range"] < 0.2
            and jk["b3_loo_min"] - 1e-9 <= fit["interaction"] <= jk["b3_loo_max"] + 1e-9
            and 0.3 < 0.5 * (jk["b3_loo_min"] + jk["b3_loo_max"]) < 0.5)
    print("(D) jackknife b3 in [%.3f, %.3f] range=%.3f brackets est=%s -> %s"
          % (jk["b3_loo_min"], jk["b3_loo_max"], jk["b3_loo_range"], jk["b3_loo_min"] <= fit["interaction"] <= jk["b3_loo_max"], d_ok))
    ok &= d_ok

    # (E) wiring: state_holl finite on a deep book
    sessions, ev_rows, rel = [], [], {}
    for k, bk in enumerate([0.7, 1.1]):
        d = f"2025-0{k+3}-15"; dk, _ = make_deep_session(bk, date=d, seed=30 + k)
        sessions.append((pd.Timestamp(d, tz=NY), "event", dk))
        ev_rows.append({"date": d, "category": "FOMC", "selection": "scheduled", "impact_blind": True, "event_et": ""})
        rel[pd.Timestamp(d).normalize()] = dk.index[len(dk) // 2]
    pan_w = onr.cross_event_onset(sessions, pd.DataFrame(ev_rows), pre_steps=900, post_steps=900, release_by_date=rel)
    e_ok = "state_holl" in pan_w.columns and pan_w["state_holl"].notna().all() and len(pan_w) == 2
    print("(E) deep-book panel state_holl=%s all finite=%s -> %s"
          % (list(np.round(pan_w["state_holl"].to_numpy(), 3)), pan_w["state_holl"].notna().all(), e_ok))
    ok &= e_ok

    print("stress-surface checks ->", ok)
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
