"""test_onset_surface_stage.py -- wiring guard for run_onset_surface (the launcher orchestration).

Builds deep-book sessions across a (basis-dislocation x touch-depth) grid so the three capacity axes
(SPY hollowness, ES hollowness, joint cost) and the basis state vary across events, then checks that
run_onset_surface returns the full report: a panel carrying the enriched columns, the leg-triple of
surface fits (robust + naive b3, identification verdict, jackknife) for every axis, the 1-D marginal,
and a decision-legible summary. Coefficient RECOVERY and the bounce artifact are validated elsewhere
(test_stress_surface, test_noise_robust_surface); this guards the end-to-end plumbing.
"""
import sys
import warnings
import numpy as np
import pandas as pd
import onset_response as onr

NY = "America/New_York"

def make_surface_session(basis_scale, q1, beta=0.8, T=2000, nlev=6, date="2025-01-01", seed=0):
    rng = np.random.default_rng(seed); o = T // 2
    es = np.empty(T - 1); es[:o - 1] = rng.normal(0, 2e-5, o - 1); es[o - 1:] = rng.normal(0, 1.6e-4, (T - 1) - (o - 1))
    # SPY tracks ES (beta) plus an idiosyncratic component scaled by basis_scale -> the SPY-ES basis
    # dislocation (and hence the predetermined basis state) grows with basis_scale across events.
    spy = beta * es + rng.normal(0, 2e-5, T - 1) + basis_scale * rng.normal(0, 6e-5, T - 1)
    me = np.exp(np.r_[np.log(5000.0), np.log(5000.0) + np.cumsum(es)])
    ms = np.exp(np.r_[np.log(500.0), np.log(500.0) + np.cumsum(spy)])
    idx = pd.date_range(f"{date} 09:30", periods=T, freq="s", tz=NY)
    cols = {}
    for a, mid, tk in (("SPY", ms, 0.01), ("ES", me, 0.25)):
        for i in range(1, nlev + 1):
            cols[f"{a}_bidprice_{i}"] = mid - (i - 0.5) * tk
            cols[f"{a}_askprice_{i}"] = mid + (i - 0.5) * tk
            qty = float(q1) if i == 1 else 200.0                 # touch=q1 vs fixed deep -> hollowness varies with q1
            cols[f"{a}_bidquantity_{i}"] = qty
            cols[f"{a}_askquantity_{i}"] = qty
    return pd.DataFrame(cols, index=idx), idx[o]

def main():
    warnings.simplefilter("ignore")
    basis_grid = [0.3, 1.0, 1.8, 2.6]
    q1_grid = [20.0, 60.0, 120.0, 220.0]                          # touch depth -> hollowness axis
    sessions, ev_rows, rel = [], [], {}
    k = 0
    for bi, bs in enumerate(basis_grid):
        for qi, q1 in enumerate(q1_grid):
            d = pd.Timestamp("2025-01-06") + pd.Timedelta(days=7 * k)   # distinct weekly dates
            ds = d.strftime("%Y-%m-%d")
            dfk, ots = make_surface_session(bs, q1, date=ds, seed=200 + k)
            sessions.append((pd.Timestamp(ds, tz=NY), "event", dfk))
            ev_rows.append({"date": ds, "category": "FOMC", "selection": "scheduled", "impact_blind": True, "event_et": ""})
            rel[pd.Timestamp(ds).normalize()] = dfk.index[len(dfk) // 2]
            k += 1
    evf = pd.DataFrame(ev_rows)
    rep = onr.run_onset_surface(sessions, evf, pre_steps=800, post_steps=800, notional=1e6, release_by_date=rel)

    panel = rep["panel"]
    have_cols = all(c in panel.columns for c in ["state", "state_holl", "state_holl_imp", "state_cost_joint",
                                                 "transmission", "transmission_robust"])
    # the capacity axes should actually vary across events (non-degenerate plumbing)
    vary = all(np.nanstd(panel[c].to_numpy()) > 0 for c in ["state_holl", "state_holl_imp", "state_cost_joint"])
    finite_robust = panel["transmission_robust"].notna().sum() >= 0.7 * len(panel)
    a_ok = have_cols and vary and finite_robust and len(panel) == k
    print("(A) panel: rows=%d cols_present=%s axes_vary=%s robust_finite=%d/%d -> %s"
          % (len(panel), have_cols, vary, int(panel["transmission_robust"].notna().sum()), len(panel), a_ok))

    axes = rep["axes"]
    triple_ok = all(name in axes for name in ["joint_cost", "ES_hollowness", "SPY_hollowness"])
    fields_ok = all(all(kk in axes[name] for kk in ["b3_robust", "b3_naive", "identified", "id_reasons", "n", "jackknife_robust"])
                    for name in axes)
    # the PRIMARY (joint_cost) axis should be well-conditioned on a proper plane; ES/SPY axes may be
    # NaN when their plane is unspanned/near-singular -- that is correct (the diagnostic flags it).
    b_ok = triple_ok and fields_ok and np.isfinite(axes["joint_cost"]["b3_robust"])
    print("(B) leg triple present=%s fields_ok=%s | b3_robust joint=%.4f (ES=%s SPY=%s) -> %s"
          % (triple_ok, fields_ok, axes["joint_cost"]["b3_robust"],
             ("nan" if not np.isfinite(axes["ES_hollowness"]["b3_robust"]) else "%.4f"%axes["ES_hollowness"]["b3_robust"]),
             ("nan" if not np.isfinite(axes["SPY_hollowness"]["b3_robust"]) else "%.4f"%axes["SPY_hollowness"]["b3_robust"]),
             b_ok))

    summ = rep["summary"]
    c_ok = (isinstance(summ, str) and "b3_robust" in summ and "joint_cost" in summ
            and isinstance(rep["marginal"]["slope"], float) and rep["n_events"] == k)
    print("(C) summary is decision-legible string=%s | marginal slope=%.4f n_events=%d -> %s"
          % ("b3_robust" in summ, rep["marginal"]["slope"], rep["n_events"], c_ok))

    # (D) parallel onset loop must be IDENTICAL to the serial path (fans the independent per-event
    # estimates across threads; order preserved, no shared state -> bit-identical panel).
    p_serial = onr.cross_event_onset(sessions, evf, pre_steps=800, post_steps=800, release_by_date=rel, n_jobs=1)
    p_par = onr.cross_event_onset(sessions, evf, pre_steps=800, post_steps=800, release_by_date=rel, n_jobs=4)
    num = list(p_serial.select_dtypes(include=[np.number]).columns)
    same = (p_serial.shape == p_par.shape and list(p_serial.columns) == list(p_par.columns)
            and all(np.allclose(p_serial[c].to_numpy(), p_par[c].to_numpy(), rtol=0, atol=0, equal_nan=True) for c in num)
            and all(p_serial[c].astype(str).tolist() == p_par[c].astype(str).tolist()
                    for c in p_serial.columns if c not in num))
    print("(D) parallel onset loop (n_jobs=4) == serial (n_jobs=1): %s  [%d events]" % (same, len(p_serial)))

    print("\n----- sample summary -----\n" + summ)
    ok = a_ok and b_ok and c_ok and same
    print("\nonset-surface-stage checks ->", ok)
    return ok

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
