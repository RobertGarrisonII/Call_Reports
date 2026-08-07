#!/usr/bin/env python3
"""test_audit_fixes.py -- the v0.9.64 final-audit fixes stay fixed.

The audit's verification pass confirmed defects across the estimation surface and the run
machinery; the behavioral ones are pinned here (the mean-group identification bar is pinned in
test_panel_svar check 7, the halt-mask policy in test_halt_masked_estimation, derivation
exactness in test_pull_once):

  (1)  run_table9_both_ways._load applies the halt mask on load (the headline table was the ONE
       estimator running halt-included since v0.9.57 made saved frames pre-mask), and
       --no-halt-mask passes the raw frame through
  (2)  run_contagion recognizes 'benchmark' as a calm label -- the label every real run uses --
       instead of silently falling back to sessions[0]
  (3)  panel_var_ols / select_lag_var_panel estimate from per-session Grams and match the
       materialized-design path EXACTLY (coefficients, Sigma, residuals, the full IC table)
  (4)  pds.panel_vecm (chunked Grams) matches a verbatim reconstruction of the old stacked
       estimation: alphas, regime Omegas (via the shares), day-clustered SEs, n_obs
  (5)  build_svar_frame memoizes per frame (same params -> the SAME object back; changed params
       -> a rebuild) and _dcc_corr honors its rho memo -- the bootstrap used to re-fit the DCC
       once per session per draw
  (6)  mask_frame returns the ORIGINAL frame (no copy) when no halt applies, and a masked COPY
       when one does
  (7)  derive_coarse_frame CEIL-anchors the coarse grid: a fine frame whose first rows were
       trimmed off-grid derives an on-grid coarse frame with real (not all-NaN) flow
  (8)  a --verify-existing MISMATCH rolls back the derived caches written in the same run
       (exit 3 used to leave them for later runs to cache-HIT)
  (9)  a dead worker pool (TerminatedWorkerError) no longer discards the batch: completed
       results are kept and the remainder finishes sequentially
  (10) driver text invariants: STAGE 6 does not forward the SVAR-frame lag to the VECM stages;
       the fine extractions pass --no-dataset; STAGE 6b passes --no-save-frames; the STAGE 3
       failure-branch sed cannot kill the script; rc=3 from derive_frames is handled distinctly
  (11) the jump split carries kappa/ec_valid per day (non-correcting days are flagged, and
       run_jumps' IS means skip them)
  (12) table_rigobon uses the exogenous session labels when two exist (with the identification
       verdict rows) and labels the endogenous fallback as such

Run: python test_audit_fixes.py
"""
from __future__ import annotations

import os
import pickle
import subprocess as sp
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TZ = "America/New_York"


# ── (1) Table 9 loader masks on load ──────────────────────────────────────────
def check_table9_loader_masks():
    import run_table9_both_ways as t9
    idx = pd.date_range("2020-03-09 09:30", periods=600, freq="s", tz=TZ)
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"SPY_mid": 500 + rng.normal(0, .05, 600).cumsum(),
                       "ES_mid": 5000 + rng.normal(0, .5, 600).cumsum()}, index=idx)
    h0, h1 = idx[100], idx[200]
    df.attrs["halt_windows_SPY"] = [(h0, h1)]
    df.attrs["halt_windows_ES"] = [(h0, h1)]
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "s.pkl")
        with open(p, "wb") as fh:
            pickle.dump([("2020-03-09", "volatile", df)], fh)
        ns = t9.build_parser().parse_args(["--source", "load", "--pickle", p])
        m = t9._load(ns)[0][2]
        masked = m.loc[h0:h1, "SPY_mid"].isna().all() and \
            m.drop(m.loc[h0:h1].index)["SPY_mid"].notna().all()
        ns2 = t9.build_parser().parse_args(["--source", "load", "--pickle", p, "--no-halt-mask"])
        raw = t9._load(ns2)[0][2]
        passthrough = raw["SPY_mid"].notna().all()
    ok = masked and passthrough
    print("(1) Table 9 loader: halt rows NaN'd on load (%s); --no-halt-mask passes the raw "
          "frame through (%s) : %s" % (masked, passthrough, ok))
    return ok


# ── (2) run_contagion knows 'benchmark' ───────────────────────────────────────
def check_contagion_benchmark_label():
    import run_contagion as rc
    df_v = pd.DataFrame({"x": [1.0]})
    df_b = pd.DataFrame({"x": [2.0]})
    sessions = [("d1", "volatile", df_v), ("d2", "benchmark", df_b)]
    in_set = "benchmark" in rc._CALM
    picked = rc._pick_regime(sessions, rc._CALM) is df_b     # pre-fix: sessions[0] fallback = df_v
    ok = in_set and picked
    print("(2) 'benchmark' in _CALM (%s) and _pick_regime returns the benchmark session, not "
          "the sessions[0] fallback (%s) : %s" % (in_set, picked, ok))
    return ok


# ── (3) Gram-based panel OLS / lag selection are exact ────────────────────────
def check_gram_exactness():
    import correlation_svar as cs
    import price_discovery_shares as pds
    rng = np.random.default_rng(7)
    Xs = [rng.normal(size=(300, 4)) + rng.normal(size=(1, 4)) * 3 for _ in range(3)]
    ok = True
    for fe in (True, False):
        for p in (0, 1, 3):
            Y, Z, groups = cs._panel_var_design(Xs, p, fe=fe)
            if Z.shape[1]:
                B_ref, r_ref = pds._ols(Y, Z)
            else:
                B_ref, r_ref = np.zeros((0, Y.shape[1])), Y
            k = Y.shape[1]
            dof = max(r_ref.shape[0] - Z.shape[1] - (len(Xs) * k if fe else 0) // max(k, 1), 1)
            off = 0 if fe else 1
            A_ref = [B_ref[off + (i - 1) * k: off + i * k, :].T for i in range(1, p + 1)]
            A, Sig, resid, grp = cs.panel_var_ols(Xs, p, fe=fe, return_resid=True, chunk=64)
            ok &= all(np.allclose(a, b, atol=1e-9) for a, b in zip(A, A_ref))
            ok &= np.allclose(Sig, r_ref.T @ r_ref / dof, atol=1e-9)
            ok &= np.allclose(resid, r_ref, atol=1e-9) and np.array_equal(grp, groups)
    # IC table: candidate p scored from the leading-column Gram submatrix == full rebuild
    for fe in (True, False):
        _p_new, tab_new = cs.select_lag_var_panel(Xs, pmax=6, fe=fe, chunk=64)
        rows = []
        for p in range(0, 7):
            Y, Z, _ = cs._panel_var_design(Xs, p, fe=fe, start=6)
            resid = pds._ols(Y, Z)[1] if Z.shape[1] else Y
            n, k = resid.shape
            Sig = resid.T @ resid / n
            _s, ld = np.linalg.slogdet(Sig + np.eye(k) * 1e-12)
            K = Z.shape[1] * k
            rows.append((ld, ld + 2 * K / n, ld + np.log(n) * K / n,
                         ld + 2 * np.log(np.log(max(n, 3))) * K / n))
        ok &= np.allclose(tab_new.to_numpy(), np.array(rows), atol=1e-8, equal_nan=True)
    print("(3) Gram-accumulated panel_var_ols/select_lag_var_panel match the materialized "
          "design exactly (fe x p grid + full IC table) : %s" % ok)
    return ok


# ── (4) panel_vecm Grams == the old stacked estimation ────────────────────────
def _ecm_pair(n, seed, halt=False):
    r = np.random.default_rng(seed)
    f = np.cumsum(r.normal(0, 4e-4, n))
    p2 = f + r.normal(0, 1e-4, n)
    p1 = np.empty(n); p1[0] = f[0] + 5e-4
    for t in range(1, n):
        p1[t] = p1[t - 1] + 0.08 * (p2[t - 1] - p1[t - 1]) + r.normal(0, 2e-4)
    m1, m2 = 500 * np.exp(p1), 5000 * np.exp(p2)
    if halt:
        m1[300:420] = np.nan; m2[310:420] = np.nan
    return m1, m2


def check_panel_vecm_gram():
    import price_discovery_shares as pds
    sessions = [("d1", "benchmark", *_ecm_pair(2000, 1)),
                ("d2", "volatile", *_ecm_pair(2000, 2, halt=True)),
                ("d3", "volatile", *_ecm_pair(2000, 3)),
                ("d4", "benchmark", *_ecm_pair(2000, 4))]
    n_lags = 5
    # verbatim reconstruction of the pre-v0.9.64 stacked estimation
    dYs, ECs, Ds, Ls, Gs = [], [], [], [], []
    for gi, (_l, regime, m_spy, m_es) in enumerate(sessions):
        p1 = np.log(np.asarray(m_spy, float)); p2 = np.log(np.asarray(m_es, float))
        z = (p1 - p2); z = z - np.nanmean(z)
        dp1 = np.diff(p1); dp2 = np.diff(p2); T = len(dp1)
        ECs.append(z[n_lags:T])
        Ds.append(np.full(T - n_lags, 1.0 if regime == "volatile" else 0.0))
        lag = [np.ones(T - n_lags)]
        for L in range(1, n_lags + 1):
            lag.append(dp1[n_lags - L:T - L]); lag.append(dp2[n_lags - L:T - L])
        dYi = np.column_stack([dp1[n_lags:T], dp2[n_lags:T]]); Li = np.column_stack(lag)
        oki = (np.all(np.isfinite(dYi), axis=1) & np.isfinite(z[n_lags:T])
               & np.all(np.isfinite(Li), axis=1))
        dYs.append(dYi[oki]); ECs[-1] = ECs[-1][oki]; Ds[-1] = Ds[-1][oki]; Ls.append(Li[oki])
        Gs.append(np.full(int(oki.sum()), gi))
    dY = np.vstack(dYs); ec = np.concatenate(ECs); D = np.concatenate(Ds)
    X = np.column_stack([ec, ec * D, np.vstack(Ls)])
    B, resid = pds._ols(dY, X)
    XtXinv = np.linalg.inv(X.T @ X)
    se_ref = [float(pds._cluster_se(X, resid[:, k], np.concatenate(Gs), XtXinv)[0][1])
              for k in range(2)]
    lo, hi, mid_b = pds.hasbrouck_is(B[0], np.cov(resid[D == 0], rowvar=False, bias=True))
    lo, hi, mid_v = pds.hasbrouck_is(B[0] + B[1], np.cov(resid[D == 1], rowvar=False, bias=True))

    new = pds.panel_vecm(sessions, n_lags=n_lags)
    ok = (np.allclose(new["alpha_benchmark"], B[0], atol=1e-10)
          and np.allclose(new["alpha_interaction"], B[1], atol=1e-10)
          and np.allclose([new["benchmark"]["IS_mid_SPY"], new["benchmark"]["IS_mid_ES"]],
                          mid_b, atol=1e-9)
          and np.allclose([new["volatile"]["IS_mid_SPY"], new["volatile"]["IS_mid_ES"]],
                          mid_v, atol=1e-9)
          and np.allclose([new["se_interaction_spy"], new["se_interaction_es"]], se_ref,
                          rtol=1e-8)
          and new["n_obs"] == dY.shape[0])
    print("(4) panel_vecm (chunked Grams, halt-masked session included) == the stacked "
          "estimation: alphas, both regime IS, day-clustered SEs, n_obs : %s" % ok)
    return ok


# ── (5) frame memo + DCC rho memo ─────────────────────────────────────────────
def check_svar_frame_memo():
    import correlation_svar as cs
    from test_panel_svar import _book_frame
    df = _book_frame(seed=5)
    a = cs.build_svar_frame(df, "standard", 3, None, "rolling", 40, "cost_to_fill")
    b = cs.build_svar_frame(df, "standard", 3, None, "rolling", 40, "cost_to_fill")
    hit = a is b                                        # identical params -> the cached object
    c = cs.build_svar_frame(df, "standard", 3, None, "rolling", 60, "cost_to_fill")
    miss = c is not a                                   # changed window -> rebuild
    df2 = _book_frame(seed=6)
    sentinel = np.full(len(df2), 0.123)
    # v0.9.65 contract: the rho memo is (data_fingerprint, rho) -- a stale or foreign
    # fingerprint must be IGNORED, a matching one served without a fit.
    df2.attrs["_dcc_rho_memo"] = (cs._frame_fingerprint(df2), sentinel)
    memo_used = cs._dcc_corr(df2, None, None) is sentinel
    df2.attrs["_dcc_rho_memo"] = (("stale",), sentinel)
    stale_ignored = cs._dcc_corr(df2, None, None) is not sentinel
    ok = hit and miss and memo_used and stale_ignored
    print("(5) build_svar_frame memo: same params -> same object (%s), changed params -> "
          "rebuild (%s); _dcc_corr serves a fingerprint-matched rho memo (%s) and ignores a "
          "stale one (%s) : %s" % (hit, miss, memo_used, stale_ignored, ok))
    return ok


# ── (6) mask_frame no-copy fast path ──────────────────────────────────────────
def check_mask_frame_no_copy():
    import market_halts as mh
    idx = pd.date_range("2024-07-24 09:30", periods=300, freq="s", tz=TZ)
    quiet = pd.DataFrame({"SPY_mid": np.ones(300), "ES_mid": np.ones(300)}, index=idx)
    out_q, rep_q = mh.mask_frame(quiet)
    same = out_q is quiet and sum(rep_q.values()) == 0
    halted = pd.DataFrame({"SPY_mid": np.ones(300), "ES_mid": np.ones(300)}, index=idx)
    halted.attrs["halt_windows_SPY"] = [(idx[50], idx[80])]
    halted.attrs["halt_windows_ES"] = [(idx[50], idx[80])]
    out_h, rep_h = mh.mask_frame(halted)
    copied = out_h is not halted and out_h["SPY_mid"].isna().sum() == 31 \
        and halted["SPY_mid"].notna().all()
    ok = same and copied
    print("(6) mask_frame: no-halt session returns the ORIGINAL frame, no copy (%s); halted "
          "session returns a masked COPY leaving the original intact (%s) : %s"
          % (same, copied, ok))
    return ok


# ── (7) ceil-anchored coarse grid on a trimmed fine frame ─────────────────────
def check_derive_ceil_anchor():
    import derive_frames as dv
    from test_pull_once import _event_stream, _frame_at
    t0, n_secs, quotes, trades = _event_stream()
    fine = _frame_at("10ms", t0, n_secs, quotes, trades)
    trimmed = fine.iloc[37:]                            # starts at 09:30:00.370 -- OFF the 1s grid
    derived = dv.derive_coarse_frame(trimmed, "1s", "10ms")
    on_grid = derived.index[0] == t0 + pd.Timedelta(seconds=1)
    aligned = (derived.index.view("int64") % 1_000_000_000 == 0).all()
    flow_cols = [c for c in derived.columns if c.endswith(dv.SUM_SUFFIXES)]
    flow_live = any(np.isfinite(pd.to_numeric(derived[c], errors="coerce")).any()
                    for c in flow_cols)
    direct = _frame_at("1s", t0, n_secs, quotes, trades)
    body = derived.index[:-1]
    book_ok = True
    for c in ("SPY_bidprice_1", "ES_askquantity_1"):
        a = derived.loc[body, c].to_numpy(float); b = direct.loc[body, c].to_numpy(float)
        book_ok &= np.allclose(a, b, atol=1e-9, equal_nan=True)
    flow_ok = True
    for c in flow_cols:
        a = pd.to_numeric(derived.loc[body, c], errors="coerce").to_numpy(float)
        b = pd.to_numeric(direct.loc[body, c], errors="coerce").to_numpy(float)
        flow_ok &= np.allclose(a, b, atol=1e-9, equal_nan=True)
    ok = on_grid and aligned and flow_live and book_ok and flow_ok
    print("(7) trimmed fine frame: coarse grid CEIL-anchored on-grid (%s/%s), flow columns "
          "live (%s), book rows (%s) and full-bin flow sums (%s) match a direct coarse "
          "extraction : %s" % (on_grid, aligned, flow_live, book_ok, flow_ok, ok))
    return ok


# ── (8) mismatch rolls back this run's derived caches ─────────────────────────
def check_mismatch_rollback():
    import derive_frames as dv
    import mstbook_loader as ml
    from test_pull_once import _event_stream, _frame_at
    ok = True
    with tempfile.TemporaryDirectory() as td:
        cache = os.path.join(td, "cache"); os.makedirs(cache)
        recs = []
        for day, seed in (("2024-07-24", 7), ("2024-07-25", 8)):
            t0, n_secs, quotes, trades = _event_stream(day=day, seed=seed)
            recs.append((day, "benchmark", _frame_at("10ms", t0, n_secs, quotes, trades),
                         _frame_at("1s", t0, n_secs, quotes, trades)))
        # session A: a CORRUPTED direct cache (book value perturbed) -> verified mismatch.
        # session B: no direct cache -> the derivation caches it, then must roll it back.
        bad_direct = recs[0][3].copy()
        bad_direct.iloc[10, bad_direct.columns.get_loc("SPY_bidprice_1")] += 1.0
        pa = ml._cache_path(cache, recs[0][0], "1s", 10, "receipt", "aggregated")
        bad_direct.to_pickle(pa)
        pb = ml._cache_path(cache, recs[1][0], "1s", 10, "receipt", "aggregated")
        fine_pkl = os.path.join(td, "fine.pkl")
        with open(fine_pkl, "wb") as fh:
            pickle.dump([(d, r, f) for d, r, f, _ in recs], fh)
        rc = dv.main(["--pickle", fine_pkl, "--target", "1s", "--fine-interval", "10ms",
                      "--cache-dir", cache, "--levels", "10", "--clock", "receipt",
                      "--verify-existing"])
        ok &= rc == 3
        ok &= os.path.exists(pa)                        # the direct cache is KEPT
        rolled = not os.path.exists(pb)                 # the derived write is ROLLED BACK
        ok &= rolled
    print("(8) verified mismatch: exit 3 (%s), direct cache kept (%s), this run's derived "
          "cache write rolled back (%s) : %s"
          % (rc == 3, os.path.basename(pa) is not None, rolled, ok))
    return ok


# ── (9) dead worker pool: keep completed, finish sequentially ─────────────────
def check_worker_pool_containment():
    import types
    import mstbook_loader as ml
    TWE = type("TerminatedWorkerError", (RuntimeError,), {})

    class _FakeParallel:
        def __init__(self, *a, **k):
            pass

        def __call__(self, gen):
            it = iter(gen)
            f, a, k = next(it)
            yield f(*a, **k)                            # one result completes...
            raise TWE("worker killed (simulated OOM)")  # ...then the pool dies

    fake = types.ModuleType("joblib")
    fake.Parallel = _FakeParallel
    fake.delayed = lambda f: (lambda *a, **k: (f, a, k))
    saved = sys.modules.get("joblib")
    sys.modules["joblib"] = fake
    try:
        done = []
        res = ml._parallel_sessions(lambda x: x * 10, [1, 2, 3, 4], n_jobs=2,
                                    backend="process", on_done=done.append)
    finally:
        if saved is not None:
            sys.modules["joblib"] = saved
        else:
            del sys.modules["joblib"]
    ok = res == [10, 20, 30, 40] and sorted(done) == [10, 20, 30, 40]
    print("(9) TerminatedWorkerError mid-batch: all 4 results present in order (%s), "
          "completion callbacks fired for every item (%s) : %s"
          % (res == [10, 20, 30, 40], sorted(done) == [10, 20, 30, 40], ok))
    return ok


# ── (10) driver text invariants ───────────────────────────────────────────────
def check_driver_invariants():
    with open(os.path.join(HERE, "run_paper_replication.sh")) as fh:
        text = fh.read()
    stage6 = text.split('say "STAGE 6  remaining')[1].split('say "STAGE 6b')[0]
    no_forward = "N_LAGS_INT" not in stage6.replace("# N_LAGS_INT is NOT forwarded", "")
    fine_no_ds = text.count("--no-dataset --only extract") >= 2
    s6b = text.split('say "STAGE 6b')[1].split('STAGE 7')[0]
    s6b_lean = "--no-save-frames" in s6b and "--no-dataset" in s6b
    sed_guarded = "qc_frames.txt\" 2>/dev/null | tr '\\n' ' ' || true" in text
    rc3 = "DERIVE_RC" in text and "rolled back" in text.lower()
    ok = no_forward and fine_no_ds and s6b_lean and sed_guarded and rc3
    print("(10) driver: STAGE 6 does not forward the SVAR lag (%s); fine extracts pass "
          "--no-dataset (%s); STAGE 6b is load-only lean (%s); STAGE 3 sed guarded (%s); "
          "derive rc=3 handled distinctly (%s) : %s"
          % (no_forward, fine_no_ds, s6b_lean, sed_guarded, rc3, ok))
    return ok


# ── (11) jump split carries kappa/ec_valid ────────────────────────────────────
def check_jump_split_ec_valid():
    import jump_robust as jr
    m1, m2 = _ecm_pair(6000, 11)
    r = jr.continuous_jump_information_shares(m1, m2, n_lags=5)
    has = "kappa" in r and "ec_valid" in r
    consistent = has and (r["ec_valid"] == (r["kappa"] > 0))
    ok = has and consistent
    print("(11) continuous_jump_information_shares returns kappa/ec_valid (%s), and the flag "
          "is kappa>0 (%s) : %s" % (has, consistent, ok))
    return ok


# ── (12) table_rigobon regimes are the session labels ─────────────────────────
def check_rigobon_exogenous_regimes():
    import paper_tables as pt
    from test_panel_svar import _book_frame
    two = [("d1", "benchmark", _book_frame(seed=5, n=2500)),
           ("d2", "volatile", _book_frame(seed=6, n=2500))]
    tbl2 = pt.table_rigobon(two)
    exo = "exogenous" in tbl2.notes and "session labels" in tbl2.notes
    verdict_rows = any("identified" in str(i) for i in tbl2.df.index)
    one = [("d1", "benchmark", _book_frame(seed=5, n=2500)),
           ("d2", "benchmark", _book_frame(seed=6, n=2500))]
    tbl1 = pt.table_rigobon(one)
    endo = "ENDOGENOUS" in tbl1.notes
    ok = exo and verdict_rows and endo
    print("(12) table_rigobon: two-label panel uses exogenous session labels (%s) with "
          "verdict rows (%s); single-label fallback is labeled ENDOGENOUS (%s) : %s"
          % (exo, verdict_rows, endo, ok))
    return ok


# ── (13) bar coverage floors exclude masked bars ──────────────────────────────
def check_bar_coverage_floors():
    import correlation_svar as cs
    from test_panel_svar import _book_frame
    full = _book_frame(seed=5, n=1200)
    Xf, _nf, _cf = cs.build_svar_frame(full, "standard", 3, None, "bar", 100,
                                       "cost_to_fill", bar_seconds=60)
    masked = _book_frame(seed=5, n=1200)
    # NaN 50 of bar #1's 60 sub-rows across ALL columns: 10 finite < min_sub=15 (0.25*60)
    # for rho, and < 30 (0.5*60) for the flow sums -- the bar must fall out of the design.
    for c in masked.columns:
        masked.loc[masked.index[60:110], c] = np.nan
    Xm, _nm, _cm = cs.build_svar_frame(masked, "standard", 3, None, "bar", 100,
                                       "cost_to_fill", bar_seconds=60)
    dropped = len(Xf) - len(Xm)
    ok = len(Xf) > 0 and 1 <= dropped <= 3       # the masked bar + diff-chain neighbours
    print("(13) bar coverage floors: a bar with 10/60 finite sub-rows is excluded from the "
          "design (%d row(s) dropped vs the unmasked build of %d) : %s"
          % (dropped, len(Xf), ok))
    return ok


# ── (14) MWCB DiD wild-cluster p-values are finite on a well-formed panel ─────
def check_mwcb_wcb_p_finite():
    import paper_tables as pt
    rel = pd.Timestamp("2020-03-09 09:34:00", tz=TZ)
    treated, control, release_by_date = [], [], {}
    for i, day in enumerate(("2020-03-09", "2020-03-12", "2020-03-16", "2020-03-18")):
        r = pd.Timestamp(f"{day} 09:34:00", tz=TZ)
        treated.append((day, pt._synth_release_book(day, r, True, 100 + i)))
        release_by_date[day] = r
    for i, day in enumerate(("2020-02-10", "2020-02-11", "2020-02-12", "2020-02-13")):
        r = pd.Timestamp(f"{day} 09:34:00", tz=TZ)
        control.append((day, pt._synth_release_book(day, r, False, 200 + i)))
        release_by_date[day] = r
    tbl = pt.table_mwcb_did(treated, control, release_by_date)
    col = tbl.df["p (wild cluster)"]
    n_finite = sum(1 for v in col if str(v) not in ("--", "nan"))
    ok = n_finite >= 2                            # DiD and RD rows must both carry real p's
    print("(14) MWCB DiD: wild-cluster bootstrap p-values are FINITE on a well-formed panel "
          "(%d of %d rows; the _wcb except-branch must not silently eat them) : %s"
          % (n_finite, len(col), ok))
    return ok


# ── (15) per-day level estimators: longest contiguous unmasked run ────────────
def check_longest_run_and_per_day_tables():
    import paper_tables as pt
    from test_panel_svar import _book_frame
    Y = np.ones((1000, 2))
    Y[300:340] = np.nan                           # two runs: 300 rows, then 660 rows
    run = pt._longest_finite_run(Y)
    picks_longer = len(run) == 660
    d1 = _book_frame(seed=5, n=2500, day="2024-07-24")
    d2 = _book_frame(seed=6, n=2500, day="2024-07-25")
    # a level shift between days: per-day estimation must not see it as a return
    for c in d2.columns:
        if "price" in c:
            d2[c] = d2[c] * 1.10
    sessions = [("2024-07-24", "benchmark", d1), ("2024-07-25", "volatile", d2)]
    t_tick = pt.table_tick_correction(sessions)
    tick_ok = not t_tick.df.empty and "PER DAY" in t_tick.notes.upper()
    t_reg = pt.table_regime_is(sessions, p=1, max_iter=30)
    reg_ok = not t_reg.df.empty and "n_days" in t_reg.df.columns
    ok = picks_longer and tick_ok and reg_ok
    print("(15) per-day level estimators: _longest_finite_run picks the longer post-gap "
          "segment (%s); tick-correction (%s) and MS-VECM regime (%s) tables build per-day "
          "across a 10%% overnight level shift : %s" % (picks_longer, tick_ok, reg_ok, ok))
    return ok


def main():
    checks = [check_table9_loader_masks, check_contagion_benchmark_label,
              check_gram_exactness, check_panel_vecm_gram, check_svar_frame_memo,
              check_mask_frame_no_copy, check_derive_ceil_anchor, check_mismatch_rollback,
              check_worker_pool_containment, check_driver_invariants,
              check_jump_split_ec_valid, check_rigobon_exogenous_regimes,
              check_bar_coverage_floors, check_mwcb_wcb_p_finite,
              check_longest_run_and_per_day_tables]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback
            traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("audit-fix checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
