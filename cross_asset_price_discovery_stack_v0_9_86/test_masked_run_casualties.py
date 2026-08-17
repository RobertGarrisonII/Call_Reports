#!/usr/bin/env python3
"""test_masked_run_casualties.py -- what the first halt-masked run broke, pinned.

The 2026-08-05 masked run was the first time every estimator received NaN-masked frames, and the
periphery failed in four ways the headline tables could not show:

  (1) `local_projection` came back ALL-NaN at every horizon: np.cumsum carried the first masked
      return through every later cumulative response, so one 09:34 halt emptied the whole day.
  (2) `cojump_from_mids` reported ZERO jumps on a crash day: the Lee-Mykland local-vol cumsum
      carried the same NaN through every window, sigma_hat went NaN, and |L| > thr was False
      everywhere -- while the per-day split (gap-free VECM residuals) found 24 jumps.
  (3) the legacy inference row printed coef = NaN and both t's = NaN while its wild-bootstrap p
      survived -- and the comovement table silently DROPPED the four MWCB days (NaN std fails the
      inclusion test without a trace).
  (4) the Rigobon identification flipped sign against every other estimator with nothing in the
      output saying whether it was identified at all -- het-ID is meaningless when the regimes
      differ only in scale, and the strength diagnostic existed but was never surfaced.

Also pinned here, because the fix cuts across runs: frames saved by v0.9.51-56 were MASKED AT
REST (the save happened after _mask_halt_rows), which silently turned --no-halt-mask into a no-op
on reloaded frames and made the halt-mask A/B impossible from a run's own artifact. v0.9.57 saves
pre-mask frames again, and `ab_halt_mask.py` refuses masked-at-rest input rather than printing an
A/A and calling it an A/B.

Run: python test_masked_run_casualties.py
"""
from __future__ import annotations

import os
import pickle
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd

import cross_impact as cimp
import irf as irfm
import jump_robust as jr
import legacy_tables as legacy
import market_halts as mh

TZ = "America/New_York"


def _book_session(day="2020-03-09", n=7000, seed=3, halt=("10:00:00", "10:14:59"),
                  jump_at=None, jump_bps=0.0, eps=None):
    """Level-1 frame with book + quantities; optionally a halt window and an injected co-jump."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(f"{day} 09:30:00", periods=n, freq="s", tz=TZ)
    if eps is None:
        eff = np.cumsum(rng.normal(0, 2e-5, n))
        r_spy = np.diff(np.concatenate([[0.0], eff])) + rng.normal(0, 4e-6, n)
        r_es = np.diff(np.concatenate([[0.0], eff])) + rng.normal(0, 4e-6, n)
    else:
        r_spy, r_es = eps
    if jump_at is not None:
        r_spy[jump_at] += jump_bps * 1e-4
        r_es[jump_at] += jump_bps * 1e-4
    spy = 280.0 * np.exp(np.cumsum(r_spy))
    es = 2800.0 * np.exp(np.cumsum(r_es))
    df = pd.DataFrame({"SPY_bidprice_1": spy - 0.005, "SPY_askprice_1": spy + 0.005,
                       "SPY_bidquantity_1": rng.integers(50, 150, n).astype(float),
                       "SPY_askquantity_1": rng.integers(50, 150, n).astype(float),
                       "ES_bidprice_1": es - 0.125, "ES_askprice_1": es + 0.125,
                       "ES_bidquantity_1": rng.integers(20, 80, n).astype(float),
                       "ES_askquantity_1": rng.integers(20, 80, n).astype(float)}, index=idx)
    if halt is not None:
        win = [(f"{day}T{halt[0]}-04:00", f"{day}T{halt[1]}-04:00")]
        df.attrs["halt_windows_SPY"] = win
        df.attrs["halt_windows_ES"] = win
    else:
        df.attrs["halt_windows_SPY"] = []
        df.attrs["halt_windows_ES"] = []
    return df


def _mid(df, a):
    return (df[f"{a}_bidprice_1"].to_numpy() + df[f"{a}_askprice_1"].to_numpy()) / 2.0


def check_local_projection_survives_the_mask():
    df = _book_session()
    masked, rep = mh.mask_frame(df)
    n_halt = rep.get("SPY", 0)
    lp = irfm.local_projection_irf(masked, impulse="ES", response="SPY", state=None,
                                   horizons=range(0, 6), n_lags=5, n_levels=1, n_boot=20)
    finite = np.isfinite(lp["theta"].to_numpy()).all()
    ok = n_halt > 100 and finite
    print("(1) local_projection on a masked day (%d halt rows): theta finite at every horizon "
          "(%s) -- was all-NaN via cumsum poisoning : %s" % (n_halt, finite, ok))
    return ok


def check_cojumps_detected_and_seam_not_a_jump():
    df = _book_session(jump_at=5000, jump_bps=50.0)
    masked, _ = mh.mask_frame(df)
    ms, me = _mid(masked, "SPY"), _mid(masked, "ES")
    out = jr.cojump_from_mids(ms, me, max_lag=2)
    found = out["n_cojump"] >= 1 and out["simultaneous"] >= 1
    # the reopen boundary: the first finite return after the NaN block is at most suspect;
    # the seam itself is a NaN return and can never be flagged
    r = np.diff(np.log(ms))
    lm = jr.lee_mykland(r)
    nan_idx = set(np.where(~np.isfinite(r))[0])
    seam_clean = not (set(map(int, lm["jump_idx"])) & nan_idx)
    # false-positive guard: same frame without the injected jump stays quiet-ish
    df0 = _book_session()
    m0, _ = mh.mask_frame(df0)
    base = jr.cojump_from_mids(_mid(m0, "SPY"), _mid(m0, "ES"), max_lag=2)
    quiet = base["n_cojump"] <= 3
    ok = found and seam_clean and quiet
    print("(2) cojump on a masked crash day: injected 50bp co-jump found (%s, n=%d, simul=%d); "
          "no jump flagged on a NaN return (%s); clean day stays quiet (n=%d) : %s"
          % (found, out["n_cojump"], out["simultaneous"], seam_clean, base["n_cojump"], ok))
    return ok


def check_legacy_inference_and_comovement():
    # The masked MWCB day gets INDEPENDENT legs (corr ~0) while the clean day is common-factor
    # (corr ~0.96). The old comovement code silently dropped the masked day (NaN std fails the
    # inclusion test) and would report the clean day's ~0.96 alone -- the discriminating
    # assertion is therefore that the mean sits near the AVERAGE, which requires the masked day
    # to be inside. (The review workflow proved the previous version of this check passed
    # against the pre-fix code; this version does not.)
    rng = np.random.default_rng(31)
    n = 7000
    indep = (rng.normal(0, 2e-5, n), rng.normal(0, 2e-5, n))
    sessions = [("2020-03-09", "volatile", mh.mask_frame(_book_session(eps=indep))[0]),
                ("2024-07-24", "benchmark", _book_session(day="2024-07-24", seed=11, halt=None))]
    tab = legacy.inference_table(sessions, n_levels=1, quick=True)
    row = tab.iloc[0]
    fin = (np.isfinite(row["coef"]) and np.isfinite(row["legacy t (iid, pooled)"])
           and np.isfinite(row["upgraded t (day-cluster)"]))
    # the masked day contributes its non-halt rows: n_obs must sit strictly between one clean
    # session and two, and the day count must still be 2
    nobs = int(row["n_obs"])
    partial = (6999 < nobs < 2 * 6999) and int(row["n_days"]) == 2
    com = legacy.comovement_table(sessions)
    pear = float(com.iloc[0, 0]); hy = float(com.iloc[0, 1])
    both_in = np.isfinite(pear) and np.isfinite(hy)
    # near the two-day average (~0.48), NOT the clean day alone (~0.96): the masked day is in.
    # And segment-wise HY tracks Pearson instead of being pushed toward 1 by the reopen gap.
    included = pear < 0.75 and hy < 0.75
    sane = abs(pear - hy) < 0.25
    ok = fin and partial and both_in and included and sane
    print("(3) legacy on masked frames: inference coef/t finite (%s) with n_obs=%d over 2 days "
          "(%s); comovement mean Pearson=%.3f HY=%.3f near the 2-day average -- the masked day "
          "is INCLUDED (%s), no gap domination (%s) : %s"
          % (fin, nobs, partial, pear, hy, included, sane, ok))
    return ok


def _structural_sessions(rel_het=True, n=6000, per_regime=2, mask_het_day=False):
    """Frames whose returns follow A y = eps with known A, regime-specific shock variances.
    Order (ES, SPY) to match identification_table. C = I - A: C[SPY,ES]=0.45, C[ES,SPY]=0.15.
    With ``mask_het_day`` the FIRST volatile session carries the halt and is masked -- and it is
    the ONLY session carrying the relative heteroskedasticity, so a code path that drops the
    whole day (the plain-.mean() demeaning bug) collapses the identifying variation."""
    g_spy_es, g_es_spy = 0.45, 0.15
    A = np.array([[1.0, -g_es_spy], [-g_spy_es, 1.0]])     # rows/cols: [ES, SPY]
    P = np.linalg.inv(A)
    out = []
    k = 0
    for reg, scales in (("benchmark", [[2e-5, 2e-5]] * per_regime),
                        ("volatile", ([[6e-5, 3e-5]] + [[2.2e-5, 2.2e-5]] * (per_regime - 1))
                         if (rel_het and mask_het_day) else
                         [[6e-5, 3e-5] if rel_het else [6e-5, 6e-5]] * per_regime)):
        for j, scale in enumerate(scales):
            rng = np.random.default_rng(100 + k); k += 1
            eps = rng.normal(0, 1, (n, 2)) * np.asarray(scale)
            y = eps @ P.T                                   # columns [r_ES, r_SPY]
            day = "2024-0%d-1%d" % ((k % 8) + 1, k % 7)
            hlt = ("10:00:00", "10:14:59") if (mask_het_day and reg == "volatile" and j == 0) else None
            df = _book_session(day="2024-07-24", seed=200 + k, halt=hlt,
                               eps=(y[:, 1], y[:, 0]), n=n)
            if hlt is not None:
                df, _ = mh.mask_frame(df)
            out.append((day, reg, df))
    return out, g_spy_es, g_es_spy


def check_rigobon_strength_is_in_the_table():
    sessions, g_spy_es, g_es_spy = _structural_sessions(rel_het=True)
    tab = legacy.identification_table(sessions)
    has_rows = all(any(key in str(i) for i in tab.index)
                   for key in ("var-ratio spread", "relative eigenvalue gap", "identified"))
    rg = tab["upgraded (Rigobon het-ID)"]
    est_se = float(rg.iloc[0]); est_es = float(rg.iloc[1])
    recovered = abs(est_se - g_spy_es) < 0.08 and abs(est_es - g_es_spy) < 0.08
    yes = str(rg.iloc[-1]).strip().startswith("yes")
    weak_sessions, _, _ = _structural_sessions(rel_het=False)
    tab2 = legacy.identification_table(weak_sessions)
    no = "NO" in str(tab2["upgraded (Rigobon het-ID)"].iloc[-1])
    # THE MASKED-DAY PIN (review finding): the relative het lives ONLY in a halt-masked volatile
    # day. Plain-.mean() demeaning turns that whole day NaN, regime_residual_cov drops it, the
    # spread collapses and the verdict flips to NO -- with nanmean the day's finite rows stay in
    # and the system remains identified with the right coefficients.
    m_sessions, _, _ = _structural_sessions(rel_het=True, mask_het_day=True, per_regime=2)
    tab3 = legacy.identification_table(m_sessions)
    rg3 = tab3["upgraded (Rigobon het-ID)"]
    masked_yes = str(rg3.iloc[-1]).strip().startswith("yes")
    masked_rec = abs(float(rg3.iloc[0]) - g_spy_es) < 0.10
    ok = has_rows and recovered and yes and no and masked_yes and masked_rec
    print("(4) Rigobon table: strength rows present (%s); with RELATIVE het recovers "
          "SPY<-ES=%.3f (true %.2f) ES<-SPY=%.3f (true %.2f) and says identified (%s); "
          "common-scale says NO (%s); het carried by a MASKED day survives the mask "
          "(identified=%s, SPY<-ES=%.3f) : %s"
          % (has_rows, est_se, g_spy_es, est_es, g_es_spy, yes, no,
             masked_yes, float(rg3.iloc[0]), ok))
    return ok


def check_cross_impact_panel_carries_drop1():
    dfc = _book_session(day="2024-07-24", seed=21, halt=None)
    dfd = _book_session(day="2024-08-05", seed=22, halt=None)
    # inject the seam-shaped leverage point: one colossal SPY-flow x ES-return co-movement
    dfd.iloc[3000, dfd.columns.get_loc("SPY_bidquantity_1")] = 250000.0
    es_j = float(dfd.iloc[3000, dfd.columns.get_loc("ES_bidprice_1")]) * 1.02
    for c in ("ES_bidprice_1", "ES_askprice_1"):
        dfd.iloc[3000, dfd.columns.get_loc(c)] = es_j
    dfd.iloc[3001:, dfd.columns.get_loc("ES_bidprice_1")] *= 1.02
    dfd.iloc[3001:, dfd.columns.get_loc("ES_askprice_1")] *= 1.02
    panel, summary = cimp.cross_impact_panel([("2024-07-24", "benchmark", dfc),
                                              ("2024-08-05", "volatile", dfd)], n_levels=1)
    cols = {"lambda_drop1", "drop1_flag"} <= set(panel.columns)
    clean_quiet = not panel.loc[panel["date"] == "2024-07-24", "drop1_flag"].any()
    dirty_loud = panel.loc[panel["date"] == "2024-08-05", "drop1_flag"].all()
    named = summary.attrs.get("drop1_flagged_dates") == ["2024-08-05"]
    ok = cols and clean_quiet and dirty_loud and named
    print("(5) cross-impact panel: drop1 columns present (%s); clean day quiet (%s), leverage "
          "day flagged (%s) and named in the summary (%s) : %s"
          % (cols, clean_quiet, dirty_loud, named, ok))
    return ok


def check_saved_frames_are_premask_and_ab_runs():
    import subprocess as sp
    here = os.path.dirname(os.path.abspath(__file__))
    with tempfile.TemporaryDirectory(prefix="abmask_") as td:
        pkl = os.path.join(td, "frames_1s.pkl")
        sessions = [("2020-03-09", "volatile", _book_session()),
                    ("2020-03-12", "volatile", _book_session(day="2020-03-12", seed=5)),
                    ("2024-07-24", "benchmark", _book_session(day="2024-07-24", seed=7, halt=None)),
                    ("2023-07-19", "benchmark", _book_session(day="2023-07-19", seed=9, halt=None))]
        with open(pkl, "wb") as fh:
            pickle.dump(sessions, fh)
        # (a) run_analysis on the pre-mask pickle: the frames it SAVES must be pre-mask again
        res = sp.run([sys.executable, "run_analysis.py", "--source", "load", "--pickle", pkl,
                      "--volatile", "2020-03-09,2020-03-12", "--interval", "1s",
                      "--only", "information_shares", "--n-lags", "5",
                      "--output-dir", td, "--quick"],
                     capture_output=True, text=True, timeout=540, cwd=here)
        saved = None
        for root, _dirs, files in os.walk(td):
            for f in files:
                if f == "frames_1s.pkl" and root != td:
                    saved = os.path.join(root, f)
        premask = False
        if saved:
            with open(saved, "rb") as fh:
                out_sessions = pickle.load(fh)
            d0 = dict((d, f) for d, _r, f in out_sessions)["2020-03-09"]
            seg = d0.loc["2020-03-09 10:05:00-04:00":"2020-03-09 10:10:00-04:00", "SPY_bidprice_1"]
            premask = bool(np.isfinite(seg.to_numpy(float)).all())
        # (b) the A/B tool runs both arms on the pre-mask pickle and diverges n_obs
        ab = sp.run([sys.executable, "ab_halt_mask.py", "--pickle", pkl,
                     "--volatile", "2020-03-09,2020-03-12", "--n-lags", "5"],
                    capture_output=True, text=True, timeout=540, cwd=here)
        ran = ab.returncode == 0 and "VERDICT" in ab.stdout
        # (c) masked-at-rest input is refused
        mpkl = os.path.join(td, "frames_masked.pkl")
        with open(mpkl, "wb") as fh:
            pickle.dump([(d, r, mh.mask_frame(f)[0]) for d, r, f in sessions], fh)
        ab2 = sp.run([sys.executable, "ab_halt_mask.py", "--pickle", mpkl,
                      "--volatile", "2020-03-09,2020-03-12", "--n-lags", "5"],
                     capture_output=True, text=True, timeout=300, cwd=here)
        refused = ab2.returncode == 2 and "MASKED AT REST" in ab2.stdout
    ok = res.returncode == 0 and premask and ran and refused
    print("(6) frames at rest: run_analysis re-saves PRE-mask frames (halt rows finite: %s); "
          "ab_halt_mask runs both arms on them (%s) and refuses masked-at-rest input with exit 2 "
          "(%s) : %s" % (premask, ran, refused, ok))
    return ok


def main():
    checks = [check_local_projection_survives_the_mask,
              check_cojumps_detected_and_seam_not_a_jump,
              check_legacy_inference_and_comovement,
              check_rigobon_strength_is_in_the_table,
              check_cross_impact_panel_carries_drop1,
              check_saved_frames_are_premask_and_ab_runs]
    res = []
    for fn in checks:
        try:
            res.append(bool(fn()))
        except Exception:
            import traceback; traceback.print_exc()
            res.append(False)
        print()
    ok = all(res)
    print("masked-run casualty checks -> %s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
