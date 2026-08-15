#!/usr/bin/env python3
"""run_halflife_experiment.py -- E4's Kalman arm: is the fine-grid half-life real
or a staleness artifact?

Per session: the ECM's kappa/half-life on the RAW (stale) mids versus the joint
Kalman-VECM MLE, whose likelihood scores only actual quote refreshes (staleness =
missing data, never fake zero returns). The pilot's twin failure modes frame the
reading: a raw stale ECM can lock onto the QUOTE-REFRESH rate instead of the
price process, and a smooth-then-regress two-step manufactures persistence --
so the comparison to quote is raw vs MLE, with the synthetic-staleness bracket
as the measured cross-check (kalman_ecm.py docstring has the derivation).

Usage
    python run_halflife_experiment.py --source demo
    python run_halflife_experiment.py --source load --pickle output/.../frames_10ms.pkl \
        --interval 100ms --out-dir output/e4 --tag 100ms
"""
import argparse
import glob
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd

import derive_frames as dfr
import kalman_ecm as ke


def _load(args):
    if args.source == "demo":
        sess, _truth = ke.staleness_demo_sessions(n_days=args.n_demo,
                                                  n_steps=args.n_demo_steps,
                                                  kappa=args.demo_kappa,
                                                  dt=dfr._interval_seconds(args.interval))
        return sess
    paths = sorted(glob.glob(args.pickle))
    if not paths:
        raise SystemExit(f"no pickle matched {args.pickle!r}")
    raw = []
    for p in paths:
        with open(p, "rb") as fh:
            raw.extend(pickle.load(fh))
    vol = {d.strip() for d in (args.volatile or "").split(",") if d.strip()}
    out = []
    for rec in raw:
        if len(rec) == 3:
            date, regime, df = rec
        else:
            date, df = rec
            regime = "volatile" if str(date)[:10] in vol else "benchmark"
        out.append((date, regime, df))
    if getattr(args, "halt_mask", True):
        import market_halts as mh
        out = [(d, r, mh.mask_frame(df)[0]) for d, r, df in out]
    if args.interval != args.fine_interval:
        derived = []
        for d, r, df in out:
            try:
                derived.append((d, r, dfr.derive_coarse_frame(df, args.interval,
                                                              args.fine_interval)))
            except ValueError as exc:
                print(f"  NOTE: {d} skipped ({exc})", file=sys.stderr)
        out = derived
    return out


def build_parser():
    ap = argparse.ArgumentParser(description="E4 Kalman arm: raw vs Kalman-VECM half-life under staleness")
    ap.add_argument("--source", choices=["demo", "load"], default="demo")
    ap.add_argument("--pickle", default="", help="glob for the FINE-grid frames pickle")
    ap.add_argument("--volatile", default="")
    ap.add_argument("--fine-interval", default="10ms")
    ap.add_argument("--interval", default="100ms",
                    help="grid the experiment runs at (derived from the fine frames; "
                         "100ms keeps the MLE to tens of seconds per session)")
    ap.add_argument("--obs-noise-ticks", type=float, default=0.5)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--halt-mask", dest="halt_mask", action="store_true", default=True)
    ap.add_argument("--no-halt-mask", dest="halt_mask", action="store_false")
    ap.add_argument("--n-demo", type=int, default=2)
    ap.add_argument("--n-demo-steps", type=int, default=30000)
    ap.add_argument("--demo-kappa", type=float, default=0.02)
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    warnings.simplefilter("ignore")
    dt = dfr._interval_seconds(a.interval)
    sessions = _load(a)
    print("sessions: %d  interval: %s (dt=%gs)" % (len(sessions), a.interval, dt))
    if a.source == "demo":
        print("NOTE: demo plants kappa=%g (half-life %.2fs at this dt); the raw stale ECM"
              % (a.demo_kappa, np.log(2) / a.demo_kappa * dt))
        print("      should MISS it and the Kalman MLE should recover it.")
    per_day, summ = ke.half_life_experiment(sessions, dt, obs_noise_ticks=a.obs_noise_ticks)
    pd.set_option("display.width", 240)
    show = [c for c in ("regime", "kappa_raw", "kappa_kalman", "half_life_raw_s",
                        "half_life_kalman_s", "stale_frac_SPY", "stale_frac_ES",
                        "mle_evals", "error") if c in per_day.columns]
    print(per_day[show].round(5).to_string())
    print()
    if summ:
        print("E4 KALMAN ARM: median half-life raw %.2fs vs Kalman %.2fs (ratio raw/kalman "
              "%.3f) over %d day(s)."
              % (summ["median_half_life_raw_s"], summ["median_half_life_kalman_s"],
                 summ["median_ratio_raw_over_kalman"], summ["n_days"]))
        print("  Reading: if the Kalman half-life at this grid closes toward the 1s-grid "
              "value, the fine-grid")
        print("  disagreement is staleness, not dynamics -- corroborate with the "
              "synthetic-staleness bracket")
        print("  before quoting (filtered, not measured).")
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
        tag = (a.tag + "_") if a.tag else ""
        stem = os.path.join(a.out_dir, f"halflife_e4_{tag}{a.interval}")
        per_day.to_csv(stem + ".csv")
        print("wrote %s.csv" % stem)
    return 0


if __name__ == "__main__":
    sys.exit(main())
