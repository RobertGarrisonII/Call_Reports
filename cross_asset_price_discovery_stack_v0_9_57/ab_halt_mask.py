#!/usr/bin/env python3
"""ab_halt_mask.py -- did the MASK kill the headline, or did the SAMPLE?

The 2026-08-05 masked run reversed the paper's decisive regime result (pooled CS_ES
0.266 -> 0.915, interaction t = 2.38 became t = -1.18, permutation p 0.055 -> 0.28) -- but two
things changed at once: the halt masking AND a revised sample. This tool runs the exact
information-shares + panel-VECM estimation TWICE on the SAME frames -- masked and unmasked --
with everything else held fixed, and prints the two columns side by side. Whatever moves between
them is the mask; whatever does not is the sample (or the lags).

    python ab_halt_mask.py --pickle frames_1s.pkl \
        --volatile 2020-03-09,2020-03-12,...  --n-lags 59

The input must be a PRE-MASK frames pickle: the extraction cache, an extraction run's
frames_<interval>.pkl, or anything saved by v0.9.57+ (which saves pre-mask frames again). Frames
saved by a v0.9.51-56 analysis run are masked AT REST -- the NaN is baked in and the unmasked arm
is impossible from them; the tool detects that (attrs['halt_masked'] or NaN inside recorded halt
windows on a leg's mid) and refuses rather than printing an A/A and calling it an A/B.

Reading the verdict:
  * masked != unmasked, unmasked ~ the old decisive numbers  -> the halts WERE the headline: the
    "migration" was 900 seconds of mechanically crossed, non-tradeable snapshots per MWCB day.
  * masked ~ unmasked, both weak                             -> the mask is innocent; the sample
    revision (or the lag change) did it. Re-run the old universe to close the question.

Exit codes: 0 = both arms ran; 2 = input frames are masked-at-rest (no unmasked arm possible).
"""
from __future__ import annotations

import argparse
import pickle
import sys

import numpy as np
import pandas as pd

import cross_asset_pd_liquidity as ca
import market_halts as mh
import price_discovery_shares as pds

EPS = 1e-12


def _load_sessions(path, volatile_csv):
    with open(path, "rb") as fh:
        raw = pickle.load(fh)
    vol = {d.strip() for d in (volatile_csv or "").split(",") if d.strip()}
    out = []
    for rec in raw:
        if len(rec) == 3:
            d, r, df = rec
            if vol:                                       # explicit list wins over stored labels
                r = "volatile" if str(d)[:10] in vol else "benchmark"
        else:
            d, df = rec
            r = "volatile" if str(d)[:10] in vol else "benchmark"
        out.append((str(d)[:10], r, df))
    return out


def _masked_at_rest(sessions):
    """True if the pickle already carries the mask: the unmasked arm would be an A/A."""
    for _d, _r, df in sessions:
        attrs = getattr(df, "attrs", {}) or {}
        if attrs.get("halt_masked"):
            return True
        for a in ("SPY", "ES"):
            wins = attrs.get(f"halt_windows_{a}") or []
            col = f"{a}_bidprice_1"
            if wins and col in df.columns:
                for w in wins:
                    a0, b0 = pd.Timestamp(w[0]), pd.Timestamp(w[1])
                    seg = df.loc[(df.index >= a0) & (df.index <= b0), col]
                    if len(seg) > 10 and not np.isfinite(seg.to_numpy(float)).any():
                        return True
    return False


def _mask(sessions):
    return [(d, r, mh.mask_frame(df)[0]) for d, r, df in sessions]


def _estimate(sessions, n_lags):
    mids = [(d, r, ca._mid(df, "SPY"), ca._mid(df, "ES")) for d, r, df in sessions]
    per_day = pds.estimate_sample(mids, n_lags=n_lags)
    reg = pds.compare_regimes(per_day, metric="CS_ES")
    pan = pds.panel_vecm(mids, n_lags=n_lags)
    kb = float(pan["alpha_benchmark"][1] - pan["alpha_benchmark"][0])
    kv = float(pan["alpha_volatile"][1] - pan["alpha_volatile"][0])
    return {
        "n_obs (pooled panel)": pan["n_obs"],
        "CS_ES pooled benchmark": pan["benchmark"]["CS_ES"],
        "CS_ES pooled volatile": pan["volatile"]["CS_ES"],
        "interaction t (ES eq, day-cluster)": pan["t_interaction_es"],
        "interaction p": pan["p_interaction_es"],
        "kappa benchmark (half-life s)": "%.4f (%.0f)" % (kb, np.log(2) / max(kb, EPS)),
        "kappa volatile (half-life s)": "%.4f (%.0f)" % (kv, np.log(2) / max(kv, EPS)),
        "per-day CS_ES mean benchmark": reg["ben_mean"],
        "per-day CS_ES mean volatile": reg["vol_mean"],
        "per-day diff (vol - ben)": reg["diff"],
        "permutation p": reg["p_perm"],
        "n ec_invalid days": int((~per_day["ec_valid"]).sum()) if "ec_valid" in per_day else 0,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pickle", required=True, help="PRE-MASK frames pickle (List[(date[,regime],df)])")
    ap.add_argument("--volatile", default="", help="comma-separated YYYY-MM-DD marked volatile "
                                                   "(omit to use the labels stored in the pickle)")
    ap.add_argument("--n-lags", type=int, default=59, help="VECM lag order (the masked run used 59)")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    sessions = _load_sessions(a.pickle, a.volatile)
    n_vol = sum(1 for _d, r, _f in sessions if r == "volatile")
    print("loaded %d session(s) (%d volatile / %d benchmark), n_lags=%d"
          % (len(sessions), n_vol, len(sessions) - n_vol, a.n_lags))
    if _masked_at_rest(sessions):
        print("\nINPUT FRAMES ARE MASKED AT REST -- the halt NaN is baked into this pickle "
              "(saved by a v0.9.51-56 analysis run), so an unmasked arm is impossible from it "
              "and the comparison would be an A/A. Point --pickle at the EXTRACTION run's frames "
              "or the extract cache; frames saved by v0.9.57+ are pre-mask again.")
        return 2

    print("estimating the unmasked arm (halts included -- the pre-v0.9.51 behaviour) ...")
    un = _estimate(sessions, a.n_lags)
    print("estimating the masked arm ...")
    ma = _estimate(_mask(sessions), a.n_lags)

    tab = pd.DataFrame({"halts included (old behaviour)": un, "halt-masked (current)": ma})
    lines = ["halt-mask attribution A/B -- same frames, same sample, same lags", "",
             tab.to_string(), ""]
    dp = abs(float(un["permutation p"]) - float(ma["permutation p"]))
    dt = abs(float(un["interaction t (ES eq, day-cluster)"])
             - float(ma["interaction t (ES eq, day-cluster)"]))
    if dt > 1.0 or dp > 0.1:
        lines.append("VERDICT: the arms DIVERGE (|dt|=%.2f, |dp|=%.2f) -- the halt rows were "
                     "carrying that much of the regime result. The masked column is the "
                     "defensible one: a halted market has no valid midpoint, so the difference "
                     "is mechanical comovement, not price discovery." % (dt, dp))
    else:
        lines.append("VERDICT: the arms broadly AGREE (|dt|=%.2f, |dp|=%.2f) -- the mask is not "
                     "what moved the headline. The remaining suspects are the SAMPLE revision "
                     "and the lag order: re-run this tool on frames restricted to the old "
                     "universe to close the question." % (dt, dp))
    text = "\n".join(lines)
    print("\n" + text)
    if a.out:
        with open(a.out, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
