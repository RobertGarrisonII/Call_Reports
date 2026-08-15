#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════════════════════════════════╗
# ║ run_ofr_improved.sh -- the IMPROVED leg of the OFR WP 19-04 / JFM reproduction.               ║
# ║                                                                                              ║
# ║ What makes this the "improved" leg, versus a flag change on the published run:                ║
# ║   * SAMPLE SELECTION IS EX-ANTE. The paper hand-picked the 5 highest expected- and 5 highest  ║
# ║     unexpected-volatility days -- selection on the realized outcome. Here the day universe is ║
# ║     chosen from a LAGGED daily vol series (prior-close VIX, or a HAR forecast) before any     ║
# ║     book is touched, and the same series drives the volatile/benchmark labels at estimation   ║
# ║     time (--vol-csv), so selection and labeling cannot disagree.                              ║
# ║   * EXCHANGE CLOCK by default. Receipt-clock SPY<->ES skew is venue-composition-dependent and ║
# ║     sign-flipping, and it correlates with which venue is aggressive -- i.e. with the event    ║
# ║     itself -- so it can load onto the price-discovery estimates. (NOTE: run_liberation_day.sh ║
# ║     still defaults CLOCK=receipt; this script does not.)                                      ║
# ║   * RECONSTRUCTED BOOKS on both legs, one clock, message-level.                               ║
# ║   * BIC-selected VECM lag length instead of a fixed order.                                    ║
# ║                                                                                              ║
# ║ WHY THREE STAGES. Selection is daily-frequency and costs nothing; extraction is the entire    ║
# ║ budget. Never pass the full 2014-01-01:2017-08-31 range to --date-range: that is ~920 trading ║
# ║ sessions of consolidated SPY+ES MBO, not the ~20-40 the design actually needs. `select`       ║
# ║ decides WHICH days on daily data, `pilot` proves the reconstruction is sound on this era      ║
# ║ before the budget is committed, and only then does `main` extract.                            ║
# ║                                                                                              ║
# ║   STAGE=select ./run_ofr_improved.sh     # cheap: pick the day universe, no extraction        ║
# ║   STAGE=pilot  ./run_ofr_improved.sh     # 2 days: crossed-book gate on the OFR-era tape      ║
# ║   STAGE=main   ./run_ofr_improved.sh     # the full improved run -> manuscript tables         ║
# ║   STAGE=all    ./run_ofr_improved.sh     # select -> pilot -> main (pilot must pass)          ║
# ╚══════════════════════════════════════════════════════════════════════════════════════════════╝
set -euo pipefail

STAGE="${STAGE:-select}"      # default is the CHEAP stage on purpose -- never extract by accident
[ "${1:-}" ] && STAGE="$1"

# ---- location / interpreter -------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="${PKG_DIR:-$SCRIPT_DIR}"
PY="${PY:-python3}"
OUT_ROOT="${OUT_ROOT:-output/ofr_improved}"

# ---- the reproduction era (paper sample: Jan 2014 - Aug 2017) ----------------
START="${START:-2014-01-01}"
END="${END:-2017-08-31}"

# ---- ex-ante selection ------------------------------------------------------
# VOL_CSV is REQUIRED and is the whole point: a daily series, columns date,vol -- prior-close VIX or
# a HAR one-step forecast. It must be information available BEFORE the session it labels. Realized
# same-day vol here would silently reinstate the ex-post selection this leg exists to remove.
VOL_CSV="${VOL_CSV:-}"
N_STRESS="${N_STRESS:-10}"                   # stress days to keep (paper used 10); controls are matched 1:1
LEVEL_Q="${LEVEL_Q:-0.90}"                   # stress = persistent-level percentile >= this, or a sharp onset
CALM_Q="${CALM_Q:-0.33}"                     # calm  = level percentile <= this AND stable
ONSET_THRESH="${ONSET_THRESH:-0.5}"          # log-change onset rule (captures spikes off a low level)
CALIPER="${CALIPER:-90}"                     # max CALENDAR-DAY gap for an era-matched control
EPISODE_GAP="${EPISODE_GAP:-10}"             # min calendar days between two selected stress days, so the
                                             # sample spans DISTINCT episodes (tau is persistent: without
                                             # this, top-N by level returns N consecutive days of ONE spike,
                                             # which is a single day-cluster for the bootstrap)
UNIVERSE="${UNIVERSE:-$OUT_ROOT/ofr_universe.txt}"

# ---- pilot (the reconstruction gate for this era) ---------------------------
# Two OFR-era sessions: one stressed, one calm. Every reconstruction fix in this stack was validated
# on 2025 tape; 2014-2017 is a different venue-protocol generation, so the crossed-book rate here is
# UNKNOWN until measured. CROSS_MAX is the pass bar (fraction of 1s snapshots with bid1 > ask1).
PILOT_DATES="${PILOT_DATES:-2015-08-24,2015-08-17}"
CROSS_MAX="${CROSS_MAX:-0.01}"

# ---- shared estimation parameters -------------------------------------------
NLEVELS="${NLEVELS:-10}"
CLOCK="${CLOCK:-exchange}"                   # <-- the memo's decision; NOT run_liberation_day.sh's receipt default
CLASSIFY="${CLASSIFY:-aggressor}"
LAGCRIT="${LAGCRIT:-bic}"
MAXLAGS="${MAXLAGS:-12}"
INTERVAL="${INTERVAL:-1s}"
OUTCOME="${OUTCOME:-auc_decay}"
NBOOT="${NBOOT:-1000}"
EVENT_CATEGORIES="${EVENT_CATEGORIES:-MWCB}"
RELEASE_ET="${RELEASE_ET:-09:30}"
read -r -a HOOKS <<< "${HOOKS_STR:-}"        # e.g. HOOKS_STR="--auction --copula"
read -r -a EXTRA <<< "${EXTRA_FLAGS:-}"

cd "$PKG_DIR"
mkdir -p "$OUT_ROOT"
TS="$(date +%Y%m%d_%H%M%S)"
banner () { printf '\n========== %s ==========\n' "$*"; }
die () { printf '\nERROR: %s\n' "$*" >&2; exit 2; }

# ---- worker sizing: autoscale.py is the single source (shell + python agree) -
export PEAK_GB_GUESS="${PEAK_GB_GUESS:-32}"
RESERVE_GB="$("$PY" autoscale.py reserve)"
BOOT_WORKERS="$("$PY" autoscale.py jobs)"    # bootstrap is core-bound, not memory-bound

# Flags shared by every extraction run in this leg
contagion_common () {
  printf '%s\n' \
    --source extract \
    --n-levels "$NLEVELS" \
    --clock "$CLOCK" \
    --classify "$CLASSIFY" \
    --book-source reconstruct \
    --progress auto
}

# ══════════════════════════════════════════════════════════════════════════════
# STAGE: select -- pick the day universe from DAILY data. No extraction.
# ══════════════════════════════════════════════════════════════════════════════
run_select () {
  [ -n "$VOL_CSV" ] || die "VOL_CSV is required for STAGE=select.
  Supply a daily ex-ante vol series (columns: date,vol) -- prior-close VIX or a HAR forecast:
     VOL_CSV=vix_daily.csv STAGE=select ./run_ofr_improved.sh
  It must be knowable BEFORE the session it labels; same-day realized vol reinstates ex-post selection."
  [ -f "$VOL_CSV" ] || die "VOL_CSV not found: $VOL_CSV"
  banner "SELECT ($START -> $END, ex-ante vol from $VOL_CSV, N_STRESS=$N_STRESS)"
  "$PY" - "$VOL_CSV" "$START" "$END" "$N_STRESS" "$LEVEL_Q" "$CALM_Q" "$ONSET_THRESH" \
        "$CALIPER" "$EPISODE_GAP" "$UNIVERSE" "$OUT_ROOT/selection_$TS.md" <<'PYEOF'
import sys
import numpy as np
import pandas as pd
import stress_index as si
import market_shocks as mk

(csv, start, end, n_stress, level_q, calm_q, onset_t, caliper, ep_gap,
 uni_path, rep_path) = sys.argv[1:12]
n_stress, level_q, calm_q = int(n_stress), float(level_q), float(calm_q)
onset_t, caliper, ep_gap = float(onset_t), float(caliper), float(ep_gap)

raw = pd.read_csv(csv)
cols = {c.strip().lower(): c for c in raw.columns}
dcol = cols.get("date", raw.columns[0])
vcol = cols.get("vol", cols.get("close", raw.columns[-1]))
s = pd.Series(pd.to_numeric(raw[vcol], errors="coerce").to_numpy(),
              index=pd.to_datetime(raw[dcol])).sort_index().dropna()
s = s.loc[str(start):str(end)]
if len(s) < 60:
    raise SystemExit("only %d daily vol points in %s..%s -- need a fuller series" % (len(s), start, end))

# The series is used AS GIVEN (it is the ex-ante signal). select_days ranks on the persistent
# component tau and flags sharp onsets, which is what picks up spikes off a low base.
panel = si.select_days(s, level_q=level_q, calm_q=calm_q, onset_thresh=onset_t)
panel = panel[[mk.is_trading_day(pd.Timestamp(i).date()) for i in panel.index]]

stress_all = panel[panel["label"] == "stress"].sort_values("tau", ascending=False)
# Thin to DISTINCT EPISODES before taking the top N. tau is the persistent component, so a single
# vol spike leaves a plateau of consecutive high-tau sessions; an unthinned top-N returns ten days of
# one episode, which the day-cluster bootstrap sees as ONE cluster. Greedy: take the highest-tau day,
# drop everything within ep_gap calendar days, repeat.
keep = []
for d in stress_all.index:
    t = pd.Timestamp(d)
    if all(abs((t - pd.Timestamp(k)).days) >= ep_gap for k in keep):
        keep.append(d)
    if len(keep) >= n_stress:
        break
stress = stress_all.loc[keep].sort_values("tau", ascending=False)
calm = panel[panel["label"] == "calm"]

# Controls are matched on CALENDAR PROXIMITY (era), 1:1 without replacement -- NOT on the vol level.
# Level matching here would be tautological: `stress` and `calm` are DEFINED by tau percentile
# (>= level_q vs <= calm_q), so the tau gap between them is large by construction and a level caliper
# rejects every pair. (stress_index.match_controls does match on tau, with replacement; run it on a
# panel whose labels are not level-defined, or leave level balance to the downstream matcher.) The
# ex-ante LEVEL-matched control balance is event_study_driver's job inside run_contagion -- this stage
# only decides which sessions to EXTRACT. Era matching is what that stage needs: a control from the
# same weeks carries the same venue protocol generation, tick regime, and contract cycle. Matching
# without replacement keeps the day-cluster bootstrap supplied with distinct control clusters; the tau
# imbalance is reported below rather than hidden, since it is the treatment, not a confound.
cal_idx = list(calm.index)
cal_ord = np.array([pd.Timestamp(i).value for i in cal_idx], float)
used, pairs, unmatched = set(), [], []
for d, lv in stress["tau"].items():
    t0 = float(pd.Timestamp(d).value)
    best, bestj = np.inf, None
    for j, cv in enumerate(cal_ord):
        if j in used:
            continue
        gap = abs(cv - t0) / 86_400_000_000_000.0            # ns -> calendar days
        if gap < best:
            best, bestj = gap, j
    if bestj is None or best > caliper:
        unmatched.append((d, best if bestj is not None else np.nan))
        continue
    used.add(bestj)
    pairs.append((d, cal_idx[bestj], best))

sd = [str(pd.Timestamp(i).date()) for i in stress.index]
cd = sorted({str(pd.Timestamp(c).date()) for _s, c, _g in pairs})
alld = sorted(set(sd) | set(cd))
open(uni_path, "w").write(",".join(alld) + "\n")

L = ["# Ex-ante day selection (%s -> %s)" % (start, end), "",
     "Source series: `%s` (used as the EX-ANTE signal; no same-day realized vol enters)." % csv, "",
     "- daily points in range: %d" % len(s),
     "- labeled stress: %d; kept %d distinct episodes (>= %.0f days apart) by persistent level tau"
     % (len(stress_all), len(stress), ep_gap),
     "- matched calm controls: %d (1:1 era match, without replacement, caliper %.0f days)" % (len(cd), caliper),
     "- unmatched stress days: %d" % len(unmatched),
     "- extraction universe: %d sessions" % len(alld), "",
     "## Stress days (kept)", "",
     "| date | vol | tau | log_change | level_pct | onset |", "| --- | --- | --- | --- | --- | --- |"]
for i, r in stress.iterrows():
    L.append("| %s | %.4g | %.4g | %+.3f | %.3f | %s |"
             % (pd.Timestamp(i).date(), r["vol"], r["tau"], r["log_change"], r["level_pct"], bool(r["onset"])))
L += ["", "## Matched pairs (stress -> control, tau gap)", "",
      "| stress | control | gap (days) | tau stress | tau control |", "| --- | --- | --- | --- | --- |"]
for a, b, g in pairs:
    L.append("| %s | %s | %.0f | %.4g | %.4g |"
             % (pd.Timestamp(a).date(), pd.Timestamp(b).date(), g,
                panel.loc[a, "tau"], panel.loc[b, "tau"]))
if unmatched:
    L += ["", "**Unmatched stress days (no control inside the caliper):** "
          + ", ".join("%s" % pd.Timestamp(d).date() for d, _g in unmatched)]
L += ["",
      "## Extraction universe", "", "```", ",".join(alld), "```", ""]
open(rep_path, "w").write("\n".join(L))
print("stress=%d (distinct episodes)  controls=%d  unmatched=%d  universe=%d sessions"
      % (len(stress), len(cd), len(unmatched), len(alld)))
if len(stress) < n_stress:
    print("NOTE: only %d episodes >= %.0f days apart exist in range (asked %d); lower EPISODE_GAP,"
          " widen the range, or accept fewer treated clusters" % (len(stress), ep_gap, n_stress))
if len(cd) < max(3, len(stress) // 2):
    print("WARNING: few distinct control clusters (%d) -- the day-cluster bootstrap will be weak;"
          " widen CALIPER or raise CALM_Q" % len(cd))
print("universe -> %s" % uni_path)
print("report   -> %s" % rep_path)
PYEOF
  echo ">>> review $OUT_ROOT/selection_$TS.md before running STAGE=pilot"
}

# ══════════════════════════════════════════════════════════════════════════════
# STAGE: pilot -- extract 2 days and GATE on the crossed-book invariant.
# A crossed top is impossible in a real matching engine, so a nonzero rate is reconstruction
# error, and it contaminates every book-derived input (spread, hollowness, cost, mid).
# ══════════════════════════════════════════════════════════════════════════════
run_pilot () {
  local out="$OUT_ROOT/pilot"; mkdir -p "$out"
  local nsess; nsess="$(awk -F, '{print NF}' <<< "$PILOT_DATES")"
  local workers; workers="$("$PY" autoscale.py workers --sessions "$nsess")"
  banner "PILOT ($PILOT_DATES, clock=$CLOCK, workers=$workers) -- crossed-book gate"
  "$PY" run_contagion.py \
    $(contagion_common) \
    --dates "$PILOT_DATES" \
    --mode descriptive \
    --interval "$INTERVAL" \
    --n-boot 0 \
    --max-workers "$workers" \
    --save-pickle "$out/pilot_sessions.pkl" \
    --output-dir "$out" 2>&1 | tee "$out/pilot_$TS.log"

  "$PY" - "$out/pilot_sessions.pkl" "$CROSS_MAX" <<'PYEOF'
import pickle, sys
import numpy as np
import onset_response as onr

path, thr = sys.argv[1], float(sys.argv[2])
sessions = pickle.load(open(path, "rb"))
print("\ncrossed-book gate (bid1 > ask1; impossible in a real book -> reconstruction error)")
print("  %-12s %-10s %10s %10s" % ("date", "regime", "SPY", "ES"))
worst = 0.0
for date, regime, df in sessions:
    x = {a: onr._crossed_frac(df, a) for a in ("SPY", "ES")}
    fin = [v for v in x.values() if np.isfinite(v)]
    worst = max([worst] + fin)
    print("  %-12s %-10s %9.2f%% %9.2f%%" % (str(date)[:10], regime, 100 * x["SPY"], 100 * x["ES"]))
if not np.isfinite(worst):
    raise SystemExit("could not measure crossing (no bid1/ask1 columns) -- inspect the frames")
print("\n  worst = %.2f%%   bar = %.2f%%" % (100 * worst, 100 * thr))
if worst > thr:
    raise SystemExit(
        "PILOT FAILED: reconstruction on this era crosses above the bar.\n"
        "  Do NOT run STAGE=main -- every state and transmission estimate would inherit this.\n"
        "  Decompose within-venue vs cross-venue (verify_crossing) before committing extraction.")
print("  PILOT PASSED -- reconstruction is sound on this era; STAGE=main is safe to run.")
PYEOF
}

# ══════════════════════════════════════════════════════════════════════════════
# STAGE: main -- extract the selected universe and build the manuscript tables.
# ══════════════════════════════════════════════════════════════════════════════
run_main () {
  [ -f "$UNIVERSE" ] || die "no universe file at $UNIVERSE -- run STAGE=select first"
  local dates; dates="$(tr -d '[:space:]' < "$UNIVERSE")"
  [ -n "$dates" ] || die "universe file $UNIVERSE is empty"
  local nsess; nsess="$(awk -F, '{print NF}' <<< "$dates")"
  local workers; workers="$("$PY" autoscale.py workers --sessions "$nsess")"
  local out="$OUT_ROOT/main_$INTERVAL"; mkdir -p "$out"

  banner "MAIN ($nsess sessions @ $INTERVAL, clock=$CLOCK, workers=$workers, boot=$BOOT_WORKERS x $NBOOT)"
  echo ">>> universe : $UNIVERSE"
  echo ">>> sizing   : RESERVE_GB=$RESERVE_GB PEAK_GB_GUESS=$PEAK_GB_GUESS (autoscale)"
  # --auto-regime + --vol-csv together are the point: labels come from the SAME ex-ante series that
  # chose the days, so selection and labeling cannot disagree. Without --vol-csv the driver would
  # fall back to per-session realized vol -- ex-post, and the thing this leg removes.
  local volflag=(); [ -n "$VOL_CSV" ] && volflag=(--vol-csv "$VOL_CSV")
  [ -n "$VOL_CSV" ] || echo ">>> WARNING: VOL_CSV unset -- regime labels will fall back to realized (ex-post) vol"

  "$PY" run_contagion.py \
    $(contagion_common) \
    --dates "$dates" \
    --auto-regime "${volflag[@]}" \
    --mode both \
    --interval "$INTERVAL" \
    --lag-criterion "$LAGCRIT" \
    --max-lags "$MAXLAGS" \
    --outcome "$OUTCOME" \
    --event-categories "$EVENT_CATEGORIES" \
    --default-release-et "$RELEASE_ET" \
    --n-boot "$NBOOT" \
    --max-workers "$workers" \
    --boot-workers "$BOOT_WORKERS" \
    "${HOOKS[@]}" "${EXTRA[@]}" \
    --output-dir "$out" 2>&1 | tee "$out/main_$TS.log"

  banner "MAIN done -> $out"
  echo "  contagion_manuscript_tables.md / .tex   (the 16-table suite)"
  echo "  contagion_run_summary.md"
}

case "$STAGE" in
  select) run_select ;;
  pilot)  run_pilot ;;
  main)   run_main ;;
  all)    run_select; run_pilot; run_main ;;
  *) die "unknown stage '$STAGE' (use: select | pilot | main | all)" ;;
esac
