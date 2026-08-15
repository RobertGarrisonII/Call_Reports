#!/usr/bin/env bash
# ==============================================================================================
# ofr_improved_job.sh -- SINGLE-FILE GRID JOB: the improved leg of the OFR WP 19-04 / JFM
# reproduction. Submit as-is; it runs unattended start to finish and exits non-zero on failure.
#
#   Phase 0  preflight   -- environment + synthetic table build (fails fast, before any extraction)
#   Phase 1  select      -- pick the day universe from DAILY ex-ante vol (no book access)
#   Phase 2  pilot       -- extract 2 era days and GATE on the crossed-book invariant
#   Phase 3  main        -- extract the selected universe, build the 16-table manuscript suite
#
# Requeue-safe: each phase skips if its output already exists, so a restarted job resumes rather
# than repeating the extraction.
#
# EXIT CODES (so the scheduler log is diagnosable without reading the run log)
#   0 ok | 10 config/path error | 11 preflight failed | 12 selection produced no universe
#   13 PILOT GATE FAILED (reconstruction crosses on this era) | 14 main run failed
# ==============================================================================================

# ---- scheduler directives: UNCOMMENT THE BLOCK FOR YOUR GRID --------------------------------
## --- Slurm ---
##SBATCH --job-name=ofr_improved
##SBATCH --cpus-per-task=32
##SBATCH --mem=256G
##SBATCH --time=48:00:00
##SBATCH --output=ofr_improved_%j.out
## --- SGE / UGE ---
##$ -N ofr_improved
##$ -pe smp 32
##$ -l h_vmem=8G
##$ -cwd -j y
## --- PBS / Torque ---
##PBS -N ofr_improved
##PBS -l select=1:ncpus=32:mem=256gb
##PBS -l walltime=48:00:00
# ---------------------------------------------------------------------------------------------

set -euo pipefail

# ##############################################################################################
# #                              CONFIGURATION -- EDIT THIS BLOCK                               #
# ##############################################################################################

# --- paths (ABSOLUTE: a grid job does not start in your home directory) ----------------------
PKG_DIR="/home/workbench/cross_asset_price_discovery_stack_v0_9_13"
VOL_CSV="/home/workbench/data/vix_daily.csv"      # daily EX-ANTE vol series, columns: date,vol
OUT_ROOT="/home/workbench/output/ofr_improved"
PY="python3"

# --- reproduction era (paper sample: Jan 2014 - Aug 2017) ------------------------------------
START="2014-01-01"
END="2017-08-31"

# --- ex-ante day selection -------------------------------------------------------------------
N_STRESS=10          # stress days to keep (paper used 10)
LEVEL_Q=0.90         # stress = persistent-level percentile >= this, or a sharp onset
CALM_Q=0.33          # calm   = level percentile <= this AND stable
ONSET_THRESH=0.5     # log-change onset rule (catches spikes off a low base)
EPISODE_GAP=10       # min calendar days between selected stress days -> DISTINCT episodes.
                     # tau is persistent, so without this, top-N returns N consecutive days of
                     # ONE spike, which the day-cluster bootstrap sees as a single cluster.
CALIPER=90           # max calendar-day gap for an era-matched control

# --- pilot gate (the reconstruction check for this era) --------------------------------------
RUN_PILOT=1                            # set 0 to skip once this era is already validated
PILOT_DATES="2015-08-24,2015-08-17"    # one stressed, one calm, both in-era
CROSS_MAX=0.01                         # max crossed-book fraction; above this the job ABORTS

# --- estimation ------------------------------------------------------------------------------
NLEVELS=10
CLOCK="exchange"     # NOT receipt: receipt-clock SPY<->ES skew is venue-composition-dependent,
                     # sign-flipping, and correlated with which venue is aggressive
CLASSIFY="aggressor"
INTERVAL="1s"
LAGCRIT="bic"
MAXLAGS=12
OUTCOME="auc_decay"
NBOOT=1000
EVENT_CATEGORIES="MWCB"
RELEASE_ET="09:30"
HOOKS_STR=""         # e.g. "--auction --copula"
EXTRA_FLAGS=""       # e.g. "--price wmid_orth"

# --- resources: leave empty to auto-detect (autoscale reads cgroup limits, so it respects the
# --- scheduler's allocation). Set only to override.
FORCE_CORES=""       # e.g. 32
FORCE_RAM_GB=""      # e.g. 256
PEAK_GB_GUESS=32     # per-worker peak RSS for one full session of MBO messages
RUN_PREFLIGHT=1      # synthetic 16-table build before touching data

# ##############################################################################################
# #                          END CONFIGURATION -- no edits needed below                         #
# ##############################################################################################

STAMP="$(date +%Y%m%d_%H%M%S)"
UNIVERSE="$OUT_ROOT/ofr_universe.txt"
PILOT_DIR="$OUT_ROOT/pilot"
MAIN_DIR="$OUT_ROOT/main_${INTERVAL}"

fail () { printf '\n[FATAL] %s\n' "$2" >&2; exit "$1"; }
step () { printf '\n=============== [%s] %s ===============\n' "$(date +%H:%M:%S)" "$*"; }

[ -d "$PKG_DIR" ] || fail 10 "PKG_DIR does not exist: $PKG_DIR"
cd "$PKG_DIR" || fail 10 "cannot cd to PKG_DIR: $PKG_DIR"
mkdir -p "$OUT_ROOT" "$PILOT_DIR" "$MAIN_DIR" || fail 10 "cannot create OUT_ROOT: $OUT_ROOT"

LOG="$OUT_ROOT/ofr_improved_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

# BLAS must be pinned to ONE thread: the bootstrap parallelizes across replicates, and a threaded
# BLAS inside each replicate oversubscribes the node and runs slower than serial.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1
export PEAK_GB_GUESS
[ -n "$FORCE_CORES" ]  && export CORES="$FORCE_CORES"
[ -n "$FORCE_RAM_GB" ] && export RAM_GB="$FORCE_RAM_GB"

trap 'rc=$?; [ "$rc" -ne 0 ] && printf "\n[ABORTED] rc=%s -- see %s\n" "$rc" "$LOG"; exit "$rc"' EXIT

step "OFR improved leg -- job start"
cat <<EOF
  host        : $(hostname)
  package     : $PKG_DIR  (version $("$PY" -c 'import __init__ as p; print(p.__version__)' 2>/dev/null || echo '?'))
  era         : $START -> $END
  vol series  : $VOL_CSV
  clock       : $CLOCK        interval: $INTERVAL      lags: $LAGCRIT (max $MAXLAGS)
  bootstrap   : $NBOOT reps
  output      : $OUT_ROOT
  log         : $LOG
EOF

CORES_D="$("$PY" autoscale.py cores)"
RAM_D="$("$PY" autoscale.py ram)"
RESERVE_D="$("$PY" autoscale.py reserve)"
BOOT_WORKERS="$("$PY" autoscale.py jobs)"
echo "  resources   : ${CORES_D} cores, ${RAM_D} GB RAM (reserve ${RESERVE_D} GB), boot-workers ${BOOT_WORKERS}"

# ---------------------------------------------------------------------------------------------
# PHASE 0 -- preflight. Cheap failures now beat expensive failures after six hours of extraction.
# ---------------------------------------------------------------------------------------------
if [ "$RUN_PREFLIGHT" -eq 1 ]; then
  step "PHASE 0/3  preflight"
  [ -f "$VOL_CSV" ] || fail 10 "VOL_CSV not found: $VOL_CSV"
  "$PY" -c "import numpy, pandas, joblib, run_contagion, paper_tables, stress_index, onset_response" \
    || fail 11 "package imports failed -- check the interpreter and PKG_DIR"
  ( cd "$PILOT_DIR" && "$PY" "$PKG_DIR/paper_tables.py" >/dev/null 2>&1 ) \
    || fail 11 "synthetic 16-table build failed -- the reporting layer is broken in this environment"
  echo "  preflight OK: imports resolve, synthetic table suite builds"
else
  echo "  preflight SKIPPED (RUN_PREFLIGHT=0)"
fi

# ---------------------------------------------------------------------------------------------
# PHASE 1 -- ex-ante selection. Daily data only; no order book is touched.
# Passing the full era to --date-range would extract ~920 sessions instead of the ~20 needed.
# ---------------------------------------------------------------------------------------------
step "PHASE 1/3  ex-ante day selection"
if [ -s "$UNIVERSE" ]; then
  echo "  universe already exists -> reusing $UNIVERSE (delete it to re-select)"
else
  "$PY" - "$VOL_CSV" "$START" "$END" "$N_STRESS" "$LEVEL_Q" "$CALM_Q" "$ONSET_THRESH" \
          "$CALIPER" "$EPISODE_GAP" "$UNIVERSE" "$OUT_ROOT/selection_${STAMP}.md" <<'PYEOF'
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

panel = si.select_days(s, level_q=level_q, calm_q=calm_q, onset_thresh=onset_t)
panel = panel[[mk.is_trading_day(pd.Timestamp(i).date()) for i in panel.index]]

stress_all = panel[panel["label"] == "stress"].sort_values("tau", ascending=False)
# Thin to DISTINCT EPISODES before taking the top N: tau is the persistent component, so one spike
# leaves a plateau of consecutive high-tau sessions and an unthinned top-N is one day-cluster.
keep = []
for d in stress_all.index:
    t = pd.Timestamp(d)
    if all(abs((t - pd.Timestamp(k)).days) >= ep_gap for k in keep):
        keep.append(d)
    if len(keep) >= n_stress:
        break
stress = stress_all.loc[keep].sort_values("tau", ascending=False)
calm = panel[panel["label"] == "calm"]

# Controls are ERA-matched (calendar proximity), 1:1 WITHOUT replacement -- not level-matched.
# Level matching is tautological here (stress/calm are DEFINED by tau percentile, so the gap is
# large by construction); ex-ante level balance is event_study_driver's job downstream. Without
# replacement keeps the day-cluster bootstrap supplied with distinct control clusters.
cal_idx = list(calm.index)
cal_ord = np.array([pd.Timestamp(i).value for i in cal_idx], float)
used, pairs, unmatched = set(), [], []
for d, lv in stress["tau"].items():
    t0 = float(pd.Timestamp(d).value)
    best, bestj = np.inf, None
    for j, cv in enumerate(cal_ord):
        if j in used:
            continue
        gap = abs(cv - t0) / 86_400_000_000_000.0
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
     "Source series: `%s`, used AS THE EX-ANTE SIGNAL (no same-day realized vol enters)." % csv, "",
     "- daily points in range: %d" % len(s),
     "- labeled stress: %d; kept %d distinct episodes (>= %.0f days apart)" % (len(stress_all), len(stress), ep_gap),
     "- matched calm controls: %d (1:1 era match, without replacement, caliper %.0f days)" % (len(cd), caliper),
     "- unmatched stress days: %d" % len(unmatched),
     "- extraction universe: %d sessions" % len(alld), "",
     "## Stress days", "",
     "| date | vol | tau | log_change | level_pct | onset |", "| --- | --- | --- | --- | --- | --- |"]
for i, r in stress.iterrows():
    L.append("| %s | %.4g | %.4g | %+.3f | %.3f | %s |"
             % (pd.Timestamp(i).date(), r["vol"], r["tau"], r["log_change"], r["level_pct"], bool(r["onset"])))
L += ["", "## Matched pairs (stress -> control)", "",
      "| stress | control | gap (days) | tau stress | tau control |", "| --- | --- | --- | --- | --- |"]
for a, b, g in pairs:
    L.append("| %s | %s | %.0f | %.4g | %.4g |"
             % (pd.Timestamp(a).date(), pd.Timestamp(b).date(), g, panel.loc[a, "tau"], panel.loc[b, "tau"]))
if unmatched:
    L += ["", "**Unmatched stress days (no control inside the caliper):** "
          + ", ".join(str(pd.Timestamp(d).date()) for d, _g in unmatched)]
ts_ = [panel.loc[a, "tau"] for a, _b, _g in pairs]
tc_ = [panel.loc[b, "tau"] for _a, b, _g in pairs]
if ts_:
    L += ["", "Mean tau: stress %.4g vs control %.4g. A large gap is EXPECTED -- tau defines the labels;"
          " ex-ante level balance is enforced downstream by event_study_driver."
          % (float(np.mean(ts_)), float(np.mean(tc_)))]
L += ["", "## Extraction universe", "", "```", ",".join(alld), "```", ""]
open(rep_path, "w").write("\n".join(L))
print("  stress=%d (distinct episodes)  controls=%d  unmatched=%d  universe=%d sessions"
      % (len(stress), len(cd), len(unmatched), len(alld)))
if len(cd) < max(3, len(stress) // 2):
    print("  WARNING: few distinct control clusters (%d) -- the day-cluster bootstrap will be weak;"
          " widen CALIPER or raise CALM_Q" % len(cd))
PYEOF
fi
[ -s "$UNIVERSE" ] || fail 12 "selection produced no universe file"
DATES="$(tr -d '[:space:]' < "$UNIVERSE")"
NSESS="$(awk -F, '{print NF}' <<< "$DATES")"
WORKERS="$("$PY" autoscale.py workers --sessions "$NSESS")"
echo "  universe : $NSESS sessions -> $UNIVERSE"
echo "  workers  : $WORKERS (memory-bound extraction budget)"

# ---------------------------------------------------------------------------------------------
# PHASE 2 -- pilot gate. Every reconstruction fix in this stack was validated on 2025 tape;
# 2014-2017 is a different venue-protocol generation, so the crossed-book rate here is UNKNOWN
# until measured. A crossed top is impossible in a real matching engine, so any material rate is
# reconstruction error contaminating the spread, hollowness, cost, and mid on every affected bar.
# ---------------------------------------------------------------------------------------------
if [ "$RUN_PILOT" -eq 1 ]; then
  step "PHASE 2/3  pilot -- crossed-book gate ($PILOT_DATES)"
  if [ -s "$PILOT_DIR/pilot_sessions.pkl" ]; then
    echo "  pilot extract already exists -> reusing (delete to re-run)"
  else
    PW="$("$PY" autoscale.py workers --sessions "$(awk -F, '{print NF}' <<< "$PILOT_DATES")")"
    "$PY" run_contagion.py \
      --source extract --dates "$PILOT_DATES" \
      --n-levels "$NLEVELS" --clock "$CLOCK" --classify "$CLASSIFY" \
      --book-source reconstruct --progress auto \
      --mode descriptive --interval "$INTERVAL" --n-boot 0 \
      --max-workers "$PW" \
      --save-pickle "$PILOT_DIR/pilot_sessions.pkl" \
      --output-dir "$PILOT_DIR"
  fi
  "$PY" - "$PILOT_DIR/pilot_sessions.pkl" "$CROSS_MAX" <<'PYEOF' || exit 13
import pickle, sys
import numpy as np
import onset_response as onr

path, thr = sys.argv[1], float(sys.argv[2])
sessions = pickle.load(open(path, "rb"))
print("\n  crossed-book gate (bid1 > ask1; impossible in a real book -> reconstruction error)")
print("    %-12s %-10s %10s %10s" % ("date", "regime", "SPY", "ES"))
worst = 0.0
for date, regime, df in sessions:
    x = {a: onr._crossed_frac(df, a) for a in ("SPY", "ES")}
    worst = max([worst] + [v for v in x.values() if np.isfinite(v)])
    print("    %-12s %-10s %9.2f%% %9.2f%%" % (str(date)[:10], regime, 100 * x["SPY"], 100 * x["ES"]))
if not np.isfinite(worst):
    raise SystemExit("could not measure crossing (no bid1/ask1 columns) -- inspect the frames")
print("\n    worst = %.2f%%   bar = %.2f%%" % (100 * worst, 100 * thr))
if worst > thr:
    raise SystemExit(
        "PILOT GATE FAILED: reconstruction on this era crosses above the bar.\n"
        "  The main run is NOT started -- every state and transmission estimate would inherit this.\n"
        "  Decompose within-venue vs cross-venue (verify_crossing) before resubmitting.")
print("    GATE PASSED -- proceeding to the main run.")
PYEOF
else
  step "PHASE 2/3  pilot SKIPPED (RUN_PILOT=0)"
fi

# ---------------------------------------------------------------------------------------------
# PHASE 3 -- main run. --auto-regime WITH --vol-csv is the point: labels come from the SAME
# ex-ante series that chose the days. Without --vol-csv the driver falls back to per-session
# realized vol -- ex-post, and exactly what this leg exists to remove.
# ---------------------------------------------------------------------------------------------
step "PHASE 3/3  main -- $NSESS sessions @ $INTERVAL, boot $NBOOT x $BOOT_WORKERS"
read -r -a HOOKS <<< "$HOOKS_STR"
read -r -a EXTRA <<< "$EXTRA_FLAGS"
"$PY" run_contagion.py \
  --source extract --dates "$DATES" \
  --n-levels "$NLEVELS" --clock "$CLOCK" --classify "$CLASSIFY" \
  --book-source reconstruct --progress auto \
  --auto-regime --vol-csv "$VOL_CSV" \
  --mode both --interval "$INTERVAL" \
  --lag-criterion "$LAGCRIT" --max-lags "$MAXLAGS" \
  --outcome "$OUTCOME" \
  --event-categories "$EVENT_CATEGORIES" --default-release-et "$RELEASE_ET" \
  --n-boot "$NBOOT" \
  --max-workers "$WORKERS" --boot-workers "$BOOT_WORKERS" \
  "${HOOKS[@]}" "${EXTRA[@]}" \
  --output-dir "$MAIN_DIR" || fail 14 "main run failed -- see $LOG"

step "JOB COMPLETE"
cat <<EOF
  selection report : $OUT_ROOT/selection_${STAMP}.md
  extraction set   : $UNIVERSE  ($NSESS sessions)
  manuscript tables: $MAIN_DIR/contagion_manuscript_tables.md
                     $MAIN_DIR/contagion_manuscript_tables.tex
  run summary      : $MAIN_DIR/contagion_run_summary.md
  full log         : $LOG
EOF
for f in "$MAIN_DIR/contagion_manuscript_tables.md" "$MAIN_DIR/contagion_run_summary.md"; do
  [ -s "$f" ] || echo "  WARNING: expected output missing or empty: $f"
done
exit 0
