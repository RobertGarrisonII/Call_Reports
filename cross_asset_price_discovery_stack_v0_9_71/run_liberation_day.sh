#!/usr/bin/env bash
# =============================================================================
# run_liberation_day.sh
#   SPY/ES liquidity-spillover & risk-contagion analysis for the April 2025
#   "Liberation Day" tariff shock (Executive Order 14257).
#
#   Extraction is LAKEQUERY-ONLY: both legs are put on one GPS *receipt* clock —
#   SPY direct (MBO order feeds + the IEX MBP feed), and the front-month E-mini
#   reconstructed from CME Globex MBO messages (price_scale 0.01). Snapshot
#   (mstbook-query) is used only for the one-off validation stage below.
#
# WHY THE WINDOW IS WHAT IT IS
#   EO 14257 was announced ~16:00 ET on 2025-04-02, AFTER the cash close, so it
#   prices on 2025-04-03 (SPY ~ -4.84%) and 2025-04-04 (~ -5.97%). The 90-day
#   tariff PAUSE on 2025-04-09 (~ +9.52%) is a SEPARATE policy event; including
#   it contaminates a clean single-shock study, so the window ENDS 2025-04-08.
#   Pre-event burn-in starts 2025-03-24 — after the ESH5->ESM5 roll (~Mar 20),
#   so the E-mini front month is ESM5 across the entire window.
#
# USAGE
#   ./run_liberation_day.sh validate   # one-off: reconstruct-vs-snapshot check (do while mstbook-query lives)
#   ./run_liberation_day.sh pilot      # 1 day, descriptive, fast — RUN FIRST, eyeball the diagnostics
#   ./run_liberation_day.sh main       # full window @ 1s: causal + descriptive + mean_variance, bootstrapped
#   ./run_liberation_day.sh robust     # full window @ 100ms robustness lens
#   ./run_liberation_day.sh all        # main then robust (only after a clean pilot)
#   (no arg defaults to 'main'; run 'pilot' first if you want the RAM probe)
#
# Heavy I/O: 'main'/'robust' reconstruct both books for ~11 full sessions. Run
# under tmux/screen or nohup. Override any setting via environment variables.
# =============================================================================
set -euo pipefail

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  PRIMARY KNOB — MIDAS submits this script with NO arguments, so choose the ║
# ║  stage HERE by editing the line below (env var STAGE=... also overrides).  ║
# ╠══════════════════════════════════════════════════════════════════════════╣
# ║   pilot    single PILOT_DATE probe — fast, measures RAM, suggests workers  ║
# ║   main     FULL window START..END @ 1s, bootstrapped  <-- the real run     ║
# ║   robust   FULL window @ 100ms robustness lens                            ║
# ║   all      main then robust                                                ║
# ║   validate one-off reconstruct-vs-snapshot check                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝
STAGE="${STAGE:-main}"       # <<<<<  default is the FULL run; set STAGE=pilot for the RAM-probe first  >>>>>
[ "${1:-}" ] && STAGE="$1"   # (a positional arg still overrides, for interactive shells)

# ---- configuration (edit / export to suit the workbench) --------------------
# This script ships inside the package (next to run_contagion.py), so default the working directory to
# the script's OWN location. That is robust to $HOME / username (here $HOME=/home/workbench, not the
# package owner), to symlinks via the cd/pwd idiom, and to being launched from any directory.
# Override with  PKG_DIR=/abs/path/to/cross_asset_price_discovery_stack ./run_liberation_day.sh ...
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="${PKG_DIR:-$SCRIPT_DIR}"
PY="${PY:-python3}"
OUT_ROOT="${OUT_ROOT:-output}"

# Event window
PILOT_DATE="${PILOT_DATE:-2025-04-03}"      # biggest single-day move; representative for the pilot
START="${START:-2025-03-24}"                # burn-in start (post ESH5->ESM5 roll)
END="${END:-2025-04-08}"                    # clean single-shock end (excludes the 04-09 pause)
CALM_DATE="${CALM_DATE:-2025-03-25}"        # a calm control date for the validation stage

# Shared estimation parameters
NLEVELS="${NLEVELS:-10}"                     # depth ladder for the reconstructed books
CLOCK="${CLOCK:-receipt}"                    # GPS source-capture clock: cross-asset lead-lag integrity
CLASSIFY="${CLASSIFY:-aggressor}"            # CME has aggressorside; SPY equities fall back to the tick rule
EVENT_CLASSES="${EVENT_CLASSES:-TRADE_POLICY}"  # market_shocks tags 04-03/04-09 TRADE_POLICY / TARIFF
RELEASE_ET="${RELEASE_ET:-09:30}"            # overnight-priced shock -> treat the cash open as the release
LAGCRIT="${LAGCRIT:-bic}"                    # data-driven VECM lag length
OUTCOME="${OUTCOME:-auc_decay}"              # liquidity-curve outcome (AUC under the decay-weighted cost curve)
NBOOT="${NBOOT:-1000}"                       # block-bootstrap reps for inference

# ---- multi-event mode (STAGE=multievent) ------------------------------------
# Pool several 2025 exogenous shocks so the design has enough TREATED day-clusters for a valid
# wild-cluster bootstrap (the single Liberation-Day window leans on one event). We extract ONLY the
# event days plus a calm control pool -- NOT a contiguous range (that would pull months of tape);
# the event-study driver matches each event's controls ex-ante from the pool.
ME_EVENTS="${ME_EVENTS:-2025-04-03,2025-04-09,2025-06-13,2025-06-23}"        # tariffs (announce+pause) + Iran (Israel 06-13, US 06-23)
ME_CONTROLS="${ME_CONTROLS:-2025-03-26,2025-03-27,2025-05-01,2025-05-02,2025-05-30,2025-07-08,2025-07-09,2025-07-10}"  # calm control pool
ME_CLASSES="${ME_CLASSES:-TRADE_POLICY,GEOPOLITICAL}"                        # set to one class for homogeneous identification

# Optional analysis hooks (set HOOKS_STR="" to disable). --auction = open/close cross
# imbalance + SPY-open -> ES-move linkage; --copula = tail-dependence / liquidity-conditional joint crash.
read -r -a HOOKS <<< "${HOOKS_STR:-"--auction --copula"}"
# Any extra flags you want appended to the contagion runs (e.g. EXTRA_FLAGS="--price wmid_orth")
read -r -a EXTRA <<< "${EXTRA_FLAGS:-}"

# ---- hardware auto-calibration ----------------------------------------------
# Detect vCPUs and RAM via system calls and size the knobs to the node. Every value stays
# overridable by exporting it (e.g. WORKERS=8 ./run_liberation_day.sh main). Two budgets:
#   * EXTRACTION is memory-bound -- each worker holds a full day of consolidated SPY+ES MBO
#     messages in pandas, and the crash days (04-03/04-04) are the RSS spikes. So WORKERS is sized
#     off RAM (not cores) and capped at the #sessions in the window (12): floor((RAM-RESERVE)/peak).
#   * The BOOTSTRAP is NOT memory-bound -- it runs after extraction with the session frames resident
#     and shared read-only across threads, and each refit is numpy/pandas C code that releases the
#     GIL. So BOOT_WORKERS = ALL detected cores (--boot-workers); this is what makes a high vCPU
#     count finally pay off, instead of sitting idle behind the 12-session extraction cap.
_min()   { [ "$1" -le "$2" ] && echo "$1" || echo "$2"; }
_clamp() { local v=$1 lo=$2 hi=$3; [ "$v" -lt "$lo" ] && v=$lo; [ "$v" -gt "$hi" ] && v=$hi; echo "$v"; }
_detect_cores() {
  # Take the MAX across signals. Plain `nproc` honors CPU affinity AND the cgroup CPU quota, so on
  # a throttled login/interactive shell (common on a managed workbench like MIDAS) it can report 1
  # even on a 64-core node -- which would silently serialize extraction AND the bootstrap. `nproc
  # --all`, `getconf _NPROCESSORS_ONLN`, and /proc/cpuinfo report the PHYSICAL online count
  # regardless of the throttle -- the right target for a job that lands on the full node. (If your
  # *job* is genuinely CPU-capped, override: CORES=8 ./run_liberation_day.sh main.)
  local n=1 c
  for c in "$(nproc --all 2>/dev/null || true)" \
           "$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)" \
           "$(grep -c '^processor' /proc/cpuinfo 2>/dev/null || true)" \
           "$(nproc 2>/dev/null || true)"; do
    case "$c" in ''|*[!0-9]*) c=0 ;; esac
    [ "$c" -gt "$n" ] && n="$c"
  done
  echo "$n"
}
_cgroup_mem_gb() {
  # The process's memory-cgroup limit in GiB, or empty if none/unlimited. A SIGKILL(-9) on a
  # worker is exactly what a cgroup OOM-killer does, so the binding budget may be a cgroup cap
  # well below the node's /proc/meminfo total -- size off the smaller of the two. Best-effort and
  # defensive: every probe is guarded so a missing file never trips `set -e`/`pipefail`.
  local lim="" path g
  [ -f /sys/fs/cgroup/memory.max ] && lim="$(cat /sys/fs/cgroup/memory.max 2>/dev/null || true)"   # cgroup v2 (unified)
  if [ -z "$lim" ] || [ "$lim" = "max" ]; then                                                     # the process's own (nested) v2 path
    if [ -r /proc/self/cgroup ]; then
      path="$(awk -F: '/^0::/{print $3}' /proc/self/cgroup 2>/dev/null | head -1 || true)"
      if [ -n "$path" ] && [ -f "/sys/fs/cgroup${path}/memory.max" ]; then
        lim="$(cat "/sys/fs/cgroup${path}/memory.max" 2>/dev/null || true)"
      fi
    fi
  fi
  if [ -z "$lim" ] || [ "$lim" = "max" ]; then                                                     # cgroup v1 fallback
    [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ] && lim="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true)"
  fi
  case "$lim" in ''|max|*[!0-9]*) return 0 ;; esac
  g=$(( lim / 1073741824 ))                                                                        # bytes -> GiB
  [ "$g" -ge 1 ] && [ "$g" -lt 1000000 ] && echo "$g"                                              # ignore v1 "unlimited" sentinels
  return 0
}
_meminfo_gb() {
  local kb="" g="0"
  kb="$(awk '/^MemTotal:/{print $2}' /proc/meminfo 2>/dev/null || true)"
  case "$kb" in ''|*[!0-9]*) kb="" ;; esac
  if [ -n "$kb" ]; then
    g=$(( (kb + 1048575) / 1048576 ))                   # kB -> GiB, round up
  else
    g="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}' || true)"
    case "$g" in ''|*[!0-9]*) g=0 ;; esac
  fi
  echo "$g"
}
DETECTED_CORES="$(_detect_cores)"
_MEMINFO_GB="$(_meminfo_gb)"; case "$_MEMINFO_GB" in ''|*[!0-9]*) _MEMINFO_GB=0 ;; esac
_CGROUP_GB="$(_cgroup_mem_gb)"; case "${_CGROUP_GB:-}" in ''|*[!0-9]*) _CGROUP_GB="" ;; esac
if   [ "$_MEMINFO_GB" -ge 1 ] && [ -n "$_CGROUP_GB" ]; then DETECTED_RAM_GB="$(_min "$_MEMINFO_GB" "$_CGROUP_GB")"  # binding = min(node, cgroup)
elif [ "$_MEMINFO_GB" -ge 1 ]; then DETECTED_RAM_GB="$_MEMINFO_GB"
elif [ -n "$_CGROUP_GB" ];     then DETECTED_RAM_GB="$_CGROUP_GB"
else DETECTED_RAM_GB=256; fi
_NPROC_RAW="$(nproc 2>/dev/null || echo 0)"; case "$_NPROC_RAW" in ''|*[!0-9]*) _NPROC_RAW=0 ;; esac

RAM_GB="${RAM_GB:-$DETECTED_RAM_GB}"                    # binding RAM budget = min(node, cgroup); override by exporting
CORES="${CORES:-$DETECTED_CORES}"                       # vCPUs (auto-detected)
PEAK_GB_GUESS="${PEAK_GB_GUESS:-32}"                    # per-worker crash-day RSS guess (the pilot stage measures the real one)
# The worker-sizing FORMULA now lives in autoscale.py -- the SINGLE source shared with the Python
# launchers (run_onset_surface, ...), so shell and Python sizing can no longer drift. The cgroup-aware
# RAM_GB / CORES detected above, and any exported WORKERS / RESERVE_GB / BOOT_WORKERS / PEAK_GB_GUESS
# override, pass through the environment (which autoscale honors), so the old "export to override"
# behavior is preserved exactly.
export RAM_GB CORES PEAK_GB_GUESS
RESERVE_GB="$("$PY" autoscale.py reserve)"                          # ~1/8 RAM held back, clamped [16,48]
WORKERS="$("$PY" autoscale.py workers --sessions 12)"              # 1s extraction = min(memory-derived, 12 sessions, cores)
WORKERS_FINE="${WORKERS_FINE:-$WORKERS}"                            # 100ms: per-worker raw-msg peak ~= 1s, so same count
BOOT_WORKERS="$("$PY" autoscale.py jobs)"                          # bootstrap: not memory-bound -> all cores

echo ">>> auto-calibrated: ${CORES} vCPU / ${RAM_GB} GB"
if [ "$_NPROC_RAW" -ge 1 ] && [ "$_NPROC_RAW" -lt "$CORES" ]; then
  echo ">>>   (this shell's nproc=${_NPROC_RAW} is a cgroup/affinity cap, not the hardware; sized to the ${CORES} physical cores -- export CORES=N to override)"
fi
if [ -n "${_CGROUP_GB:-}" ] && [ "$_MEMINFO_GB" -ge 1 ] && [ "$_CGROUP_GB" -lt "$_MEMINFO_GB" ]; then
  echo ">>>   (memory cgroup caps this job at ${_CGROUP_GB} GB of the node's ${_MEMINFO_GB} GB; sizing workers off the ${_CGROUP_GB} GB cap)"
fi
echo ">>>   extraction WORKERS=${WORKERS} (1s) / ${WORKERS_FINE} (100ms), RESERVE_GB=${RESERVE_GB} GB;  bootstrap BOOT_WORKERS=${BOOT_WORKERS}   [all overridable]"
echo ">>>   per-worker RSS assumed ${PEAK_GB_GUESS} GB; for a crash-heavy window run 'STAGE=pilot' once to measure and pin WORKERS"

# Process-level parallelism + multithreaded BLAS would oversubscribe (W procs x ~32 threads). Every fit
# in this stack is 2-asset (2x2 matrices), so pinning BLAS to 1 thread/process costs ~nothing and avoids
# thrashing. Export before any import. Override by setting these in the environment if you ever need them.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
export PYTHONUNBUFFERED=1            # live-flush stdout so the tee'd log updates during long runs

TS="$(date +%Y%m%d_%H%M%S)"

# GNU time (for the pilot RSS probe); falls back to no instrumentation if absent
if /usr/bin/time -v true >/dev/null 2>&1; then TIMEIT=(/usr/bin/time -v); else TIMEIT=(); fi

# ---- helpers ----------------------------------------------------------------
banner () { printf '\n========== %s ==========\n' "$*"; }

# Flags shared by every contagion extraction run
contagion_common () {
  printf '%s\n' \
    --source extract \
    --n-levels "$NLEVELS" \
    --clock "$CLOCK" \
    --classify "$CLASSIFY" \
    --book-source reconstruct \
    --auto-regime \
    --event-classes "$EVENT_CLASSES" \
    --default-release-et "$RELEASE_ET" \
    --lag-criterion "$LAGCRIT" \
    --progress auto
}

cd "$PKG_DIR"
echo "package: $PKG_DIR"
echo "python : $($PY --version 2>&1)"
echo "stage  : $STAGE"

# ---- stages -----------------------------------------------------------------

run_validate () {
  # One-off reconstruct-vs-snapshot benchmark for the consolidated SPY book, plus ES leadership
  # under each book. Do this while the (sunsetting) mstbook-query snapshot path still exists; a
  # sub-millisecond L1/mid delta is EXPECTED (clock offset) — price/size LEVELS should match.
  local out="$OUT_ROOT/liberation_day_validation"; mkdir -p "$out"
  banner "VALIDATION (reconstruct vs snapshot)"
  "$PY" validate_reconstruction.py \
    --volatile "$PILOT_DATE,2025-04-04" \
    --benchmark "$CALM_DATE" \
    --interval 1s \
    --n-levels "$NLEVELS" \
    --clock "$CLOCK" \
    --classify "$CLASSIFY" \
    --with-dcc \
    --output-dir "$out" 2>&1 | tee "$out/validate_$TS.log"
}

run_pilot () {
  # Single day, descriptive-only, no bootstrap: confirms extraction + reconstruction end-to-end and
  # exercises the --auction/--copula hooks fast. One worker, so the measured peak RSS is the per-session
  # footprint you multiply against total RAM (RAM_GB). Inspect the first-session diagnostic (see checklist).
  local out="$OUT_ROOT/liberation_day_pilot"; mkdir -p "$out"
  banner "PILOT ($PILOT_DATE, descriptive, 1 worker, RSS-probed)"
  "${TIMEIT[@]}" "$PY" run_contagion.py \
    $(contagion_common) \
    --date-range "${PILOT_DATE}:${PILOT_DATE}" \
    --mode descriptive \
    --interval 1s \
    --outcome "$OUTCOME" \
    --n-boot 0 \
    --max-workers 1 \
    "${HOOKS[@]}" "${EXTRA[@]}" \
    --output-dir "$out" 2>&1 | tee "$out/pilot_$TS.log"

  # Turn the measured single-session peak RSS into a memory-safe --max-workers for the full window.
  local peak_kb peak_gb sugg
  peak_kb="$(grep -i 'Maximum resident set size' "$out/pilot_$TS.log" | grep -oE '[0-9]+' | tail -1 || true)"
  if [ -n "${peak_kb:-}" ]; then
    peak_gb="$(awk "BEGIN{printf \"%.1f\", $peak_kb/1048576}")"
    sugg="$(awk "BEGIN{h=$RAM_GB-$RESERVE_GB; w=int(h/($peak_kb/1048576)); if(w>12)w=12; if(w<1)w=1; print w}")"
    printf '\n>>> measured single-session peak RSS: %s GB  ->  suggested --max-workers ~ %s (cap 12)\n' \
           "$peak_gb" "$sugg"
    printf '>>> set it for the full run, e.g.:  WORKERS=%s ./run_liberation_day.sh main\n' "$sugg"
  else
    echo ">>> (install GNU 'time' for an automatic RSS->worker suggestion; else watch 'free -g' during main)"
  fi
  cat <<'CHECK'

------------------------------------------------------------------
PILOT complete. Before running 'main', confirm in the log:
  * E-mini front month resolved to ESM5 (NOT ESH5 / ESU5)
  * ES/SPY median-mid ratio ~ 10  (ES in index points ~5700 vs SPY ~570)
      -> if ES prints ~570000, the 0.01 futures_scale was not applied
  * mbp_feeds includes iex_deep ; mbo_feeds includes
    bats_edgx, xdp_arca_integrated, total_view
  * run-completion manifest lists auction_imbalance.{csv,md},
    auction_linkage.csv, copula_garch_summary.md, copula_selection.csv
Then:  ./run_liberation_day.sh main
------------------------------------------------------------------
CHECK
}

run_main () {
  # Full window @ 1s: causal (event-study + spillover) + descriptive + mean_variance, bootstrapped.
  # --upgraded runs the re-floored v0.6-0.8 analysis (day-clustered/wild-cluster inference,
  # Hayashi-Yoshida + Rigobon leadership, the two-axis liquidity-stress conditioner, the event-clock
  # study, and the LEGACY before/after tables) IN-PROCESS on the SAME in-memory sessions -- no pickle,
  # no second extraction/load. Its output lands under $out/upgraded/.
  local out="$OUT_ROOT/liberation_day_1s"; mkdir -p "$out"
  banner "MAIN ($START -> $END @ 1s, --max-workers $WORKERS, --boot-workers $BOOT_WORKERS, +upgraded)"
  "$PY" run_contagion.py \
    $(contagion_common) \
    --date-range "${START}:${END}" \
    --mode both \
    --interval 1s \
    --max-lags 12 \
    --outcome "$OUTCOME" \
    --n-boot "$NBOOT" \
    --max-workers "$WORKERS" \
    --boot-workers "$BOOT_WORKERS" \
    --upgraded \
    "${HOOKS[@]}" "${EXTRA[@]}" \
    --output-dir "$out" 2>&1 | tee "$out/main_$TS.log"
  echo "MAIN done -> $out  (contagion tables + re-floored analysis in $out/upgraded/)"
}

run_robust () {
  # 100ms robustness lens: finer grid, longer max-lag horizon to cover the same wall-clock lead-lag.
  # --upgraded mirrors the re-floored analysis here too (in-process, no pickle); --upgraded-skip dcc
  # drops the cDCC stage, which is the expensive one at a 100ms grid.
  local out="$OUT_ROOT/liberation_day_100ms"; mkdir -p "$out"
  banner "ROBUST ($START -> $END @ 100ms, --max-workers $WORKERS_FINE, --boot-workers $BOOT_WORKERS, +upgraded)"
  "$PY" run_contagion.py \
    $(contagion_common) \
    --date-range "${START}:${END}" \
    --mode both \
    --interval 100ms \
    --max-lags 80 \
    --outcome "$OUTCOME" \
    --n-boot "$NBOOT" \
    --max-workers "$WORKERS_FINE" \
    --boot-workers "$BOOT_WORKERS" \
    --upgraded --upgraded-skip dcc \
    "${HOOKS[@]}" "${EXTRA[@]}" \
    --output-dir "$out" 2>&1 | tee "$out/robust_$TS.log"
  echo "ROBUST done -> $out  (contagion tables + re-floored analysis in $out/upgraded/)"
}

run_multievent () {
  # Pooled multi-event study: more treated days -> enough day-clusters for a valid wild-cluster
  # bootstrap (the single Liberation-Day window leans on ONE event). Reuses contagion_common with a
  # local EVENT_CLASSES override and a curated --dates list (event days + a calm control pool, not a
  # contiguous range). --upgraded runs the re-floored analysis in-process on the same sessions.
  local EVENT_CLASSES="$ME_CLASSES"                 # override the single-event class for this stage
  local out="$OUT_ROOT/multievent_1s"; mkdir -p "$out"
  banner "MULTI-EVENT (events=$ME_EVENTS | controls=$ME_CONTROLS | classes=$ME_CLASSES, +upgraded)"
  echo ">>> NOTE: pooling across event TYPES (tariff + geopolitical) assumes a COMMON treatment effect"
  echo ">>>       and mixes timing regimes (overnight / intraday / weekend). For homogeneous"
  echo ">>>       identification run ONE class:  ME_CLASSES=TRADE_POLICY STAGE=multievent ./run_liberation_day.sh"
  echo ">>>       Pooling buys cluster count for the bootstrap, not identification cleanliness."
  "$PY" run_contagion.py \
    $(contagion_common) \
    --dates "${ME_EVENTS},${ME_CONTROLS}" \
    --mode both \
    --interval 1s \
    --max-lags 12 \
    --outcome "$OUTCOME" \
    --n-boot "$NBOOT" \
    --max-workers "$WORKERS" \
    --boot-workers "$BOOT_WORKERS" \
    --upgraded \
    "${HOOKS[@]}" "${EXTRA[@]}" \
    --output-dir "$out" 2>&1 | tee "$out/multievent_$TS.log"
  echo "MULTI-EVENT done -> $out  (contagion tables + re-floored analysis in $out/upgraded/)"
}

run_onset_surface () {
  # Cross-event ONSET stress-surface on the impact-blind scheduled backbone (FOMC by default): the 2-D
  # liquidity-spiral interaction across the LEG TRIPLE (leg-neutral joint round-trip cost / ES book /
  # SPY book) on the NOISE-ROBUST transmission. The scheduled calendar drives the date universe, so
  # sessions are extracted only for those release days. Tune via env: ONSET_RANGE, ONSET_CATS,
  # ONSET_PRE, ONSET_POST, ONSET_NOTIONAL.
  local out="$OUT_ROOT/onset_surface_1s"; mkdir -p "$out"
  local range="${ONSET_RANGE:-2023-01-01:2025-12-31}"
  local cats="${ONSET_CATS:-FOMC}"
  banner "ONSET SURFACE (leg triple, robust transmission; backbone=$cats, window=$range)"
  "$PY" run_onset_surface.py \
    --source extract \
    --date-range "$range" \
    --categories "$cats" \
    --n-levels "$NLEVELS" \
    --pre-steps "${ONSET_PRE:-600}" --post-steps "${ONSET_POST:-600}" \
    --notional "${ONSET_NOTIONAL:-1000000}" \
    --max-workers "$WORKERS" --onset-jobs "$BOOT_WORKERS" \
    --output-dir "$out" 2>&1 | tee "$out/onset_surface_$TS.log"
  echo "ONSET SURFACE done -> $out  (summary above; panel in $out/onset_surface_panel.csv)"
}

case "$STAGE" in
  validate)   run_validate ;;
  pilot)      run_pilot ;;
  main)       run_main ;;
  robust)     run_robust ;;
  multievent) run_multievent ;;
  onset)      run_onset_surface ;;
  all)        run_main; run_robust ;;
  *) echo "unknown stage '$STAGE' (use: validate | pilot | main | robust | multievent | onset | all)" >&2; exit 2 ;;
esac

# -----------------------------------------------------------------------------
# SEPARATE STUDY (not Liberation Day): the March 2020 market-wide circuit breakers
# are a different shock class. Run them on their own, e.g.:
#   STAGE-equivalent command:
#   python run_contagion.py --source extract --date-range 2020-03-09:2020-03-18 \
#     --mode both --auto-regime --event-categories MWCB --default-release-et 09:30 \
#     --lag-criterion bic --max-lags 12 --outcome auc_decay --n-boot 1000 \
#     --n-levels 10 --clock receipt --classify aggressor --auction --copula \
#     --max-workers "$WORKERS" --boot-workers "$BOOT_WORKERS" \
#     --output-dir output/mwcb_2020_1s
# -----------------------------------------------------------------------------
