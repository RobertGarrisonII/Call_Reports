#!/usr/bin/env bash
# ==============================================================================
# run_paper_replication.sh
#
# Replicate Garrison, Jain & Paddrik, "Cross-Asset Tandem Trading and
# Extraordinary Volatility", using the CORRECTED reconstruction and the
# corrected inference. One command, seven stages, two of which are gates.
#
# What differs from the original pipeline, and why the order below matters:
#
#   1. The book reconstruction was silently wrong. mstbook_loader pruned the
#      'sequencenumber' column before lob_reconstruct saw it, so the replay fell
#      back to clock-only intra-feed ordering, a Modify could resurrect an
#      already-Cancelled order, and the phantom pinned the top crossed on ~99.9%
#      of snapshots. Nothing raised. STAGE 3 therefore GATES on debug_crossing:
#      no table is produced from a book that fails its own invariant.
#
#   2. The Table 5 null was rejected by the MARGINALS, not by cross-market
#      trading. Binomial(n, 1/2) bundles "each market is a fair coin" with "the
#      markets are independent", and it is the first that fails -- the ETF is
#      directional in 68.7% of baseline seconds against a 2.4% prediction. That
#      inflates all four corner cells with no cross-market linkage at all.
#      STAGE 4 reports the marginal-preserving benchmark and the corner log odds
#      ratio alongside the published numbers.
#
#   3. The Table 7 null is frequency-dependent. The per-second calibration
#      (n = 505 / 112) gives 0.4% in each corner, but a ten-millisecond bar
#      (~5.1 / ~1.1 orders) gives ~23% and action time ~50%. STAGE 4 computes the
#      null at the actual per-bar counts for every aggregation reported.
#
#   4. Delta-rho is a grid-sampled Pearson correlation, which the Epps effect
#      attenuates -- by an amount that depends on trading intensity, itself a
#      regressor in Eq. (5). STAGE 5 reports Table 9 on both that and the
#      Hayashi-Yoshida correlation, with the difference as the artifact estimate.
#
#   5. The SVAR lag length was a stated compromise, not a choice: footnote 17
#      records AIC pointing at 60 lags and 6 being used because the full model
#      would not run there. That makes p a researcher degree of freedom, so
#      STAGE 4c picks it by criterion (--n-lags bic, the default) ONCE on the
#      pooled SVAR frame, prints the whole AIC/BIC/HQ table, and reuses that one
#      number everywhere downstream -- so the p in the table note is the p that
#      was fitted. BIC rather than AIC: AIC is not consistent for lag order and
#      on ~23k-bar intraday samples runs away (the paper's own 60); the log(T)
#      penalty is what keeps it finite. Pass --n-lags 6 to reproduce the paper.
#
#   6. Extraction is the longest and least reliable stage (hours of vendor I/O),
#      and it used to be all-or-nothing in both directions: one un-retried
#      lakequery rc=1 aborted the batch and discarded every finished day, while a
#      FAILED fetch of a single message type was swallowed into an empty frame
#      and the resulting fabricated book -- crossed on up to 100% of snapshots --
#      went into the dataset with only a warning that did not name the date. Now
#      a vendor error is retried, a failed critical fetch refuses that session, a
#      failed session cannot take the batch down, every session is QC'd against
#      the no-crossing invariant before it is saved, and each good day is cached
#      so a re-run pays only for the days that are missing (--extract-cache).
#
# Usage
#   ./run_paper_replication.sh                          # demo: runs anywhere, no data needed
#   ./run_paper_replication.sh --source extract         # full paper sample via MayStreet
#   ./run_paper_replication.sh --source load --pickle output/frames_*.pkl
#   ./run_paper_replication.sh --stages 1,4,5           # re-run selected stages
#   ./run_paper_replication.sh --dry-run                # print the commands only
#   ./run_paper_replication.sh --n-lags 6               # the paper's fixed lag, for comparison
#   ./run_paper_replication.sh --n-lags bic --pmax 20   # data-driven lag (default), wider search
#   ./run_paper_replication.sh --source extract --qc-action drop   # exclude crossed sessions
#   ./run_paper_replication.sh --source extract --extract-cache /scratch/sessions
#   ./run_paper_replication.sh --paper-sample           # the published 2014-2017 universe
#   ./run_paper_replication.sh --volatile D,D,... --baseline D,D,...   # your own sample
#
# Environment
#   PYTHON=python3            interpreter (default: python3)
#   N_JOBS=8                  bootstrap/extraction workers (default: all cores)
#   MST_LAKEQUERY_RETRIES=3   vendor-query attempts before a fetch is believed to have failed
#   MST_LAKEQUERY_BACKOFF=5   seconds before the first retry (doubles each attempt)
# ==============================================================================
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
SOURCE="demo"
PICKLE=""
OUT_ROOT="output"
INTERVAL="1s"
FINE_INTERVAL="10ms"
N_BOOT=499
CORR_WINDOW=100
N_LAGS="bic"          # integer, or an information criterion: bic | aic | hq
PMAX=12
N_LAGS_INT=""        # resolved integer, filled in by STAGE 4c
T9_DCC=""            # "--with-dcc", set by STAGE 4c if the lag order tracks the corr window
STAGES="0,1,2,3,4,5,6,7"
DRY=0
QUICK=0
NJ="${N_JOBS:-}"
# Cache extracted sessions OUTSIDE the per-run output folder: the point is that a re-run (new
# RUN_ID) reuses days a previous run already paid for. One day is 10-25 minutes of vendor I/O.
CACHE_DIR="${EXTRACT_CACHE:-${OUT_ROOT}/extract_cache}"
RESUME=1
QC_ACTION="warn"      # warn | drop | raise -- what to do with a session whose book crosses

# ── the sample ────────────────────────────────────────────────────────────────
# RE-SAMPLED onto 2022-2026 so the paper speaks to the current market. The
# construction rule is unchanged from Appendix A.1: volatile = the largest
# intraday-range days in the window, baseline = the same weekday roughly one
# year prior to each (paired POSITIONALLY, list against list), MWCB = the four
# March-2020 halt days, which are fixed by history and do not move.
#
# Appendix A.1, and every "2014-2017" / "N days" statement in the text, has to
# change with this. STAGE 0 validates the universe before anything is extracted:
# a weekend, an exchange holiday or a half day costs 10-25 minutes of vendor I/O
# each to return an empty or one-legged frame. That is not hypothetical -- the
# first real run included 2026-01-19, which is MLK Day (NYSE closed, CME Globex
# on an abbreviated session), and it extracted to `median SPY=nan ES=6913.75`.
#
# Pass --paper-sample to restore the published 2014-2017 universe verbatim.
VOLATILE="2024-12-18,2026-06-05,2025-10-10,2024-09-03,2025-04-03,2024-08-05,2024-07-24,2025-01-07,2023-03-09,2026-01-19"
BASELINE="2023-12-20,2025-06-13,2024-10-18,2023-09-05,2024-04-04,2023-08-07,2023-07-19,2024-01-29,2022-03-24,2025-01-13"
MWCB="2020-03-09,2020-03-12,2020-03-16,2020-03-18"
# the published sample, Appendix Table A.1 (--paper-sample)
PAPER_VOLATILE="2015-03-18,2015-10-02,2016-01-08,2016-01-27,2016-06-24,2015-08-21,2015-08-24,2015-09-01,2016-01-13,2016-01-20"
PAPER_BASELINE="2014-03-26,2014-10-10,2015-01-09,2015-02-04,2015-06-26,2014-08-22,2014-08-25,2014-09-02,2015-01-14,2015-01-21"
ALLOW_BAD_DATES=0

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source)        SOURCE="$2"; shift 2 ;;
    --pickle)        PICKLE="$2"; shift 2 ;;
    --out-dir)       OUT_ROOT="$2"; shift 2 ;;
    --interval)      INTERVAL="$2"; shift 2 ;;
    --fine-interval) FINE_INTERVAL="$2"; shift 2 ;;
    --n-boot)        N_BOOT="$2"; shift 2 ;;
    --corr-window)   CORR_WINDOW="$2"; shift 2 ;;
    --n-lags)        N_LAGS="$2"; shift 2 ;;
    --pmax)          PMAX="$2"; shift 2 ;;
    --volatile)      VOLATILE="$2"; shift 2 ;;
    --baseline)      BASELINE="$2"; shift 2 ;;
    --mwcb)          MWCB="$2"; shift 2 ;;
    --stages)        STAGES="$2"; shift 2 ;;
    --extract-cache) CACHE_DIR="$2"; shift 2 ;;
    --no-cache)      CACHE_DIR=""; shift ;;
    --no-resume)     RESUME=0; shift ;;
    --qc-action)     QC_ACTION="$2"; shift 2 ;;
    --paper-sample)  VOLATILE="$PAPER_VOLATILE"; BASELINE="$PAPER_BASELINE"; shift ;;
    --allow-bad-dates) ALLOW_BAD_DATES=1; shift ;;
    --dry-run)       DRY=1; shift ;;
    --quick)         QUICK=1; N_BOOT=49; shift ;;
    -h|--help)       sed -n '2,45p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

RUN_ID="$(date +%Y%m%d_%H%M%S)"
OUT="${OUT_ROOT}/replication_${RUN_ID}"
LOG="${OUT}/replication.log"
mkdir -p "$OUT"

# Size the pools from the MACHINE via autoscale, not from a default buried in a library. Leaving
# --n-jobs unset used to fall through to inference.py's os.cpu_count(), which ignores CPU affinity
# and so reports the whole node's cores inside a pinned container. autoscale takes the max across
# cpu_count / /proc/cpuinfo / sched_getaffinity and honours the same env overrides the other shell
# drivers use, so bash and Python cannot disagree about how wide to run.
N_SESS=$(echo "${VOLATILE},${BASELINE},${MWCB}" | tr ',' '\n' | grep -c . || echo 24)
AS_CORES="$($PY autoscale.py cores 2>/dev/null || echo 1)"
AS_RAM="$($PY autoscale.py ram 2>/dev/null || echo 0)"
AS_EXTRACT="$($PY autoscale.py workers --sessions "$N_SESS" 2>/dev/null || echo 1)"
[ -z "$NJ" ] && NJ="$($PY autoscale.py jobs 2>/dev/null || echo 1)"
# --n-jobs is accepted by run_table9_both_ways.py ONLY. run_analysis.py sizes its bootstrap from
# autoscale.cpu_jobs(), which honours BOOT_WORKERS, so the width is exported rather than passed as a
# flag that argparse would reject. STAGE 6 passed --n-jobs to run_analysis and had therefore never
# completed on ANY source; the earlier partial runs (--stages 0,1,4) simply never reached it.
JOBS_FLAG=""; [ -n "$NJ" ] && JOBS_FLAG="--n-jobs $NJ"
[ -n "$NJ" ] && export BOOT_WORKERS="$NJ"

# Resolve the frames path HERE, not inside STAGE 2. Stages 4c/5/6 all read it, and --stages lets
# any of them run without STAGE 2 -- in which case the old in-stage assignment left FRAMES pointing
# at a file that was never written, and every downstream stage failed on a missing pickle.
FRAMES="${OUT}/frames.pkl"
if [ "$SOURCE" = "load" ] && [ -n "$PICKLE" ]; then FRAMES="$PICKLE"; fi

# BLAS must stay single-threaded: the bootstrap parallelises over replicates, and
# nested BLAS threads oversubscribe every core and slow the whole run down.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

have_stage() { case ",${STAGES}," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
say()  { printf '\n\033[1m== %s\033[0m\n' "$*" | tee -a "$LOG"; }
info() { printf '   %s\n' "$*" | tee -a "$LOG"; }
run()  {
  printf '   + %s\n' "$*" | tee -a "$LOG"
  [ "$DRY" -eq 1 ] && return 0
  # shellcheck disable=SC2068
  if ! "$@" >>"$LOG" 2>&1; then
    printf '   \033[31mFAILED\033[0m (see %s)\n' "$LOG"
    printf '   --- last 15 lines ---\n'; tail -n 15 "$LOG" | sed 's/^/   | /'
    return 1
  fi
}
# for the stages whose OUTPUT is the deliverable: show it on the console as well as the log
run_show() {
  printf '   + %s\n' "$*" | tee -a "$LOG"
  [ "$DRY" -eq 1 ] && return 0
  # shellcheck disable=SC2068
  if ! "$@" 2>&1 | tee -a "$LOG"; then
    printf '   \033[31mFAILED\033[0m (see %s)\n' "$LOG"; return 1
  fi
}
# same, but the exit code is the result rather than a failure (used by the gates)
run_rc() { printf '   + %s\n' "$*" | tee -a "$LOG"; [ "$DRY" -eq 1 ] && return 0
           set +e; "$@" >>"$LOG" 2>&1; local rc=$?; set -e; return $rc; }

echo "cross-asset replication  run=${RUN_ID}  source=${SOURCE}  out=${OUT}" | tee "$LOG"
echo "interval=${INTERVAL} (+${FINE_INTERVAL})  corr_window=${CORR_WINDOW}  n_lags=${N_LAGS} (pmax=${PMAX})  n_boot=${N_BOOT}" | tee -a "$LOG"

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 0 — preflight
# ══════════════════════════════════════════════════════════════════════════════
if have_stage 0; then
  say "STAGE 0  preflight"
  info "python: $($PY -V 2>&1)"
  run $PY - <<'EOF'
import sys
import numpy, pandas
print("numpy", numpy.__version__, "| pandas", pandas.__version__)
try:
    import scipy; print("scipy", scipy.__version__)
except ImportError:
    sys.exit("scipy is required (pip install scipy)")
import lob_reconstruct, mstbook_loader, correlation_svar, tandem_order_flow, paper_tables  # noqa: F401
print("stack imports OK")
EOF
  info "cores=${AS_CORES}  ram=${AS_RAM}GiB  extraction_workers=${AS_EXTRACT} (${N_SESS} sessions)  cpu_jobs=${NJ}"
  info "  sizing comes from autoscale.py; override with CORES/RAM_GB/WORKERS/BOOT_WORKERS/PEAK_GB_GUESS"
  if [ "${AS_EXTRACT}" -lt "${AS_CORES}" ] 2>/dev/null && [ "${AS_EXTRACT}" -lt "${N_SESS}" ] 2>/dev/null; then
    info "  NOTE: extraction is capped at ${AS_EXTRACT} by the per-worker MEMORY budget"
    info "  (PEAK_GB_GUESS), not by cores. STAGE 2 measures the real peak and prints the value"
    info "  to export -- that guess is the one number here that is not measured."
  fi

  # Every flag this script passes must exist in the tool that receives it. argparse exits 2 on an
  # unrecognized option, so a typo kills its stage the moment it is REACHED -- STAGE 6 passed
  # --n-jobs to run_analysis.py (which has no such option) and had therefore never completed on any
  # source. A dry run cannot catch it: printing a command does not parse it. One --help per tool.
  # Version consistency is ADVISORY here, deliberately. It belongs to release hygiene, not to
  # correctness: a mislabelled archive is a real defect but it cannot make a table wrong, and the
  # STAGE 1 gate is reserved for corrections whose absence is SILENT in the output. Putting it in
  # that gate aborted a run at minute two over three old zips sitting in the working directory,
  # which is what a working directory looks like. package.sh runs it with --strict, which is where
  # an unswept archive can actually be shipped by mistake.
  if run_rc $PY check_version.py; then :; else
    info "  (advisory) version metadata is inconsistent -- see above. The run continues; fix before"
    info "  packaging, where package.sh gates on it."
  fi
  if run_rc $PY check_driver_flags.py "$0" --quiet; then :; else
    echo "" | tee -a "$LOG"
    echo "DRIVER FLAG CHECK FAILED -- see the log. A stage would die on argparse the moment it ran." | tee -a "$LOG"
    tail -n 20 "$LOG" | sed 's/^/   | /' | tee -a /dev/null
    exit 1
  fi

  # ── the sample is checked BEFORE anything is extracted ──────────────────────
  # A weekend, an exchange holiday or a 13:00 half day costs 10-25 minutes of vendor I/O each and
  # returns an empty or one-legged frame. Checking the calendar takes milliseconds and has to happen
  # here, not after three hours: the first real run carried 2026-01-19 (MLK Day, NYSE closed, CME
  # Globex on an abbreviated session) all the way to `median SPY=nan ES=6913.75`.
  if [ "$SOURCE" = "extract" ]; then
    VS_FLAGS=""; [ "$ALLOW_BAD_DATES" -eq 1 ] && VS_FLAGS="--allow-bad-dates"
    # shellcheck disable=SC2086
    if run_show $PY validate_sample.py --volatile "$VOLATILE" --baseline "$BASELINE" \
         --mwcb "$MWCB" $VS_FLAGS --out "${OUT}/sample_validation.txt"; then
      :
    else
      echo "" | tee -a "$LOG"
      echo "SAMPLE REJECTED — see ${OUT}/sample_validation.txt" | tee -a "$LOG"
      echo "Replace the flagged date(s) with tradable sessions, or pass --allow-bad-dates" | tee -a "$LOG"
      echo "if you intend to extract them anyway. Whichever date replaces one must keep the" | tee -a "$LOG"
      echo "matching rule: same weekday, roughly one year prior to its volatile partner." | tee -a "$LOG"
      exit 1
    fi
  fi
  info "OK"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — GATE: the corrections must actually be present
#
# Every one of these failed silently before. A green stack is a precondition for
# believing any number downstream, so this stage is a hard gate rather than a
# smoke test: if the sequencenumber fix has been reverted, or the Table 5 null
# regressed, the run stops here instead of emitting plausible-looking tables.
# ══════════════════════════════════════════════════════════════════════════════
if have_stage 1; then
  say "STAGE 1  correctness gate (regression tests for each correction)"
  FAILED=""
  for t in test_crossed_root_cause.py \
           test_reconstruct_ordering.py \
           test_crossed_regression.py \
           test_debug_crossing.py \
           test_tandem_null.py \
           test_hy_correlation.py \
           test_extract_resilience.py \
           test_validate_sample.py \
           test_feed_reset.py \
           test_validate_aggregated.py \
           test_halt_aware_qc.py \
           test_driver_flags.py \
           test_svar_lag_artifact.py \
           test_market_state.py ; do
    if [ "$DRY" -eq 1 ]; then info "(dry-run) would run $t"; continue; fi
    if run_rc $PY "$t"; then info "PASS  $t"; else info "FAIL  $t"; FAILED="$FAILED $t"; fi
  done
  if [ -n "$FAILED" ] && [ "$DRY" -eq 0 ]; then
    echo "" | tee -a "$LOG"
    echo "GATE FAILED:$FAILED" | tee -a "$LOG"
    echo "A correction is missing or has regressed. Fix before replicating -- the" | tee -a "$LOG"
    echo "failure modes these guard are all SILENT in the output tables." | tee -a "$LOG"
    exit 1
  fi
  info "all corrections verified present"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — reconstruct the books
#
# SPY is a CONSOLIDATED multi-venue book replayed from messages (hybrid MBO+MBP);
# ES is a CME order-by-order replay. Both go through the single extraction path so
# they share one clock -- a vendor snapshot for one leg and a message replay for
# the other would corrupt the cross-asset lead-lag by construction.
# ══════════════════════════════════════════════════════════════════════════════
if have_stage 2; then
  say "STAGE 2  reconstruct / load the session books"
  case "$SOURCE" in
    demo)
      info "synthetic sessions (no MayStreet needed); numbers are illustrative only"
      ;;
    load)
      [ -n "$PICKLE" ] || { echo "--source load needs --pickle" >&2; exit 2; }
      info "using existing frames: $PICKLE"
      FRAMES="$PICKLE"
      ;;
    extract)
      info "extracting ${INTERVAL} books for $(echo "$VOLATILE,$BASELINE,$MWCB" | tr ',' '\n' | wc -l) sessions"
      CACHE_FLAGS=""
      if [ -n "$CACHE_DIR" ]; then
        CACHE_FLAGS="--extract-cache $CACHE_DIR"
        [ "$RESUME" -eq 0 ] && CACHE_FLAGS="$CACHE_FLAGS --no-resume"
        info "session cache: ${CACHE_DIR} (resume=$([ "$RESUME" -eq 1 ] && echo yes || echo no))"
        info "  a day already extracted there is reused, not re-fetched -- so a run that dies"
        info "  partway costs only the missing days on the retry. --no-cache disables it."
      fi
      # shellcheck disable=SC2086
      run_show $PY autoscale.py measure -- $PY run_analysis.py --source extract \
          --dates "${VOLATILE},${BASELINE},${MWCB}" \
          --volatile "${VOLATILE},${MWCB}" --benchmark "${BASELINE}" \
          --interval "$INTERVAL" --n-levels 10 --max-workers "$AS_EXTRACT" \
          --qc-action "$QC_ACTION" $CACHE_FLAGS \
          --output-dir "$OUT" --save-dataset --save-objects --only extract
      # A session that failed or that violates the book invariant is named here, not only in the
      # scrollback of a multi-hour log -- the SAMPLE is a result, and a universe that quietly
      # shrank from 24 days to 22 produces tables indistinguishable from a clean run's.
      if [ -s "${OUT}/extract_report.txt" ]; then
        info "EXTRACTION WAS NOT CLEAN -- ${OUT}/extract_report.txt:"
        sed 's/^/     /' "${OUT}/extract_report.txt" | tee -a "$LOG"
      fi
      # run_analysis writes the frames into a TIMESTAMPED SUBDIRECTORY of --output-dir, so a
      # flat glob on ${OUT} finds nothing. That is not hypothetical: an 85-minute extraction
      # completed, wrote its run folder, and STAGE 3 then died on "no pickle matched
      # .../frames.pkl" with every downstream stage unable to start. Search RECURSIVELY, newest
      # first, and accept either name.
      FOUND="$(find "${OUT}" -name 'frames_*.pkl' -o -name '*aggregated*.pkl' 2>/dev/null \
               | xargs -r ls -1t 2>/dev/null | head -1 || true)"
      if [ -n "$FOUND" ]; then
        FRAMES="$FOUND"
      else
        echo "" | tee -a "$LOG"
        echo "STAGE 2 produced no session-frames pickle under ${OUT}." | tee -a "$LOG"
        echo "Extraction may still have succeeded -- check for final_dataset.parquet -- but every" | tee -a "$LOG"
        echo "downstream stage loads List[(date, regime, df)], which the flat dataset is not." | tee -a "$LOG"
        echo "Re-run STAGE 2 with a run_analysis that supports --save-frames (v0.9.25+)." | tee -a "$LOG"
        exit 1
      fi
      info "frames: $FRAMES"
      ;;
    *) echo "unknown --source $SOURCE" >&2; exit 2 ;;
  esac
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — GATE: crossed-book invariant, per session
#
# A matching engine cannot cross. A crossed reconstructed top therefore always
# means the replay is wrong, and the original failure produced a FULL-LENGTH
# frame of garbage with no error.
#
# Two steps, cheap one first:
#   3a  qc_frames gates the SAVED frames -- the exact objects the tables will be
#       estimated on -- in seconds, with no vendor I/O. This is the gate.
#   3b  debug_crossing runs ONLY on the sessions 3a flags, for the root cause
#       (sequencenumber present? orphaned cancels inside the window? a Modify
#       resurrecting a Cancelled order?). It re-pulls the raw messages, which is
#       why it cannot be the gate: gating that way meant a SECOND multi-hour pass
#       over the whole sample, and it judged a freshly fetched book rather than
#       the one on disk -- so a session could pass the gate and still be estimated
#       from a broken frame. debug_crossing exits 0 clean / 2 DATA / 3 CODE.
# ══════════════════════════════════════════════════════════════════════════════
if have_stage 3 && [ "$SOURCE" != "demo" ]; then
  say "STAGE 3  data-integrity gate (crossed-book invariant on the saved frames)"
  if [ "$DRY" -eq 1 ]; then
    printf '   + %s\n' "$PY qc_frames.py --pickle $FRAMES --crossed-tol 0.005 --out ${OUT}/qc_frames.txt" | tee -a "$LOG"
    info "(dry-run) would then run debug_crossing.py on any flagged session"
  else
    if run_show $PY qc_frames.py --pickle "$FRAMES" --crossed-tol 0.005 \
         --out "${OUT}/qc_frames.txt"; then
      info "every session satisfies the invariant"
      # The invariant says the book is SELF-consistent. It does not say the book is RIGHT.
      #
      # NOTE ON WHAT THIS NOW PROVES. The ES leg is built FROM the venue ladder
      # (mt_aggregated_price_update) since v0.9.42, so this is no longer a check on the shipped
      # futures book -- that would be comparing the ladder with itself. It is the converse and it is
      # still worth running: an INDEPENDENT reconstruction from the raw message tape is compared
      # against the ladder, and agreement is evidence that the ladder is a faithful book rather than
      # a lossy summary of one. Read the result that way in the paper: it validates the SOURCE
      # choice, not the extraction. One session is enough for the robustness table; --stages 3
      # re-runs it on another day.
      if [ "$SOURCE" = "extract" ]; then
        VDATE="$(echo "$VOLATILE" | cut -d, -f1 | tr -d -)"
        VCON="$($PY -c "import mstbook_loader as ml,datetime;print(ml.get_front_month_contract('ES', as_of_date=ml._parse_yyyymmdd('$VDATE')))" 2>/dev/null || true)"
        if [ -n "$VCON" ]; then
          info "cross-checking an independent message replay against the venue ladder the ES leg"
          info "  is built from, on ${VDATE} (${VCON}) -- evidence the ladder is a faithful book"
          run_rc $PY validate_aggregated.py --date "$VDATE" --product "$VCON" \
                 --product-type futures --price-scale 0.01 --interval "$INTERVAL" \
                 --out "${OUT}/validate_${VCON}_${VDATE}.txt" \
            && info "  the independent replay reproduces the ladder (${OUT}/validate_${VCON}_${VDATE}.txt)" \
            || info "  DISAGREEMENT replay vs ladder -- see ${OUT}/validate_${VCON}_${VDATE}.txt"
        fi
      fi
    else
      BAD="$(sed -n 's/^BAD \([0-9-]*\).*/\1/p' "${OUT}/qc_frames.txt" | tr '\n' ' ')"
      if [ -n "$BAD" ] && [ "$SOURCE" = "extract" ]; then
        info "root-causing the flagged session(s) -- both tools re-fetch that day's raw messages"
        for d in $BAD; do
          ymd="${d//-/}"
          # FIRST: is the capture even complete? A venue gap report or a decoder error means the
          # adds never arrived, so the removals referencing them are orphans and the book crosses
          # through no fault of the replay. No amount of reconstruction work fixes a lost packet.
          run_rc $PY feed_health.py --date "$ymd" --product SPY \
                 --out "${OUT}/feed_health_${ymd}.txt" || true
          # THEN the replay-side root cause. --clock must MATCH the extraction (receipt): diagnosing
          # a book built on the other clock answers a question about a book you did not save, and
          # some feeds stamp a whole burst with one exchange timestamp, which makes that ordering
          # degenerate for them.
          run_rc $PY debug_crossing.py --date "$ymd" --product SPY \
                 --clock receipt --ab-ordering --out "${OUT}/crossing_${ymd}.txt" || true
          info "  reports: ${OUT}/feed_health_${ymd}.txt, ${OUT}/crossing_${ymd}.txt"
          # The gate fails on EITHER leg, so diagnosing only SPY can answer the wrong question.
          # The futures leg has its own capture, its own resets and its own halts -- CME Velocity
          # Logic pauses ES for 5-10 s on exactly these days.
          ECON="$($PY -c "import mstbook_loader as ml;print(ml.get_front_month_contract('ES', as_of_date=ml._parse_yyyymmdd('$ymd')))" 2>/dev/null || true)"
          if [ -n "$ECON" ]; then
            run_rc $PY feed_health.py --date "$ymd" --product "$ECON" --product-type futures \
                   --out "${OUT}/feed_health_${ymd}_${ECON}.txt" || true
            info "  ES leg: ${OUT}/feed_health_${ymd}_${ECON}.txt"
          fi
        done
      fi
      echo "" | tee -a "$LOG"
      echo "GATE FAILED — crossed/incomplete books:${BAD:- (see ${OUT}/qc_frames.txt)}" | tee -a "$LOG"
      echo "CHECK 4/4b/8 in the per-session reports name the cause. Re-extract these" | tee -a "$LOG"
      echo "days (the session cache means the clean ones are not re-fetched) or drop" | tee -a "$LOG"
      echo "them with --qc-action drop; do not estimate on a book that violates its" | tee -a "$LOG"
      echo "own invariant." | tee -a "$LOG"
      exit 1
    fi
  fi
elif have_stage 3; then
  say "STAGE 3  data-integrity gate — SKIPPED (--source demo has no reconstructed book)"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4 — Table 5 / Table 7 with the corrected nulls
# ══════════════════════════════════════════════════════════════════════════════
if have_stage 4; then
  say "STAGE 4  Table 5 + Table 7 with the corrected nulls"
  info "reports: raw corner shares (as published), independence GIVEN the observed"
  info "marginals, the corner log odds ratio, and the frequency-matched binomial null"
  run_show $PY - "$OUT" <<'EOF'
import sys, warnings; warnings.simplefilter("ignore")
import numpy as np, pandas as pd, tandem_order_flow as tof
out = sys.argv[1]

# The paper's published Table 5.II, so the correction is auditable without re-running
# anything. Replace with your own frequency matrices (or table5_from_series on the real
# per-bar new-order counts) once the message counts are extracted.
panels = {"A baseline": [[21.24, 12.01, 7.73], [4.82, 6.97, 4.88], [7.86, 12.07, 21.50]],
          "B volatile": [[24.55, 11.12, 3.52], [5.73, 9.66, 4.04], [4.65, 13.63, 20.99]],
          "C MWCB":     [[9.96, 20.89, 6.00], [4.94, 18.06, 6.77], [2.86, 17.46, 10.68]]}
rows = []
for k, M in panels.items():
    f = pd.DataFrame(np.array(M, float), index=tof.DIR3, columns=tof.DIR3)
    d = tof.dependence_summary(f, n_bars=23400)
    rows.append({"panel": k, "PCMOF": d["PCMOF"], "PCMOF_indep": d["PCMOF_indep"],
                 "PCMOF_ratio": d["PCMOF_ratio"], "NCMOF": d["NCMOF"],
                 "NCMOF_indep": d["NCMOF_indep"], "NCMOF_ratio": d["NCMOF_ratio"],
                 "log_OR": d["log_OR"], "log_OR_z": d.get("log_OR_z", np.nan)})
t5 = pd.DataFrame(rows).set_index("panel").round(3)
print("\nTable 5 -- cross-market dependence, separated from each market's own directionality")
print(t5.to_string())
print("\n  binomial null (n=505/112, one second): PCMOF = NCMOF = 0.4% per corner pair")
print("  'indep' = independence GIVEN the observed marginals -- the benchmark that isolates")
print("  cross-market trading. log_OR is marginal-free, so it is comparable across panels.")
t5.to_csv(f"{out}/table5_corrected_null.csv")

# Table 7: the same binomial null evaluated at each aggregation's actual order counts.
rng = np.random.default_rng(0); N = 400_000
freqs = [("1 second", 505.0, 112.0), ("10 millisecond", 5.05, 1.12), ("action time", 1.0, 1.0)]
obs = {"1 second": 11.88, "10 millisecond": 30.14, "action time": 48.40}   # paper, both off-corners
rows = []
for lbl, le, lf in freqs:
    ne = np.full(N, 1) if lbl == "action time" else rng.poisson(le, N)
    nf = np.full(N, 1) if lbl == "action time" else rng.poisson(lf, N)
    nul = tof.independence_null_from_counts(ne, nf)
    rows.append({"aggregation": lbl, "orders_ETF": le, "orders_FUT": lf,
                 "NCMOF_null_%": nul["NCMOF"], "NCMOF_observed_%": obs[lbl],
                 "obs/null": obs[lbl] / max(nul["NCMOF"], 1e-9)})
t7 = pd.DataFrame(rows).set_index("aggregation").round(2)
print("\nTable 7 -- independence null at the ACTUAL per-bar counts of each aggregation")
print(t7.to_string())
print("\n  The published comparison uses the per-second null (0.4%) for all three rows.")
t7.to_csv(f"{out}/table7_frequency_matched_null.csv")
EOF
  info "wrote ${OUT}/table5_corrected_null.csv, ${OUT}/table7_frequency_matched_null.csv"
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 4c — resolve the SVAR lag order ONCE
#
# The paper's p=6 is a stated compromise (footnote 17: AIC suggested 60, 6 was
# used because the full model would not run there), which makes it a researcher
# degree of freedom. Pick it by criterion on the SAME pooled frame Eq. (5) is
# fitted on, print the whole IC table so the choice is inspectable, and reuse
# that ONE integer in every downstream stage -- so the p reported in a table
# note is provably the p that was fitted, and Pearson/HY differ only in the
# estimator rather than also in the model.
# ══════════════════════════════════════════════════════════════════════════════
if have_stage 4 || have_stage 5 || have_stage 6; then
  case "$N_LAGS" in
    ''|*[!0-9]*)                                  # a criterion, not an integer
      say "STAGE 4c  SVAR lag order by ${N_LAGS^^} (pmax=${PMAX})"
      if [ "$DRY" -eq 1 ]; then
        info "(dry-run) would resolve --n-lags ${N_LAGS} on the session frames"
      elif [ "$SOURCE" = "demo" ]; then
        info "demo mode: each driver resolves ${N_LAGS} on its own synthetic frames"
      else
        N_LAGS_INT="$($PY - "$FRAMES" "$N_LAGS" "$PMAX" "$CORR_WINDOW" <<'EOF' 2>>"$LOG" || true
import pickle, sys, glob, warnings
warnings.simplefilter("ignore")
import correlation_svar as cs
path, crit, pmax, win = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
raw = []
for f in sorted(glob.glob(path)):
    with open(f, "rb") as fh:
        raw.extend(pickle.load(fh))
sess = [r if len(r) == 3 else (r[0], "benchmark", r[1]) for r in raw]
p, tab = cs.select_svar_lag(sess, spec="informational", corr_window=win,
                            criterion=crit, pmax=pmax)
if p is None:
    sys.exit(1)
sys.stderr.write(tab[["aic", "bic", "hqic"]].round(3).to_string() + "\n")
# Is the selected lag an answer or an artifact of how dCorr is built? See lag_diagnosis:
# d(W-bar rolling correlation) carries an MA term at exactly lag W, and the criterion finds it.
d = cs.lag_diagnosis(p, corr_window=win, pmax=pmax, corr_method="rolling")
sys.stderr.write("LAG DIAGNOSIS: " + d["text"] + "\n")
print("%d %d %d" % (int(p), int(d["window_artifact"]), int(d["at_boundary"])))
EOF
)"
        LAG_ART=""; LAG_EDGE=""
        if [ -n "$N_LAGS_INT" ]; then
          set -- $N_LAGS_INT
          N_LAGS_INT="$1"; LAG_ART="${2:-0}"; LAG_EDGE="${3:-0}"
          info "selected p=${N_LAGS_INT} by ${N_LAGS^^} over p<=${PMAX} (IC table in the log)"
          if [ "$N_LAGS_INT" = "0" ]; then
            info "NOTE: p=0 means the criterion found no dynamics. A VAR(0) has no impulse response,"
            info "so p=1 is used downstream and the response is impact-only. On a wide"
            info "--corr-window this usually means the criterion could not see the window's own MA"
            info "spike rather than that the system is empty."
            N_LAGS_INT=1
          fi
          if [ "$LAG_ART" = "1" ]; then
            info "WARNING: p EQUALS --corr-window (${CORR_WINDOW}). That is an artifact, not a"
            info "finding: dCorr is the difference of a ${CORR_WINDOW}-bar rolling correlation, so it"
            info "carries an MA term at exactly lag ${CORR_WINDOW} and the criterion is locating it."
            info "On simulated data with a CONSTANT correlation and iid liquidity this reproduces"
            info "exactly (test_svar_lag_artifact.py). Re-run STAGE 5 with --with-dcc: the DCC"
            info "conditional correlation has no fixed-width box to difference and returns p*~1 on"
            info "that same data. NOTE that 'both ways' does NOT fix this on its own -- Pearson and"
            info "HY differ on asynchronicity but both difference the same ${CORR_WINDOW}-bar box."
            T9_DCC="--with-dcc"
          fi
          if [ "$LAG_EDGE" = "1" ] || { [ "$N_LAGS_INT" -ge "$PMAX" ] 2>/dev/null; }; then
            info "WARNING: p == pmax, so the criterion is still improving at the edge of the search."
            info "That is a BOUND, not an optimum. Note that with --corr-window ${CORR_WINDOW} the"
            info "induced spike sits at lag ${CORR_WINDOW}, so raising --pmax walks TOWARD the window"
            info "rather than converging -- change the dependent variable, not the search."
          fi
        else
          info "selection failed (too few usable rows?); downstream stages fall back to their defaults"
        fi
      fi
      ;;
    *) N_LAGS_INT="$N_LAGS"; info "SVAR lag order: fixed p=${N_LAGS_INT} (no criterion)" ;;
  esac
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 5 — Table 9 both ways, at both aggregations
#
# Attenuation grows as the bar shrinks, so the ten-millisecond specification is
# where the Pearson/HY gap should be largest.
# ══════════════════════════════════════════════════════════════════════════════
if have_stage 5; then
  say "STAGE 5  Table 9 on Pearson AND Hayashi-Yoshida d-correlation"
  T9ARGS="--spec informational --n-lags ${N_LAGS} --pmax ${PMAX} --n-boot ${N_BOOT} --out-dir ${OUT}"
  # set by STAGE 4c when the selected lag turned out to equal the correlation window
  [ -n "${T9_DCC:-}" ] && T9ARGS="$T9ARGS ${T9_DCC}"
  [ -n "$NJ" ] && T9ARGS="$T9ARGS --n-jobs $NJ"
  if [ "$SOURCE" = "demo" ]; then
    # shellcheck disable=SC2086
    run_show $PY run_table9_both_ways.py --source demo --corr-window "$CORR_WINDOW" $T9ARGS
  else
    # 1-second: the paper's headline window (100 bars = 100 seconds)
    # shellcheck disable=SC2086
    run_show $PY run_table9_both_ways.py --source load --pickle "$FRAMES" \
        --volatile "${VOLATILE},${MWCB}" --corr-window "$CORR_WINDOW" $T9ARGS
    # 10-millisecond: the paper uses a 1-second window there, i.e. 100 bars again
    if [ -f "${FRAMES/1s/${FINE_INTERVAL}}" ]; then
      # shellcheck disable=SC2086
      run_show $PY run_table9_both_ways.py --source load --pickle "${FRAMES/1s/${FINE_INTERVAL}}" \
          --volatile "${VOLATILE},${MWCB}" --corr-window "$CORR_WINDOW" $T9ARGS
    else
      info "no ${FINE_INTERVAL} frames found — re-run STAGE 2 with --interval ${FINE_INTERVAL} for the"
      info "fine-grid pair, which is where the Epps gap should be largest"
    fi
  fi
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 6 — the remaining tables (full stack)
# ══════════════════════════════════════════════════════════════════════════════
if have_stage 6; then
  say "STAGE 6  remaining paper + revamp tables"
  QFLAG=""; [ "$QUICK" -eq 1 ] && QFLAG="--quick"
  case "$SOURCE" in
    demo)  # shellcheck disable=SC2086
           run $PY run_analysis.py --source demo $QFLAG --legacy --output-dir "$OUT" ;;
    *)     # shellcheck disable=SC2086
           run $PY run_analysis.py --source load --pickle "$FRAMES" \
               --volatile "${VOLATILE},${MWCB}" --benchmark "${BASELINE}" \
               --interval "$INTERVAL" ${N_LAGS_INT:+--n-lags "$N_LAGS_INT"} --legacy \
               --output-dir "$OUT" --save-dataset $QFLAG ;;
  esac
fi

# ══════════════════════════════════════════════════════════════════════════════
# STAGE 7 — manifest
# ══════════════════════════════════════════════════════════════════════════════
if have_stage 7; then
  say "STAGE 7  manifest"
  {
    echo "# Replication run ${RUN_ID}"
    echo
    echo "source=${SOURCE}  interval=${INTERVAL}  corr_window=${CORR_WINDOW}  n_boot=${N_BOOT}"
    echo "lag order: requested=${N_LAGS} (pmax=${PMAX})  resolved p=${N_LAGS_INT:-per-driver}"
    echo "sizing:    cores=${AS_CORES} ram=${AS_RAM}GiB extraction_workers=${AS_EXTRACT} cpu_jobs=${NJ}"
    echo "volatile=${VOLATILE}"
    echo "baseline=${BASELINE}"
    echo "mwcb=${MWCB}"
    echo
    echo "## Corrections applied"
    echo "- sequencenumber preserved through the loader; intra-feed replay order restored"
    echo "- crossed-book invariant gated per session before any estimation (STAGE 3)"
    echo "- Table 5 benchmarked against independence GIVEN the observed marginals + corner log OR"
    echo "- Table 7 null computed at each aggregation's actual per-bar order counts"
    echo "- Table 9 reported on Pearson AND Hayashi-Yoshida d-correlation, with the difference"
    echo "- SVAR lag order chosen by ${N_LAGS} on the pooled frame (paper fixed it at 6; see fn.17)"
    echo "- cluster SE falls back to Newey-West at G=1 instead of returning a silent NaN"
    echo
    echo "## Artifacts"
    ls -1 "$OUT" | sed 's/^/- /'
  } > "${OUT}/MANIFEST.md"
  cat "${OUT}/MANIFEST.md"
fi

say "done — ${OUT}"
[ "$SOURCE" = "demo" ] && echo "   NOTE: --source demo is synthetic. Re-run with --source extract for paper numbers."
exit 0
