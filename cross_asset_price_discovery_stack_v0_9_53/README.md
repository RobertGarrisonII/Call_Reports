# Cross-Asset Liquidity & Price Discovery — Code Guide

Research code extending **Garrison, Jain & Paddrik, *Cross-Asset Market Order Flow, Liquidity, and Price Discovery* (OFR WP 19-04)** toward a JF/RFS-grade revision. The stack replaces the paper's spread-only liquidity measure with the full limit-order-book depth curve, adds the information-share machinery the title promises, and fuses the two into **liquidity-state-conditional price discovery** — the new contribution.

---

## 0. Replicating the paper (start here)

```bash
./run_paper_replication.sh                     # demo: runs anywhere, no data needed
./run_paper_replication.sh --source extract    # the full paper sample via MayStreet
./run_paper_replication.sh --stages 4,5        # re-run selected stages
./run_paper_replication.sh --dry-run           # print the command sequence only
```

Seven stages, **two of which are gates**. The gates exist because every defect this pipeline
was corrected for was *silent* — the run "succeeded" and wrote a full-length frame of garbage:

| stage | what it does |
|---|---|
| 0 | preflight: interpreter, deps, stack imports |
| **1** | **GATE** — the regression test for each correction must pass, or the run stops |
| 2 | reconstruct the SPY consolidated book + ES on one clock (or load/demo) |
| **3** | **GATE** — `debug_crossing` per session; no table is built from a book that violates the crossed-book invariant |
| 4 | Table 5 against independence *given the observed marginals* + corner log OR; Table 7 null at each aggregation's actual per-bar counts |
| 5 | Table 9 on **both** the Pearson and Hayashi–Yoshida Δρ, with the difference as the artifact estimate |
| 6 | the remaining paper + revamp tables (`run_analysis.py`) |
| 7 | manifest listing the corrections applied and every artifact written |

The paper's sample (Appendix Table A.1 — 10 volatile, 10 matched baseline, 4 MWCB dates) is
baked in; override with `--volatile/--baseline/--mwcb`. See `CHANGELOG.md` v0.9.15 for what
each correction changes and why.

---

## 1. Requirements

```
python >= 3.10
numpy
pandas
scipy            # MLE for GARCH/DCC, optimizers
```

No `statsmodels` or `arch` needed — GARCH-X, the DCC recursion, and HAC/Newey-West inference are implemented from scratch. Keep **all `.py` files in one folder** (they import each other).

```bash
pip install numpy pandas scipy
```

Smoke-test every module (each has a self-contained synthetic check):

```bash
for m in liquidity_curve_metrics price_discovery_shares cross_asset_pd_liquidity \
         cross_impact dcc_garch robustness irf noise_robust_cov \
         microstructure_diagnostics functional_liquidity jump_robust \
         ecm_sde robust_prices; do
  echo "== $m =="; python "$m.py"; done
```

---

## 2. Files

| File | Role |
|---|---|
| `run_paper_replication.sh` | **The paper, end to end, with every correction applied.** Seven stages, two gates (see §0). |
| `run_table9_both_ways.py` | Table 9 on the Pearson and Hayashi–Yoshida Δρ side by side, with the artifact delta; writes .csv/.md/.tex. |
| `run_analysis.py` | **Driver — one command runs the whole stack.** Loads (or extracts) the session frames, runs every stage below, and writes a per-run folder `output/run_<ts>/` with the consolidated **final dataset** and the **result tables** (CSV). Sources: `--source demo` (synthetic, runs anywhere), `load` (a saved pickle), `extract` (live MayStreet). |
| `market_analysis_fixed.py` | **Starting point (step 1).** MayStreet/Athena extraction (the `bidprice_1` fix applied; per-level book projected through to the final frame; per-`(date, product)` Athena fetches run concurrently under a `max_workers` cap). Produces the per-session book frames the rest of the stack consumes — returned in memory as `List[(date, DataFrame)]`; it writes no files itself (persistence is the driver's job). |
| `liquidity_curve_metrics.py` | Depth-curve functionals: entropy, arc-length shape, slope, center-of-mass, cost-to-fill, L2 / min-max normalization, cross-asset shape cosine. |
| `price_discovery_shares.py` | VECM information shares: Hasbrouck IS (bounds + midpoint), Lien–Shrestha order-invariant IS, Gonzalo–Granger CS. Per-session, panel VECM, permutation tests. **Lag order by BIC/AIC/HQ (pooled across sessions) or fixed.** |
| `cross_asset_pd_liquidity.py` | **Integration layer.** OFI, realized moments, the liquidity-conditional VECM, the windowed panel join, day-clustered panel regression, state-split shares. |
| `cross_impact.py` | Multi-asset / multi-level cross-impact matrices (Cont–Kukanov–Stoikov, Cont–Cucuringu–Zhang) with HAC inference. |
| `dcc_garch.py` | DCC-GARCH-X: time-varying SPY–ES conditional correlation with exogenous variance drivers. |
| `robustness.py` | Lag, β, subsample, window-size, state-variable, and bootstrap robustness checks. |
| `irf.py` | Dynamic layer: Jordà local projections (state-dependent, block-bootstrap bands), structural VECM IRFs identified off the cross-impact matrix, and FEVD. |
| `noise_robust_cov.py` | Sub-second covariance: two-scale & realized-kernel variance (noise-robust), Hayashi-Yoshida & refresh-time + realized-kernel covariance (async-robust), Epps / signature diagnostics. |
| `microstructure_diagnostics.py` | Staleness (zero-return fraction), tick-discreteness, and noise-to-signal diagnostics with a frequency sweep — tells you the grid below which the information shares stop being trustworthy. |
| `functional_liquidity.py` | Functional-data extension: FPCA on the depth-by-level curve. Emits the leading book-shape eigen-modes as a data-driven liquidity-state series — a drop-in for `relative_depth_state` in the conditional VECM. |
| `jump_robust.py` | Continuous-vs-jump price discovery. Jump-robust realized measures (bipower / MedRV / MinRV, Mancini truncation, BNS & Lee-Mykland tests, threshold/bipower covariation, co-jump lead-lag), and an information-share split (built on `price_discovery_shares`) into a continuous and a jump component — who leads diffusive price discovery vs who reacts first to jumps. |
| `ecm_sde.py` | **Liquidity-conditional price discovery as a state-dependent error-correction SDE** — the continuous-time form of the thesis. State-interacted VECM with polynomial-in-state loadings α(S) (the coef on z·S, with day-clustered HAC t-stats, is the single-number test of liquidity-conditioning), kernel-local varying-coefficient VECM (the full IS(S)/CS(S)/κ(S) curves), and a Euler-Maruyama pseudo-MLE with state-dependent diffusion. Takes a `price_fn` so the SDE can run on any observable. |
| `robust_prices.py` | **Noise-robust observables that feed `ecm_sde`.** Depth-aware prices (microprice, book-centroid, area-under-curve / cost-to-fill mid) and curve-length-normalized book states, plus sparse-sampling / pre-averaging (`coarsen`) for a noise-robust *measure*. `price_fn` / `state_fn` / `noise_robust_is` wire these into the SDE. The naive top-of-book mid carries bounce + tick discreteness that bias every information share toward 0.5; these replace it. |
| `tandem_order_flow.py` | **Liquidity-spillover revamp.** Reproduces the paper's Table 5 cross-market order-imbalance contingency (§5.14), and the corrected nulls: independence *given the observed marginals*, the marginal-free corner log odds ratio, a permutation null, and the binomial null evaluated at the actual per-bar order counts (the published one is a per-second calibration and cannot be reused at 10 ms or in action time). |
| `correlation_svar.py` | Weighted-spread observables + the Table-9-style correlation-IRF with spreads, bootstrap SE and Romano–Wolf stars (§5.15). |
| `cross_flow.py` | Continuous, signed, size-aware cross-market order flow (χ, PCMOF/NCMOF, common/divergent), incl. the VPIN-style volume clock (§5.16). |
| `stress_index.py` | Turns a daily conditional-vol series into the continuous stress state, regime labels, and corrected day-selection for the contagion design (§5.17). |
| `mwcb_event_study.py` | The MWCB reopening-asymmetry event study: per-bar liquidity outcomes (`cost_to_fill`/`inside_depth`/`composite`/`auc_decay`/`curve_length`), DiD, RD-in-time, cross-market spillover (§5.18). |
| `rigobon_id.py` | Order-free SVAR identification through heteroskedasticity; Forbes–Rigobon contagion-vs-interdependence test (§5.19). |
| `hawkes_cross.py` | Bivariate Hawkes on liquidity-event times: branching matrix/ratio = directed contagion intensity and reflexivity (§5.20). |
| `markov_switching_vecm.py` | Calm/stress as a latent Markov state with regime-dependent error-correction and information shares (§5.21). |
| `pricing_error.py` | Hasbrouck permanent/transitory decomposition → σ_s mispricing magnitude and its dollar welfare cost (§5.22). |
| `tick_correction.py` | Tick-size-aware information shares (common-grid + rounding-corrected): is futures dominance genuine or grid-induced (§5.23). |
| `inference.py` | Generated-regressor & moving-block bootstrap SEs, Romano–Wolf / Holm / Benjamini–Hochberg multiple-testing control (§5.24). |
| `paper_tables.py` | Reporting layer: builds the 16-table reproduction+revamp suite and renders Markdown/LaTeX (§5.25). |
| `market_shocks.py` | Annotated shock registry (6 event classes) + level-matched treatment/control selection by event timing (§5.26). |
| `event_study_driver.py` | Registry → matched controls → estimators in one call per identification regime (§5.27). |
| `run_contagion.py` | **Top-level contagion pipeline + CLI.** One command runs the causal event study, the descriptive vol-conditioning, or the mean+variance framework over a date range; writes the manuscript tables and run summaries (§5.28). |
| `mean_variance.py` | **Mean+variance framework.** Per-session VECM (information shares + innovations) → DCC-GARCH-X with per-asset liquidity covariates → time-varying ρ(SPY,ES) with a calm-vs-stress split and the liquidity→vol loadings (§5.29). |
| `mstbook_loader.py` | **Fast CLI data pull (MayStreet `mstbook-query` + `mstwx-lakequery`) in the canonical schema — the `--source extract` backend (no SQL).** Streams the book *and* the trade tape faster than Athena, emits `{ROOT}_{bid\|ask}{price\|quantity}_{i}` plus trade-tape buy/sell counts directly, with dtype + joint-finite hygiene at parse time, busted/corrected-print scrubbing, and features via the stack's own functions (§5.30). |
| `lob_reconstruct.py` | **Consolidated book + NBBO reconstructed from messages — the sole extraction path for both legs.** Hybrid MBO+MBP per-venue replay: SPY consolidated across order feeds *and* IEX-style price-level feeds → an odd-lot-inclusive 10-level ladder plus a strict round-lot Reg NMS NBBO; ES a single-venue CME **order-by-order (MBO)** replay (`price_scale` → index points). One GPS-synchronized capture clock for both, which the cross-asset lead-lag requires. `validate_against_snapshot` benchmarks the rebuild against the (now-legacy, different-clock) vendor book (§5.31). |
| `auction_imbalance.py` | **Opening/closing-cross imbalance features + the cash-open → futures linkage.** `mt_order_imbalance` is auction-only in this lake, so this turns the primary-listing (ARCA for SPY) open/close cross trajectory into per-(session, auction) features — signed imbalance ratio, indicative-vs-reference dislocation (bps), pre-cross imbalance slope — and relates the SPY **opening** cross to the contemporaneous E-mini move on the same reconstructed clock. ES has no auction (CME runs no cross), so the linkage is auction → continuous move. The `--auction` hook in `run_contagion` (§5.33). |
| `copula_garch.py` | **Copula dependence with tail dependence + a liquidity-conditional joint-crash hook.** GARCH(1,1)-X margins → rank PIT → constant-copula selection {gaussian, t, clayton, gumbel, BB1, SJC} by BIC with lower/upper tail-dependence coefficients (BB1 = the two-parameter Clayton-Gumbel, both tails, analytic density, nesting Clayton at δ=1 for a clean nested LR test); a **t-copula-DCC** dynamic baseline (time-varying ρ_t *and* λ_t, vs Gaussian DCC whose tail dependence is identically 0); and the headline — lower-tail dependence re-fit on **stress-vs-calm days** (open/close-trimmed), under both an AUC_decay liquidity split and a realized-volatility split, testing whether **joint-crash dependence rises under stress**. The `--copula` hook in `run_contagion` (§5.34). |
| `validate_reconstruction.py` | **Driver — snapshot-vs-reconstruction benchmark table.** For each date builds SPY+ES under both book sources, reports the book-faithfulness match rates *and* re-estimates ES leadership (Hasbrouck IS, order-invariant MIS, Gonzalo–Granger CS, Hayashi–Yoshida lead-lag, return correlation) on each book, then the reconstruct−snapshot delta — the robustness table that says whether the snapshot bias moves the leadership conclusion (§5.32). |
| `onset_response.py` | **The identified headline — cross-event onset estimate of the stress-response function f(state).** Per-event ONSET transmission (ES→SPY contemporaneous coefficient, Rigobon heteroskedasticity-ID across the pre/post-release boundary) paired with the predetermined basis-dislocation and book-capacity state, pooled across events to trace f(state). Contains the 2-D **stress surface** `fit_stress_surface` (the liquidity-spiral interaction b3 = basis×capacity), its identification gate `interaction_identified`, the **noise-robust** transmission (TSRV+Hayashi–Yoshida covariances — defusing the bid-ask-bounce attenuation that mimics a spiral), the three round-trip cost aggregations (sum / bottleneck-max / product), the line- and surface-excess tests of the salience-curated tail (`excess_over_benchmark` / `excess_over_surface`), Ibragimov–Müller inference, and the `run_onset_surface` orchestration (§8). |
| `onset_sensitivity.py` | Validation diagnostics for the onset estimate: the post-window sweep (`post_window_profile` / `window_sweep` — is the estimate window-robust, or has the window reached the cascade?), the identification-strength census (`id_strength_profile` / `id_threshold_sweep` — how many events are weakly identified, does the slope survive dropping them?), and the order-free-vs-ordered `cholesky_bracket` (§8). |
| `run_onset_surface.py` | **CLI launcher for the onset stress-surface** (the `onset` stage in `run_liberation_day.sh`). Builds the impact-blind scheduled backbone (FOMC by default), loads the matching sessions, runs `run_onset_surface` across the capacity axes on the robust transmission, prints the decision-legible summary, and writes the onset panel CSV (§8). |

The first block (through `robust_prices.py`) is the original price-discovery core; the second block (`tandem_order_flow.py` onward) is the liquidity-spillover / contagion revamp; the **third block** (`onset_response.py` onward) is the cross-event **onset identification spine** (§8). `run_contagion.py` and `mean_variance.py` are the two top-level entry points for the contagion work.

Dependency order: `liquidity_curve_metrics` and `price_discovery_shares` are leaves; `cross_asset_pd_liquidity` builds on both; `cross_impact`, `robustness`, and `irf` build on the integration layer; `dcc_garch` and `noise_robust_cov` are standalone (the latter feeds `dcc_garch` and replaces `realized_moments` below ~1s); `microstructure_diagnostics` uses `noise_robust_cov` and annotates the information-share tables; `functional_liquidity` is standalone (numpy/pandas) and feeds its FPC state into `liquidity_conditional_vecm`; `jump_robust` builds on `price_discovery_shares` (it reuses the VECM fit and `hasbrouck_is`) to add the continuous/jump decomposition; `ecm_sde` builds on `price_discovery_shares` (`hasbrouck_is`, Gonzalo–Granger) and `cross_asset_pd_liquidity` (the state) for the continuous-time ECM-SDE; `robust_prices` builds on `liquidity_curve_metrics` and `cross_asset_pd_liquidity` and composes with `ecm_sde` through its `price_fn` hook (the depth-aware, noise-robust observable the SDE is computed on).

**Package & layout.** The bundle ships an `__init__.py`, so it can be used either way: run a module directly from inside the folder (`python irf.py`), or import it as a package (`from cross_asset_price_discovery_stack import irf`) — the `__init__.py` puts its own directory on `sys.path` so the modules' flat imports of one another resolve in both cases. Keep all `.py` files in the one directory. All persisted output is owned by the driver: `run_analysis.py` writes a per-run folder `output/run_<ts>/` holding the consolidated **final dataset** (`final_dataset.parquet` if a parquet engine is installed, else `final_dataset.csv.gz`) and the **result tables** under `tables/`, plus `report.md` and `summary.json`. `market_analysis_fixed.py` itself writes nothing — it returns the frames in memory. The `output/` directory is exposed as `cross_asset_price_discovery_stack.OUTPUT_DIR`.

---

## 3. Data contract

Every analysis below (except `dcc_garch`, which takes return series) consumes a **per-session wide book frame** `df`:

- **Index:** tz-aware `DatetimeIndex` on a 1-second grid (one trading session per frame).
- **Columns**, for each `ASSET ∈ {SPY, ES}` and level `i = 1..N`:
  `{ASSET}_bidprice_{i}`, `{ASSET}_askprice_{i}`, `{ASSET}_bidquantity_{i}`, `{ASSET}_askquantity_{i}`.

Mid prices are derived internally as `mid = (bidprice_1 + askprice_1)/2` (`cross_asset_pd_liquidity._mid`). SPY and ES need not share a price scale — distances are computed in percent (scale-free) and the VECM absorbs the level/basis via a per-day mean.

> **Per-level book — implemented.** `market_analysis_fixed.py` projects the snapshot book through `time_buckets → aggregated → filled → final` (helpers `_book_time_buckets_cols / _book_aggregated_cols / _book_filled_cols / _book_final_cols`, mirroring the AUC passthrough), emitting `bidprice_{i} / askprice_{i} / bidquantity_{i} / askquantity_{i}` for `i = 1..N`. After the asset-prefix rename in `fetch_aggregated_product` these become exactly the `{ASSET}_…` columns in the contract above. The per-bucket book is the **latest NBBO state in the bucket** (`MAX_BY` by exchange timestamp), **forward-filled** across empty buckets — the same convention used for `spread`. (Validated at the SQL-string level; a live Athena run is still the remaining empirical step.)

A **session list** used by the panel/robustness functions is:

```python
sessions = [(date, regime, df), ...]   # regime ∈ {"volatile", "benchmark"}
```

`price_discovery_shares.estimate_sample` instead takes `(label, regime, mid_spy, mid_es)` arrays; convert with `robustness._mid_sessions(sessions)`.

---

## 4. Quickstart (one session)

### Run everything — the driver

One file drives the entire stack. `run_analysis.py` loads (or extracts) the session frames, runs every stage, and writes one per-run folder `output/run_<ts>/` containing: the consolidated **final dataset** (`final_dataset.parquet`, or `final_dataset.csv.gz` when no parquet engine is installed), the **result tables** as one CSV per output under `tables/`, a readable `report.md`, and `summary.json` (headline scalars). No intermediate pickle or per-day CSVs are written; pass `--save-objects` if you also want the full Python objects pickled for reload, `--dataset-format {auto,parquet,csv}` to force a dataset format, and `--no-dataset` to skip the dataset (it is written for `load`/`extract` by default, and for `demo` only with `--save-dataset`).

```bash
# smoke-test the whole pipeline anywhere (synthetic data, fast):
python run_analysis.py --source demo --quick

# real data you've already extracted to a pickle:
python run_analysis.py --source load --pickle output/1s_aggregated_*.pkl \
       --volatile 2024-08-05,2024-10-31

# extract fresh from MayStreet and run (MIDAS only):
python run_analysis.py --source extract --interval 1s --n-levels 10

# sub-second: --interval sets the grid AND the estimation path (auto lags/horizons/
# fleeting filter, and the Lee-Mykland jump classifier). Loaded frames are resampled
# to the grid; a finer grid than the data's native one is kept (no upsampling).
python run_analysis.py --source load --pickle output/100ms_aggregated_*.pkl --interval 100ms
```

Stages run defensively (one failure is logged, the rest continue) and can be selected with `--only liquidity_conditional,panel` or `--skip dcc,robustness`. **`--interval` is the master frequency switch**: it sets the aggregation grid (`1s`, `100ms`, `10ms`, …) *and* the estimation path — VECM lags, IRF horizons, and the bootstrap block auto-scale to hold wall-clock roughly constant (via `frequency_defaults`), the fleeting-quote filter engages on the OFI inputs sub-second, and the jump split switches to the Lee-Mykland local-volatility classifier at ≤100ms. For `--source extract` it drives the MayStreet aggregation; for `--source load` the frames are resampled to the grid (coarsen-only — a finer target than the data's native grid is kept, with a warning). The resolved grid/lags/horizons/filter/jump-method are logged at startup and written to the report header and `summary.json`. Other useful flags: `--n-lags` (override the auto value), `--jump-method {auto,truncation,lee_mykland}`, `--window`, `--max-workers`, and `--quick` (smaller bootstrap/horizons). The pipeline order is: `liquidity_curves → information_shares → liquidity_conditional` (runs **both** the relative-depth and the FPCA state) `→ ecm_sde → panel → cross_impact → dcc → irf → jumps → robustness → microstructure`. The headline (the liquidity-conditional information share rising in the relative-depth/FPCA state, with its panel slope and bootstrap CI) lands at the top of the report and the JSON. Note `dcc` is the slow stage — the GARCH MLE dominates wall-clock (tens of seconds per fit); `irf`/`dcc` run on the first session by design, the panel/robustness stages pool all of them.

The sections below show each piece individually; the driver simply calls them in order.

### Starting point — generating the session frames

Everything downstream consumes the per-session book frames emitted by `market_analysis_fixed.py` (step 1 of the revision plan). Run it directly:

```bash
python market_analysis_fixed.py        # extracts the event-day frames, returns them in memory (writes nothing)
```

It fans out one Athena query per `(date, product)` across a thread pool (`max_workers`, default 4 — raise toward your Athena concurrency limit, lower if throttled), then reassembles in date order, joining SPY and ES per day into one wide frame. It returns a `List[(date, DataFrame)]` in memory — exactly the `sessions` object the panel and robustness functions take, and each `DataFrame` is the per-session frame described in §3. (Running it directly no longer writes files; use `run_analysis.py --source extract` to persist the consolidated dataset and the result tables.) Programmatically:

```python
from market_analysis_fixed import main
results = main(interval="1s", max_workers=4)     # [(date, df), ...]
date, df = results[0]                            # df -> feed the calls below
```

For the 100ms / 10ms variants, pass a finer `interval`; see §5.10 and the §8 sampling-frequency note. (The per-level book is now projected into the final frame — see the §3 "Per-level book" note — so the depth-curve metrics work directly off the output.)

### One session, end to end

```python
import numpy as np
import liquidity_curve_metrics as lcm
import price_discovery_shares as pds
import cross_asset_pd_liquidity as ca
import cross_impact as ci

# df = one session's book frame (see §3; e.g. results[0][1] above)
mid_spy, mid_es = ca._mid(df, "SPY"), ca._mid(df, "ES")
N = 10

# (a) liquidity-curve features for one side/asset
bid_feats = lcm.side_curve_metrics(df, "ES", "bid", N, fill_targets=(500, 2000))

# (b) price-discovery shares (IS bounds+mid, MIS, CS for both markets)
shares = pds.estimate_day(mid_spy, mid_es, n_lags=5)

# (c) HEADLINE: price discovery as a function of the book state
state = ca.relative_depth_state(df, N)                     # log(depth_ES) - log(depth_SPY)
lc = ca.liquidity_conditional_vecm(mid_spy, mid_es, state, n_lags=5)
print(lc["shares_by_state"])      # CS_ES / IS_ES at the 10/50/90th pct of the state
print(lc["t_delta_es"])           # significance of the liquidity x error-correction interaction

# (d) cross-impact matrix (rows = return, cols = OFI)
print(ci.cross_impact_matrix(df, n_levels=N)["Lambda"])
```

---

## 5. Per-analysis usage

### 5.1 Liquidity curve (`liquidity_curve_metrics`)

```python
lcm.side_curve_metrics(df, "SPY", "ask", N)          # scalar features per 1s
lcm.normalized_curves(df, "ES", "bid", N)            # L2-normalized / min-max / share vectors
lcm.book_asymmetry(df, "ES", N)                      # bid-vs-ask shape cosine, depth imbalance
lcm.cross_asset_curve_similarity(df, "SPY", "ES", "bid", N)   # SPY<->ES shape cosine, depth log-ratio
# near-book: the SAME AUC / arc-length, restricted to [touch -> weighted price]
lcm.near_book_metrics(df, "SPY", "bid", N)           # nb_span, nb_auc_shape, nb_auc_depth, nb_arc_len
lcm.near_book_metrics(df, "SPY", "bid", N, mode="fill", target=Q)   # fill-VWAP endpoint variant
```

`near_book_metrics` confines the curve functionals to the economically active band between the touch and the **weighted (center-of-mass) price** — the part of the book withdrawn first under stress, so far more stress-responsive than the full-N-level versions and the natural early-warning liquidity signal for the contagion redo (it sharpens the state for `ecm_sde`/`spread_conditioned_is` and the outcome for `mwcb_event_study`). The window is endogenous, so the shape metrics (`nb_auc_shape` ∈ [0,1] convexity, `nb_arc_len` the scale-free ||L||) are normalized and the **`nb_span`** (touch→weighted distance) is reported as its own concentration scalar; its bid-vs-ask asymmetry is near-book directional pressure. `mode="fill"` bounds the band by a fill-VWAP for a target size instead. When the book is so concentrated that the weighted price falls inside the first level gap (fewer than 2 levels in the band), the shape metrics return NaN — there is no near-book curvature to measure, and `nb_span ≈ 0` flags it.

### 5.2 Price-discovery shares (`price_discovery_shares`)

```python
mids = robustness._mid_sessions(sessions)            # -> (label, regime, mid_spy, mid_es)
per_day = pds.estimate_sample(mids, n_lags=5)        # per-session IS/MIS/CS distribution
pds.compare_regimes(per_day, metric="CS_ES")         # permutation test volatile vs benchmark
pds.panel_vecm(sessions=mids, n_lags=5)              # pooled day-FE estimation + regime interaction
```

**Lag-order selection (information criterion).** The VECM lag is no longer a hard-coded constant — pass `criterion="bic"` (or `"aic"`/`"hqic"`) and the order is chosen by that criterion instead of the fixed `n_lags`. BIC is the right default here: it is consistent for the true order (AIC over-fits asymptotically), and a parsimonious VECM keeps the parameter count in $\Omega$ and the $\Psi\alpha_\perp$ map down, which tightens the Hasbrouck bounds. Every candidate is fit on the **common sample** of the largest model (so the criterion can't favour higher $p$ just by scoring fewer rows), and $p$ here is the VECM lagged-difference order (one less than the levels-VAR order; $p=0$ is a pure error-correction model).

```python
p_star, ic = pds.select_lag(mid_spy, mid_es, pmax=12, criterion="bic")   # one session + IC table
p_pool, _  = pds.select_lag_pooled(price_pairs, pmax=12, criterion="bic")# one order across sessions
# pooled is the default for the panel: ONE order (min of the summed criterion) for comparability
per_day = pds.estimate_sample(mids, criterion="bic", pmax=12, lag_selection="pooled")
per_day.attrs["pooled_lag"]                          # the chosen order; per_day["n_lags"] echoes it
```

`lag_selection` ∈ {`"pooled"` (default; one order across all sessions — what you want when you pool/compare days), `"per_day"` (each session selects its own), `"fixed"`}. **`criterion=None` reproduces the fixed-`n_lags` results exactly**, so nothing downstream changes unless you opt in. The same hooks reach every estimator in the stack: `markov_switching_vecm.fit_ms_vecm(..., criterion="bic")` selects on the single-regime VECM then fits the switching model at that order; `correlation_svar.correlation_irf(..., criterion="bic")` selects the SVAR order on the (regime-pooled) variable matrix via the generic `pds.select_lag_var`; `cross_asset_pd_liquidity.liquidity_conditional_vecm(..., criterion="bic")` and `build_window_panel(..., criterion="bic")` select on the base bivariate VECM (one pooled order applied to every window); and `mean_variance.run_mean_variance(..., criterion="bic")` selects one pooled order across sessions for the layered mean+variance fit. From the CLI, `run_contagion --lag-criterion {bic,aic,hqic} --max-lags 12` drives the descriptive, panel, **and** mean-variance paths (`fixed` = use `--n-lags`). The Rigobon ID inherits the SVAR's order (it consumes those residuals). Cointegrating rank stays fixed at $r=1$ for the SPY/ES pair; the Johansen rank test (a robustness toggle, and structural once the asset count grows) is the second-pass item, not part of this change.

`information_leadership_share()` (Putniņš ILS) is intentionally a **stub** — the measure is contested (Shrestha & Lee 2023; Shen, Zhang & Zivot 2025). Report **IS + MIS + CS**; handle the noise point qualitatively.

### 5.3 Integration (`cross_asset_pd_liquidity`)

```python
# windowed panel joining per-window shares + liquidity features + OFI + RV
panel = ca.build_window_panel(sessions, window="30min", n_levels=N, n_lags=5)

# does the deeper/flatter book lead? within-day FE, day-clustered SEs
ca.panel_regression(panel, y="IS_mid_ES", X_cols=["rel_logdepth", "rcorr"])

# information shares within terciles of the book state
ca.state_split_shares(panel, state="rel_logdepth")
```

### 5.4 Cross-impact (`cross_impact`)

```python
ci.cross_impact_matrix(df, n_levels=N, hac_lags=10)  # K x K Lambda + HAC t-stats
ci.compare_impact_depth(df, n_levels=N)              # best-level vs multi-level (cross terms shrink)
ci.cross_impact_panel(sessions, n_levels=N)          # per-session matrices + regime means
```

**How to read it.** Λ is the K×K contemporaneous (within-bar) map from order flow to returns: the diagonal Λ_ii is *own* price impact (own OFI → own return), the off-diagonal Λ_ij is *cross*-impact (asset j's OFI → asset i's return), each with a HAC t-stat. A significant off-diagonal says ES order flow moves SPY's price *contemporaneously* — the simultaneous cross-market link the recursive ordering would otherwise assume away (it is exactly what identifies the structural IRF in §5.7). `compare_impact_depth` shows the cross terms *shrink* as you add depth beyond the best level (Cont-Cucuringu-Zhang): much of the apparent cross-impact is really deeper own-book pressure, a spec/robustness result. `cross_impact_panel`'s calm-vs-stress means say whether cross-impact strengthens under stress. Pitfall: Λ is contemporaneous structure, *not* lead-lag — for "who moves first," read the IRF (§7.8); Λ answers "who moves whom within the bar."

### 5.5 DCC-GARCH-X (`dcc_garch`)

```python
import dcc_garch as dg
r = np.column_stack([np.diff(np.log(mid_spy)), np.diff(np.log(mid_es))])
X = realized_vol_proxy[1:]                            # optional predetermined variance driver
fit = dg.dcc_garch_x(r, X=X)
fit["rho"]          # time-varying SPY-ES conditional correlation
fit["dcc_a"], fit["dcc_b"]                            # DCC persistence (a + b < 1)
fit["marginals"]["SPY"]["gamma"]                      # GARCH-X covariate loading
```

**How to read it.** `rho` is the time-varying SPY-ES correlation of the *standardized* (devolatilized) residuals — split it calm-vs-stress for the Forbes-Rigobon contagion read (a stress rise here is contagion, not the mechanical vol effect; full reading in §7.6). `dcc_a + dcc_b` is the correlation persistence (near 1 = ρ_t moves slowly and persistently). `gamma` is the GARCH-X variance loading: positive = the covariate (e.g. AUC_decay, a realized-vol proxy) forecasts higher own-return volatility. Pitfall: the variance driver X must be *predetermined* (lagged); a contemporaneous X biases γ and conflates with the very volatility it is meant to forecast.

### 5.6 Robustness (`robustness`)

```python
import robustness as rb
rb.lag_sensitivity(sessions, lags=(3, 5, 10, 20))
rb.beta_fixed_vs_estimated(sessions)                 # beta_hat ~ 1 validates (1,-1)
rb.subsample_am_pm(sessions)
rb.window_size_sensitivity(sessions, windows=("10min", "20min", "30min"))
rb.alt_state_variable(sessions, kinds=("rel_logdepth", "rel_slope", "rel_entropy"))
rb.stationary_bootstrap_ci(per_day["CS_ES"].to_numpy(), mean_block=3)
```

**How to read it.** Each function asks whether the headline (ES leads / CS_ES > 0.5) survives one modeling choice: `lag_sensitivity` — CS_ES stable across lag lengths (not an artifact of L); `beta_fixed_vs_estimated` — β̂ ≈ 1 validates the imposed (1,−1) cointegration; `subsample_am_pm` — the lead holds in both halves of the session (not an open/close effect); `window_size_sensitivity` / `alt_state_variable` — the conditioning result is robust to the window and to how the book state is defined; `stationary_bootstrap_ci` — a dependence-robust CI on CS_ES. Read them as a panel, not one at a time: the result is credible when CS_ES stays on the same side of 0.5 with overlapping magnitudes across all of them. A sign flip under any single choice is itself a finding — the leadership is specific to that specification, and you report *which* one.

### 5.7 Impulse responses & FEVD (`irf`)

```python
import irf
state = ca.relative_depth_state(df, N)

# cross-asset LP IRF: SPY price response to a 1-SD ES-OFI impulse, by book state
irf.local_projection_irf(df, impulse="ES", response="SPY", state=state,
                         horizons=range(0, 16))        # theta_low / theta_high = state-dependent IRF

# structural IRF identified off the cross-impact matrix (not Cholesky)
si = irf.structural_vecm_irf(df, horizons=range(0, 16), n_lags=5)
irf.fevd_from_irf(si["return_irf"], H=15)              # rows sum to 1: own vs cross OFI shares
```

**How to read it.** `theta` at horizon *h* is the cross-asset price response (SPY to a 1-SD ES-OFI impulse, *h* seconds out); `theta_low`/`theta_high` are that response at a relatively thin vs deep book; `fevd_from_irf` rows are the own-vs-cross OFI variance shares. The full reading — the spillover half-life (where the band first covers zero), the state-dependence as the dynamic face of the liquidity-conditional result, and the off-diagonal FEVD as the dynamic counterpart of the cross-asset information share — is in §7.8.

### 5.8 Sub-second covariance (`noise_robust_cov`)

At 100ms/10ms, naive realized variance explodes (microstructure noise) and naive realized correlation is attenuated (Epps effect from SPY/ES asynchronicity). Use the noise- and async-robust estimators; the covariance functions want each asset's **irregular** quote-update times `(t, logprice)` (extract from the raw feed), while the variance estimators take any fine series.

```python
import noise_robust_cov as nrc

# noise-robust variance (cross-check two estimators)
nrc.realized_variance(logmid_es, method="tsrv")          # Two-Scale RV
nrc.realized_variance(logmid_es, method="kernel")        # Realized Kernel (Parzen)

# async-robust covariance and robust correlation (drop-in for realized_moments)
nrc.hayashi_yoshida(t_spy, lp_spy, t_es, lp_es)
m = nrc.noise_robust_moments(t_spy, lp_spy, t_es, lp_es) # rv_spy, rv_es, cov, corr

# multivariate route: refresh-time sync + realized kernel
tau, P = nrc.refresh_time_prices([t_spy, t_es], [lp_spy, lp_es])
nrc.realized_kernel_cov(np.diff(P, axis=0))

# diagnostics for the frequency-robustness exhibit
nrc.epps_curve(t_spy, lp_spy, t_es, lp_es, intervals=[0.01, 0.1, 1.0])
nrc.signature_plot(t_es, lp_es, intervals=[0.01, 0.1, 1.0])
```

Feed `m["corr"]` / the realized covariance into `dcc_garch` and the panel `rcorr` control instead of `realized_moments` whenever the grid is sub-second.

### 5.9 Staleness & discreteness diagnostics (`microstructure_diagnostics`)

Attach these to the information-share tables: they tell you where staleness and tick discreteness make the shares untrustworthy, and in which direction the bias runs.

```python
import microstructure_diagnostics as md

# per-asset staleness / discreteness / noise for the current grid
md.staleness_report(df)              # zero_return_frac, frac_le_1tick, mean_abs_ticks_nz, noise_to_signal

# the exhibit: how the three pathologies scale with sampling interval
md.frequency_diagnostics(t_es, mid_es, tick_size=0.25,
                         intervals=[0.01, 0.05, 0.1, 0.5, 1.0])
```

A high `zero_return_frac` for SPY relative to ES means SPY looks "stale" and its Gonzalo-Granger share is biased upward — read a high-frequency `CS_SPY` against this column. Rising `frac_le_1tick` / falling `mean_abs_ticks_nz` flags the discreteness regime where the Gaussian VECM weakens.

### 5.10 Frequency scaling & fleeting-quote filtering (sub-second)

The stack is grid-agnostic; these helpers hold the wall-clock window fixed and suppress flicker as you move to 100ms / 10ms.

```python
ca.frequency_defaults(df)     # {dt, n_lags, horizons, block, min_rest_steps} scaled to the grid

# fleeting-quote (flicker) filter on the OFI inputs; inert at 1s, engages sub-second
ca.order_flow_imbalance(df, "ES", n_levels=10, min_rest_steps=5)

# event-level debounce for a RAW best-quote stream, upstream of gridding
ca.resting_time_filter(t, best_bid, min_rest=0.05)
```

`build_window_panel`, `local_projection_irf`, and `structural_vecm_irf` accept `n_lags=None` / `horizons=None` (auto-scaled from the grid via `frequency_defaults`) and a `min_rest_steps` passthrough, so the same call adapts across 1s / 100ms / 10ms. Defaults reproduce the 1s behavior exactly (`min_rest_steps=0`, `n_lags=5`). The driver exposes all of this through a single `--interval` flag (see §4): it resolves the grid to `frequency_defaults`, scales the lags/horizons/block/fleeting-filter, switches the jump classifier to Lee-Mykland sub-second, and resamples loaded frames to the grid — so the whole 1s→100ms→10ms progression is one command-line argument.

### 5.11 Functional PCA on the book (`functional_liquidity`)

A functional-data view: each timestamp's depth-by-level profile is a curve, and FPCA returns the dominant eigen-modes of book-shape variation. The leading FPC score is a principled, data-driven replacement for the hand-built `relative_depth_state`.

```python
import functional_liquidity as fl

# data-driven relative liquidity state (drop-in for ca.relative_depth_state)
state = fl.relative_fpc_state(df, n_levels=N)            # 1s series; 'log' = level+shape, 'share' = pure shape
ca.liquidity_conditional_vecm(mid_spy, mid_es, state)

# inspect the modes: variance explained + the eigen-curves themselves
r = fl.asset_fpca(df, "ES", n_levels=N)
r["explained_variance_ratio"]                            # how many modes matter
fl.modes_of_variation(r, n_sd=2)                         # mean +/- 2 sqrt(lambda) * eigenfunction
fl.fpc_state_series(df, "ES", n_levels=N, k=1)           # one asset's FPC1 as a scalar state
```

It also slots into `robustness.alt_state_variable` as another candidate state. Domain is the level index 1..N (uniform quadrature → PCA on the sampled curves, with the functional interpretation retained); the machinery generalizes to a physical-distance / percent grid by passing `weights`.

---

### 5.12 Continuous vs jump price discovery (`jump_robust`)

Quote prices are jump-diffusions, and a vanilla information share silently blends the diffusive and the jump parts of the common factor's quadratic variation. This module separates them: it fits the same fixed-(1,-1) VECM as `price_discovery_shares`, splits the realized innovation covariance into a continuous part and a jump part (PSD-guaranteed truncation), and runs `hasbrouck_is` on each. The common-factor direction is shared, so the total IS is unchanged (and equals `pds.estimate_day`); what the split adds is **who leads diffusive price discovery (`ISc`) vs who reacts first to jumps (`ISj`)**, plus the jump share of the common factor's variance.

```python
import jump_robust as jr

# per-session continuous/jump information-share split (parallels pds.estimate_sample)
split = jr.estimate_sample_jump_split(mid_sessions, n_lags=5)     # ISc_*, ISj_*, jump_frac_cf, bns_cf_p
jr.continuous_jump_information_shares(mid_spy, mid_es, n_lags=5)   # one session; reports jump_frac_cf (trunc) AND jump_frac_cf_lm
jr.continuous_jump_information_shares(mid_spy, mid_es, method="lee_mykland")  # whole split on the local-vol classifier

# the estimators on their own (any return series r = diff(log mid))
jr.jump_variation(r)            # RV, IV (bipower), JV; jump_fraction (RV-BV) AND jump_fraction_lm (Lee-Mykland)
jr.bns_jump_test(r)             # Barndorff-Nielsen-Shephard / Huang-Tauchen ratio test
jr.lee_mykland(r)               # intraday jump TIMES (Gumbel threshold)
jr.continuous_jump_cov(rx, ry)  # PSD continuous & jump (co)variation matrices

# who reacts first to jumps — Lee-Mykland jump-time alignment across the two assets
jr.cojump_from_mids(mid_spy, mid_es, max_lag=2)   # n_cojump, lead_SPY, lead_ES, sign_agreement
```

The thesis prediction is testable here twice over: a deeper book should both carry the larger continuous information share *and* absorb jumps with less price impact, and the jump lead should itself shift with the relative-depth state under stress. **Jump fraction: prefer `jump_frac_cf_lm` (Lee-Mykland) over the truncation/`(RV−BV)` estimate.** The integrated and global-threshold measures scale jumps by a single per-day volatility, so at high frequency they load fat tails, microstructure noise, and tick discreteness into the jump bucket and inflate the fraction (on a Student-t, no-jump series the realized fraction reads ~0.16 against a true zero); Lee-Mykland normalizes each return by a *local* bipower volatility with a FWER-controlled Gumbel threshold, which deflates that bias (~0.07 on the same series) while still catching genuine multi-sigma jumps, and it cleanly drives the whole split via `method="lee_mykland"`. Run it on **100ms** returns so the trailing window (K≈√n) has enough points to estimate local vol. The truncation multiple `c` and the LM `alpha` are the size knobs. This is the residual-innovation analogue of the high-frequency QV split — the model is stated once and the economic split is foregrounded, not Itô-with-jumps formulae for decoration. A full bivariate Hawkes treatment of the order flow (whose static linear shadow is the cross-impact matrix) is deliberately left as a separate paper.

### 5.13 The ECM-SDE & noise-robust observables (`ecm_sde`, `robust_prices`)

The liquidity-conditional VECM is the discretization of a state-dependent error-correction SDE, `dY = α(S)(β′Y)dt + Σ(S)^½ dW (+ dJ)`, β=(1,−1). `ecm_sde` estimates it: the loadings α(S) become a function of the book state, so the price-discovery "derivative" `ψ(S)=α(S)⊥` and the information share `IS_j(S)=[ψ(S)′Σ(S)^½]_j² / ψ(S)′Σ(S)ψ(S)` are now *curves*. The coefficient on z·S (with day-clustered HAC t-stats) is the single-number test that the loadings move with liquidity — the continuous-time statement of the whole thesis. The information share is a scale-invariant ratio, so `dt` cancels and the discrete estimate targets the continuous-time object directly. The SDE earns the notation only through this state-dependence (in a constant-coefficient model every "derivative" is a constant); the coupled-book SPDE — lifting prices to the full depth profile `D(t,x)` with a moving free boundary — is a separate theory paper, not this one.

The information share is only as clean as the price it is computed on, and the naive `(b1+a1)/2` carries bid-ask bounce and tick discreteness that act as i.i.d. measurement noise and bias every share toward 0.5. `robust_prices` replaces it on two axes: the **observable** integrates the depth curve (microprice, book centroid, area-under-curve/cost-to-fill mid — *weighted midpoints, area under curve, curve length*), and the **measure** is taken on a coarser grid (sparse sampling / pre-averaging) where noise is a smaller share of the innovation. Pre-averaging must be paired with subsampling — differenced at the fine grid it injects an MA that collapses IS to 0.5 (`coarsen` does it correctly; the step must stay below the error-correction half-life or the dynamics wash out).

```python
import ecm_sde as es, robust_prices as rp

# the thesis test as a continuous-time ECM-SDE: a1 = d alpha / dS, day-clustered t-stat
fit = es.estimate_sample(sessions, state_fn=lambda df: ca.relative_depth_state(df, 10), degree=1)
fit["a1_SPY"], fit["t_a1_SPY"]      # loading gradient (the single-number liquidity-conditioning test)
fit["curve"]                         # IS(S)/CS(S)/kappa(S) over the state grid (DataFrame, has IS_ES)
es.fit_em_mle(mid_spy, mid_es, S)    # Euler-Maruyama pseudo-MLE with state-dependent diffusion

# run the SAME SDE on a noise-robust observable (depth-aware microprice)
pf  = rp.price_fn("microprice", n_levels=10)         # also 'com', 'auc', 'mid'
fit_rob = es.estimate_sample(sessions, state_fn=rp.state_fn("rel_logdepth"), price_fn=pf)
rp.noise_robust_is(sessions, scheme="microprice", step=10, pre_average=True)   # + coarser-grid measure

# the observables / states on their own
rp.microprice(df, "ES", 10); rp.auc_mid(df, "ES", 10); rp.center_of_mass_price(df, "ES", 10)
rp.book_state(df, 10, kind="rel_arclen")             # curve-length-normalized, scale-free
```

The driver's `ecm_sde` stage reports the loading gradient and IS(S) curve on both the raw mid and the microprice observable, so any gap between them is itself a microstructure-noise diagnostic (they coincide only when the book carries no sub-tick information). Real-data note: choose the coarsening `step` from the estimated basis half-life (`ln2/κ` from `fit["curve"]`), and reach for `robust_prices` whenever the raw-mid and microprice information shares disagree.

### 5.14 Reproducing Table 5 — cross-market order-imbalance contingency (`tandem_order_flow`)

`tandem_order_flow` rebuilds **Table 5** of the published tandem-trading paper: the 3×3 (Sell/Neutral/Buy)² cross-market order-imbalance matrix, in all three layers. This is the order-*flow* counterpart to the price-discovery VECM, and it consumes a **different input object** — the per-bar new-order buy/sell *counts* (Eq. 1), i.e. the harmonized message stream — **not** the depth book frames the rest of the stack uses. Returns/correlation for 5.III come from the mids. Pure numpy/pandas.

```python
import tandem_order_flow as tof
# 5.I theoretical nulls (rows = Future, cols = ETF, percent):
tof.theoretical_null(rho=0.0)                 # Panel A independent binomial (null 1A)
tof.theoretical_null(rho=1.0)                 # Panel B comonotone rho=1 (null 1B)
tof.expected_return_signs()                   # the (ETF, Future) expected-sign matrix
# 5.II + 5.III from per-bar arrays (one sample):
res = tof.table5_from_series(buy_etf, sell_etf, buy_fut, sell_fut,
                             ret_etf, ret_fut, corr_window=100, ret_in_bps=False)
res["frequency"]          # 5.II: % of bars per cell, with All margins
res["pcmof_ncmof"]        # PCMOF = Sell-Sell + Buy-Buy ; NCMOF = Sell-Buy + Buy-Sell
res["dcorr_x1000"]        # 5.III: mean Δ(100-bar return corr) ×1000 per cell
res["ret_etf_bps"], res["ret_fut_bps"]        # 5.III: mean returns (bps) per cell
# pool a session list, split by regime, via an extractor you supply:
tof.table5_from_sessions(sessions, counts_fn=lambda df: (.../* 6 per-bar arrays */))
```

The bins are the paper's: `Sell < 0.45 ≤ Neutral ≤ 0.55 < Buy`, with `NA` when a market has no new orders in the bar (kept as its own row/col, not folded into Neutral). The nulls use the normal-approximation binomial tail the paper used (`Φ((0.45−½)/√(¼N))` with N≈505 ETF, 112 Future), which reproduces the printed marginals (1.23/97.54/1.23 and 14.50/71.01/14.50); pass `method="exact"` to `binom_state_probs` for the exact pmf. The only practical work on your side is `counts_fn`: the depth frame alone does not carry new-order side counts, so point it at your per-second message aggregation (new-buy / new-sell per market). The IRF tables (9, 11, 13) are a *separate* object — the SVAR of Eqs. (5)–(6), not this contingency analysis.

### 5.15 Reproducing Table 9 — correlation IRF with spreads (`correlation_svar`)

`correlation_svar` rebuilds **Table 9** (impulse response of price-return correlation to order-flow / liquidity), but with **spread-based**, depth-frame-native liquidity proxies instead of the paper's message-type proportions, and it offers the structurally cleaner replacement. Spreads are in bps of the mid: the **standard** quoted spread (`quoted_spread`, a₁−b₁), a **weighted** spread (`weighted_spread`, the depth-weighted round-trip cost-to-fill `slippage_ask(Q)+slippage_bid(Q)`, or a size-weighted half-spread average), plus **informational proxies** — multi-level OFI (book pressure, replacing the crude NCMOF/PCMOF message states), realized variance, and the microprice−mid deviation.

```python
import correlation_svar as csv
# spread observables (one asset, bps):
csv.quoted_spread(df, "ES"); csv.weighted_spread(df, "ES", target_qty=None, kind="cost_to_fill")
# Table-9-style IRF: rows = shock variables, cols = regimes, impact response of d-corr x100
csv.correlation_irf(sessions, spec="standard")         # quoted spread + OFI
csv.correlation_irf(sessions, spec="weighted")         # + weighted spread
csv.correlation_irf(sessions, spec="informational",    # + OFI / RV / microprice deviation
                    ident="cholesky", corr_method="rolling")
csv.correlation_irf(sessions, spec="informational", ident="identity")     # the paper's shortcut
csv.correlation_irf(sessions, spec="informational", corr_method="dcc")    # DCC conditional corr LHS
csv.fevd_correlation(sessions, spec="informational")   # variance shares (sum to 1)
# the SAME table with inference built in: bootstrap SE + Romano-Wolf joint stars
res = csv.correlation_irf_inference(sessions, spec="weighted", n_boot=499)  # res["table"] is publication-ready
res["table"]; res["padj"]; res["joint"]               # coef*** (se) cells, FWER p-values, per-regime joint test
# the better object: spread as the conditioning STATE in IS(S), via ecm_sde
fit = csv.spread_conditioned_is(sessions, spread_kind="weighted"); fit["curve"][["S","IS_ES","CS_ES"]]
```

Two methodological dials matter. **Dependent variable**: `corr_method="rolling"` differences a 100-bar rolling return correlation (the paper); `"dcc"` uses the DCC conditional correlation (via `dcc_garch`) with no look-ahead-window artifact. **Identification**: the paper sets the shock covariance to identity and orders Δρ first, which would null the contemporaneous effects it reports; `ident="cholesky"` (the default) orders correlation **last** so every liquidity/flow shock hits it contemporaneously (recursive, futures-lead), and `ident="identity"` reproduces the paper's shortcut so the gap is visible. The variables enter in causal order (futures ES block, then equities SPY block, then Δcorrelation last); liquidity levels are differenced, OFI/RV/MicroDev enter as levels. Returns and spreads come from the **book frames**, so the only inputs Table 9 still needs from the message tape are trade volume and the NCMOF/PCMOF states — and the continuous OFI subsumes the latter.

**The better way (and the point of this stack).** Table 9 asks what *correlates with* return correlation; the structural question is how *price discovery* depends on the liquidity state. `spread_conditioned_is` answers that directly: it makes the relative (weighted) spread the conditioning state `S = log(spread_ES) − log(spread_SPY)` in the information-share curve and returns `IS(S)` via the ECM-SDE. A negative `IS_ES` gradient in `S` says ES leads price discovery precisely when its book is relatively tighter — the thesis, read off a spread state, replacing the reduced-form correlation SVAR with the state-dependent price-discovery object.

**IRF tables with joint significance built in.** `correlation_irf_inference` wraps `correlation_irf` with the inference layer so the table comes out publication-ready: each cell is the impact (or cumulative) response ×100 with a bootstrap SE and stars from a **Romano–Wolf** family-wise-error-controlled p-value (jointly over all shock × regime cells, or `rw_by_regime=True` within each regime). The resampling unit is the **trading day**: with a `(date, regime, df)` session list it runs a day-cluster bootstrap (resampling whole sessions), which — because each replication rebuilds the SVAR frame — also propagates the generated-regressor uncertainty in Δρ; with a single frame it falls back to a moving-block bootstrap of the built frame rows (Δρ then treated as fixed). It returns `point`, `se`, `tstat`, `padj`, `reject`, percentile `ci_lower/ci_upper`, a rendered `table`, and a per-regime `joint` summary (`n_shocks`, `n_reject`, `min_padj`). This is what `paper_tables.table_correlation_irf` calls under the hood, so the revamped Table 9 (continuous χ via `extra_fn`) prints `PCMOF 0.169*** (0.032)` etc. with the stars reflecting joint, not per-cell, significance. Note the bootstrap needs enough day-clusters to have power — a handful of sessions yields heavy-tailed resample outliers that make Romano–Wolf very conservative.

### 5.16 Continuous cross-market order flow (`cross_flow`)

`cross_flow` replaces the categorical PCMOF / NCMOF (Eqs. 1–2) with a continuous, signed, size-aware measure, fixing the leakage in the count-based, add-only, top-of-book imbalance by building on the multi-level CKS OFI (which already nets adds/cancels/trades). Three layers:

```python
import cross_flow as cf
g_s = cf.signed_ofi(df, "SPY"); g_e = cf.signed_ofi(df, "ES")
chi = cf.cross_flow(g_s, g_e)                 # in [-1,1]: + same-direction, - opposed
pcmof, ncmof = cf.pcmof_ncmof(chi)            # rectified halves; PCMOF - NCMOF == chi (drop-in for Eqs. 5-6)
cf.discretize(chi)                            # the paper's 3-state categorical as the thresholded special case
cd = cf.common_divergent(g_s, g_e)            # Hasbrouck-Seppi: common (PCMOF) vs divergent (NCMOF) flow
cd["common_share"], cd["divergent_share"]     # = (1±rho)/2, orthogonal, threshold-free
cf.comovement_rolling(g_s, g_e); cf.comovement_dcc(g_s, g_e)   # H3 stress object, measured directly
cf.return_correlation_hy(t_spy, lp_spy, t_es, lp_es)          # Epps-corrected return-corr LHS for Table 9
cf.cross_flow_features(df)                    # all regressors as a frame -> feed correlation_svar(spec=[...])
cf.bucketed_cross_flow(df, n_buckets=50)      # VPIN-style volume-clock PCMOF/NCMOF (per-bucket)
cf.cross_flow_features(df, clock="volume")    # same per-bar frame, but co-flow computed on the volume clock
```

`chi_t` is the product of bounded per-leg directional pressures `tanh(z(OFI))`: positive when both books push the same way, negative when opposed, with magnitude the joint strength. Its rectified halves `PCMOF = max(chi,0)`, `NCMOF = max(-chi,0)` keep the paper's same-/opposite-direction split and two-regressor structure but continuous and signed-via-magnitude, with the `0.45/0.55` dead-zone removed (and `discretize` recovers the categorical, showing exactly what it drops). The `common_divergent` rotation is the principled version — a fixed 45° rotation into orthogonal common flow `(z_SPY+z_ES)/√2` (market-wide, lifts correlation) and divergent flow `(z_SPY−z_ES)/√2` (relative/arbitrage, lowers it), with variance shares `(1±ρ)/2`. The conditional-comovement series (rolling + DCC of the two OFI series; Hayashi-Yoshida for the async return correlation) measure the H3 stress breakdown directly rather than inferring it from shrinking corner cells. Because `cross_flow_features` returns a frame, you can run Table 9 on the continuous flow via the `extra_fn` hook — `correlation_svar.correlation_irf(sessions, spec="weighted", extra_fn=lambda df: cross_flow.cross_flow_features(df)[["PCMOF","NCMOF"]])` (pick a non-collinear subset, since `chi = PCMOF − NCMOF`) — and let the information share depend on `chi` as a state to recover Figure 2's M/W nonlinearity without return dummies.

**VPIN-style volume clock (`bucketed_cross_flow`, `clock="volume"`).** PCMOF/NCMOF live on 1-second *clock* bars, where asynchronous trading (the Epps effect) biases cross-market co-directionality toward zero and intraday volume seasonality manufactures spurious co-direction. `bucketed_cross_flow` reformats them on a VPIN-style **equal-notional clock**: bars are bucketed on *combined* SPY+ES notional (one shared clock, not two per-instrument clocks — the cross-market wrinkle VPIN itself never faces), each market's signed OFI is summed within a bucket, and co-directionality is read off the bucket pair. The transferable piece of VPIN is exactly this — the volume clock and the `n`-bucket trailing mean (`PCMOF_roll`/`NCMOF_roll`) — *not* bulk-volume classification (which signs flow from returns and would make a return-correlation LHS circular; the size/cancel-aware OFI signing is kept) and *not* the absolute single-market imbalance (PCMOF/NCMOF is about the joint sign of two markets, so an absolute value would discard the measure). Each bucket carries `O_/I_` per market, the magnitude co-flow `chi` and its rectified `PCMOF`/`NCMOF`, the categorical `PCMOF_cat`/`NCMOF_cat`, the trailing means, and a `weight = min(activity_SPY, activity_ES)` (co-flow counts only when both legs trade — the thin-bucket guard). Real per-bar trade notional goes in via `notional`/`notional_fn`; absent that, a book-native `|OFI|·mid` activity-clock proxy is used. `cross_flow_features(df, clock="volume")` runs the bucketing and broadcasts each bucket's co-flow back to its bars, so the frame stays per-bar and usable as an SVAR regressor while the co-directionality is synchronized in information time. The volume clock is the uncontroversial half of VPIN; the contested half (BVC + the absolute-imbalance toxicity claim, cf. Andersen–Bondarenko) is deliberately not imported.

### 5.17 Stress index for the contagion redo (`stress_index`)

`stress_index` turns a daily conditional-volatility series (NYU V-Lab MF2-GARCH on SPY) into the stress objects the liquidity-contagion design needs, and encodes the recommended fixes to a one-day-change day-selection rule. The contagion regressions should condition on the **continuous** stress state, not binned days; binning is reserved for where discrete regimes are genuinely needed (Rigobon ID, Markov-switching).

```python
import stress_index as si
tau, g = si.long_short_decomposition(vol, long_run=vlab_long_run)   # persistent vs transient (pass V-Lab's long-run if available)
S = si.stress_state(vol, use="long_run")                            # continuous standardized state -> ecm_sde / spread conditioning
si.classify_regimes(tau, q_low=0.33, q_high=0.90)                   # calm / elevated / stress (Rigobon #2, Markov-switching #4)
si.select_days(vol, level_q=0.90, onset_thresh=0.5)                 # labels: stress = high level OR sharp onset; calm = low & stable
si.stress_surprise(realized_vol, forecast_vol)                      # ln(realized/forecast): look-ahead-free innovation
si.match_controls(panel)                                            # nearest-vol control per stress day (DiD #1)
si.broadcast_to_session(S, session.index)                          # daily state -> per-bar conditioning state
```

Three corrections to a `change > 0.5 / change < 0` rule, encoded here: (i) the one-day log-change is volatility *acceleration* — a good **onset** flag but it misses sustained-stress plateaus, so the **level** (ideally MF2's long-run `tau`) carries the regime; (ii) a vol *decline* is an outcome, not a treatment, and is endogenous to the contagion being measured, so it is never a treatment label — the MWCB treatment is the reopening asymmetry (§6 / item #1), dated at the release boundary; (iii) use the ex-ante one-step forecast or the realized−forecast **surprise** rather than full-sample smoothed vol for anything causal. The self-test demonstrates (i) directly: on a synthetic crisis path the pure-change rule flags 1 onset day while the recommended labeling flags 13 stress days (adding the plateau).

### 5.18 MWCB reopening-asymmetry event study (`mwcb_event_study`)

`mwcb_event_study` is the causal-identification layer (item #1): the March-2020 MWCB **reopening asymmetry** — futures book frozen-then-empty at release vs the ETF primary listing accumulated auction-style — as a sharp, exogenously-timed liquidity-contagion experiment. The outcome is a parameter, defaulting to the cost-to-fill spread:

```python
import mwcb_event_study as ev
# outcome in {'cost_to_fill' (default), 'inside_depth', 'composite'}
panel = ev.build_panel(treated, control, release_by_date, asset="SPY", outcome="cost_to_fill", target_qty=Q)
ev.did(panel)                       # 2x2 + regression DiD (treated*post coef = causal effect), date-clustered SE
ev.event_study(panel, bin_sec=30)   # outcome time-profile around release (Figure-3 analogue for liquidity)
ev.rd_in_time(panel, bandwidth_min=5)            # sharp RD at the release boundary; the jump at tau=0
ev.cross_market_spillover(treated, control, release_by_date, source="ES", target="SPY")  # contagion core
```

`treated` / `control` are `(date, df)` session lists, `release_by_date` maps each to its (pseudo-)release Timestamp (controls get the matched time-of-day; pair them with `stress_index.match_controls`). Outcomes are oriented so the sign is interpretable: `cost_to_fill` (bps) ↑ = more illiquid, `inside_depth` (summed top-`n_inside` bid+ask size) ↑ = more liquid, `composite` = `z(cost_to_fill) − z(inside_depth)` ↑ = more illiquid. The DiD's `treated*post` coefficient is the abnormal post-release liquidity move attributable to the reopening; `rd_in_time` is the discontinuity at the boundary (local-linear each side); and `cross_market_spillover` is the triple difference whose `spill_post_treated` coefficient is the *extra* source→target illiquidity transmission post-release on MWCB days — futures illiquidity spilling into the ETF book, the contagion the redo is about. For cross-day comparability pass a fixed `target_qty` to the cost-to-fill outcome. The self-test recovers a planted spillover (0.2 baseline, 0.8 post-treated amplification) and the correct DiD signs on both outcomes.

### 5.19 Order-free SVAR identification (`rigobon_id`)

`rigobon_id` removes the paper's arbitrary "futures lead" recursive ordering (item #2) by identifying the contemporaneous structural matrix through heteroskedasticity (Rigobon 2003): with the propagation matrix constant but the structural shock variances differing across the `stress_index` regimes, the regime-varying residual covariances over-determine the matrix and pin it down with no ordering.

```python
import rigobon_id as rg
# U = residuals of one reduced-form VAR on pooled data; regimes from stress_index.classify_regimes
res = rg.rigobon_identify(U, regimes, calm="calm", stress="stress", names=["ETF","ES"])
res["contemp_rigobon"]        # order-free contemporaneous spillover matrix C[i,j] = effect of j on i
res["contemp_cholesky_fwd"], res["contemp_cholesky_rev"]   # the two recursive orderings, for contrast
res["var_ratios"], res["eig_separation"]                   # identification strength (distinct ratios needed)
res.get("overid")             # 3+ regimes: off-diagonal mass that should be ~0 if constant-A holds
# contagion vs interdependence (Forbes-Rigobon 2002):
rg.forbes_rigobon_corr(x_crisis, y_crisis, x_calm, y_calm)   # rho_adjusted strips the vol-driven rise
```

The identification is the generalized-eigenvalue (simultaneous-diagonalization) solution: `M = Ω_calm Ω_stress⁻¹` has the columns of `A⁻¹` as eigenvectors and the structural shock-variance ratios as eigenvalues, so it needs **distinct** ratios (the regimes must differ in *relative* heteroskedasticity, reported as `eig_separation`). The headline contagion result — the contemporaneous cross-market coefficient and whether it is **asymmetric** (futures→ETF vs ETF→futures) — is now an estimate, with the two Cholesky orderings shown beside it to make the ordering dependence the paper assumes away visible (each ordering forces one direction of the pair to zero; Rigobon keeps both). `forbes_rigobon_corr` is the canonical contagion test: a crisis rise in comovement that survives the heteroskedasticity adjustment is contagion, one that does not is interdependence amplified by volatility. The self-test recovers a known asymmetric matrix order-free, separates a correct A from a wrong one via the over-ID statistic, and correctly declines to call pure heteroskedasticity contagion.

### 5.20 Bivariate Hawkes for liquidity-event contagion (`hawkes_cross`)

`hawkes_cross` is the continuous-time generative model of the spillover (item #3): a multivariate Hawkes process whose marks are **liquidity events** (depth-withdrawal or spread-jump times per market). The clustering the paper documents (Table 5 corners, Table 6 continuation) is self- and cross-excitation; the cross kernel is the spillover and the branching ratio is reflexive fragility.

```python
import hawkes_cross as hk
e_spy = hk.liquidity_events(df, "SPY", kind="depth_withdrawal", q=0.95)   # event times (s)
e_es  = hk.liquidity_events(df, "ES",  kind="depth_withdrawal", q=0.95)
fit = hk.fit_hawkes([e_spy, e_es], T)        # MLE: mu, alpha, beta, branching_matrix, branching_ratio
fit["branching_matrix"]                       # G_ij = alpha_ij/beta_j = type-i events per type-j event
fit["branching_ratio"]                        # spectral radius: endogeneity / reflexivity (<1 stationary)
hk.poisson_loglik([e_spy, e_es], T)           # no-excitation benchmark Hawkes should beat
hk.stress_conditional_fit(e_calm, T_c, e_stress, T_s)   # H3: does cross-excitation/branching rise in stress
hk.simulate_hawkes(mu, alpha, beta, T)        # Ogata thinning (self-test / parametric bootstrap of the ratio)
```

The conditional intensity is `λ_i(t) = μ_i + Σ_j Σ_{t_k^j<t} α_ij exp(−β_j(t−t_k^j))`, fit by exact recursive MLE (exponential kernels, O(N·D)) via L-BFGS-B. The **branching matrix** `G_ij = α_ij/β_j` is the expected number of type-i liquidity events directly triggered by one type-j event, so its off-diagonal is the directed contagion intensity (ES→SPY vs SPY→ES asymmetry) and its **branching ratio** (spectral radius, < 1 for stationarity, → 1 near-critical) is the fraction of events that are endogenously triggered — reflexivity. `stress_conditional_fit` is the dynamic H3 test: the self-test shows the branching ratio rising 0.44→0.55 and ES→SPY cross-excitation 0.18→0.36 from calm to stress. Marks are book-native (depth/spread), so no message tape is needed; the estimator returns ~0 branching when events are genuinely independent, so it does not manufacture contagion. This is the dynamic counterpart to the static `cross_impact` λ-matrix.

### 5.21 Markov-switching VECM (`markov_switching_vecm`)

`markov_switching_vecm` makes the calm/stress regimes a **latent state** estimated endogenously (item #4), replacing the paper's hand-picked high-vol days and medium-return dummies. It is the latent-discrete-state complement to the continuous-observable-state `ecm_sde`: the stress regime is where the SPY–ES error correction (arbitrage) weakens and the lead shifts — the H3 breakdown, dated by the model.

```python
import markov_switching_vecm as ms
fit = ms.fit_ms_vecm(Y, K=2, p=1, beta=1.0)   # Y = log prices [SPY, ES]; beta=1 for the log-price spread
ms.regime_summary(fit)                          # per regime: EC speeds, vol, GG shares, ergodic prob, calm/stress label
fit["P"], fit["ergodic"], fit["smoothed"]       # transition matrix, stationary probs, smoothed regime path
fit["alpha"][fit["stress_regime"]]              # error-correction speeds in the stress regime
ms.gonzalo_granger_shares(fit["alpha"][k])      # who leads price discovery in regime k
ms.single_regime_loglik(Y, p=1, beta=1.0)       # no-switching benchmark the model should beat
```

The model is `dY_t = c(k) + α(k)(β'Y_{t-1}) + Σ Γ_i(k) dY_{t-i} + ε_t`, `ε_t ~ N(0,Σ(k))`, `S_t ~ Markov(P)`, estimated by exact EM (Hamilton filter for the likelihood and filtered probabilities, Kim smoother for the smoothed and joint probabilities, weighted-LS M-step). β is the cointegrating vector (estimated by a first-stage regression, or supplied — `beta=1` for the log-price spread). The stress regime is identified as the high-innovation-variance state; `regime_summary` reports its **error-correction speed** (which should be weaker — arbitrage breaking down) and its **Gonzalo-Granger information share** (the lead shifting). The self-test recovers exactly that: EC strength 0.42 (calm) → 0.07 (stress), the smoothed stress probability tracks the true regime at corr 0.95, the SPY information share moves 0.47 → 0.94 across regimes, and the switching log-likelihood beats the single-regime VECM by ~1185.

### 5.22 Hasbrouck pricing error + dollar welfare cost (`pricing_error`)

`pricing_error` decomposes price into a random-walk efficient component and a transitory pricing error (item #5), and prices the dislocation in dollars. σ_s = sd of the transitory component is the magnitude of mispricing the MWCB reopening asymmetry injects; it is reduced-form identified (Beveridge–Nelson), no ordering needed.

```python
import pricing_error as pe
dec = pe.market_quality(df, "SPY")            # sigma_s, sigma_w (bps), noise_ratio for one market
pe.pricing_error_decomposition(z, p=15)       # core: z = (Delta p, order flow); returns sigma_s, sigma_w
pe.welfare_cost(sigma_s_bps, notional)        # dollar cost = notional x avg fractional transitory deviation
pe.dislocation_cost(dec_calm, dec_stress, notional_stress)   # rise in sigma_s + its incremental dollar cost
pe.relative_pricing_error(df, "SPY", "ES")    # transitory SPY-ES basis dislocation (bps) -- the contagion object
```

From the VMA of `(Δp, order flow)`, `σ_w² = θ(1)Ωθ(1)'` is the permanent (efficient) variance and `σ_s² = Σ_k θ*_k Ω θ*_k'` (tail sums `θ*_k = Σ_{j>k} θ_j`) is the transitory pricing-error variance — the Beveridge–Nelson cycle, identified from the reduced form. `market_quality` runs it per market on `(mid return, signed OFI)`; the self-test recovers the Roll-model truth (σ_s 0.20, σ_w 0.30) essentially exactly and gives ~0 σ_s for an efficient price. `welfare_cost` turns σ_s (bps) into dollars via notional × expected absolute deviation (`√(2/π)·σ_s`), and `dislocation_cost` reports the calm→stress rise in σ_s and its incremental dollar cost — the "1–2% of volatility" claim recast as a price-efficiency welfare number. `relative_pricing_error` is the cross-market object: the transitory (high-pass) deviation of the SPY–ES basis, whose stress-increase is the contagion-driven dislocation (a full bivariate-VECM pricing-error decomposition is the rigorous extension; the high-pass deviation avoids over-differencing the stationary basis).

### 5.23 Tick-size-aware price-discovery correction (`tick_correction`)

`tick_correction` referee-proofs the futures-dominance result (item #6). SPY and ES sit on different price grids — ES's tick is the coarser one in fractional terms (0.25/5500 ≈ 0.45 bps vs SPY's 0.01/550 ≈ 0.18 bps), consistent with "1 ES tick ≈ 2.5 SPY ticks" in index points. A coarser grid injects rounding noise that contaminates the Hasbrouck information share, so the asymmetry could be a discreteness artifact. This module re-estimates the IS on equal footing.

```python
import tick_correction as tk
tk.information_share(Y, beta=1.0)                       # raw Hasbrouck IS (lower/upper/mid) + GG from a VECM
res = tk.tick_corrected_information_share(Y, ticks=(0.01, 0.25), prices=(550, 5500), beta=1.0)
res["table"]            # per market: raw / common_grid / rounding_corrected IS_mid, tick-noise share, frac tick (bps)
res["leader_raw"], res["leader_common_grid"], res["leader_rounding_corrected"], res["lead_survives"]
```

Two corrections, both reusing the VECM + `hasbrouck_is_from_alpha` machinery: **common-grid** rounds both log prices to the same (coarser) fractional tick so neither instrument has a discreteness advantage, then refits (assumption-light); **rounding-corrected** subtracts each market's rounding-noise variance `δ_i²/6` (`δ_i = tick_i/price_i`) from the innovation-covariance diagonal — a first-order de-rounding, a tractable stand-in for the Hasbrouck-1999 rounded-observation state space. It also reports each market's **tick-noise share** of return variance (the discreteness disadvantage, mechanically larger for the coarser grid). The self-test shows both halves of the argument: a coarse ES grid imposed on a symmetric DGP creates a spurious SPY lead (raw IS 0.78) that both corrections remove (→ 0.48 / 0.50), while a genuine ES lead survives all three estimators (≈ 0.58). Run it on the real ticks to report whether the paper's futures dominance is genuine or grid-induced.

### 5.24 Inference: generated-regressor & bootstrap SEs, multiple testing (`inference`)

`inference` supplies the standard errors and multiple-testing control the paper omits (the rest of item #7). Δρ is an estimated rolling/DCC correlation used as both the SVAR dependent variable and a Table-14 regressor, so naive OLS SEs are wrong twice over — they ignore its first-stage estimation error and its overlapping-window serial correlation — and the IRF tables star dozens of coefficients with no family-wise control.

```python
import inference as inf
boot = inf.moving_block_bootstrap(data, statistic, n_boot=499, block_len=None)   # dependent bootstrap engine
inf.bootstrap_summary(boot, point)                       # SE, percentile & basic CIs, bootstrap t
inf.generated_regressor_bootstrap(data, generate_and_fit)  # two-stage: regenerate Delta-rho each replication
inf.romano_wolf_from_boot(point, boot_draws)             # step-down FWER, correlation-robust, > Bonferroni
inf.holm(pvals); inf.benjamini_hochberg(pvals, q=0.05)   # FWER / FDR on a vector of p-values
```

`moving_block_bootstrap` resamples overlapping row-blocks (length ~ `n^{1/3}`, inflated for persistence via `auto_block_length`) to preserve dependence; when the statistic re-generates Δρ inside it, the first-stage uncertainty is propagated automatically (`generated_regressor_bootstrap` is the documented two-stage wrapper). For the IRFs this is already packaged in `correlation_svar.correlation_irf_inference` (day-cluster bootstrap + Romano–Wolf joint stars, returning a publication-ready table); reach for the primitives here only when you need a bespoke statistic. `romano_wolf_from_boot` runs the Romano–Wolf step-down on the studentized statistics, recentering the bootstrap draws to the null and using the bootstrap **max-statistic** distribution — robust to cross-correlation among coefficients and more powerful than Bonferroni/Holm (a lower critical value under positive dependence). The self-test confirms each piece: the block bootstrap recovers the AR(1) long-run SE (≈ 2× the iid SE), the naive OLS SE on a rolling-correlation regression is a 4–5× understatement of the two-stage bootstrap SE, and Romano–Wolf rejects all planted signals with zero false positives at a critical value below Bonferroni's.

### 5.25 Reporting layer: paper tables, reproduced and revamped (`paper_tables`)

`paper_tables` turns every calculation into a publication table and emits a combined report. It builds two families — faithful **reproductions** of the working paper's tables and **revamped** versions that fold in the upgrades — and renders each to console, GitHub-Markdown, or booktabs LaTeX with no third-party dependency.

```python
import paper_tables as pt
tables = pt.build_all_tables(sessions, counts_fn, vol, mwcb_treated, mwcb_control,
                             release_by_date, calm_session, stress_session, n_boot=200)
pt.write_report(tables, "tables_report.md", "tables_report.tex")
print(tables["t9_revamp"].to_latex())            # one table to LaTeX
```

Builders cover Table 5 (`table_imbalance_contingency`), Table 9/11/13 (`table_correlation_irf`, with a day-cluster bootstrap + Romano–Wolf stars when `n_boot>0`), and the Section-6 reopening OLS (`table_mwcb_reopening_ols`) on the reproduction side; and on the revamp side the cross-flow IRF (`table_crossflow_irf` — spreads + continuous χ), the bucketed-vs-timed PCMOF/NCMOF comparison (`table_pcmof_clock_comparison` — VPIN-style volume clock vs per-bar, with the Epps-correction OFI correlation), the spread-conditioned IS curve (`table_spread_conditioned_is`), the MWCB DiD + triple-difference spillover (`table_mwcb_did`), Rigobon vs Cholesky (`table_rigobon`), the Markov-switching regime table (`table_regime_is`), the bivariate-Hawkes contagion table (`table_hawkes`), the σ_s + dollar-welfare table (`table_pricing_error`), the tick-corrected IS (`table_tick_correction`), and the stress-selection summary (`table_stress_selection`). `build_all_tables` isolates each builder so one failure never aborts the run (it is recorded as a table note), and the bundled synthetic generators (`_synth_session`, `_counts_fn`, `_synth_release_book`, `_synth_vol`) exercise every contract. Running the module writes `paper_tables_report.{md,tex}`; point the builders at real MIDAS sessions to produce the exact-replacement tables for the manuscript.

### 5.26 Treatment/control selection & shock registry (`market_shocks`)

`market_shocks` fixes how treatment and control days are chosen for the event study. The error it replaces: selecting controls on `ln(σ_t/σ_{t-1}) < 0` (days where vol fell). That conditions on a *vol outcome* — endogenous to the event and mechanically lower-vol than the "vol rose" treatment — so the difference-in-means is biased by the volatility level and mean reversion before any market-structure channel operates. Treatment must be an **exogenous event**, control must be **matched on the ex-ante level**.

```python
import market_shocks as mk
mk.shocks_frame()                                  # annotated registry across 6 event classes
mk.event_dates(classes=("GEOPOLITICAL","MONETARY"))        # filter by theme ...
mk.event_dates(categories=("MWCB",), spy_es_only=True)     # ... or by mechanism
mk.release_times(categories=("MWCB",))             # {date -> reopen Timestamp} (halt boundary)
mk.release_times(classes=("MONETARY",))            # {date -> 14:00 ET} for scheduled FOMC (HF surprise window)
design = mk.treatment_control_assignment(daily_state, classes=("MARKET_STRUCTURE",), n_controls=2,
                                         buffer_days=5, window_days=60, caliper=0.25)
mk.compare_control_rules(vol)                      # balance: level-matched vs the ln(sigma)<0 rule
```

The registry treats halts as one *mechanism* of a single theme — liquidity and price discovery under stress — and spans six **event classes**: `MARKET_STRUCTURE` (the 1997 and four March-2020 MWCB halts, the 2010 flash crash, the 2013 Nasdaq/UTP-SIP freeze, the 2015 NYSE outage, the 2015-08-24 mass-LULD/ETF-dislocation open), `VOL_REGIME` (2018 Volmageddon, 2024-08-05 yen-carry unwind), `MONETARY` (the 2020 intermeeting emergency cuts, the 2022-06-15 first 75bp hike, the 2024-09-18 50bp cut), `SOVEREIGN_CREDIT` (the S&P-2011, Fitch-2023, Moody's-2025 US downgrades), `TRADE_POLICY` (2019 yuan/currency-manipulator, the 2025-04-02 "Liberation Day" tariffs), and `GEOPOLITICAL` (Russia-Ukraine 2022, the 2023-10-07 Hamas attack, Iran's 2024-04-13 strike, Israel-Iran 2025). Each row carries an `event_class`, a specific `category`, a `timing` tag, an `event_et` (identifying timestamp), the S&P move, a `spy_es` relevance flag, an annotation, and a source.

The `timing` field is the methodologically important addition, because **identification differs by how the shock arrives**, and the same DiD machinery does not apply uniformly. `halt_reopen` events have an exogenous intraday reopen boundary → the reopening-asymmetry RD/event study (the paper's Section-6 design). `scheduled` events (FOMC at 14:00 ET) → an intraday event study in a tight window bracketing the announcement, i.e. the high-frequency monetary-surprise identification (Gürkaynak–Sack–Swanson, Nakamura–Steinsson). `overnight`, `weekend`, and `multi_day` shocks (downgrades after the close, weekend strikes, the tariff selloff-then-reversal) have no intraday boundary at all → identify off the close-to-open gap and the next session's path. `release_times` returns a usable boundary only for the first two (`event_et` populated); for the rest it deliberately returns nothing, signalling the next-open design. `event_dates`, `is_event_day`, and `treatment_control_assignment` all accept `classes=` (theme) and/or `categories=` (mechanism), so you can build, e.g., a clean MWCB-only treatment set, or a monetary-surprise set with a 14:00 window, or pool all `spy_es=True` events.

`treatment_control_assignment` marks event days as treatment, excludes any day within `buffer_days` trading days of *any* event, and matches each treatment day to the nearest non-event days on an **ex-ante** stress level (prior-close VIX, a vol forecast, or the long-run vol level from `stress_index` — never a contemporaneous or one-day-change vol) within a calendar window, optionally same-weekday and within a `caliper`, without replacement. `compare_control_rules` quantifies the payoff: the level-matched controls are balanced (standardized mean difference ≈ 0.03) while the `ln(σ)<0` controls are badly imbalanced (≈ 1.1), which is exactly the bias the matched design removes. **The dates and intraday times are approximate and must be verified against primary sources** (SEC MWCB and Reg SCI notices, FOMC statements, ratings-agency releases, exchange/FINRA halt records) before publication; the module says so in its header and in the registry table's note. Two events are flagged `spy_es=False` (the 2013 SIP and 2015 NYSE venue outages) because SPY/ES were not halted — they are a different, data-availability shock and make natural placebo events.

### 5.27 Event-study driver (`event_study_driver`)

`event_study_driver` closes the loop from the registry to the estimators, so a contagion event study is **one call per identification regime**. `run_event_study(daily_state, sessions_by_date, classes=…)` selects an event subset (one `timing` regime), matches controls via `market_shocks.treatment_control_assignment`, resolves each treated session's release boundary, gives every control the same wall-clock release on its own date (the same-time-of-day counterfactual), assembles the `(date, df)` lists and the release map `mwcb_event_study.build_panel` consumes, and runs the DiD, the RD-in-time, and the cross-market spillover.

```python
import event_study_driver as esd
res = esd.run_event_study(daily_state, sessions_by_date,        # daily_state: ex-ante level; sessions: {date -> book}
                          categories=("MWCB",),                  # one regime (mechanism) ...
                          asset="SPY", outcome="cost_to_fill", source="ES", target="SPY",
                          pre_min=5, post_min=5, target_qty=100)
res["did"]["did_coef"], res["rd"]["jump"], res["spillover"]["spill_post_treated"]  # the contagion estimates
res["regimes"], res["mixed_regime"], res["dropped"]            # which timing regime, and what had no session
# assembly only (no estimation), e.g. to feed a custom estimator:
a = esd.build_event_study_panel(daily_state, sessions_by_date, classes=("MONETARY",), default_release_et="09:30")
```

The release boundary is resolved per `timing`: a `halt_reopen` event uses the registry reopen time (RD at the reopen), a `scheduled` FOMC uses 14:00 ET (intraday event study in the surprise window), and an `overnight`/`weekend` event — which has no intraday boundary — falls back to the session open (`default_release_et`, the open-to-open anchor). Controls get the *same* wall-clock release on their own dates, so the DiD/RD ask whether the discontinuity at that instant is specific to event days. Because identification is not uniform across `timing`, the driver returns `regimes` (the set of timing tags it assembled) and a `mixed_regime` flag — keep one regime per call. It also returns `dropped` (events or controls with no session in `sessions_by_date`) so coverage gaps are explicit rather than silent. The synthetic self-test runs the full chain on compact books placed on the real March-2020 MWCB dates and recovers a sharp post-reopen cost-to-fill jump (DiD t≈21, RD t≈10), confirming the registry → matched-controls → `build_panel` → estimators path end to end.

### 5.28 End-to-end pipeline (`run_contagion`)

`run_contagion` is the top-level driver — one call runs the whole pipeline on a real session set and writes the manuscript table file. It sequences four stages, each isolated so a failure is recorded rather than fatal:

```python
import run_contagion as rc
res = rc.run_contagion(
    sessions,                 # (date, regime, df) panel
    daily_vol,                # daily vol Series (also the ex-ante matching state)
    sessions_by_date=None,    # {date -> book df} for the event study (derived from sessions if omitted)
    counts_fn=my_counts_fn,   # Table-5 order-flow counts (optional)
    event_categories=("MWCB",),  # the identification regime to study
    target_qty=100, n_levels=10, n_boot=200,
    out_dir="…", write=True)
res["event_study"]["did"], res["event_study"]["rd"]      # the causal estimates
res["report"]["markdown"], res["report"]["summary"]      # contagion_manuscript_tables.md, contagion_run_summary.md
```

Stage 1 (**selection**) runs `stress_index.select_days` for regime labels/onsets, summarizes the `market_shocks` registry, and computes the level-matched-vs-`ln(σ)` control balance. Stage 2 (**event study**) calls `event_study_driver.run_event_study` for the requested regime, and — the key integration — **feeds the assembled registry-matched treated/control design and release map into the table suite**, so the MWCB tables are estimated on the same matched design rather than by-construction books (if the session set contains no registry event days, it falls back to synthetic release books and flags it). Stage 3 (**tables**) runs the full 16-table `paper_tables.build_all_tables` suite on the real sessions and that design. Stage 4 (**emit**) writes the combined `contagion_manuscript_tables.{md,tex}` and a `contagion_run_summary.md` (stage statuses, headline DiD/RD/spillover, control balance, and the table inventory), and returns a structured results bundle (`selection`, `event_study`, `tables`, `report`, `stages`). The synthetic self-test runs the entire chain — registry-driven event study (4 treated / level-matched controls) through all 16 tables — and recovers a significant post-reopen cost-to-fill jump (DiD t≈17, RD t≈6) with zero build errors, end to end.

**Command line.** `run_contagion.py` has an `argparse` CLI that loads sessions (reusing `run_analysis`'s loader, so the date-universe selection and book alignment apply), gets or derives the daily vol, and runs the **causal**, **descriptive**, **both**, or **mean_variance** analysis:

```bash
# causal event study + descriptive vol-conditioning over the MWCB window (date-range expands to business days)
python run_contagion.py --source extract --date-range 2020-02-20:2020-03-31 \
    --mode both --event-categories MWCB --outcome auc_decay \
    --target-qty 100 --n-levels 10 --n-boot 500 --buffer-days 2 --output-dir out_mwcb

# price discovery + volatility spillover: VECM information shares + DCC-GARCH-X rho_t, calm-vs-stress split
python run_contagion.py --source extract --date-range 2020-02-20:2020-03-31 \
    --mode mean_variance --auto-regime --price mid --n-levels 10 --output-dir out_mwcb

# a SEPARATE identification regime — Liberation Day tariffs are overnight, not a halt RD
python run_contagion.py --source extract --date-range 2025-03-24:2025-04-21 \
    --mode causal --event-classes TRADE_POLICY --default-release-et 09:30 \
    --outcome auc_decay --target-qty 100 --n-levels 10 --output-dir out_liberation

python run_contagion.py --selftest        # synthetic end-to-end regression
```

`--mode` picks the analysis: **causal** runs the registry-driven event study and the 16-table suite (writing `contagion_manuscript_tables.{md,tex}`); **descriptive** runs `run_descriptive` — the per-session vol-conditioned VECM (CS_ES at low/median/high trailing vol) plus the cross-day OLS of per-day CS_ES on the ex-ante daily vol level (`descriptive_vol_conditioning.{csv,md}`); **mean_variance** runs the §5.29 framework — VECM information shares + DCC-GARCH-X ρ_t with the calm-vs-stress split and the liquidity→vol γ loadings (`mean_variance_summary.md`); **both** = causal + descriptive. Sessions come from `run_analysis`'s loader, so `--source {demo,load,extract}`, `--pickle`, `--volatile`/`--dates`, **`--date-range START:END`** (business-day expansion, combines with `--dates`), `--n-levels`, and book alignment all apply. Daily vol is `--vol-csv`/`--vol-pickle` or computed per session (annualized secondly realized vol); the descriptive and matching designs condition/select on a *predetermined* vol, never the contemporaneous realized vol. Regime labels — for the regime-split tables and the mean_variance calm/stress split — come from `--volatile` or, preferably, **`--auto-regime`** (top-tercile ex-ante vol within the window). `--event-categories`/`--event-classes` pick the **one** causal identification regime per call; **`--price {mid,wmid,wmid_orth,depth_wmid,mid_depth}`** picks the mean_variance cointegration/anchor configuration (§5.29); `--no-counts` skips the Table-5 contingency (otherwise a clearly-labelled book-OFI proxy fills in when no trade tape is available). See §7.7 for how to read the two-episode (MWCB vs Liberation Day) output side by side.

### 5.29 Mean + variance framework (`mean_variance`)

`mean_variance.run_mean_variance` composes the existing blocks (it adds no new estimators) into the layered conditional-mean + conditional-variance object: the price VECM is the mean of the cointegrated system, the SVAR/VECM of liquidity (elsewhere) is the mean of the cost system, and a DCC-GARCH-X sits on top as the variance layer. It returns the information shares, the time-varying SPY–ES correlation ρ_t with a calm-vs-stress split, and the liquidity→volatility loadings γ.

```python
import mean_variance as mvm
res = mvm.run_mean_variance(sessions,            # list of (label, regime, df) or a single df
                            n_lags=5, n_levels=10,
                            decay_scale=None,     # AUC_decay scale Q0 (None -> 5x inside size)
                            price="mid",          # "mid" | "wmid" | "wmid_orth" (see below)
                            stress_regimes=("volatile", "stress", "crisis"))
res["info_shares"]                  # per-session IS_mid_*, CS_ES, alpha_*
res["rho"], res["rho_calm"], res["rho_stress"], res["rho_diff_stress_minus_calm"]
res["gamma_liquidity_to_vol"]       # {asset: {auc_decay, curve_length}} GARCH-X loadings
res["dcc_a"], res["dcc_b"]          # DCC persistence
```

CLI (the calm/stress split is driven by the session regime labels, so pair it with `--volatile` or `--auto-regime`):

```bash
python run_contagion.py --source extract --date-range 2020-02-20:2020-03-20 \
    --mode mean_variance --auto-regime --price mid --n-levels 10   # writes mean_variance_summary.md
```

**Estimation, layer by layer.**

1. **Mean — price VECM (per session).** For log prices `p = (p^SPY, p^ES)` the cointegrating vector is *fixed* at β = (1, −1) (the basis is stationary), so the model is a one-equation-per-asset reduced form
   Δp_t = c + α (p^SPY_{t-1} − p^ES_{t-1} − c̄) + Σ_{ℓ=1..L} Γ_ℓ Δp_{t-ℓ} + ε_t,
   estimated by OLS (`pds._fit_vecm_fixed`); Ω = Cov(ε_t). The error-correction loadings α = (α_SPY, α_ES) give the common-factor weights ψ = (α_ES, −α_SPY). From (α, Ω):
   - **Hasbrouck information shares** IS_j: the share of efficient-price innovation variance attributable to market j, computed at the two Cholesky orderings of Ω and reported as a [lo, hi] band plus the midpoint (`hasbrouck_is`);
   - **Gonzalo–Granger component share** CS_j = |ψ_j| / Σ|ψ| — the permanent-component weight, leadership = *not* adjusting (`gonzalo_granger`);
   - **Lien–Shrestha modified IS** (MIS), the order-invariant variant via the symmetric correlation factorization (`lien_shrestha_is`).
   The fit also yields the reduced-form return innovations ε_t, which are what the variance layer consumes. (The Putniņš ILS is deliberately *not* used — it is orientation-disputed; see `price_discovery_shares.information_leadership_share`.)

2. **Variance — DCC(1,1)-GARCH(1,1)-X (`dcc_garch.dcc_garch_x`).** Two-stage. Stage 1, a GARCH-X marginal per asset on its innovation:
   h_{i,t} = ω_i + a_i ε_{i,t-1}^2 + b_i h_{i,t-1} + γ_i' x_{i,t-1},
   where x_{i,t-1} = (AUC_decay_i, ‖L‖_i) is asset i's **own** lagged liquidity (per-asset covariates via `X_per_asset`); γ_i is the **liquidity→volatility loading** (does a thinner / more convex book forecast higher own variance?). Stage 2, standardize η_t = ε_t / √h_t and fit Engle DCC(1,1):
   Q_t = (1−a−b) Q̄ + a η_{t-1} η_{t-1}' + b Q_{t-1},   R_t = diag(Q_t)^{-1/2} Q_t diag(Q_t)^{-1/2},
   so ρ_t = R_{t,12} is the time-varying SPY–ES correlation. Both stages are Gaussian-QMLE (`L-BFGS-B`/`Nelder-Mead`); non-finite rows are dropped jointly so η stays aligned, and a too-small clean sample returns NaNs rather than raising.

3. **Contagion split (Forbes–Rigobon).** ρ_t is the correlation of the **devolatilized** residuals η_t, so its stress-vs-calm difference is *not* the mechanical rise in raw correlation that accompanies higher volatility. Rows are tagged by their session's regime; `rho_stress`, `rho_calm`, and `rho_diff_stress_minus_calm` report the contagion contrast. For an endogenous regime split, drive the labels with `stress_index.classify_regimes` (CLI `--auto-regime`) or replace the split with `markov_switching_vecm`; for variance-based identification of the *mean* spillover, see `rigobon_id`.

**The microprice question — three configurations (`price=`).** The microprice (size-weighted mid) decomposes as m_w = m + (spread/2)·(Q^bid − Q^ask)/(Q^bid + Q^ask), i.e. m_w − m = −(spread/2)·I where I is the signed inside-size imbalance. So ln(microprice) on the LHS carries I, and the curve metrics (AUC_decay, ‖L‖) anchored at the microprice also carry I — putting it on *both* sides of an equation makes the VECM innovation and the covariate share a component, biasing α and contaminating the information shares. Each `price` value is a *complete* configuration that keeps the microprice off both sides:

| `price` | LHS (cointegrated) price | curve anchor | covariates | when to use |
|---|---|---|---|---|
| `mid` (default) | ln(simple mid) | microprice | — | cleanest; LHS carries no imbalance, so IS are uncontaminated |
| `wmid` | ln(microprice) | simple mid | — | microprice's better efficient-price proxy on the LHS, RHS imbalance-free |
| `wmid_orth` | ln(microprice) | microprice | **orthogonalized** | microprice on both, each covariate residualized on its asset's inside imbalance I (OLS residual) to purge the shared term |
| `depth_wmid` | ln(depth-weighted mid, n_levels) | simple mid | — | depth-aware fair price (smoother, reflects deeper resting liquidity), RHS imbalance-free |
| `mid_depth` | ln(simple mid) | depth-weighted mid | — | discovery of the plain mid, cost referenced to a *deep* fair value (near-touch marginal cost may be negative) |

The **depth-weighted mid** (`lcm.depth_weighted_mid(df, asset, n_levels)`) generalizes the microprice from the touch to the cumulative book: each side's size-weighted average price to `n_levels`, cross-weighted by the opposite side's total size, so it nests the inside microprice exactly at `n_levels=1`. It is the only "weighted mid" object that uses `--n-levels` (the inside microprice and the simple mid are level-1 / two-quote objects). `depth_wmid` is also available directly as `anchor="depth_wmid"` in `decay_weighted_cost`/`impact_curve_length`; unlike the inside microprice it can sit outside the touch when the book is imbalanced, so a near-touch marginal cost can be negative (cost relative to deep fair value), which is fine for the normalized ‖L‖ but worth knowing for AUC.

The microprice is a one-step-ahead predictor of the mid, so `wmid`/`wmid_orth` typically tighten the cointegration and reduce bid-ask-bounce in ε, at the cost of the extra handling above. The configuration the code *prevents* is the accidental fourth case — microprice LHS with microprice-anchored, non-orthogonalized curves — which double-counts I.

**Design invariants (all configs).** Covariates enter **lagged** (x_{t-1}, predetermined), never contemporaneously. Residuals are pooled across sessions for one DCC (per-session VECM means), so ρ_t spans calm and stress days; the few overnight-boundary rows are negligible intraday. Liquidity covariates are *not* themselves GARCH'd (they are persistent, bounded levels, not returns) — they enter only as exogenous variance regressors. Caveats that bite this design specifically: both metrics carry a strong intraday U-shape, so deseasonalize the innovations *and* the covariates (time-of-day spline) before a multi-day run or the diurnal pattern masquerades as both volatility clustering and spillover; at 1 s the squared-return innovation is a noisy variance proxy, so a Realized-GARCH/HEAVY measurement equation is the natural upgrade; and ‖L‖ ∈ [√2, 2] is bounded, so if you ever move it to the LHS of a GARCH use the logit transform.

---

### 5.30 Fast data pull in the canonical schema (`mstbook_loader`)

A faster ingestion path than the Athena / `maystreet_data` loader (`market_analysis_fixed`): it streams the book via MayStreet's `mstbook-query` CLI and emits the **canonical stack schema directly**, so it feeds `run_contagion` / `run_analysis` / `mean_variance` without an adapter.

```python
import mstbook_loader as ml

# one product, canonical columns (SPY_bidprice_1, ..., SPY_askquantity_10), float, tz-aware
spy = ml.query_canonical("20200312", "SPY", "direct", levels=10)

# SPY + front-month ES merged into one session frame, ready for the pipeline
date, regime, df = ml.query_pair("20200312", regime="stress", es_symbol="ES", levels=10)

# a list of (date, regime, df) sessions, then drive the contagion pipeline programmatically
sessions = ml.load_sessions([("20200312", "stress"), ("20200316", "stress"), ("20200224", "benchmark")])
# import run_contagion as rc; rc.run_mean_variance(...) / rc.run_contagion(sessions, daily_vol, ...)

feat = ml.attach_features(spy, "SPY", n_levels=10)     # correct wmid / depth_wmid / auc_decay / curve_length
vw, filled, used = ml.vwap_to_lot(spy, "SPY", "ask", lot=100, n_levels=10)   # vectorized cost to buy 100
```

**What it does.** Raw `{prefix}.bid{n}/bsize{n}/ask{n}/asize{n}` columns map to `{ROOT}_{bid|ask}{price|quantity}_{i}` (ROOT = `SPY` for direct, `get_futures_root` for futures, e.g. `ES`). Front-month contract selection (`get_front_month_contract`, third-Friday expiry, `rollover_days` before) picks the active ES contract; note a window spanning an expiry covers two contracts (ESH0→ESM0 across mid-March 2020).

**How to read it / why it's built this way.** The module enforces the same hygiene as the rest of the stack at parse time: every column is coerced to float (Athena often returns Decimal/None as object), and the **"a level counts only if both its price and size are finite"** rule is applied per level (a torn level from a shift → NaN on both legs = absent). Derived features are computed by **calling the stack's own functions** (`lcm.weighted_mid`, `depth_weighted_mid`, `decay_weighted_cost`, `impact_curve_length`) so there is exactly one definition of each metric — in particular the microprice uses the correct opposite-side size weighting (a heavy bid leans fair value **up**, the convention a own-side-weighted `wmid` gets backwards). `vwap_to_lot` is a vectorized, NaN-on-partial-fill, torn-level-safe replacement for a per-row book walk. The `mstbook-query` subprocess needs the MayStreet binary; `_to_canonical` / `attach_features` / `vwap_to_lot` operate on already-pulled data and are unit-tested on synthetic output.

**Message-type layer (the trade tape).** `mstbook_loader` also wraps `mstwx-lakequery` for the L3 message stream — `mt_add_order` / `mt_cancel_order` / `mt_modify_order` / `mt_trade` — and buckets it to the analysis interval:

```python
agg = ml.query_messages("20200312", "SPY", "direct", "mt_trade", interval="1s")  # counts, qty, trade_vwap
flow = ml.trade_flow("20200312", "SPY", "direct", interval="1s")  # {ROOT}_trade_buy/sell/px/vwap per bucket
```

`receipttimestamp` (ns) is read as UTC and converted to the same tz as the books so the two merge cleanly; missing values are coerced to NaN (never silently 0); the quantity column is selected per message type (`previousquantity` for cancel/modify, confirmed against the real headers). **Trade signing is validated against the live schema:** CME futures carry an explicit `aggressorside` (`Buy`/`Sell`, with `Buy↔side=Bid`, `Sell↔side=Ask`), so trades are signed from it directly; equities leave `aggressorside` blank, so the default (`classify="aggressor"`) falls back to the **Lee-Ready tick rule** for them (`classify="tick"` forces one cross-asset-consistent rule; `classify="side"` uses the `side` field with `Bid`=buy). `NonPrintable` prints (hidden / certain conditions) are dropped from the tape and VWAP; odd lots are kept. **Price scale:** CME equity-index futures print integer hundredths of an index point (e.g. `ESU5 543775` = 5437.75), so `query_canonical`/`trade_flow`/`extract_sessions` take a `price_scale`/`futures_scale` (pass `0.01` for index-point units); the stack's log/return/bps math is invariant to a constant price scale, so this only affects interpretability and notional, and `extract_sessions` logs the median SPY/ES mids (warning if `ES/SPY` is far from ~10×, which flags an unscaled feed).

**Busted / corrected prints are scrubbed before aggregation** (`trade_flow(..., scrub=True)`, on by default). Exchanges bust and correct trades, and do so most on volatile days — 2025-04-03 has real SPY busts and a \$10 correction (`oldprice 553.45 → 543.45`), and ES busts recur across the April-2025 window — so a tape that keeps a print which never settled contaminates trade signing and flow exactly when it matters. `_scrub_trades` fetches `mt_trade_break` and `mt_trade_correction`, drops trades whose id is in the break set (matched by `matchid`, else `tradereferencenumber`), and overwrites price/size for `matchid`-matched corrections. (It scrubs the same-day tape; a late bust disseminated the next session is not caught — rare. The reconstruction book is deliberately *not* scrubbed: a handful of busted reductions are negligible on a level snapshot, and MBP feeds self-correct via level updates.)

**This replaces SQL.** `extract_sessions(date_specs)` pulls SPY + front-month ES books **and** attaches the trade-tape counts in one CLI pass, returning `[(date, regime, df)]` exactly like the old `market_analysis_fixed.main`. **`run_analysis`/`run_contagion`'s `--source extract` now routes here** (no Athena/`maystreet_data`): pass an explicit universe (`--dates`, `--date-range`, `--volatile`, `--benchmark`). When the session frames carry the trade columns, `run_contagion` automatically uses `counts_from_frame` (the real trade-tape buy/sell counts) for Table 5 instead of the book-OFI proxy — `has_trade_flow(sessions)` is the switch. `market_analysis_fixed` is retained only for reading legacy extracts; the live path is now CLI-only and substantially faster.

**Parallel bootstrap (a separate budget from extraction).** The `--n-boot` step is the other long pole, and it parallelizes on a **different constraint**. The day-cluster bootstrap in `correlation_svar.correlation_irf_inference` (the engine behind Table 9-revamp / `t9_revamp`, the `--n-boot 1000` cost) and the moving-block bootstrap in `inference.moving_block_bootstrap` now route through `inference.parallel_bootstrap(draw_one, n_boot, n_jobs, backend, seed)`. Unlike extraction, the bootstrap runs **after** the sessions are resident and **shares the frames read-only across threads**, and each replicate's refit is numpy/pandas C code that releases the GIL — so it is **not memory-bound and can use every vCPU** (default backend `threading`, `n_jobs=None` → all cores). This is wired to `run_contagion --boot-workers` (default: all cores) and plumbed through `paper_tables.build_all_tables(..., n_jobs=)` → `table_crossflow_irf(..., n_jobs=)`. Reproducibility is preserved across worker counts: each replicate gets its own spawned `SeedSequence`, so the *set* of draws — hence every SE, CI and Romano-Wolf p-value — is identical whether you run `--boot-workers 1` or `64` (the self-tests assert `serial == threaded` bit-for-bit). One consequence: because the seeding scheme changed from a single shared stream to per-replicate spawns, exact bootstrap SEs differ from the pre-parallel code by Monte-Carlo noise (nothing systematic). Keep BLAS pinned to one thread (the driver does) so the bootstrap threads don't oversubscribe inside each linear-algebra call; a `process` backend (joblib loky) is available for pure-Python statistics where the GIL would bind, but it needs a picklable top-level `draw_one`, so the closure-based SVAR path stays on threads.

**The two budgets, and how `run_liberation_day.sh` auto-calibrates them.** Extraction is **memory-bound** (size workers off RAM, cap at the ~12 sessions); the bootstrap is **core-bound** (use all vCPUs). The driver now detects both at runtime via system calls — `nproc` (falling back to `getconf`/`/proc/cpuinfo`) for cores and `/proc/meminfo` `MemTotal` (falling back to `free -g`) for RAM — and sizes the knobs accordingly: `WORKERS = min(floor((RAM−RESERVE)/peak_RSS), 12, cores)` for `--max-workers`, and `BOOT_WORKERS = cores` for `--boot-workers`, with `RESERVE_GB ≈ RAM/8` clamped to [16, 48]. Every value stays overridable by exporting it. So on a 64-vCPU / 256-GB node it auto-selects `WORKERS=12` (all sessions in one wave) and `BOOT_WORKERS=64` — the configuration where the high core count finally pays off, since the spare cores that sit idle behind the 12-session extraction cap are exactly the ones the bootstrap saturates.

**Parallel extraction.** Each trading day is an independent, I/O-heavy unit (multiple `lakequery` pulls + a full-day book replay), so `extract_sessions(..., max_workers=N)` fans the session loop across `N` workers — wired to `run_contagion --max-workers` (the flag was previously parsed but unused, so the extract ran serially). The binding constraint is **RAM, not cores**: peak memory ≈ `max_workers ×` one full day of resident messages, and the window has ~12 sessions, the natural cap — so size `N` to the memory-safe count (a crash-day pilot's peak RSS ÷ available RAM), not the core count. `--max-workers 1` keeps the serial path and its per-fetch heartbeat (use it for the RSS-probe pilot); `>1` streams a per-session completion line as each day lands. `--extract-backend` selects the engine: `process` (default; joblib loky spawned workers, falling back to a stdlib `ProcessPoolExecutor` if joblib is absent — spawn is the safe choice for numerical workers), `threading`, or `sequential`. Pin BLAS to one thread per worker (`OMP_NUM_THREADS=1` etc.) so `N` processes don't each spawn a thread pool; the `run_liberation_day.sh` driver sets this automatically.

---

### 5.31 Consolidated SPY book + NBBO from messages (`lob_reconstruct`)

For SPY the vendor snapshot (`mstbook-query`) is a convenience product: it bakes in conflation, a time grid, and level-aggregation choices that, at sub-second horizons, bias Hasbrouck information shares, Gonzalo–Granger component shares, and lead-lag toward whichever series is sampled finer or carries less noise — exactly the quantities a SPY-vs-ES leadership claim turns on. `lob_reconstruct` rebuilds the **consolidated SPY book and NBBO from the messages themselves** so the construction is auditable end-to-end, and keeps the snapshot only as a benchmark.

**What it does.** SPY messages span several venues, each with its own sequence and order-reference namespace, so the engine maintains **one book per feed** and merges them into a single consolidation layer. Order-by-order (MBO) venues (`bats_edgx`, `xdp_arca_integrated`, `total_view`, …) are keyed on `(feed, order_reference_number)` and replayed per order:

* `mt_add_order` → insert a resting order;
* `mt_cancel_order` (carries `previousquantity`) → delete the order;
* `mt_modify_order` (carries `price`, `quantity`, `previousprice`, `previousquantity`, `maintainpriority`, `orderupdateaction`, and possibly a new `orderreferencenumber` — `total_view` re-IDs the order on modify, BATS keeps the same ref) → locate the old order via `previousorderreferencenumber`/`previousprice` and re-insert as the new order;
* `mt_trade` → decrement the referenced resting order (fallback: reduce the price level on that feed for hidden/odd-lot prints).

**Hybrid MBO + MBP (the IEX fix).** Some venues publish *only* a price-aggregated (MBP) depth feed, not an order feed — most importantly **IEX** (`iex_deep`), which carries no add/cancel/modify messages at all. An MBO-only engine silently drops these venues, so the "consolidated" book is missing real liquidity and the NBBO/depth are biased. The engine therefore also consumes `mt_price_level_update` (and the CME MBP `mt_modify_price_level` / `mt_delete_price_level`): each message carries the **new aggregate size at a price level**, applied directly via `set_level` (assign, not accumulate; `quantity` 0 or `admindelete` deletes the level). MBP and MBO venues have different feed keys, so the two paths never collide; a level is a level however it was built, and they aggregate identically in the consolidation. Trades on an MBP feed are not double-applied (the level update already reflects post-trade size). The engine warns if any feed somehow appears on both paths (a double-count risk), and records `df.attrs['mbo_feeds']`/`['mbp_feeds']` so you can see which venues entered which way. Without this, IEX is absent from the consolidated SPY book — a correctness gap, not a refinement.

Every message is ordered by a single **GPS-synchronized clock with a common UTC reference**, and the consolidated book is sampled **as-of** each grid point (the snapshot at *t* reflects all events with timestamp ≤ *t*). For the book *state* a modify is remove-old + add-new; `maintainpriority` only governs queue position (recorded in `df.attrs['lob_stats']`, not used for the ladder, which is correct — NBBO and depth do not depend on time priority).

**The clock (`--clock {receipt,exchange}`, default `receipt`).** LSEG/MayStreet captures at the source — the PCAP tap and switches sit next to the CME matching engine in the CyrusOne Aurora I data center, and each equity feed is tapped at its own venue colo — with every packet hardware-stamped to GPS-UTC. So this is **not** a single-observer-at-one-location frame: it is multi-point source capture on a common UTC reference, which for a *price-discovery* question is the right frame, because no inter-site propagation is baked into any series ("ES leads SPY" means the ES event carries the earlier GPS-UTC stamp, transport-free). `receipt` (`receipttimestamp`) is therefore the default and is *one uniform methodology applied identically to all four feeds*, whereas `exchange` (`exchangetimestamp`) carries heterogeneous per-venue publication semantics (CME MDP transaction time vs. PITCH/XDP/ITCH conventions) — so receipt is the more *consistent* cross-venue clock, not the less. Because capture is co-located with the engines, receipt ≈ exchange per venue, so `--clock exchange` is a **robustness lens** that should barely move results (reporting that invariance is itself a clean robustness line); the engine warns when `exchange` is used to order a multi-venue book. `df.attrs['clock']` records the choice.

**Two top-of-book objects, by design.** The canonical columns `{ASSET}_{bid|ask}{price|quantity}_{i}` (i = 1..levels) are the **odd-lot-inclusive consolidated ladder** (displayed size summed across venues at each distinct price, 10 best prices per side). Alongside them it emits a **strict round-lot Reg NMS NBBO** in `{ASSET}_nbbo_bid`/`{ASSET}_nbbo_ask`: each venue must show ≥ `round_lot` (100 for SPY) at its best, then best across venues. SPY trades near \$740, so a 100-share round lot is ≈ \$74k of notional and a large share of best-priced liquidity sits in odd lots invisible to the SIP NBBO (Bartlett–McCrary–O'Hara, *RFS* 2023) — reporting price discovery under **both** definitions, and the gap between them, is itself a liquidity-spillover result. Genuine cross-venue locked/crossed states are legitimate at sub-second scale (two venues' bests momentarily inverted before the SIP reconciles) and are **retained, not "corrected"** (counted as `consolidated_locked_or_crossed` in `lob_stats`). A *strictly* crossed **consolidated** top (best bid > best ask, `consolidated_crossed`) is a different animal — a real matching engine never rests crossed, so a *persistent* crossed top is a reconstruction artifact, not a market state. That artifact was the symptom that surfaced the event-ordering bug fixed in v0.9.5–v0.9.7 (see *The event-ordering fix* below); the reconstructor now carries a hard never-crossed invariant guard, and `consolidated_crossed` is read as a data-cleaning diagnostic, not a retained state.

```python
import lob_reconstruct as lob

# live (needs the MayStreet binary): consolidated SPY book for one day
spy = lob.reconstruct_session("20250423", "SPY", levels=10, interval="1s",
                              round_lot=100, odd_lot_inclusive=True)
#   -> SPY_bidprice_1.. / SPY_askquantity_10 (consolidated ladder)
#      + SPY_nbbo_bid / SPY_nbbo_ask (strict round-lot)  + SPY_mid (L1 consolidated)
#   spy.attrs["lob_stats"] -> events, modify_reprioritized, trade_no_ref,
#                             cancel_no_order, consolidated_locked_or_crossed, ...

# benchmark the rebuild against the vendor snapshot (run on a calm day, a March-2020
# circuit-breaker day, and an April-2025 day; report as a robustness table)
v = lob.validate_against_snapshot(spy, snapshot_df, asset="SPY", levels=10)
#   -> {bid1_match, ask1_match, mid_match, mid_mean_abs_diff, level_price_match, n}
```

**Wired into the pipeline — the sole extraction path.** `run_contagion --source extract` now reconstructs **both legs** from `mstwx-lakequery` messages: SPY as the consolidated multi-venue NBBO/ladder (hybrid MBO+MBP), ES as a single-venue CME price-level (MBP) replay (integer-hundredths → index points via `futures_scale`). The reason is correctness, not just tooling: the vendor `mstbook-query` snapshot sits on a **different clock** from the message lake, so snapshotting ES while reconstructing SPY would put the two legs on different timestamps and silently corrupt the SPY↔ES lead-lag — the very thing the paper rests on. Reconstructing both from one feed keeps them on a single GPS-disciplined capture clock. `mstbook-query` is also being sunset; it is retained (`query_canonical`) only so `validate_reconstruction` can run a one-time historical bias benchmark, never on the live path. `--book-source` is kept only for back-compat: `snapshot` now warns and reconstructs anyway. `--round-lot`/`--round-lot-only` set the NBBO ladder convention; `--classify {aggressor,tick,side}` selects the trade-direction rule (default `aggressor`: CME tag 5797 for ES, Lee-Ready tick fallback for equities); `--clock {receipt,exchange}` selects the ordering clock (default `receipt`).

**Validation protocol (do this before the full run).** Reconstruct a calm day, a March-2020 MWCB day, and an April-2025 day; sample the rebuild on the snapshot grid and tabulate L1 bid/ask/mid match rates and per-level price agreement (`validate_against_snapshot`); then re-estimate the headline statistics (Hasbrouck IS bounds, Gonzalo–Granger CS, ILS, Hayashi–Yoshida lead-lag, the DCC inputs) on **both** books and report the difference. If the leadership conclusion is stable, the snapshot bias is immaterial and you cite the robustness; if it moves or flips, you have both justified the reconstruction and produced a methods contribution.

**Caveats.** (i) A full SPY day is millions of messages; the replay is pure-Python (numpy/pandas only) and runs in roughly seconds-to-minutes per day — cache the reconstructed frames or coarsen `interval`. (ii) Confirm two things against MayStreet's gated reference (docs.maystreet.com) before locking the pipeline: the exact `maintainpriority`/`orderupdateaction` semantics, and that the `receipttimestamp` you read is the **source-NIC GPS hardware-capture time** (not a downstream Data-Lake/normalization arrival stamp, which would reintroduce transport and processing jitter) — and that each equity feed is genuinely tapped at its own colo rather than back-hauled to Aurora and captured there (a back-hauled feed would carry the NJ→Aurora delay, the one asymmetry to rule out). (iii) Source-GPS capture removes *transport*, but **not** per-venue engine-internal latency (match → packet-on-wire, microseconds to tens of µs, differing across venues) or GPS sync error (sub-µs). At the 1 s grid — and even at millisecond lead-lag — these sit far below the effect, but they are the binding floor if you push lead-lag to the microsecond scale, which is why **ILS and Hayashi–Yoshida** (not a single VECM Cholesky ordering) are the right tools. Separately, the Aurora–NJ light cone (~4 ms one-way) is real for any **tradeability** claim — an arbitrageur sits at one location — but that is a different question from price *discovery* and is the only place a single-observer frame and the latency shift reappear; state this distinction in the methods section rather than "correcting" the discovery measure for it. (iv) **Both legs are reconstructed from messages** — SPY as the consolidated multi-venue book (hybrid MBO+MBP), ES as a single-venue CME order-by-order (MBO) replay (`price_scale` → index points) — on one capture clock; the snapshot is retained only as the §5.32 benchmark, never on the live path. (An earlier draft of this caveat said ES stayed on the snapshot; that predates the single-feed extraction path described above.)

**The event-ordering fix (v0.9.5–v0.9.7) — why the reconstructed consolidated top is never spuriously crossed.** A first pass on the real tape returned a consolidated top that was *strictly crossed on ~100% of snapshots* — not a market state but a replay bug, and the reason the reconstruction is now treated as a data-cleaning contribution rather than plumbing. UDP multicast arrives out of packet order, so ordering each feed by any *timestamp* inverts ~10% of adjacent events: a Cancel slips ahead of the Modify it follows, the Modify (which is remove+add) resurrects the deleted order, and that phantom level pins the consolidated top crossed on every snapshot. **v0.9.5** makes intra-feed order follow the venue's authoritative `sequencenumber` (per feed), bumps the clock to be non-decreasing along that order (per-feed cummax), and interleaves feeds by that monotone clock — a k-way merge of per-feed streams keyed by clock. That alone clears the SPY equity leg to 0% crossed. **v0.9.6** ships `verify_crossing.py`: a one-command before/after that rebuilds one session twice — legacy clock ordering vs sequence ordering — on identical events and prints the crossed-fraction and trade/cancel reference-miss rates side by side (the number for the data-cleaning appendix; ~99.6% → 0% on the synthetic resurrection demo). **v0.9.7** closes the residual: CME's `sequencenumber` is *packet-level*, so one value can carry a Modify and the Cancel of the order it references, and the within-tie fallback (message concatenation order) applied the Cancel first — the same resurrection on a single packet, which left ES crossed on ~88% of snapshots even after v0.9.5. The fix ranks liquidity removals (cancel / trade / level-delete) **last** within a tie, so the removal is the final word in its packet — a strict no-op wherever `sequencenumber` is strictly increasing per feed (SPY: 0% → 0%), biting only on genuine ties (the CME packet case). A hard never-crossed invariant now guards every reconstructed book. This is verified by synthetic known-answer guards (`test_reconstruct_ordering`; `test_verify_crossing`; and `test_crossed_regression`'s `single_stranding` / `multilevel_sweep` / `packet_resurrection` / `referenced_unaffected` / `reduce_at_price_units`), the full suite (25/25 green), and the `lob_reconstruct` self-test. The **real-tape** crossed audit (`smoke_test_crossed`) on the now-in-hand co-temporal **2025-04-03 09:29–09:32 ET** window (both legs) is the immediate next step and has not yet been run.

---

### 5.32 Snapshot-vs-reconstruction benchmark (`validate_reconstruction`)

The reconstruction in §5.31 is only worth the effort if you can show what it buys — so this driver answers the referee's question directly: *does building the consolidated SPY book from messages, rather than taking the vendor snapshot, change the SPY-vs-ES price-discovery conclusion?* For each date it builds SPY+ES **both ways** (`--book-source reconstruct` and `snapshot`) and emits one row combining book faithfulness and a leadership re-estimate under each book.

**What each row contains.** (a) *Faithfulness* from `lob_reconstruct.validate_against_snapshot` — level-1 bid/ask/mid match rates, the mid mean-absolute-difference, and per-level price agreement (how close the rebuild is to the vendor book). (b) *Leadership under each book* — the Hasbrouck IS midpoint for ES, the order-invariant Lien–Shrestha unique information share (`MIS`), the Gonzalo–Granger component share (`CS_ES`), a **Hayashi–Yoshida lead-lag** in bars (positive = ES leads SPY), and the return correlation (the DCC input; `--with-dcc` adds the DCC mean ρ). (c) The **reconstruct−snapshot delta** on every leadership metric — the headline.

**Read it as a historical sanity check, not an ongoing control, and expect a small timing-driven mismatch.** Because `mstbook-query` (snapshot) and `mstwx-lakequery` (messages) are on *different clocks* and the snapshot tool is being sunset, the right reading is: the price/size **levels** should agree closely (per-level price match near 1.0), but the level-1/mid *match rates* will fall short of 100% by an amount consistent with the cross-tool clock offset — that gap is expected, not a reconstruction bug. Run it once on an archived calm day, a March-2020 circuit-breaker day, and an April-2025 day to certify the rebuild reproduces the vendor levels; thereafter the production path is reconstruction-only (both legs, one clock), and this benchmark is not re-run per analysis.

```python
import validate_reconstruction as vr

# live: needs the MayStreet binary; builds each date under both book sources
table = vr.run_validation([("2025-04-23", "volatile"),     # April-2025 "Liberation Day"
                           ("2020-03-12", "volatile"),     # a March-2020 MWCB day
                           ("2025-06-10", "benchmark")],    # a calm control day
                          n_lags=5, max_lag_bars=10)
print(vr.format_report(table))
#   per date: faithfulness (match rates) then IS_mid_ES / MIS_ES / CS_ES / HY-lead under
#   recon vs snap, with the Δ. A near-zero Δ => snapshot bias immaterial (cite the robustness);
#   a large Δ or a HY-lead SIGN change => the reconstruction is load-bearing (a methods result).
```

`hayashi_yoshida_leadlag(price_x, price_y, max_lag_bars)`, `leadership_metrics(df, …)`, and `compare_books(recon_df, snap_df, …)` are exposed for one-off use; the lead-lag uses the standardized cross-correlation of log-returns at integer-bar shifts (the Hayashi–Yoshida estimator's reduction on the common grid the books share — for a genuinely event-time HY, feed the un-gridded series).

From the shell (mirrors the other drivers; `--volatile`/`--benchmark`/`--dates` are comma-lists, the regime is a label column only):

```bash
python validate_reconstruction.py \
  --volatile 2020-03-12,2025-04-09 --benchmark 2025-06-10 \
  --interval 1s --clock receipt --out output/snapshot_vs_recon.csv
#   prints the report and writes one row per date (match rates + leadership + deltas) to CSV.
#   --with-dcc adds the DCC mean rho per book; --selftest runs the synthetic check without the binary.
```

**On ILS:** Putniņš' Information Leadership Share is a deliberate `NotImplementedError` in `price_discovery_shares` (contested orientation; Shen–Zhang–Zivot 2025), so this driver uses MIS as the order-invariant share and HY as the asynchronous cross-check rather than a hand-rolled ILS; if a validated ILS is later added to `pds`, it surfaces here without change. The synthetic self-test exhibits the intended behavior: on a one-bar-lag DGP the faithful book shows ES leading SPY by +1 bar, while a conflated "snapshot" inflates the apparent lead and raises `CS_ES` — exactly the bias the table is meant to catch.

### 5.33 Auction imbalance and the cash-open → futures linkage (`auction_imbalance`)

The `mt_order_imbalance` message is **auction-only** in this lake — it carries the opening/closing (and venue periodic) cross state, not a continuous-session order-flow imbalance (that comes from the reconstructed book's depth changes). So this module is a *feature* layer, not an estimator: it turns the cross trajectory into a few per-(session, auction) numbers and relates the SPY opening cross to the E-mini.

**Why it earns a place in a spillover paper.** The overnight shock (Liberation Day tariffs, announced after the 04-02 close) is absorbed by the continuously-trading E-mini overnight; the cash market only re-opens at 09:30, and the **opening auction is the mechanism by which cash catches up to where the future already is**. The signed opening imbalance and the indicative-vs-reference dislocation quantify the cash-side pressure at the open; relating them to the ES move around the open is a clean, on-thesis cross-asset linkage. ES has **no** auction imbalance (CME runs no opening cross — the futures `mt_order_imbalance` query returns empty), so the linkage is necessarily **SPY-auction → ES-*continuous* move**, not auction-to-auction. Both sides sit on the same GPS-disciplined capture clock (ES from the reconstructed book), which is what makes the comparison legitimate.

**Primary listing.** Several venues run crosses for SPY — NYSE Arca (its listing market), Nasdaq's own cross, and Cboe periodic auctions — but the price-forming one is the **primary listing** (ARCA for SPY). `select_primary_feed` picks it by `primarylistingmic` when populated, else by the feed carrying the most complete auction book (populated `indicativeprice`); periodic auctions are excluded from the open/close features.

**What it computes.** Per (session, auction ∈ {open, close}), from the primary listing's trajectory (signed Bid +, Ask −): the final signed imbalance and a bounded ratio `signed / (|signed| + paired)`; a bid/ask-quantity skew; the **indicative-vs-reference dislocation in bps** (how far the cross clears from the prior reference — a large negative value at the open is a gap-down); and the **pre-cross trajectory** (imbalance slope per second and the indicative-price drift over the last *N* messages — is pressure building or resolving as the cross approaches). The linkage pairs the opening signed-imbalance ratio and the indicative-vs-reference bps with the ES mid log-return (bps) over a pre-open and a post-open window, and `run_auction_analysis` reports the **cross-sectional** relationship (Pearson *r* and OLS slope of the ES post-open return on the SPY opening imbalance), overall and split by regime. The post-open ES move is measured **first→last valid observation inside the window** (robust to a stale or NaN-carried quote landing on an edge — the failure that previously returned a spurious `0.0` on days the future clearly moved); a genuinely flat or empty window returns `NaN` (so it drops from the regression rather than biasing it toward zero), and an `es_post_open_n` column reports how many ES observations populated the window, so a zero is distinguishable from an under-populated book.

**A read.** A positive cross-sectional `corr`/`slope` means a sell (buy) opening imbalance co-moves with a negative (positive) post-open ES return — the cash open and the future agree in direction, consistent with the cash market re-pricing toward the future. The economically sharp Liberation-Day pattern is a large *sell* opening imbalance with the indicative clearing well below the reference (a deep gap-down cross) on 04-03, against an E-mini that had already fallen overnight.

**Limitation to respect.** The reconstructed session starts at 09:30, so the **pre-open** ES window has no book data and the linkage returns `NaN` for it; the default uses a *post-open* window, which is always available. To study the pre-open futures path explicitly, reconstruct from ~09:25 (`extract_sessions(start_time="9:25")`) and the pre-open return populates.

```python
import auction_imbalance as ai
# sessions = [(date_label, regime, df), ...] as returned by run_analysis.load_sessions (df carries ES_mid)
res = ai.run_auction_analysis(sessions, out_dir="output/liberation_day_1s", write=True, clock="receipt")
res["summary"]["overall"]      # {'n':…, 'corr':…, 'slope':…}  SPY opening imbalance -> ES post-open return
res["auction_panel"]           # per (session, auction) features ; res["linkage_panel"] per session
```

From the pipeline, add `--auction` to a `run_contagion --source extract` call: it computes the panel and linkage for the session dates and writes `auction_imbalance.{csv,md}` and `auction_linkage.csv` alongside the contagion tables (and prints the cross-sectional linkage line).

### 5.34 Copula-GARCH dependence and liquidity-conditional tail risk (`copula_garch`)

The DCC-GARCH in `mean_variance`/`dcc_garch` gives a time-varying **linear** correlation ρ_t. Its tail-dependence coefficient is **zero for any ρ<1** — a Gaussian DCC literally asserts that far enough into the tail SPY and ES decouple, which is backwards for a crash. `copula_garch` keeps the GARCH(-X) margins but replaces the dependence with a copula that can carry **tail dependence** and **asymmetry** — the dimension a tariff gap-down actually lives in.

**Staging (IFM / pseudo-ML).** Fit GARCH(1,1)-X margins per asset (reusing `dcc_garch.garch_x_fit`), take rank-PIT pseudo-observations (robust to margin tail mis-specification), then fit dependence. Three layers:

- **Constant-copula selection** — `{gaussian, t, clayton, gumbel, BB1, SJC}` by BIC, each reporting its lower/upper tail-dependence coefficients λ_L, λ_U. Clayton has λ_L>0, λ_U=0 (joint-crash only); Gumbel the reverse; t is symmetric (one ν). **BB1 (Joe's two-parameter Clayton-Gumbel)** carries *both* tails with an analytic density — λ_L=2^(−1/(θδ)), λ_U=2−2^(1/δ) — and **nests Clayton at δ=1**, so it is the both-tails workhorse and gives a clean *nested* likelihood-ratio test for upper-tail dependence beyond the lower tail (the δ=1 null is on the parameter boundary, so the p-value uses the ½χ²₀+½χ²₁ mixture, Self–Liang 1987). SJC parameterizes (λ_L, λ_U) directly via a symmetrized Joe-Clayton and is kept mainly for the *dynamic* time-varying-tail extension; for the static selection its density is evaluated by a clamped numerical CDF mixed-difference, so BB1 is the preferred analytic both-tails fit. *Clayton/BB1/SJC winning over Gaussian/t — with λ_L large — is a joint-crash signature*; Gaussian forces λ_L=λ_U=0, so the gap between it and the best copula is the tail content the linear model misses.
- **t-copula-DCC (dynamic baseline)** — the DCC correlation recursion evaluated under a t-copula, so you get time-varying ρ_t **and** a time-varying symmetric tail dependence λ_t = λ(ρ_t, ν). Reported against the Gaussian-DCC ρ_t as a delta: same correlation path, but a finite ν gives a non-zero, stress-rising joint-crash probability the linear correlation cannot express. (Two-step: ν from the constant t-copula, then the DCC recursion on the t-scores; a one-step joint MLE or a score-driven (GAS) time-varying *asymmetric* copula is the heavier extension.)
- **Liquidity- and volatility-conditional tail dependence (the headline)** — split the sample and re-fit the **selected** copula (not a hard-wired Clayton) in each half, asking whether joint-crash dependence intensifies under stress. The split is **day-level by default** (stress-vs-calm *days*, not per-second observations), with the first/last `tod_trim_min` minutes (default 15) **trimmed** to drop the open/close auction transition — because a naive per-second split on a 1s grid is dominated by intraday seasonality (the thinnest books are mechanically at the open/close), not the crash channel. Two complementary states are reported: a **liquidity** split on SPY AUC_decay (higher = thinner) and a **volatility** split on per-day realized volatility (higher = stress), each giving λ_L stress/thin vs calm/deep, the delta, and the across-day corr(per-day λ_L, day-state), with **sign-aware** prose (a positive delta RISES = the liquidity-contagion direction; negative FALLS = tail *decoupling*). **Empirical caveat learned on the Liberation-Day tape:** on crash days the reconstructed book is densely quoted at all ten levels, so decay-weighted cost can label those days as *deep*, which inverts the AUC_decay split; when the liquidity and volatility splits disagree in sign the markdown flags the AUC_decay one as confounded by book density and points to the **volatility split as the cleaner stress read**. This is a more direct liquidity-*contagion* statement than the liquidity→vol γ in `mean_variance`, and one not previously done with reconstructed consolidated depth.

```python
import copula_garch as cg
# explicit returns + a liquidity state, or sessions (pools mid log-returns; "auto" derives SPY AUC_decay):
res = cg.run_copula_analysis(sessions=sessions, liq_state="auto", out_dir="output/liberation_day_1s", write=True)
res["selection"]["table"]       # BIC ranking with lambda_L / lambda_U per family
res["dynamic"]["lambda_mean"]   # mean time-varying tail dependence (t-copula-DCC) vs Gaussian-DCC rho
res["liquidity_conditional"]    # day-level lambda_L stress vs calm (AUC_decay split) + delta + across-day corr
res["volatility_conditional"]   # same day-level split on per-day realized vol -- the cleaner stress proxy
```

Add `--copula` to a `run_contagion` call to write `copula_garch_summary.md` and `copula_selection.csv` and print the selection/tail/conditional summary. **Caveats:** for SPY/ES the calm-period dependence is near-elliptical, so Gaussian DCC and the copulas look alike *most of the time* — the copula's value concentrates in the **stress tail**, which is exactly where the paper lives, so present it as an event-window/stress result, not an all-sample one. n=2 is the clean copula case (tail-dependence coefficients are well-defined bivariately); the deferred crypto n≥3 work would need vine copulas, so doing this now is cheap.

---

| Paper section | Augmentation | Module |
|---|---|---|
| §1 Variables | Depth curve replaces spread; multi-level OFI replaces message-count proportions | `liquidity_curve_metrics`, `cross_asset_pd_liquidity.order_flow_imbalance` |
| §2 Cross-asset order flow | Cross-impact matrices (own vs spillover) | `cross_impact` |
| §3 Liquidity | Depth-curve liquidity state; DCC-GARCH-X comovement | `liquidity_curve_metrics`, `dcc_garch` |
| §4 Price discovery | Hasbrouck IS / MIS / Gonzalo–Granger CS (delivers the title) | `price_discovery_shares` |
| §4 Order-flow contingency | Reproduces **Table 5** (3×3 cross-market imbalance: nulls, frequencies, conditional Δcorr/returns; PCMOF/NCMOF) | `tandem_order_flow` |
| §5 Correlation IRF | Reproduces **Table 9** with spread-based liquidity (standard + weighted spread + informational proxies); DCC-corr LHS option; recursive vs identity ID; **and** the spread-conditioned IS(S) replacement | `correlation_svar` |
| §2/§4 Cross-flow measure | Continuous signed cross-flow χ (size/cancel-aware) replacing categorical PCMOF/NCMOF; rectified halves + Hasbrouck–Seppi common/divergent rotation + DCC/HY conditional comovement | `cross_flow` |
| §3/§6 Stress selection | Continuous stress state + percentile regimes from V-Lab MF2-GARCH (fixes the day-selection rule; look-ahead-safe) | `stress_index` |
| §6 MWCB causal ID | Reopening-asymmetry **event study / DiD / RD-in-time / cross-market spillover** on cost-to-fill or inside-depth (item #1) | `mwcb_event_study` |
| §5 SVAR identification | **Order-free** contemporaneous ID via heteroskedasticity (Rigobon) + Forbes–Rigobon contagion test (item #2) | `rigobon_id` |
| §2/§6 Liquidity-event contagion | **Bivariate Hawkes** on depth-withdrawal/spread-jump marks; cross-excitation kernel + branching ratio; stress-conditional H3 test (item #3) | `hawkes_cross` |
| §5/§6 Latent regimes | **Markov-switching VECM**: endogenous calm/stress regimes, regime-specific error-correction & Gonzalo–Granger shares (item #4) | `markov_switching_vecm` |
| §6 Pricing error / welfare | **Hasbrouck σ_s** transitory-mispricing decomposition + **dollar welfare cost** of the dislocation; cross-market basis dislocation (item #5) | `pricing_error` |
| §5 Tick-size robustness | **Tick-aware IS correction** (common grid + rounding-noise correction); referee-proofs the futures-dominance result (item #6) | `tick_correction` |
| §5 Inference | **Generated-regressor & moving-block bootstrap SEs** for Δρ, **Romano–Wolf / Holm / BH** multiple testing across IRF & DiD coefficients (item #7) | `inference` |
| Tables 5/9/11/13, §6 | **Reporting layer**: reproduces every paper table and emits revamped versions (spreads + χ, bootstrap+RW stars, DiD, Rigobon, regime IS, σ_s welfare, tick correction) to Markdown/LaTeX | `paper_tables` |
| §6 Day selection | **Treatment/control assignment** (exogenous-event treatment, ex-ante level-matched controls) + annotated **shock registry** across 6 classes (market-structure, vol-regime, monetary, sovereign-credit, trade-policy, geopolitical) with timing-based identification | `market_shocks` |
| §6 Event study | **End-to-end driver**: registry → matched controls → release boundary → `build_panel` → DiD / RD-in-time / spillover, one call per identification regime | `event_study_driver` |
| Whole paper | **Top-level pipeline**: selection → event study → 16-table suite → manuscript `.md`/`.tex` + run summary, one call on a real session set | `run_contagion` |
| Layers 1+2 | **Mean+variance framework**: per-session VECM (information shares + innovations) → DCC-GARCH-X with per-asset AUC_decay/‖L‖ as variance covariates → time-varying ρ(SPY,ES) with a calm-vs-stress (Forbes-Rigobon) split and the liquidity→vol loadings γ | `mean_variance` |
| **New** | **Liquidity-conditional price discovery + windowed panel** | `cross_asset_pd_liquidity` |
| **New** | **State-dependent IRFs, cross-impact-identified structural IRFs, FEVD** | `irf` |
| **New** | **Continuous vs jump price discovery (ISc / ISj, co-jump lead-lag)** | `jump_robust` |
| **New** | **Liquidity-conditional price discovery as a state-dependent ECM-SDE (the α(S), IS(S) curves; the z·S loading-gradient test)** | `ecm_sde` |
| **New** | **Noise-robust observables (microprice / AUC / centroid) + coarser-grid measure feeding the SDE** | `robust_prices` |
| **New** | **Auction (open/close) imbalance features + SPY cash-open → E-mini move linkage** (same reconstructed clock) | `auction_imbalance` |
| **New** | **Copula-GARCH dependence**: tail-dependence selection, t-copula-DCC (λ_t vs Gaussian-DCC ρ_t), liquidity-conditional joint-crash dependence | `copula_garch` |
| §5 Fragility | Depth evaporation + shape-cosine breakdown at events; SIP-outage DiD | all of the above |

**Contribution:** price discovery is liquidity-state-dependent — the market whose book is relatively deeper/flatter carries the larger information share, intensifying under stress. The original paper measured liquidity by spread alone and never computed information shares; this stack supplies both and links them.

---

## 7. Reading the output — interpretation vignettes

Methods are in §5; this section is about turning the numbers into sentences, in the spirit of an R package vignette. For each output object: what it is, how to read direction and magnitude, what a plausible result looks like, and the interpretation traps.

### 7.1 Information shares (IS) and component share (CS)

**What you get.** Per session: `IS_mid_ES` (and SPY) with an `[IS_lo, IS_hi]` band, `MIS_ES`, `CS_ES`, and the error-correction loadings `alpha_spy`, `alpha_es`.

**How to read it.** IS_j ∈ [0,1] is market j's share of the variance of the *permanent* (efficient-price) innovation — how much of true price discovery happens in j. CS_j is the Gonzalo-Granger permanent-component weight ψ_j/Σψ: leadership = the market that does **not** error-correct (α ≈ 0), while the laggard is the one that adjusts back to the common price (large |α|). IS is a *variance-contribution* statement (who innovates); CS is a *who-adjusts* statement. Both above 0.5 for ES is the paper's "ES leads." The `[lo, hi]` band is the two Cholesky orderings — Hasbrouck IS is not point-identified; report the midpoint and the width. A wide band means the contemporaneous SPY/ES innovations are highly correlated (ordering matters — common at high frequency); a narrow band means the share is order-robust. `MIS` (Lien-Shrestha) is the order-invariant single number.

**What a result looks like.** `IS_mid_ES = 0.55 [0.48, 0.62]`, `CS_ES = 0.62`, `alpha_es ≈ 0`, `alpha_spy ≈ −0.3`: ES contributes ~55% of efficient-price innovations and barely error-corrects, while SPY does the adjusting (negative α_spy = SPY pulls back toward ES). ES leads — modestly by IS, more clearly by CS.

**Pitfalls.** (i) A stale / less-frequently-updating series looks like it doesn't adjust → spuriously high CS; check `staleness_report` (§5.9) before trusting a gap, especially sub-second. (ii) CS uses only α (adjustment speed); a market can innovate heavily yet still adjust, so IS and CS can disagree — the disagreement is informative, not a bug. (iii) IS ≈ 0.5 with a wide band means "can't tell who leads from this sample/ordering," not "they are equal."

### 7.2 Comparing price references (`mid` / `wmid` / `depth_wmid`) — the IS-spread diagnostic

**What you get.** The same IS/CS under each `price` config.

**How to read it.** The *spread* of CS_ES across references is itself the result. Narrow (all agree ES leads) = leadership is robust to how price is defined — a strong claim. Wide = leadership is reference-specific, and *which* reference it lives in is the mechanism: touch only (`mid`/`wmid` lead, `depth_wmid` collapses) = fast, marginal, informational leadership; microprice only (`wmid` ≫ `mid`) = inside-imbalance leadership (the queue tilts first); deep only (`depth_wmid` strongest) = repricing happens in the book, not the touch.

**What a result looks like.** `CS_ES = 0.62 / 0.63 / 0.58` (mid/wmid/depth) → robustly ES-led, slightly touch-concentrated. Versus `0.62 / 0.71 / 0.51` → the lead is largely an inside-imbalance phenomenon, weak in the deep consensus price.

**Pitfall.** `depth_wmid` depends on `--n-levels`; hold it fixed when comparing references and sweep it as a separate robustness check.

### 7.3 Liquidity-conditional & regime price discovery

**What you get.** Descriptive mode / `liquidity_conditional_vecm`: CS_ES at low / median / high values of the conditioning state (ex-ante vol or book depth) plus `t_delta_es`; the cross-day slope of per-day CS_ES on ex-ante vol.

**How to read it.** The object is CS_ES as a *function* of the state, not one number. A significant `t_delta_es` with high-state CS_ES > low-state CS_ES means ES leads *more* when the book is thin / vol is high — the central contagion claim (discovery concentrates in the future under stress). The cross-day slope is the same statement across days.

**What a result looks like.** `CS_ES 0.55 (low vol) → 0.78 (high vol)`, `t_delta_es = 4.2`: ES's dominance roughly doubles its lead in the high-vol state.

**Pitfall.** Condition on *ex-ante* (lagged) vol, never contemporaneous realized vol, or you select on the outcome (wired correctly in descriptive mode). Within a pure-crisis window the "low" state is not calm, so the contrast is high-vs-higher — widen the window or use a longer vol history.

### 7.4 Liquidity outcomes — AUC_decay (bps) and ‖L‖ (convexity)

**What you get.** Per bar, `auc_decay` (bps) and `curve_length` (dimensionless, [√2, 2]).

**How to read it.** AUC_decay is the decay-weighted average cost in bps to demand liquidity — the *level* of illiquidity, no fixed fill size; higher = more expensive. ‖L‖ is the *convexity* of the impact curve: √2 = linear impact (cost grows proportionally with size), → 2 = sharply convex / L-shaped (cheap, then a wall). They are near-orthogonal — AUC says *how expensive*, ‖L‖ says *how nonlinear*. A book can be cheap-but-fragile (low AUC, high ‖L‖: size at the touch, a cliff behind) or expensive-but-smooth (high AUC, ‖L‖ ≈ √2: uniformly wide).

**What a result looks like.** AUC_decay 0.8 → 1.4 bps and ‖L‖ 1.45 → 1.70 across a reopen: liquidity got both more expensive *and* more nonlinear — a stiffening of the whole surface, not just a wider spread.

**Pitfall.** With `anchor="depth_wmid"`, near-touch marginal cost can be negative (cost vs deep fair value), so AUC there is net-of-deep-lean — don't plot it on the same axis as a mid-anchored AUC. ‖L‖, being normalized, is anchor-robust.

### 7.5 Event study — DiD, RD, and spillover

**What you get.** Causal mode: `did_coef`/`did_t`, the RD `jump`/`jump_t`, and a `spillover` block.

**How to read it.** DiD = (post − pre)_treated − (post − pre)_control on the outcome — the causal effect of the event (e.g. the reopening asymmetry) on illiquidity. Orientation follows the outcome: for `cost_to_fill`/`auc_decay`/`curve_length`, higher = worse, so a positive significant DiD = the event made the target market more illiquid / more convex relative to matched controls. The RD jump is the discontinuity at the reopen instant — a sharper, local version of the same effect. The spillover block asks whether the *source* market's illiquidity transmits to the *target*: a positive post×treated interaction = ES illiquidity spills into SPY at the event.

**What a result looks like.** `DiD = +0.9 bps (t=4.5)`, `RD jump = +1.2 bps (t=3.1)`, spillover post×treated `+0.05 (t=2.0)`: the reopen raised SPY cost-to-demand ~1 bp vs controls, concentrated at the instant, and futures illiquidity measurably fed the ETF.

**Pitfalls.** (i) Check `n_treated`/`n_control` and whether the run fell back to synthetic books (`fallback_tables`) — a clean causal estimate needs the registry-matched design, not the fallback. (ii) The matched-control standardized mean difference should be ≈ 0; if it's large you matched on the wrong thing. (iii) One identification regime per run — never read a DiD that pooled a halt RD with an overnight gap.

### 7.6 Mean-variance — ρ_t, the calm-vs-stress contagion split, and γ

**What you get.** `rho_mean`, `rho_calm`, `rho_stress`, `rho_diff_stress_minus_calm`, `dcc_a`/`dcc_b`, and `gamma_liquidity_to_vol` per asset.

**How to read it.** ρ_t is the time-varying SPY–ES correlation of the **standardized (devolatilized)** residuals. The headline is `rho_stress − rho_calm`: positive and material = **contagion** in the Forbes-Rigobon sense — comovement rises in stress *beyond* the mechanical rise that higher volatility alone produces (the residuals are already devolatilized). A near-zero gap with high ρ throughout = **interdependence**, not contagion: they are always tightly coupled and stress adds none. `a + b` is the correlation persistence (near 1 = ρ_t moves slowly). γ is the GARCH-X liquidity→vol loading: positive = a thinner / more convex book (higher AUC_decay / ‖L‖) forecasts higher own-return volatility — the liquidity-drives-vol channel.

**What a result looks like.** `rho_calm 0.65 → rho_stress 0.82`, `a+b = 0.98`, `γ_AUC(SPY) = +0.4`: SPY/ES comovement jumps ~0.17 in stress over and above the vol effect (genuine contagion), the correlation is highly persistent, and SPY's execution cost positively forecasts its volatility.

**Pitfalls.** (i) The split is only as good as the regime labels — use `--auto-regime` (ex-ante vol tercile) or an event-defined stress window; a within-crisis window gives high-vs-higher, not calm-vs-stress. (ii) γ ≈ 0 does not mean liquidity is irrelevant — it can mean the covariate barely moves *within the window* (e.g. a near-constant-shape book); check whether AUC/‖L‖ actually vary. (iii) A rolling *raw* correlation would rise in stress mechanically — the entire reason for the DCC ρ_t here is that it does not.

### 7.7 A worked two-episode reading (MWCB 2020 vs Liberation Day 2025)

Reading the two windows from the previous answer side by side. The causal runs (`out_mwcb`, `out_liberation`) each give a DiD/RD on `auc_decay`: a positive, significant MWCB jump localized at the reopen is the circuit-breaker reopening-asymmetry effect; the Liberation Day effect, anchored at the open, is the overnight tariff-shock illiquidity — compare magnitudes to say which episode stressed SPY/ES execution more. The mean-variance runs give, per episode, ES's information share and the ρ_calm→ρ_stress jump. If *both* episodes show ES leadership rising into stress **and** a positive devolatilized ρ gap, that is consistent contagion across two very different shocks (a structural halt vs a macro-policy surprise). If Liberation Day shows the ρ jump but *not* the ES-lead increase, the tariff shock moved correlation without changing who discovers price — a distinct signature worth reporting on its own. Within each episode, the `--price mid` vs `--price wmid` IS spread (§7.2) says whether that episode's leadership is touch- or imbalance-driven. Remember the regime-split caveat: Feb–Mar 2020 is near-pure crisis, so for a genuine *calm* baseline in the descriptive/regime read, widen the window into quiet January or supply a longer `--vol-csv`; the causal event studies don't need this (their controls are the nearby matched days).

### 7.8 Dynamic results — IRFs & FEVD

The dynamic results are deliberately a *supporting* layer — the claim that the deeper book leads rests on the information shares (§5.2) and the panel regression (§5.3). IRFs and the FEVD do two jobs: they repair the identification of the dynamics the paper already reports, and they give the liquidity-conditional result a *dynamic* face. Read them as follows.

**Local-projection IRF.** `theta` at horizon *h* is the cumulative price response of the response asset to a one-standard-deviation OFI impulse in the impulse asset, *h* seconds out. A positive response that decays back toward zero is a transient cross-asset spillover; the horizon at which the bootstrap band first covers zero **is** the spillover half-life — the disciplined version of the paper's "spillover decays within a second." Because LP is a sequence of separate horizon regressions, it is robust to the dynamic misspecification that contaminates a VAR/VECM IRF at longer horizons, which is why it is the modern default; at this sample size its efficiency cost is irrelevant.

**State-dependent IRF.** `theta_low` and `theta_high` are the same response evaluated at the 1-SD-low and 1-SD-high values of the relative book state. If `theta_high` exceeds `theta_low` with non-overlapping bands, cross-asset transmission is stronger when the impulse market's book is relatively deeper — depth governs not only *who* discovers price but *how* a shock propagates. This is the dynamic analogue of the liquidity-conditional VECM and the most direct visual evidence for the contribution.

**Identification.** The structural IRF rotates the reduced-form VECM innovations with `B` built from the estimated **cross-impact matrix** (the contemporaneous OFI→return map), not a Cholesky ordering. This matters because at the SPY–ES horizon the contemporaneous link is genuinely simultaneous; an ordering would assume the answer. The alternative is Rigobon identification-through-heteroskedasticity, where the volatile-vs-benchmark variance shift pins down the structural shocks — the event sample is built for exactly that. Either way the point is to *estimate* the contemporaneous structure rather than impose it, which is the single most defensible upgrade to the paper's empirics.

**IRF ↔ information shares.** The IRFs and the information shares come from the *same* VMA representation of the VECM, so reporting identified IRFs makes the IS/CS numbers transparent rather than black-box. In particular the off-diagonal FEVD entry — the share of one asset's return-forecast-error variance driven by the *other* asset's OFI shock — is the dynamic counterpart of the cross-asset information share.

**FEVD.** Entry `[i, j]` is the fraction of asset *i*'s *H*-second return-forecast-error variance attributable to OFI shock *j* (rows sum to one). Own (diagonal) versus cross (off-diagonal) is the spillover decomposition; running it on best-level versus multi-level OFI shows the cross shares shrink as depth is added (Cont–Cucuringu–Zhang) — itself a robustness result. Keep *H* short: at 1s the action is in the first few lags, and short-horizon responses with tight bands are the point; long horizons add noise, not insight.

### 7.9 Auction imbalance — the cash-open → futures linkage

**What it is.** Two objects from `auction_imbalance.run_auction_analysis`: a per-(session, auction) feature panel, and a cross-sectional linkage summary relating the SPY **opening** cross to the E-mini move. It is an *event-window* view of the open and close, not a continuous-session measure.

**Direction and sign.** Imbalances are signed: Bid = buy pressure (+), Ask = sell pressure (−). `indic_ref_bps_final` is the indicative clearing price versus the prior reference — **negative at the open means the cross clears below the reference (a gap-down)**. The linkage summary's `corr`/`slope` (ES post-open return regressed on the SPY opening imbalance ratio) is **positive when the cash open and the future agree in direction**: a sell imbalance (−) sits with a negative post-open ES return, a buy imbalance (+) with a positive one. That co-movement is the signature of the cash market re-pricing toward where the continuously-traded future already is.

**A plausible Liberation-Day reading.** On 04-03 the SPY opening cross should show a pronounced **sell** imbalance with a deep negative `indic_ref_bps` (the open gaps down to meet the overnight E-mini decline), and the `imb_slope_per_s` says whether that pressure was still building into the 09:30 cross or resolving. The close on 04-03 carried a real signed imbalance too, so the close-auction row is informative, not boilerplate. Across the window, a positive cross-sectional `corr` is the expected result; the magnitude (`slope`, in bps of ES move per unit imbalance ratio) is the economic content.

**Traps.** (1) The default linkage uses a *post-open* ES window because the reconstructed session starts at 09:30 — `es_pre_open_ret_bps` is `NaN` unless you reconstruct from ~09:25, so don't read the pre-open column as "no effect" when it's simply "not in the data." The post-open return is measured first→last *inside* the window, so a stale edge quote no longer produces a spurious `0.0`; a flat/empty window is `NaN` (dropped from the regression), and **`es_post_open_n`** shows whether the window was populated — if it is near zero on a day the future clearly moved, the ES book is under-populated there (an upstream reconstruction issue to chase), not a genuine null. (2) This is a small-N cross-section (one open per session), so treat `corr`/`slope` as descriptive, with the regime split (calm vs stress) more suggestive than tested. (3) It is *not* a continuous-session imbalance — for intraday order-flow imbalance use the reconstructed book's depth-change OFI (`cross_asset_pd_liquidity.order_flow_imbalance`), not this. (4) Only the **primary listing**'s cross is price-forming; the secondary Nasdaq/Cboe crosses are excluded by design, so the feature reflects the official open/close, not a pooled-venue average.

### 7.10 Copula dependence — tail dependence and the liquidity-conditional joint crash

**What it is.** Three objects from `copula_garch.run_copula_analysis`: the constant-copula **selection table** (BIC, with λ_L/λ_U per family, plus the nested BB1-vs-Clayton LR), the **t-copula-DCC** dynamic summary (mean ρ_t and mean λ_t against the Gaussian-DCC ρ_t), and the **conditional** λ_L on **stress-vs-calm days** (a realized-volatility split and an AUC_decay liquidity split, open/close-trimmed).

**How to read it.** The headline contrast is *which copula wins and what tail it carries*. Gaussian always reports λ_L=λ_U=0 — it is the "no tail dependence" null. If **clayton** (λ_L>0, λ_U=0) or **SJC/BB1 with λ_L≫λ_U** beats Gaussian/t/gumbel by BIC, the dependence is **lower-tail**: SPY and ES crash together more than a correlation implies. λ_L is a probability-scale number — λ_L=0.4 means that, conditional on one being in its extreme lower tail, the other is there ~40% of the time. The **nested LR (BB1 vs Clayton)** line is the sharp test of the *upper* tail: BB1 nests Clayton at δ=1, so a small p (boundary-corrected, ½χ²₀+½χ²₁) says joint melt-ups co-occur *beyond* the joint crashes — and an *insignificant* one is the clean statement that the dependence is lower-tail-only, which for a crash window is itself the result you want. The dynamic λ_t says that joint-crash probability moves over time even when ρ_t looks stable; the gap between mean λ_t and the Gaussian DCC's implicit zero is the tail content the linear model throws away. The conditional block is the contagion punchline, and it is reported **two ways on stress-vs-calm days** (open/close-trimmed): a **volatility** split (per-day realized vol) and a **liquidity** split (AUC_decay). A positive λ_L(stress) − λ_L(calm) and a positive across-day corr say joint-crash dependence intensifies under stress. Read the **volatility split as primary**: on crash days the reconstructed book is densely quoted at all levels, so AUC_decay can mislabel them as "deep" and *invert* the liquidity split — when the two disagree, the markdown flags the AUC_decay one as confounded by book density.

**What the April-2025 tape actually showed.** On the 12-day window the **t-copula won decisively** (ν≈2.7, mean λ_t≈0.4) — i.e. *symmetric* fat-tailed dependence, with Clayton (lower-only) the worst fit and the BB1-vs-Clayton LR *rejecting* (so upper-tail dependence is present too). That is the expected shape for a tightly-arbitraged index/future pair: SPY and ES co-move violently in **both** tails, and a window that also contained sharp rebounds will not look lower-tail-only. The headline that survives regardless is that the Gaussian DCC's λ=0 is badly misspecified — the joint-extreme probability is ~0.4, not zero. Caveat from the same run: at 1-second frequency a low ν and high λ partly reflect HFT/arbitrage *synchronicity*, so report how λ_t scales as you coarsen to 100ms / 1-min, and read the conditional split's **volatility** arm (the AUC_decay arm inverted because crash-day books were densely quoted — see the trap below).

**Traps.** (1) For SPY/ES the calm-period joint distribution is nearly elliptical, so on the **full sample** Gaussian and the copulas often tie — the copula earns its keep in the stress tail, so report it as an event-window/stress exhibit, not an all-sample headline. (2) Tail-dependence estimates are noisy in the very tail (few joint extremes); lean on the *direction* (which tail, does it rise) more than the third decimal of λ_L. (3) The PIT is rank-based, so this is dependence *given* the GARCH margins — a margin that is itself fat-tailed does not by itself create copula tail dependence, which is the point of separating them. (4) The conditional split is **day-level and open/close-trimmed** by design — a per-second split on a 1s grid measures intraday seasonality, not the crash channel. It is run on both AUC_decay (liquidity) and realized vol (stress); if your session frames lack the depth ladder the AUC_decay split is skipped (the volatility split and selection+dynamic still run) rather than failing.

---

## 8. Onset stress-surface — the identified estimate of f(state)

The identification spine (v0.8.7–v0.9.3): a cleanly identified estimate of the **stress-response function** f(state) — how cross-asset price-discovery transmission behaves as the SPY–ES arbitrage link comes under stress — built to a tier-1 identification bar, with the confounds that would fake the result demonstrated and removed rather than assumed away.

**The object.** Not a binary high-vs-low-vol contrast but the continuous *function* f(state). The state is the predetermined **basis dislocation** (the demeaned (1,−1) cointegrating residual, |z_{t-1}|) jointly with **book capacity** (hollowness / round-trip cost) — the health of the arbitrage link itself, not realized vol (vol is the symptom, the basis is the disease). The headline stays reduced-form ("price discovery degrades / the arb link loosens under stress"); it does not claim the book *causes* it.

**Identification: cross-event onset, not the within-crisis cascade.** Each market event (every FOMC/CPI/NFP/quad-witching, plus scheduled-uncertainty and curated marquee shocks) contributes ONE onset observation — the jump in ES→SPY contemporaneous transmission at the release boundary, identified *within-event* off the onset variance shift (Rigobon heteroskedasticity-ID, pre-feedback). Pooling the (state, transmission) pairs across events traces f(state). Identification sits at the onset because the within-crisis *cascade* endogenizes the slope (state and response co-evolve once the loop engages); that trajectory is *described* (`markov_switching_vecm`), not fit as f(state). Inference across the few, heterogeneous events is **Ibragimov–Müller**, not the wild bootstrap.

**The surface (the spiral).** `fit_stress_surface` fits the bilinear f(basis, capacity) = b0 + b1·basis + b2·capacity + b3·basis·capacity on the impact-blind benchmark. b3 is the liquidity-spiral interaction — the Brunnermeier–Pedersen / limits-of-arbitrage complementarity (a wide basis bites harder when the book is hollow). The additive model cannot represent a spiral at all, so the interaction is the *minimum* mechanism-bearing specification. States are centred before the product (removing mechanical collinearity); `interaction_identified` is the b3-analog of a weak-ID test — cross-event corr(basis, capacity), the VIF of the centred interaction, and the off-diagonal quadrant coverage (the corners that pin b3). A fit on an unspanned plane returns NaN, not an exploded coefficient.

**The bid-ask-bounce confound, defused (v0.9.0).** The transmission is a Rigobon coefficient off the pre→post mid-return variance shift; mid returns carry bid-ask bounce, and post-release the touch thins and the BBO flips faster, so the bounce variance *rises at the onset* — inflating the post diagonal and attenuating transmission most exactly where the spiral predicts, manufacturing a spurious negative b3. `transmission_robust` re-estimates each regime's covariance with noise-corrected TSRV variances + a Hayashi–Yoshida cross, so the bounce cannot enter the variance shift. The guard `test_noise_robust_surface` constructs a NO-spiral DGP with onset-scaled bounce: the naive surface fakes a significant negative b3, the robust transmission neutralizes it, and — the sharp lesson — a *pre-window* noise covariate cannot rescue it (the confound is a post-onset shift), so the noise-robust covariance, not a control, is the fix.

**The leg / cost-aggregation decision (v0.9.1–v0.9.2).** Arbitrage capacity is two-legged. The panel carries SPY-book hollowness, ES-book hollowness, and the round-trip cost aggregated three ways — **sum** (legs stack linearly), **max** (the bottleneck — the binding leg gates the arb), and **product** (the multiplicative comparator). Run all and let the diagnostics adjudicate: on synthetic co-moving data the product is *dominated* (more leverage-fragile than the sum, no compensating signal — a disguised three-way the event count cannot fund), the bottleneck max gives the strongest-but-most-fragile signal (the binding leg carries information the average washes out), and the sum is the stable baseline. Read **sum and max as co-headline**: b3 materially stronger under max than sum is itself a joint-liquidity-withdrawal signature.

**The curated tail.** `excess_over_benchmark` (1-D) and `excess_over_surface` (2-D) test the salience-curated marquee shocks as *excess* over the impact-blind-fitted benchmark — transmission beyond what a scheduled release at the same dislocation (and capacity) predicts. The benchmark carries the exogeneity, so the excess is identified even when the tail is curated on salience. `within_support_frac` flags the curated points that sit *outside* the benchmark's box — where a bilinear surface extrapolates (and a 2-D extrapolation into the high-basis/hollow-book corner compounds), so out-of-box excess must be read as extrapolation.

**Run it.**
```
# smoke (synthetic, runs anywhere):
python run_onset_surface.py --source demo --demo-events 16
# real FOMC backbone over a window (the FOMC calendar IS the date universe):
python run_onset_surface.py --source extract --date-range 2023-01-01:2025-12-31 --categories FOMC
# or as a launcher stage (tune via ONSET_RANGE / ONSET_CATS / ONSET_PRE / ONSET_POST / ONSET_NOTIONAL):
STAGE=onset ./run_liberation_day.sh
```
The summary prints b3 on robust *and* naive transmission side by side per capacity axis, with the identification verdict and the leave-one-out jackknife band. A spiral that survives on b3_robust *and* is 'identified' is real; otherwise it is bounce, extrapolation, or an unspanned plane.

**Guards.** `test_onset_response` (onset recovery + cross-event slope + line excess), `test_onset_sensitivity` (cascade-drift discrimination + weak-ID detection), `test_stress_surface` (b3 recovery + the identification gate), `test_noise_robust_surface` (the bounce artifact and its fix), `test_onset_surface_stage` (the launcher plumbing), `test_cost_aggregation` (the aggregation invariants + the product's fragility), `test_surface_excess` (curated-tail excess vs the plane + the extrapolation flag).

## 9. Caveats

- **Synthetic-tested; the reconstruction layer now has its first real window.** Every module ships with a known-DGP self-test, and the message-level reconstruction including the v0.9.5–v0.9.7 event-ordering fix (§5.31) is verified on the full guard suite. The onset f(state) results still require the multi-event MayStreet run with the per-level passthrough (§3); what has changed is that the first co-temporal stress window — SPY + ES, 2025-04-03 09:29–09:32 ET — is now in hand, so the reconstruction/crossed-book audit can run on real data even while the full event panel cannot yet.
- **Units:** keep within-asset distance work in ticks (`unit="ticks"`, pass `tick_size`); use the default `unit="pct"` for cross-asset comparability.
- **Small-N inference:** with ~14 event days, prefer the windowed panel (many obs) with day-clustered SEs and the permutation/bootstrap helpers over day-level asymptotics.
- **Estimated β:** `robustness.beta_fixed_vs_estimated` computes shares via the (1,−1) common-factor weights; this is exact at β=(1,−1) and β̂≈1 for SPY–ES. For materially different β, use the general β⊥ weighting.
- **Sampling frequency.** The stack is grid-agnostic — 100ms/10ms is just a smaller bucket in `market_analysis_fixed.py`. But the scale changes what is trustworthy: the depth-curve *state* features are robust at any frequency, whereas message-count *flow* measures suffer empty-bucket sparsity sub-second, and the information shares are vulnerable to **price staleness** (the less-frequently-updating series looks like it doesn't adjust → spuriously high CS) and **tick discreteness** (the Gaussian VECM degrades when changes are mostly 0/±1 tick). Recommended grid: **1s baseline, 100ms practical high-frequency primary, 10ms reserved for cross-impact and the LP IRF** (where the latency structure is the point — `frequency_defaults` scales `n_lags`/horizons/block and the panel and IRF wrappers auto-apply them via `n_lags=None`/`horizons=None`). Below ~1s, **always** source realized second moments from `noise_robust_cov` (naive RV/`rcorr` are biased), attach `microstructure_diagnostics.staleness_report` / `frequency_diagnostics` to the information-share section (zero-return fraction per series + the discreteness/noise sweep), and engage the fleeting-quote filter on the OFI inputs (`order_flow_imbalance(..., min_rest_steps=...)`, or `resting_time_filter` on the raw stream upstream). Present the 1s→100ms→10ms progression as a frequency-robustness exhibit.
- **The onset estimate is identified; the cascade is described.** The headline f(state) is the pre-feedback onset response; the within-crisis trajectory is endogenous and is shown descriptively, not as a causal slope. The onset b3 is the *static complementarity* the spiral predicts (its cross-sectional shadow), with the dynamic feedback living in the cascade tier — do not read it as the dynamic spiral itself.
- **The plane must be spanned.** b3 is identified off the off-diagonal corners (high basis / deep book, and low basis / hollow book). If your events cluster on the diagonal (stress in basis *and* book together — the natural correlation), the interaction is unidentified however clean the fit looks; `interaction_identified` flags it, and the remedy is impact-blind events that load the corners, an extraction target, not a tighter estimator.
