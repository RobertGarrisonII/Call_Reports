# Approaches to the 2026-08-14 Findings-Memo Items

Status ledger for the "matters for further examination" (E1–E8) and qualifications
(Q1–Q7) raised in the findings-of-record report on the 2026-08-14 run. Three items
shipped as code in v0.9.73; this file records the designed approach for the rest, so
the next implementation session starts from a decision, not a discussion.

## Shipped in v0.9.73

- **E2 (co-jump lead-lag on all sessions).** `jump_robust.cojump_by_day` runs the
  Lee–Mykland alignment on every session; `cojump_lead_test` is a day-level sign-flip
  test on the per-day lead share r = (lead_ES − lead_SPY)/(lead_ES + lead_SPY) — the
  share, not the raw count, so one jump-heavy session cannot carry the verdict. Wired
  into the jumps stage (`cojump_per_day` table + `cojump_lead_test` dict; headline
  scalars in summary.json). Gate: `test_memo_items.py` (1)–(2).
- **E3 / Q2 (IS-primary, κ-weighted, low-κ flags).** `compare_regimes` gains a
  `weights=` argument (labels permuted with (value, weight) pairs intact); the
  information-shares stage now reports `regime_test_IS` (metric = IS_mid_ES) and
  `regime_test_kappa_weighted` beside the CS test, and stamps a `low_kappa` flag
  (κ < 0.25 × median) into the per-day table. The regime migration is quotable when
  it survives all three. Gate: `test_memo_items.py` (3)–(4).
- **Q6 adjacent (contract choice).** The activity-based contract rule
  (`select_contract`, `--contract-rule activity`) extracts the volume/OI leader in
  roll windows instead of the calendar pick. Gate: `test_contract_rule.py`.

## Designed, not yet implemented

### E1 — The propagation horizon (cross-λ against aggregation interval)

**Question.** F4 brackets the SPY←ES cross-impact horizon between 10ms (λ ≈ 0) and 1s
(λ ≈ 0.4). Where between?

**Approach.** One driver (`run_horizon_profile.py`) that (i) takes the 10ms frames,
(ii) derives intermediate grids at 50ms / 100ms / 250ms / 500ms with the existing
`derive_frames` coarsening (pull-once: no new extraction), (iii) runs the existing
`cross_impact` estimator per grid per session, and (iv) writes one table:
λ_own / λ_cross by direction × interval × regime, with day-clustered CIs. The
half-impact horizon (interval where cross-λ reaches half its 1s value) is the
headline number. Cost: five cross-impact passes over derived frames — hours, not
days; no new estimator code. Caveat to encode: the fleeting-quote filter must scale
its `min_rest_steps` with the grid (the frequency_defaults mechanism already does).

### E4 — Grid-dependence of the error-correction half-life (9–12s vs 177–205s)

**Question.** Same adjustment process, twenty-fold half-life disagreement across
grids. Truncation artifact (60 lags = 0.6s of memory at 10ms) or staleness
attenuation of α (the Epps logic applied to adjustment speeds)?

**Approach.** Two discriminating experiments, both cheap:
1. **Lag-depth sweep at 10ms:** re-fit the ECM at n_lags ∈ {60, 300, 1500} (0.6s /
   3s / 15s of memory) on a few sessions. If the implied half-life falls toward the
   1s value as memory grows, mechanism (i); if it barely moves, mechanism (ii).
2. **Synthetic staleness bracket:** simulate a cointegrated pair with a KNOWN κ at
   10ms, thin quote updates to the measured refresh rates (79.6% / 89.2% zeros), and
   measure the recovered κ. The attenuation factor calibrated on synthetic data is
   then the correction factor candidate for the real 10ms α — and, if it reconciles
   the two grids, a new paper subsection: measurement error attenuates adjustment
   speeds exactly as it attenuates correlations.
The DGP machinery exists (`test_hy_correlation._frame` has the refresh-probability
staleness model); the sweep is a flag on the existing ECM driver.

### E5 — Rigobon identification through intraday variance windows

**Question.** Day-level regimes moved the relative SPY/ES variance by only 0.055
(gate threshold 0.10) — identification refused.

**Approach.** Replace day-level regimes with intraday windows chosen for RELATIVE
variance contrast: (a) MWCB reopening windows (±8 min) vs same-day mid-session
windows; (b) the 2020-03-16 open; (c) macro-release minutes (FOMC 14:00) vs the
prior hour. The existing `rigobon` estimator takes regime masks — the change is a
window-mask builder plus the same relative-eigenvalue gate. Decision rule stays: if
the gate still refuses, the paper reports Cholesky with the refusal documented (the
honest-gate precedent), and het-ID is dropped rather than forced.

### E6 — Microprice-referenced discovery at the fine grid

**Question.** The microprice reference moves 10ms IS_ES from 0.669 to 0.741 — the
mid lags the book's information.

**Approach.** A reporting change, not new estimation: the ecm_sde stage already fits
both curves. Promote the microprice curve to the primary 10ms exhibit and demote the
mid curve to robustness, with one caveat paragraph: both legs are tick-constrained,
so the microprice adjustment is bounded by half a tick and its linear-in-imbalance
form is the first-order approximation (the Stoikov fixed-point nonlinearity is a
referee-response upgrade, not a default). Requires co-author sign-off on which
number the paper quotes — flagged as a decision, not a task.

### E7 — Reconciling the depth-state and FPCA-state sign conflict at 10ms

**Question.** Mean day-level t of +1.11 (depth state) vs −3.97 (FPCA state) for the
same response.

**Approach.** Decompose before deciding: (i) report the per-day t pairs and identify
the days driving the divergence; (ii) project the FPCA leading component onto
{log total depth, centroid, HHI, τ} per day — the book-geometry module shipped in
this release exists partly for this: if the FPCA component loads on depth *shape*
(centroid/HHI) while the depth state is depth *level*, the two states measure
different objects and BOTH can be kept with distinct names; if the loadings are
level-dominated with a flipped sign on specific days, those days are the bug hunt.
No estimator change until the decomposition says which.

### E8 — Completion of the record

Table 9 both-ways (with the v0.9.72 lag-band verdict), flow-correlation, copula,
and now book-geometry outputs all land in the structured tree
(`<run>/<grid>/{table9,flow,copula,geometry}/`) with a recursive MANIFEST.json —
the next full run delivers the complete record in one archive by construction.
