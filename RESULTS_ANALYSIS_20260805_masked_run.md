# Analysis run 2026-08-05 21:55 — the halt-masked re-run (v0.9.53 corrections active)

`source=load` · 24 sessions · 1s grid · lags = 59 · all 12 stages green · 35 table CSVs.

This is the run the last three releases were built for: halt masking wired into every estimator,
the five splice sites removed, `run_dcc`/`run_irf` pooled and labelled, `ec_valid` live, and the
drop-one leverage diagnostic in the cross-impact machinery. The sample also changed — see §6.

**Verdict in one paragraph: the corrections worked, and they were consequential. The three
contradictions of the previous report are resolved — two of them (the FEVD headline and the DCC
anomaly) were the halt-included `sessions[0]` estimates, and the third (the 2020 cross-impact
reversal) was the halt seam, exactly as the drop-one diagnostic predicted. But the same masking
that fixed the artifacts also took most of the paper's headline regime result with it: the
pooled CS_ES migration (0.266 → 0.915, t = 2.38) is gone (interaction t = −1.18, p = 0.25;
per-day permutation p = 0.28). The thesis survives in a different, arguably better place: at the
FLOW level, the share of SPY return variance attributable to ES order flow nearly doubles under
stress (35.8% → 66.3%), while the reverse stays ~5%. Three estimators came back NaN-broken on
masked frames and need the same finite-row treatment the core got. One new contradiction opened:
the Rigobon identification flipped sign.**

---

## 1. The masking is verifiably in

* MWCB-day `n_obs`: 22,381 / 22,381 / 22,439 / 22,380 against 23,341 on clean days — the halt,
  the seam, and the contaminated lag windows are out. The pooled panel's n_obs = 556,401 is
  exactly 20 × 23,341 + the four masked counts: the row accounting closes to the observation.
* 2020-03-16's jump share, the previous report's caveat 2: truncation put **44%** of that day's
  common-factor QV in jumps when the reopen gap was inside; masked it is **16.8%**, against
  Lee–Mykland's 4.7%. Sample-wide, truncation-vs-LM narrowed to 13.8% vs 4.8%.
* Staleness improved as a by-product of the cleaner frames: ES zero-return fraction 18.8%
  (was 22%), SPY 6.6% (was 10%).

## 2. The three contradictions, resolved

**(a) The FEVD now points the right way — and is a better exhibit than before.** The old 30.6%
"ES return variance from SPY flow" was `sessions[0]` (2020-03-09), halt-included. The per-regime
median across all sessions:

| | benchmark | volatile |
|---|---|---|
| SPY returns explained by **ES flow** | 35.8% | **66.3%** |
| ES returns explained by SPY flow | 5.6% | 4.3% |

The cross-market flow asymmetry is one-directional and it *sharpens* under stress. The labelled
single-session matrix shows what the old headline actually was: on 2020-03-09 alone, 88% of SPY
return variance loads on ES flow — a crash-day extreme, now presented as such.

**(b) The 2020 cross-impact reversal is gone — it was the seam.** ES←SPY on the four 2020 days:
0.073, 0.178, 0.149, 0.105 (t of 3–7), against 0.29–0.30 with t up to 7.7 before masking. The
textbook asymmetry (SPY←ES large, ES←SPY near zero) now holds on every 2020 day. Do not put
crisis bidirectionality in the draft; the halt seam manufactured it. One genuine exception
remains: **2024-08-05** (the yen-carry unwind) shows ES←SPY = 0.40 (t = 17) — larger than its own
ES←ES = 0.33 — the only bidirectional day in the panel. Check `drop1_flag` on that day's matrix
in the tables CSV before quoting it; if it survives, it is a finding about that day, not 2020.

**(c) The DCC anomaly is gone — it was the single crash day.** Pooled across 24 sessions:
a = 0.194, b = 0.617, persistence 0.81 (comfortably inside the unit circle), mean_rho = 0.923
against realized 0.932. No divergence warning. ρ_t is now usable as the comovement exhibit.

## 3. The cost: the headline regime result did not survive

Before masking (and on the old sample): pooled CS_ES 0.266 benchmark → **0.915** volatile,
interaction t = 2.38 (p = 0.026, day-clustered), per-day permutation p = 0.055. The previous
report led with "price discovery migrates to the futures under stress, decisively."

Now:

* Per-day means: CS_ES 0.354 benchmark vs 0.469 volatile — right direction, diff +0.116,
  **permutation p = 0.278**.
* Pooled panel: interaction t_ES = **−1.18** (p = 0.25); and the volatile-regime α_SPY is
  *positive* (+0.0004) — the "pure-leader configuration" (ES stops adjusting) is gone.
* The κ half-life story survives in direction but not in drama: benchmark κ = 0.0456 →
  half-life ≈ 15 s; volatile κ = 0.0104 → ≈ 67 s. Cointegration binding weakens ~4×, not ~25×.
* A subtler, related observation: **three of the four `ec_valid = False` days are volatile**
  (2020-03-18, 2024-07-24, 2024-12-18, plus 2026-06-05) — on those days the basis did not
  correct at all within the day. That is the κ-degradation thesis expressed as a frequency, and
  it may be the honest version of the migration claim: under stress, error correction weakens to
  the point of vanishing on some days, rather than cleanly handing leadership to ES.

**Attribution caution: two things changed at once.** This run masked the halts AND ran a revised
sample (§6) at 59 lags. The collapse cannot yet be attributed to the mask alone. The A/B is
cheap and should be run before any conclusion enters the draft: same frames, same sample,
`--no-halt-mask` — one flag, cached frames, minutes. If the unmasked run on the *new* sample
reproduces the old decisive result, the masking did it (and the old result was the halt rows —
mechanical comovement, not price discovery, exactly what caveat 1 of the previous report
warned). If not, the sample revision did it.

**The pooled-vs-per-day tension is now large and needs a diagnostic.** The pooled panel says
CS_ES ≈ 0.03–0.05 in BOTH regimes (SPY overwhelmingly leads through the error-correction
loading: pooled α_ES = +0.043 benchmark), while the per-day mean is 0.42. The pooled estimate
imposes one α per regime, so it is dominated by the high-|α| days (2023-08-07: α_ES = 0.112;
2023-12-20, 2023-09-05 similar); per-day CS weights every day equally, including near-zero-κ
days whose CS is noise. Neither is wrong; they answer different weightings. The draft should
pick one estimand and defend it — or report κ (a rate, poolable) rather than CS (a ratio,
fragile) as the regime-varying quantity.

## 4. A new contradiction: the Rigobon identification flipped

Previous run: Rigobon het-ID *validated* the recursive ordering (SPY←ES 0.859 vs 0.900 assumed;
ES←SPY 0.008 vs 0). This run: **SPY←ES = −0.33, ES←SPY = +1.01** — the opposite direction, with
a negative sign on the channel every other estimator finds large and positive (cross-impact:
SPY←ES 0.20–0.62 with t up to 58 on all 24 days; FEVD: ES flow explains 36–66% of SPY returns).

When one identification disagrees with every other exhibit AND its own previous run, the prime
suspect is the identification, not the market: heteroskedasticity-ID is identified only up to
column sign and permutation, and the regime split that supplies the variance contrast changed in
this run (2024-07-24 relabelled volatile; new volatile days added). A weak or reordered variance
contrast can flip the rotation's labelling. Action: inspect `rigobon_id`'s normalization —
impose the sign convention (diagonal of B positive / own-effects positive) and check the
regime-variance ordering it assumes; re-run on the old regime labels as a control. Until then,
neither Rigobon number goes in the draft, and the previous run's "Rigobon validates the
recursive ordering" claim is also suspended — it may have been sample-dependent.

## 5. Three estimators came back NaN-broken on masked frames

The same defect class the v0.9.53 splice sweep fixed in the core estimators, now visible in the
periphery — estimators that receive halt-masked frames but never got finite-row handling:

1. **`irf.local_projection_irf` returned all-NaN** at every horizon (the state-dependent LP on
   2020-03-09). The FEVD path survived because its VECM goes through the masked design and OFI
   is `nan_to_num`'d; the LP's return regressions are not.
2. **`cojump_from_mids` found 0 jumps on 2020-03-09** — a crash day where the per-day split
   finds 24 LM jumps. The local-volatility normalizer almost certainly went NaN inside the halt
   window and silenced the detector.
3. **The legacy inference row** (`SPY_ret ~ ES_OFI`) reports coef = NaN, both t's = NaN — while
   its wild-bootstrap p (0.001) survived. n_obs = 561,599 says the NaN rows went into the
   design.

All three are the identical one-line fix the core got: finite-row mask after differencing, never
compression before. None affects the headline tables above, but the LP is the lag-robust IRF
exhibit and should be working before the 10 ms run leans on it.

## 6. The sample changed — document it

This run's universe is not the stack's documented default: 2026-01-19 (MLK, correctly gone) and
2025-01-07/2025-01-13 are out; **2025-01-27** (the DeepSeek/NVDA selloff), **2025-08-01**, and
**2024-08-09** are in; **2024-07-24 moved from benchmark to volatile**. Consequences:

* `SAMPLE_UNIVERSE.md` and Appendix A.1 must be regenerated to match what was actually run,
  including the intraday-range ranks that admitted the new days.
* The regime relabelling alone moves both regime means — one more reason the §3 A/B needs to
  hold the sample fixed.
* 2026-06-05 is degenerate (α_SPY = 8×10⁻⁹, CS_ES = 6×10⁻⁶, fixed-vs-estimated-β CS swings
  0.000006 → 0.389). It is `ec_valid = False` and should be excluded from any CS mean quoted;
  `mean_CS_ES_ec_valid` = 0.430 already does this.
* Six of 24 sessions still sit in ES roll windows; the roll measurement now runs at extraction
  (v0.9.54) but these frames predate it — the QC table will show `n/m`.

## 7. The mechanism (memo §3) is now clearly regime-only

Every window-level exhibit weakened further with the halts out:

* IS-on-depth: coef 0.017, t = 0.71 at 30 min (was 2.45); 10/20 min: t = 0.21 / 1.71. No window
  length is significant.
* Depth-tercile split: 0.452 / 0.410 / 0.448 — flat.
* `ecm_sde`: mid-price ∂α_SPY/∂S t = 0.25 (was −3.86); microprice t = 1.35; the IS_ES level
  still moves 0.21 → 0.34 with the price proxy.
* Alt-state agreement 54–67% — coin-flip territory.

The half-life curve (§ecm_sde) is at least now internally sane — 9–11 s across the state axis,
consistent with the pooled benchmark κ — instead of the 1,000–140,000 s of the halt-included
run. The window-level IS(S) claim has exactly one remaining rescue: the 10 ms grid (STAGE 6b,
already wired). If it fails there too, the memo's §3 should present liquidity-conditioning as
regime description, not mechanism.

## 8. What goes in the draft today, and what waits

**Solid now:** the flow-level FEVD migration table (§2a) with day-clustered context; the
corrected cross-impact asymmetry holding on all 24 days including 2020; the DCC comovement
exhibit; the jump split quoted from Lee–Mykland (4.8%); the inference upgrade
(wild-boot p = 0.001 — once the NaN coef is fixed); the spread-vs-book-state R²
(0.138 vs 0.318, partial 0.179); κ half-lives 15 s → 67 s with the ec_valid frequency
observation.

**Waits on the A/B (§3):** any statement about CS/IS migration across regimes.
**Waits on the rotation fix (§4):** both Rigobon numbers, this run's and the last one's.
**Waits on repairs (§5):** the LP exhibit, co-jump lead-lag, the legacy t-statistics.

## 9. Actions, ranked

1. **Attribution A/B**: re-run information_shares + panel on the same frames with
   `--no-halt-mask` (same sample, same lags). Decides whether the mask or the sample killed the
   headline. Minutes on cached frames.
2. **Fix the three NaN casualties** (`local_projection_irf`, `cojump_from_mids`, legacy
   inference row) with finite-row masks; pin each with a masked-frame test.
3. **Fix the Rigobon sign/permutation normalization** and re-run under both regime labelings.
4. **Check `drop1_flag` on 2024-08-05's cross-impact** before quoting the one bidirectional day.
5. **Run the 10 ms pass** (`--with-fine`, staged subset first) — the IS bounds and the mechanism
   rescue now carry the remaining weight of the paper's novelty.
6. Regenerate `SAMPLE_UNIVERSE.md` / Appendix A.1 for the revised universe.
