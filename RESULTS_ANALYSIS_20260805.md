# Analysis run 2026-08-05 — the paper's results, on real data, for the first time

`source=load` · 24 sessions · 1s grid · VECM lags = 60 · all 12 stages green · 33 table CSVs.

**Verdict: the paper's central claim now has real-data support with honest inference — price
discovery migrates to the futures under stress, decisively. The memo's *mechanism* (continuous
liquidity-conditioning) is the weak link: strong at regime level, fragile at window level. Three
internal contradictions need reconciling before drafting, and one estimation caveat — halt
snapshots were not excluded — must be fixed and the four MWCB days re-run before any of their
numbers are quoted.**

---

## 1. The headline result

Pooled panel VECM with day-clustered SEs (24 clusters):

| | benchmark | volatile |
|---|---|---|
| CS_ES (Gonzalo–Granger) | 0.266 | **0.915** |
| IS_mid_ES (Hasbrouck midpoint) | 0.462 | **0.836** |
| MIS_ES | 0.406 | **0.863** |

On calm days price discovery is shared, tilted toward SPY. On volatile days it is overwhelmingly
in ES. The regime interaction on the SPY error-correction loading is significant with proper
inference (t = 2.38, p = 0.026, day-clustered); the day-level permutation test on CS_ES gives
diff = +0.231, p = 0.055 with only 24 days. This is Gap 2 of the memo closed, with the answer the
thesis predicted.

**The most quotable economic number is hiding in the alphas.** The error-correction speed
κ = α_ES − α_SPY implies a basis half-life of roughly **8 seconds on benchmark days and ~190
seconds on volatile days** — cointegration binding weakens ~25× under stress. Arbitrage capacity
degrades exactly when the price-discovery load concentrates in one venue. That is a sentence the
paper can lead with, and it comes straight from `panel_vecm`'s α estimates
(benchmark [−0.023, +0.064]; volatile [−0.0033, +0.0003] — on volatile days ES essentially stops
adjusting: the pure-leader configuration).

## 2. The mechanism: regime-strong, window-weak

The memo's thesis is that IS is a *function* of the book state. The evidence splits:

**For it:** `panel_IS_on_depth` — IS on relative log depth across 312 thirty-minute windows:
coef 0.070, t = 2.45. The sign is right and it survives day clustering.

**Against it, honestly:**

* Window-size sensitivity: t = −0.58 at 10 min, 1.19 at 20 min, 2.45 at 30 min. **Significant at
  exactly one of three window lengths.**
* The depth-tercile split of CS_ES is flat: 0.458 / 0.416 / 0.461. No monotonicity.
* Alternative state variables agree with the depth story 50–58% of the time — a coin flip.
* The `ecm_sde` observables disagree with each other: the mid gives ∂α_SPY/∂S with t = −3.86 and
  median IS_ES = 0.49; the microprice gives t = −1.70 and IS_ES = **0.91**. A mechanism claim
  should not move that much with the choice of price proxy.
* The IS(S) half-lives in the state curve run 1,000–140,000 s — at 1s within-day the
  error-correction term barely binds along most of the state axis.

**Reading:** liquidity-conditioning is real as a *regime* phenomenon and not (yet) identified as a
smooth within-day curve at 1 s. Two candidate rescues before weakening the claim: the 10 ms grid
(where the state variation is sharper and IS bounds tighten), and estimating IS(S) on the pooled
panel rather than per window. The memo's §3 wording should be softened until one of those lands.

## 3. Referee exhibits that landed

* **Inference:** the legacy iid pooled t on SPY_ret ~ ES_OFI is **94.8**; day-clustered it is
  **4.72** (wild-bootstrap p = 0.001). The effect is real and the paper's original inference
  overstated it by a factor of twenty. This single row justifies the entire inference upgrade.
* **Identification:** Rigobon heteroskedasticity-ID vs the legacy Cholesky: contemporaneous
  SPY←ES is 0.859 (vs 0.900 assumed), ES←SPY is 0.008 (vs 0 assumed). **The recursive ordering
  happened to be approximately true — and now that is a finding, not an assumption.** This is the
  cleanest possible answer to Gap 3.
* **Liquidity beyond the spread:** book-state R² on |SPY ret| is 0.280 vs 0.177 for the quoted
  spread; partial R² of the state given the spread is 0.103. Gap 1 closed.
* **Epps at 1 s, levels vs responses:** pooled Pearson and HY correlation are identical to four
  decimals (0.8231). The Table 9 estimator differences are about *dynamics*, not levels — worth a
  precise sentence in the paper so the two results are not read as contradicting.

## 4. Contradictions to reconcile before drafting

**(a) The OFI-FEVD points the opposite way from the price-level identification.** The FEVD says
30.6% of ES return variance is attributable to SPY order flow while only 1.5% of SPY variance
comes from ES flow — a SPY→ES flow story. The Rigobon price-level result says contemporaneous
causality is almost entirely ES→SPY. These are different objects (flow shocks vs price shocks),
but a referee will put them side by side. The likely culprit is the FEVD's recursive ordering
(SPY OFI ordered first absorbs the common component). Re-run the FEVD under the Rigobon rotation
before either number goes in the draft.

**(b) The 2020 cross-impact reverses the asymmetry.** On 2022–2026 days the pattern is textbook:
SPY←ES cross-impact large (0.26–0.57, t up to 58), ES←SPY near zero or negative. On the four
2020 days ES←SPY jumps to 0.29–0.30 with t of 5–7.7. Either 2020 stress genuinely made impact
bidirectional — a big, interesting claim — or the ladder-built 2020 ES book interacts differently
with the OFI construction than the MBO-built 2022–2026 books. Now that the family-selected replay
exists for 2020 (MBP types), one session re-run with `--es-book-source replay` decides it. Do not
put the bidirectionality claim in the draft until that check runs.

**(c) DCC mean_rho = 0.357 against a realized correlation of 0.82.** Persistence a+b = 0.9999 —
essentially integrated — so the fitted ρ_t path is dominated by initialization and drifts rather
than mean-reverting. The DCC-X stage needs a look (targeting, or a variance-targeting constraint)
before ρ_t is used as the comovement exhibit.

## 5. Estimation caveats found while reading

1. **Halt snapshots are inside every estimation.** n_obs = 23,341 = 23,401 − 60 on all sessions,
   and no analysis stage consumes the halt attrs. On the four MWCB days the 900-snapshot halt —
   where neither leg has a tradeable midpoint and the book is legitimately crossed — is in the
   VECM, the IS, the OFI regressions, and the jump statistics. The machinery to mask exists
   (`market_halts.halt_mask`, per-leg windows in `df.attrs`); the stages never call it. **Fix and
   re-run at least the four MWCB days before quoting any of their numbers.**
2. **2020-03-16's jump share is the halt, not jumps.** Truncation puts 44% of that day's
   common-factor QV in jumps (BNS z = 91); Lee–Mykland says 10.8%. The reopen gap after an
   at-the-open halt is one giant "jump" that truncation loads and LM's local-vol normalization
   absorbs. The memo already prefers LM for exactly this reason — the draft should quote LM, and
   the halt fix in (1) will bring the two closer.
3. **Day-level CS is unstable; pool it.** Several benchmark days have wrong-signed αs (both legs
   moving away from equilibrium — e.g. 2023-08-07, 2023-12-20), which makes their day-level CS
   meaningless, and the fixed-vs-estimated-β robustness swings CS_ES by 0.4–0.5 on five days.
   The pooled/regime panel with day-clustered SEs is the estimand to trust; per-day CS belongs in
   an appendix figure with its bounds, not in the text.
4. **IS bounds at 1 s are wide because rcorr ≈ 0.97.** Many days have Hasbrouck bounds near
   [0, 1] (2022-03-24: [0.08, 1.00]). This is the standard contemporaneous-correlation problem
   and the strongest internal argument for the 10 ms grid, where asynchrony breaks the
   simultaneity and the bounds tighten.
5. **Lags = 60.** The mains ran at the paper's footnote-17 number. The lag-sensitivity table says
   CS is flat across 3/5/10/20 (0.446–0.485), so nothing turns on it — but the draft should quote
   a BIC-chosen order with the sensitivity table in the appendix, not inherit 60.
6. **Staleness asymmetry:** ES zero-return fraction 22% vs SPY 10% (ES's 0.25-point tick is
   ~4–10× SPY's relative tick). Worth one sentence when interpreting ES-favoring shares: coarser
   ticks *attenuate* the ES-side variance, so the ES dominance under stress is unlikely to be a
   tick artifact — the bias runs the other way.

## 6. What this means for the memo/draft

The June memo's §3 headline survives contact with the data in its regime form: **the deeper,
stress-absorbing venue carries price discovery, overwhelmingly, when it matters.** The
window-level IS(S) curve — the version pitched as the novel mechanism — is not yet supported at
1 s and should be presented as regime-conditional until the 10 ms run or pooled-curve estimation
says otherwise. The identification section is stronger than hoped: Rigobon validating the
recursive ordering is a better exhibit than either identification alone.

## 7. Actions, ranked

1. **Mask halts in the estimation stages and re-run the four MWCB days** (also fixes the 03-16
   jump share). Cheap: frames are cached; `--source load`.
2. **Re-run the FEVD under the Rigobon rotation** — resolves contradiction (a).
3. **One 2020 session with `--es-book-source replay`** — resolves contradiction (b) and doubles
   as the still-open ladder-vs-replay validation for the 2020 era.
4. **The 10 ms extraction** — now motivated three ways: Epps responses, IS bound width, and
   co-jump lead-lag (16 of 17 co-jumps are "simultaneous" at 1 s — unresolvable by construction).
5. **Look at the DCC-X fit** (integrated persistence, mean_rho far below realized).
6. Soften memo §3's window-level wording until (4) runs.


---

## Corrections and resolutions (added 2026-08-05, v0.9.53)

Working through §4's contradictions produced two corrections to this report's own hypotheses and a
set of code fixes.

**Correction to (a):** I suspected the FEVD's "recursive ordering." The FEVD is **not** Cholesky —
its B is identified from the cross-impact matrix. The real problem was worse: `run_irf` computed
the headline FEVD on **`sessions[0]` alone**, which sorted-first is **2020-03-09** — a
circuit-breaker day, estimated halt-included at the time, whose cross-impact matrix carried the
reopen seam as a leverage point. The 30.6% was one contaminated crash day presented as a headline.
It is now the per-regime median across all sessions, with the single-session matrix retained and
labelled by date.

**Correction to (b):** I suggested the 2020 cross-impact reversal might be "ladder-built 2020 vs
MBO-built 2022–2026." Impossible — **all 24 sessions used the ladder** (aggregated has been the
default since v0.9.42). The live suspects are the halt/reopen seam inside the pre-v0.9.51
estimations, and genuine crisis bidirectionality. Two things now separate them: the halt-masked
re-run, and a **drop-one leverage diagnostic** in `impact_regression` (refit without the single
largest |OFI×return| co-movement; flag when the coefficient moves by more than its SE and half its
own size). On synthetic data one injected seam observation manufactures λ(ES←SPY)=0.23 from a true
zero, and the diagnostic exposes it.

**On (c):** the DCC was also a **single-session fit on 2020-03-09**, halt-included, which is most
of the mystery. `run_dcc` now pools all sessions (differenced per session — no overnight
pseudo-returns), reports the realized correlation of the same returns beside `mean_rho`, and warns
in its own output when the two diverge by more than 0.2 with near-integrated persistence.

**Also fixed while here:** five surviving NaN-compression sites (`select_lag`, `estimate_day`,
`estimate_sample`, `panel_vecm`, `liquidity_conditional_vecm`) spliced the halt seam back into
exactly the estimators the halt masking was built for — the same defect removed from `jump_robust`
in v0.9.51, found by pattern-sweeping for it. And per-day CS now carries an `ec_valid` flag
(κ = α_ES − α_SPY > 0): on days where both alphas share a sign — 2023-08-07, 2023-12-20 among them
— CS is a quotient of noise, and `mean_CS_ES_ec_valid` reports the mean over days where it is a
share.