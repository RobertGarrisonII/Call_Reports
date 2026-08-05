# Scope triage — WP 19-04 → JF/RFS

**Organizing principle.** One falsifiable thesis is the spine: *cross-asset price discovery is governed by
the relative shape of the limit-order book — the deeper, flatter venue leads — identified causally off the
Oct-2014 SIP-outage.* Everything else earns its place as (i) reproduction/bridge to the original paper,
(ii) robustness, or it is cut / spun into a second paper. The repo is method-complete and over-built; the
revision's risk is dilution, not missing methods.

## Two structural facts (objective, from the import graph)

1. **The 28-module analysis footprint is an artifact of one stray edge.** `mstbook_loader.py:1164`
   (`import paper_tables as pt`, a Table-5 smoke check in the loader's self-test) makes the *data layer*
   import the *manuscript table-builder*, which imports everything. Severing that single edge collapses
   `run_analysis`'s transitive closure from **28 → 15** modules — and the 15 are the genuine spine. A
   low-level loader must not import the top-level table builder; move that cross-check into a separate
   integration test (or `paper_tables`' own self-test).

2. **The spine is split across entrypoints, and there are six of them.** `run_analysis` (15 once cleaned)
   runs liquidity + information shares + the IS(S) mechanism — Gaps 1–2 and the headline. But the *Gap-3
   identification* (`rigobon_id`) and the *causal exhibit* (`mwcb_event_study`, `market_shocks`,
   `stress_index`) are only reachable through `paper_tables` / the event drivers. `run_contagion` (33) is a
   kitchen-sink superset adding `copula_garch`, `mean_variance`, `auction_imbalance`; `run_mean_variance`
   is its own off-thesis driver. **Collapse to one spine driver** that runs descriptive → headline →
   identification → causal end-to-end, with `paper_tables` building the manuscript tables from its outputs.

## The four tiers

| Tier | Modules | Role / why | Action |
|---|---|---|---|
| **SPINE — data/infra** | `mstbook_loader`, `lob_reconstruct`, `market_analysis_fixed`, `tick_correction`, `validate_reconstruction`, `smoke_test_crossed` | Produces the per-session book frames; item-#1 fix lives here. | Keep. Methods/data appendix, not an exhibit. Sever the `mstbook_loader→paper_tables` edge. |
| **SPINE — liquidity (Gap 1)** | `liquidity_curve_metrics`, `functional_liquidity` | The book is a curve, not a spread; FPCA gives the data-driven state variable. | Keep. Foreground. |
| **SPINE — price discovery (Gap 2) + mechanism** | `price_discovery_shares`, `cross_asset_pd_liquidity`, `ecm_sde` | Hasbrouck/GG shares; the liquidity-conditional VECM / IS(S) — the contribution. Items #2 fix here. | Keep. The headline. |
| **SPINE — observables/noise** | `robust_prices`, `noise_robust_cov`, `microstructure_diagnostics` | IS on the top-of-book mid is biased toward ½; these supply the microprice and the trust diagnostics. | Keep. |
| **SPINE — identification (Gap 3) + causal** | `cross_impact`, `rigobon_id`, `market_shocks`, `stress_index`, `mwcb_event_study`, `event_study_driver` | Non-recursive id (cross-impact estimate + Rigobon heteroskedasticity) and the SIP-outage DiD — what moves it from a field paper to top-3. | Keep; **wire into the spine driver** (currently only via `paper_tables`). |
| **SPINE — dynamics/inference/tables** | `irf`, `inference`, `robustness`, `paper_tables` | Error bands + FEVD; the day-cluster/Romano-Wolf engine; the robustness battery; the table builder. | Keep. Trim `paper_tables` to spine tables. |
| **REPRODUCTION / bridge** | `correlation_svar`, `cross_flow`, `tandem_order_flow`, `auction_imbalance` | Recovers/extends the original NCMOF/PCMOF (Table 5) and Table-9 SVAR; auctions tie to the MWCB halts. | Keep, but framed as "reproduction + bridge," not the contribution. |
| **APPENDIX (robustness)** | `jump_robust`, `dcc_garch`, `markov_switching_vecm`, `pricing_error` | Continuous-vs-jump IS split; time-varying ρ; latent-regime breakdown (the H3 story, complementing the continuous-state ECM-SDE); σ_s + dollar-welfare "so what." | Keep, demote to appendix / a results subsection. |
| **CUT / spin off** | `hawkes_cross`, `copula_garch`, `efficient_price`, `state_space_price`, `state_space_efficient_price`, `mean_variance` (+ `run_mean_variance`, `run_contagion`) | Off-thesis or explicitly deferred — see below. | Remove from this paper. |

## The real call: this is two papers

The repo carries two distinct contributions. **Paper A** (this one): liquidity-conditional price discovery
— the spine above. **Paper B**: cross-asset *contagion / tail-dependence under stress* — `copula_garch`
(asymmetric tail dependence) plus the local-level / Forbes-Rigobon permanent-vs-transitory comovement
decomposition (`efficient_price`, `state_space_price`, `state_space_efficient_price`). Carrying B's modules
as appendices to A dilutes the spine and invites the "this is two papers" referee report.

**Resolution (with the counter built in).** A top-3 referee will want evidence the channel *matters*
(economic importance) — so the *breakdown* and *welfare* angles earn an appendix slot in Paper A, but
carried by the price-discovery-family tools, not the contagion-family ones:

- Breakdown in Paper A → `markov_switching_vecm` (regime-dependent error correction; dates the H3 breakdown
  inside the cointegration frame). Appendix.
- Importance/welfare in Paper A → `pricing_error` (σ_s, dollar figure). Appendix or a results subsection.
- Tail-dependence contagion → `copula_garch` → **Paper B.**
- The local-level / Forbes-Rigobon "is the stress comovement fundamental or microstructure" decomposition →
  **Paper B.** Note the VECM already imposes one common trend for SPY/ES (permanent correlation ≈ 1 by
  no-arbitrage), so the local-level model is *not* redundant — it answers a different (contagion) question
  the VECM assumes away. That question belongs to B. Keep at most **one** of the three (collapse to
  `state_space_price`; cut `efficient_price` and `state_space_efficient_price` as superseded).
- `hawkes_cross` → already designated "separate paper" in the memo. Its static linear shadow,
  `cross_impact`, is the Paper-A version.
- `mean_variance` → off-thesis (portfolio optimization); cut from both.

## Concrete actions, ordered

1. **Sever** `mstbook_loader → paper_tables` (move the Table-5 smoke check to an integration test). Footprint 28→15, verified.
2. **One spine driver.** Extend `run_analysis` (or a thin wrapper) to run identification (`rigobon_id`, `cross_impact`) and the SIP/MWCB causal exhibit, so the contribution + the top-3 pieces run end-to-end. Retire `run_contagion` and `run_mean_variance`.
3. **Trim `paper_tables`** to spine + reproduction + appendix tables; drop `table_hawkes`; move the copula/local-level table builders to the Paper-B repo.
4. **Collapse the local-level trio to one** (`state_space_price`); delete `efficient_price`, `state_space_efficient_price`.
5. **Spin Paper B**: `copula_garch` + `state_space_price` + their tables into a separate package.
6. **Delete** `mean_variance`, `run_mean_variance`.

## Manuscript section map (Paper A)

1. Intro / the unifying claim.
2. Data + book reconstruction (methods appendix; the crossed-book fix is a footnote on data integrity).
3. Reproduction: NCMOF/PCMOF and the Table-9 SVAR in the depth-frame (bridge to WP 19-04).
4. Liquidity is the curve (functionals + FPCA state).
5. Price discovery: IS/MIS/CS, then the headline — IS(S), the z·S gradient (day-clustered), the IS(S) curve.
6. Identification: cross-impact contemporaneous map; Rigobon heteroskedasticity cross-check; the SIP-outage DiD (the causal exhibit).
7. Robustness appendix: jumps, frequency (1s→100ms→10ms) with noise/Epps diagnostics, the Markov-switching regime table, σ_s/welfare, the battery.

---

## Status — v0.3.0 (what has been executed)

Executed and verified in this archive:
- Severed `mstbook_loader -> paper_tables` (run_analysis closure 28 -> 15).
- Deleted `run_contagion.py`, `run_mean_variance.py`, `mean_variance.py`; `run_analysis.py` is the sole
  analysis driver.
- (Plus the v0.2.0 fixes to `lob_reconstruct.py` and `ecm_sde.py`, and the v0.3.0 `panel_vecm` SE fix.)

Still to do (require real data or a larger refactor):
- Wire `rigobon_id` (identification) + the SIP/MWCB causal exhibit into `run_analysis` so the spine runs
  end-to-end (currently reachable only via `paper_tables`; causal stage needs real event dates to verify).
- Split Paper B (`copula_garch` + the local-level family) into its own package.
- Re-wire `auction_imbalance` (now orphaned by the run_contagion deletion) into the MWCB section.
- Collapse the local-level trio to one module; delete the superseded two.
