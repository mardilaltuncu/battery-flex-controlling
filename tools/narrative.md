<!-- title -->
# Battery flexibility trading: a controller's toolkit
**Interval reconciliation · variance bridge · rolling forecast and scenarios · product profitability, built on real German day-ahead prices**

A battery optimizer trades storage assets in 15-minute intervals across several markets. The controlling questions are always the same:
*do our MWh and euros agree with the exchange, interval by interval? Why did we miss the forecast? What happens in a bad year? Which product actually earns money?*
This notebook answers each of them end to end, from the raw data pull to an Excel model with built-in checks.

The forecasting part builds on my earlier short-term forecasting work (ISPR 2022, published in *Towards Industry 5.0*, Springer): walk-forward evaluation against naive benchmarks, and judging a forecast by the decision it drives, not only by its error metric.

**Results at a glance** (details in each section)
| Area | Result |
|---|---|
| Data audit | 273,500 price intervals since 2019, all 15 DST days complete; a widely used alternative file was rejected after testing (no negative prices, two series mixed) |
| Forecast | Walk-forward gradient boosting: MAE 24.2 vs 29.6 EUR/MWh for the naive benchmark; worth about 2.2-2.9k EUR per MW and year in captured spread |
| Reconciliation | 23,274 intervals / 16.6M EUR over 12 months; 99.3 % within tolerance; every injected break type found, plus a systematic truncation and a 0.3 % meter bias that no single interval reveals |
| Variance | Exact price-volume-mix bridge; one-month-ahead errors are 62 % market spread; the flat reserve-price assumption is biased in 16 of 21 months |
| Scenarios | 12-month net trading revenue: stress 2.9M / base 8.0M / upside 10.5M EUR; revenues are right-skewed, so the normal approximation overstates downside risk |
| Profitability | Intraday looks loss-making under one allocation key and profitable under the other; dropping it would cut company result by 153k EUR (death spiral test) |

<!-- provenance -->
## 0 · Scope: what is real and what is synthetic

| Element | Status | Source |
|---|---|---|
| DE-LU day-ahead prices (hourly until 30 Sep 2025, 15-minute after), load, residual load, wind, solar | **real** | Energy-Charts API (Fraunhofer ISE), data from Bundesnetzagentur / SMARD, CC BY 4.0 |
| Battery dispatch (perfect foresight and forecast-based) | **computed** from real prices | linear programme per local delivery day |
| Assets, contracts, ledger, exchange statements, schedules (Fahrplan), meters | **synthetic** | generated here, with injected breaks and a ground-truth log |
| Reserve capacity prices, intraday value, imbalance price (reBAP) | **modelled** | parameters in the assumptions table |

Nothing here describes a real company. Every parameter is declared once, with a provenance label (`observed data`, `domain primer`, `illustrative assumption`), and the table below is also written to the Excel `Inputs` sheet. **Non-goals:** accounting treatment, tax, project finance, margining.

The economics have two layers. **Layer 1** is what the *assets* earn in the markets; this is the base of every analysis here. **Layer 2** is what the *optimizer company* keeps (e.g. a revenue share). Layer 2 depends entirely on contract terms, so it is a separate, optional module (`contracts.py`) with illustrative terms. It is used only where a company view is unavoidable: company EBITDA in the scenarios and the product P&L.

<!-- s1 -->
## 1 · Data: pull, clean, and test before trusting

Prices and fundamentals are pulled with `src/fetch_data.py` (Energy-Charts API, yearly / quarterly chunks, cached and resumable because the API rate-limits). All arithmetic runs in **UTC**. Local time is used only for reporting, because a German delivery day has 92, 96 or 100 quarter-hours. The first checks are therefore structural: completeness per local day, duplicates, the DST days, and the switch from hourly to 15-minute day-ahead products.

<!-- s1_audit -->
### 1.1 Deep dive: the more popular file that does not add up
Before settling on a source I tested a widely shared Hugging Face dataset that is described as "DE-LU day-ahead prices, 15-minute resolution". I checked it against the reference series year by year instead of trusting its README.

<!-- s1_audit_after -->
**Finding.** That file is not a day-ahead price series:
- **Two series mixed.** There are 1.25-1.45 rows per timestamp.
- **No negative prices.** It has not a single negative price before 2026, although Germany had 70-575 negative hours per year.
- **Impossible maxima.** It peaks at 2,999.99 EUR/MWh where the day-ahead maximum was 871.
- **Inflated means.** Annual means are 3-9 % too high.

Any battery-revenue analysis built on it would overstate spreads and miss every negative-price charging opportunity. This is the kind of check the role asks for: *question the numbers, not just run them*.

<!-- s2 -->
## 2 · Short-term price forecast and what it is worth

**Information discipline.** The day-ahead auction for day D closes at 12:00 on D-1. The forecast may therefore use:
- day-ahead prices up to D-1 (published on D-2);
- actual load, wind and solar only up to D-2, because D-1 actuals are incomplete at forecast time.

**Models.** A weekday-aware naive benchmark (D-1 on Tue-Fri, D-7 otherwise) and a gradient-boosting model. The model is re-fitted every quarter on all earlier data and predicts the next quarter strictly out of sample (2024-01 to 2026-10).

<!-- s2_metrics -->
**Reading the metrics.**
- **Error.** The model cuts MAE from 29.6 to 24.2 EUR/MWh (-18 %), and WAPE from 31.7 % to 25.9 %.
- **Bias.** Bias stays small (+1.7 EUR/MWh), so the error is mostly noise rather than a systematic offset.
- **MAPE.** Plain MAPE is useless here (about 4,400 %): prices near zero or negative blow it up. Error metrics for power prices should be WAPE / MAE plus bias, never MAPE alone.

<!-- s2_dispatch -->
**From error to euros.** The same battery (2 h, fees and degradation in the objective) is dispatched three ways:
- with perfect foresight (the theoretical maximum);
- with the ML forecast;
- with the naive forecast.

The **capture rate** (realised / theoretical) is 85-88 % for the ML forecast vs 82-85 % for naive, about **2.2-2.9k EUR per MW and year**. For a 100 MW portfolio that is roughly 0.2-0.3M EUR a year, which is the number a forecast improvement should be judged by.

<!-- s3 -->
## 3 · The portfolio (Layer 1)

Three operating assets with different economics, plus one contracted asset that is not built yet (pre-COD, only used in the forward view):
- **BESS-A:** 60 MW / 2 h, stand-alone, 30 % of power in reserve markets.
- **BESS-B:** 30 MW / 2 h, co-located with PV. Discharge is capped in PV hours.
- **BESS-C:** 10 MW / 1 h. Planned COD March 2025, actual June 2025.
- **BESS-D:** 20 MW, planned COD March 2027.

Revenue lines per asset and month:
- day-ahead arbitrage (from the forecast-based dispatch);
- intraday value (modelled);
- reserve capacity (modelled price path that falls as storage saturates the market);
- minus exchange fees, degradation, and imbalance cost (modelled reBAP).

<!-- s3_after -->
Net trading revenue is about 66k EUR/MW/yr in 2024-25 and 86k annualised in 2026, driven by wider day-ahead spreads (average price 112 vs 89 EUR/MWh in 2025). Reserve revenue moves the other way: 1.34M EUR in 2025, 0.70M in the first nine months of 2026. That is the saturation effect, and it comes back in the variance and scenario sections.

<!-- s4 -->
## 4 · Interval-by-interval reconciliation (the core)

Every 15-minute interval exists in up to four records:

| Record | Owner | Format here |
|---|---|---|
| Trading ledger | internal (the "CSV from Data Science") | UTC timestamp, net MWh (sell +), price, amount |
| Exchange / CCP settlement statement | exchange / clearing house | **local clock time**, separate buy / sell columns, A/B marker for the repeated hour |
| Schedule notified to the TSO (Fahrplan) | optimizer / BRP | UTC, MWh |
| Meter | asset | UTC, MWh |

Window: the first 12 months of native 15-minute day-ahead products (Oct 2025 to Sep 2026), three assets. Breaks were injected on purpose, with a ground-truth log, so the engine can be *scored* rather than just demonstrated. Sample CSVs for one month are written to `outputs/raw_csv_sample/`.

<!-- s4_parse -->
### 4.1 Time first
Statements speak local time; on 26 Oct 2025 the hour 02:00-03:00 happens twice. The engine converts every row to UTC, using the A/B marker or, where an export lost it, the row sequence (and reports that it had to). For comparison, the classic shortcut of joining on local clock time breaks: it produces keys with several matches and rows that find no partner.

<!-- s4_two_way -->
### 4.2 Two-way match: ledger vs exchange statement
Each interval is compared on volume, price and amount. Differences inside a written tolerance (0.005 MWh; amount tolerance scaled by price) clear automatically. Anything else gets a signature-based classification:

| Signature | Classification |
|---|---|
| Interval on one side only | missing in statement / ledger |
| Row more than once | duplicate |
| Volume ratio 0.25 or 4 | **MW vs MWh** (a 15-minute interval is a factor of 4) |
| Ratio -1 | buy / sell **sign convention** |
| Same volume, different price | price correction |
| Otherwise | partial fill / volume difference |

Small differences that *always point the same way* are tested per asset and month. Individually harmless, collectively a systematic error.

<!-- s4_two_way_after -->
**Result.**
- **Coverage.** 23,274 intervals and 16.6M EUR gross were compared; 99.3 % sit within tolerance.
- **Breaks.** 156 intervals form 12 break events, each with an owner, a root cause and a next step.
- **Systematic finding.** One account's statement **truncates** volumes to three decimals instead of rounding. Every single difference is below tolerance and the money is tiny, but it is one-sided in 100 % of cases, in every month. That is a process issue to raise with the data feed, not noise.

Next, the fee lines on the statement are checked against the tariff the budget assumes.

<!-- s4_deep -->
**Fees.** From April 2026 the statement charges 0.12 instead of 0.10 EUR/MWh, about 250 EUR a month. It is not a break, it is a **tariff change**. It belongs in the variance bridge (tariff effect) and in the next forecast.

**Traceability.** Every euro in the register leads back to interval rows. Below, the largest break is traced to the eight quarter-hours the statement does not contain.

<!-- s4_four -->
### 4.3 Four-way match: what the two-way check cannot see
Adding the schedule and the meter separates three kinds of difference:

| Difference | Meaning | Who bears it |
|---|---|---|
| ledger ≠ statement | data or trade error | data correction |
| ledger ≠ schedule | **process error**: the trade was notified wrongly or not at all | optimizer (avoidable imbalance) |
| schedule ≠ meter | **imbalance**: the asset delivered something else | asset side (availability) |

The case of a trade that was never notified *and* an asset that stayed idle is invisible to a meter-vs-schedule check: both are zero. Only the four-way match catches it.

<!-- s4_score -->
**Four-way findings.**
- **Unplanned outages.** 88 outage events cost about 123k EUR of imbalance; these sit on the asset side and feed the availability KPI.
- **Process errors.** 3 notification errors and 4 trades that were never notified carry about 5.5k EUR of exposure. Small, but avoidable.
- **Meter bias.** One meter reads about 0.3 % low. No single interval is outside tolerance; it surfaces only as a statistical test per month (12 of 12 months flagged).

**Scorecard against the injected truth.**
- Every injected break type is found with the right classification.
- Outages are caught at 560 of 561 intervals.
- There is one false positive (a single interval flagged as a notification error), reported rather than hidden.

<!-- s5 -->
## 5 · Variance analysis: an exact price-volume-mix bridge

Every revenue and cost line is generated from a handful of **drivers**:
- MW online and the reserve share;
- MWh discharged per MW;
- the theoretical spread and the capture rate;
- the reserve price;
- the fee rate, degradation rate, imbalance volume and imbalance price.

The bridge therefore adds up **by construction**, and the control `budget + effects - actual = 0` is asserted, here and again in Excel.

Arbitrage revenue is split the way a trading business needs it:

| Effect | Formula |
|---|---|
| cycles (volume) | `(q_A - q_B) × spread_B` |
| market spread (not controllable) | `q_A × (theo_A - theo_B) × capture_B` |
| capture rate (algorithm quality, controllable) | `q_A × theo_A × (capture_A - capture_B)` |

Costs are split into volume and rate (tariff) effects. The order is fixed and stated, because PVM results depend on it.

<!-- s5_after -->
**June 2026.**
- **Overall.** Actual net trading revenue was 884k EUR against a budget of 523k (the December rolling forecast).
- **Market spread.** +344k, not controllable. Almost the same gap remains against the latest (May) forecast, so a stale budget is not the cause.
- **Capture rate.** +32k, controllable: the algorithm extracted more of the theoretical spread than planned.
- **Reserve price.** -29k (saturation).
- **Tariff effect.** The new exchange tariff shows up as a small, separate tariff effect.
- **Materiality.** Three steps exceed both thresholds (5 % and 25k EUR) and need a written explanation.

<!-- s5_attr -->
**Forecast error attribution.** Running the same bridge over all 21 one-month-ahead forecasts shows *why* the rolling forecast misses:
- **Market spread.** 62 % of the average absolute error, mostly in one direction (actual above forecast in 14 of 21 months).
- **Reserve price.** Actual fell below forecast in **16 of 21 months**. The flat "last three months" assumption is systematically wrong in a saturating market. Action: put a downward trend into the reserve-price assumption.

<!-- s6 -->
## 6 · Rolling forecast, scenarios and probabilistic view

At every month-end the next 12 months are re-forecast from trailing data:
- **Spread and cycles:** trailing-12-month level × seasonal index.
- **Capture, fees, imbalance:** trailing 12 months.
- **Reserve price:** last three months.
- **COD:** for assets not yet online, the planned COD is probability-weighted with typical delays (delays are the rule, not the exception).

The backtest compares every forecast with what actually happened, by horizon.

<!-- s6_backtest -->
**Backtest.**
- **Accuracy.** Monthly MAPE grows from 33 % one month ahead to 45 % twelve months ahead.
- **Bias.** It is consistently negative (-10 % to -16 %): in a market whose spreads widened through 2026, a trailing method lags.
- **What it means.** MAPE is far larger than |bias|, so most of the error is genuine month-to-month volatility, but the persistent part is systematic and fixable. Monthly trading revenue cannot be forecast tightly, which is exactly why the scenario and probability views below matter more than the point forecast.

<!-- s6_season -->
**A model choice tested, not assumed.** The seasonal index was first built from the *mean* of yearly profiles. The 2021-22 crisis winters distort that mean, so a *median* index was tested. It improves every horizon (e.g. 34.0 % → 32.7 % one month ahead) and is used throughout. The remaining error is a level shift, not seasonality.

<!-- s6_scen -->
**Scenarios are consistent worlds, not single-variable tweaks.** The spread level comes from history: for every month since 2019, the ratio of next-12-month to trailing-12-month spread. This distribution is centred on its median so the base keeps the current level. Downside and upside use P25 / P75, stress uses P10. The other levers move in the same direction:
- capture rate;
- reserve price (saturation);
- imbalance price;
- COD delay of the pre-COD asset.

**What the stress case shows.**
- **Asset revenue.** Layer 1 falls to 2.9M EUR (base 8.0M).
- **Floor payout.** The floor in the C2 contract pays out about 0.82M EUR. A floor is a *put option written by the company*: invisible in good years, decisive in bad ones.
- **Company result.** Company revenue turns negative and EBITDA reaches -1.17M EUR (base +0.36M).

<!-- s6_tornado -->
**Tornado (one driver at a time, P10 / P90 or ± as labelled).**
- **Market spread** dominates both views.
- **Revenue-share rate** comes second for company EBITDA (±0.24M EUR). It is a commercial lever, not a market one.
- **Cycles per MW** come next (±0.1-0.12M EUR of EBITDA).
- **COD timing and capture rate** follow, each worth about 0.1M EUR of EBITDA swing. COD timing of the pre-COD asset alone moves Layer 1 by about 1M EUR.
- **Exchange fees** are immaterial.

<!-- s6_mc -->
**Monte Carlo (4,000 draws)** combines empirical spread ratios, capture noise, reserve and imbalance price multipliers and a COD-delay distribution.

| | Empirical | Normal approximation |
|---|---|---|
| P50 | 8.0M (≈ base) | |
| P90 (exceeded with 90 % probability) | 4.2M | 3.5M |
| P99 | +3.3M | -1.2M |

The distribution is **right-skewed** (skewness 1.25). The normal approximation is too pessimistic at P90 and absurd at P99, so percentiles should be read from the simulation, not from mean minus k·sigma. The floor pays out in about 20 % of draws, with an average top-up of about 0.5M EUR when it does.

<!-- s7 -->
## 7 · Product and customer profitability (company view)

**Products.** Each asset's net revenue is split by market product (day-ahead, intraday, reserve capacity). Company revenue (Layer-2 module) is attributed to products in proportion to their contribution.
- **CM1** = company revenue − product-specific direct costs (data, order-book feeds, prequalification).
- **Joint costs** (trading team, data & IT, balance group) are then allocated twice: by revenue and by activity (trader hours, messages, MWh through the balance group).

**Customers.** Cost-to-serve is shown per customer and per MW.

<!-- s7_after -->
**The key decides the verdict.**
- **Intraday** earns a positive CM1 of 153k EUR. After joint costs it shows +46k under the revenue key but **-193k under the activity key**, because it consumes about two thirds of trader time.
- **Day-ahead** looks about 1.7 times as good under the activity key (616k) as under the revenue key (354k).

The total is identical in both cases. Allocation moves profit between products, it does not create it.

<!-- s7_drop -->
**Death-spiral test.** Dropping the "loss-making" intraday product:
- **Company result.** It falls from 440k to 287k EUR, exactly the lost CM1.
- **Remaining products.** The joint costs stay and move to the other products, so reserve capacity now looks loss-making too.

Drop decisions belong on *avoidable* costs. The better levers are automation (fewer trader hours per intraday MWh) and product-specific pricing.

**Cost to serve.** It is 3.6k EUR per MW for the small 1-hour asset vs 0.5k for the large stand-alone one. That argues for the minimum monthly fee in that contract.

<!-- s8 -->
## 8 · Excel deliverable with built-in checks

The workbook (`outputs/flex_controlling_case.xlsx`) follows the controlling primer's rules:

**Sheets**
- **Inputs** on their own sheet, editable cells coloured.
- **Live formulas, not pasted numbers.** The bridge and the product P&L recalculate in Excel; the allocation key is a drop-down.
- **Break register** with a status drop-down.
- **Change log.**

**Checks:** a `Checks` sheet with 12 tests and one master flag:
- cross-foots;
- sign convention;
- DST interval counts;
- full allocation;
- bridge residual;
- reconciliation totals.

I verified the file by recalculating it in Excel: all 12 checks return OK. A deliberate negative test (one cost entered with the wrong sign, one allocation key broken) flips the master flag to ERROR.

<!-- s9 -->
## 9 · Summary for the CFO

<!-- limits -->
## 10 · Limitations and next steps

**Limitations**
- **Synthetic records.** The operational records are synthetic. The engine is proven on known breaks, not yet on real statements.
- **Modelled markets.** Intraday value, reserve capacity prices and reBAP are modelled. Replacing them with real data would sharpen sections 3, 5 and 6 most: published imbalance prices, intraday indices (ID3) and reserve auction results.
- **Forecast inputs.** The price forecast has no day-ahead wind / solar forecasts; adding them is the obvious next gain.
- **Contracts.** Contract terms (Layer 2) are illustrative. The module is deliberately separable.

**Operational next step:** run the reconciliation daily (parameterised notebook, e.g. Papermill) instead of at month-end, with the break register as a shared, ageing list.
