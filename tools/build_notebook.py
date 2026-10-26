"""Build (and optionally execute) the Kaggle notebook from the cell list below.

Usage: python -P tools/build_notebook.py [--execute]
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "flex_controlling_case.ipynb"
TEXT = ROOT / "tools" / "narrative.md"


def load_narrative() -> dict[str, str]:
    """Narrative blocks live in tools/narrative.md as '<!-- key -->' sections."""
    out, key, buf = {}, None, []
    for line in TEXT.read_text(encoding="utf-8").splitlines():
        if line.startswith("<!-- ") and line.endswith(" -->"):
            if key:
                out[key] = "\n".join(buf).strip()
            key, buf = line[5:-4].strip(), []
        else:
            buf.append(line)
    if key:
        out[key] = "\n".join(buf).strip()
    return out


CODE = {}

CODE["setup"] = r'''
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
warnings.filterwarnings("ignore", category=FutureWarning)

def find_root() -> Path:
    cands = [Path.cwd(), Path.cwd().parent]
    if Path("/kaggle/input").exists():
        cands += [p for pat in ("*", "*/*", "*/*/*", "*/*/*/*") for p in Path("/kaggle/input").glob(pat) if p.is_dir()]
    for c in cands:
        if (c / "src" / "fcc").exists():
            return c
    raise FileNotFoundError("project root with src/fcc not found")

ROOT = find_root()
sys.path.insert(0, str(ROOT / "src"))
OUT = Path("/kaggle/working") if Path("/kaggle/working").exists() else ROOT / "outputs"
CACHE = OUT / "cache"
OUT.mkdir(parents=True, exist_ok=True)

from fcc import (config, market, forecast, dispatch, portfolio, synth, recon, variance, scenarios,
                 profitability, excel_out, viz)
from fcc.config import ASSETS, TZ
viz.style()
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.max_colwidth", 70)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
eur = lambda v: f"{v:,.0f} EUR"
print("root:", ROOT.name, "| outputs:", OUT.relative_to(ROOT) if OUT.is_relative_to(ROOT) else OUT)
'''

CODE["assumptions"] = r'''
A = config.assumptions_table()
print(A["source"].value_counts().to_string())
display(A[["parameter", "value", "unit", "source", "note"]])
'''

CODE["load"] = r'''
price_df = market.load_price(ROOT / "data/energy_charts/price_de_lu.parquet")
P = price_df["price_eur_mwh"]
F = market.load_fundamentals(ROOT / "data/energy_charts/fundamentals_de.parquet")
LAST_DAY = P.index.max().tz_convert(TZ).strftime("%Y-%m-%d")
audit = market.audit_price(price_df, "2019-01-01", LAST_DAY)
display(pd.Series(audit, name="value").to_frame())
display(market.annual_stats(price_df).loc[2019:])
ic = market.interval_count_check(P.index, "2019-01-01", LAST_DAY)
print("DST days (only these may differ from 96 intervals):")
display(ic[ic["n_expected"] != 96][["n_expected", "n_found", "gap"]])
print("native resolution:", price_df["native_res_min"].value_counts().to_dict(),
      "| first 15-minute day-ahead interval:", P.index[price_df["native_res_min"] == 15].min())
'''

CODE["hf_audit"] = r'''
hf = market.audit_secondary_source(ROOT / "data/raw/entsoe_delu_dayahead_prices.parquet", price_df)
display(hf)
'''

CODE["forecast"] = r'''
t0 = time.time()
feat = forecast.build_features(P, F)
ml_path = CACHE / "ml_forecast.parquet"
if ml_path.exists():
    ml = pd.read_parquet(ml_path)["ml"]
else:
    ml = forecast.walk_forward(feat, config.BACKTEST_START, LAST_DAY, step_months=3)
    CACHE.mkdir(parents=True, exist_ok=True); ml.to_frame().to_parquet(ml_path)
print(f"walk-forward forecast: {len(ml):,} intervals, {time.time() - t0:.0f}s")
y = feat.loc[ml.index, "price"]
fm = pd.DataFrame({
    "naive (D-1 / D-7 weekday rule)": forecast.metrics(y, feat.loc[ml.index, "naive"]),
    "naive D-1": forecast.metrics(y, feat.loc[ml.index, "lag_24h"]),
    "gradient boosting (walk-forward)": forecast.metrics(y, ml),
}).T
display(fm.round(2))
yr = ml.index.tz_convert(TZ).year
by_year = pd.DataFrame({f"{k} {v}": [forecast.metrics(y[yr == yy], (ml if k == "ML" else feat.loc[ml.index, "naive"])[yr == yy])[v]
                                    for yy in sorted(set(yr))] for k in ["ML", "naive"] for v in ["MAE", "bias (mean error)"]},
                       index=sorted(set(yr)))
display(by_year.round(2))
'''

CODE["dispatch"] = r'''
t0 = time.time()
sets = portfolio.dispatch_sets(P, ml, CACHE, config.BACKTEST_START)
naive_px = feat.loc[ml.index, "naive"].fillna(ml)
naive_sched = portfolio.cached(CACHE / "sched_naive_d2_cap1.parquet", lambda: dispatch.run_dispatch(naive_px, 2.0, 1.0))
print(f"dispatch (perfect foresight 2019+, forecast-based {config.BACKTEST_START}+): {time.time() - t0:.0f}s")

ref = sets[(2.0, 1.0)]
idx = ml.index
val = {name: dispatch.settle(s.reindex(idx), P.reindex(idx)) for name, s in
       [("perfect foresight", ref["pf"]), ("ML forecast", ref["fc"]), ("naive forecast", naive_sched)]}
yr = idx.tz_convert(TZ).year
cap = pd.DataFrame({n: v["net_eur"].groupby(yr).sum() for n, v in val.items()})
cap["capture ML %"] = 100 * cap["ML forecast"] / cap["perfect foresight"]
cap["capture naive %"] = 100 * cap["naive forecast"] / cap["perfect foresight"]
cap["ML uplift vs naive, EUR/MW"] = cap["ML forecast"] - cap["naive forecast"]
print("Day-ahead arbitrage, 2-hour battery, net EUR per MW (fees and degradation included); 2026 = Jan-Oct")
display(cap.round(1))
mon = pd.DataFrame({n: dispatch.monthly(v)["net_eur"] for n, v in val.items()})
mon = mon[mon.index <= "2026-09-01"]                      # complete months only (October 2026 is partial)
ax = viz.lines(mon, list(mon.columns), "Net day-ahead arbitrage value per MW, 2-hour battery", "EUR per MW and month", scale="k")
plt.show()
'''

CODE["portfolio"] = r'''
display(pd.DataFrame([vars(a) for a in ASSETS]))
reb = portfolio.full_rebap(P)
pm = portfolio.portfolio_monthly(sets, P, reb, config.BACKTEST_START, "2026-09-30")
yr_tab = pm.assign(year=pm["month"].dt.year).groupby("year")[
    ["gross_da", "id_rev", "res_rev", "fees", "deg", "imb_cost", "net_rev"]].sum()
yr_tab["MW (avg)"] = pm.assign(year=pm["month"].dt.year).groupby(["year", "month"])["mw"].sum().groupby("year").mean()
yr_tab["EUR/MW/yr"] = yr_tab["net_rev"] / yr_tab["MW (avg)"] / pm.assign(year=pm["month"].dt.year).groupby("year")["month"].nunique() * 12
print("Layer 1 - asset net trading revenue by year (EUR); 2026 = Jan-Sep")
display(yr_tab.round(0))
wide = pm.pivot_table(index="month", columns="asset_id", values="net_rev", aggfunc="sum")
viz.lines(wide, list(wide.columns), "Net trading revenue per asset (Layer 1)", "EUR per month", scale="k")
plt.show()
'''

CODE["ops"] = r'''
fc_by_variant = {k: v["fc"] for k, v in sets.items()}
ops = synth.build_ops(P, fc_by_variant, config.RECON_START, config.RECON_END, seed=config.RANDOM_SEED)
print({k: f"{len(getattr(ops, k)):,} rows" for k in ["ledger", "ecc", "fahrplan", "meter"]})
raw_dir = OUT / "raw_csv_sample"; raw_dir.mkdir(exist_ok=True)
m = ops.ledger["ts_utc"].dt.tz_convert(TZ).dt.strftime("%Y-%m") == "2025-10"
ops.ledger[m].to_csv(raw_dir / "ledger_2025-10.csv", index=False)
ops.ecc[ops.ecc["delivery_date"].str.startswith("2025-10")].to_csv(raw_dir / "ecc_statement_2025-10.csv", index=False)
print("Trading ledger (internal export, UTC, net MWh: sell +):")
display(ops.ledger.head(4))
print("Exchange / CCP statement (as delivered: local clock time, buy and sell columns, A/B marker for the repeated hour):")
display(ops.ecc[ops.ecc["delivery_date"] == "2025-10-26"].iloc[6:12])
'''

CODE["parse"] = r'''
ep, notes = recon.parse_ecc(ops.ecc)
print("parsing notes:", notes)
print("naive join on local clock time:", recon.naive_local_join_stats(ops.ledger, ops.ecc))
'''

CODE["two_way"] = r'''
rec = recon.reconcile_two_way(ops.ledger, ep)
summ = recon.summary(rec)
display(pd.Series(summ, name="value").to_frame())
display(rec["status"].value_counts().to_frame("intervals"))
reg2 = recon.break_register(rec)
print("Break register (two-way), largest first:")
display(reg2[["break_id", "asset_id", "local_date", "type", "n_intervals", "mwh_diff", "eur_impact", "severity", "owner",
              "root_cause", "age_days"]].head(14))
print("Systematic small differences (one-sided):")
display(rec.attrs["systematic"])
'''

CODE["fees"] = r'''
ep["month"] = ep["ts_utc"].dt.tz_convert(TZ).dt.strftime("%Y-%m")
fee = ep.assign(vol=ep["sell_mwh"] + ep["buy_mwh"]).groupby("month").agg(volume_mwh=("vol", "sum"), fees_eur=("fee_eur", "sum"))
fee["EUR per MWh on statement"] = fee["fees_eur"] / fee["volume_mwh"]
fee["expected (budget tariff)"] = config.FEE_DA_EUR_MWH
fee["difference EUR"] = fee["fees_eur"] - fee["volume_mwh"] * config.FEE_DA_EUR_MWH
display(fee.round(4))
'''

CODE["deep_dive"] = r'''
b = reg2.iloc[0]
trace = rec[(rec["asset_id"] == b["asset_id"]) & (rec["local_date"] == b["local_date"]) & (rec["status"] == b["type"])]
print(f"{b['break_id']}: {b['type']} on {b['local_date']} ({b['asset_id']}), {b['n_intervals']} intervals, {eur(b['eur_impact'])}")
display(trace[["ts_utc", "net_led", "price_led", "amount_led", "net_ecc", "amount_ecc", "diff_eur", "status"]])
'''

CODE["four_way"] = r'''
x4 = recon.reconcile_four_way(rec, ops.fahrplan, ops.meter, ops.rebap)
display(x4["status"].value_counts().to_frame("intervals"))
print("Systematic meter deviation (every single interval is inside tolerance):")
display(x4.attrs["meter_bias"].round(3))
reg4 = recon.breaks_four_way(x4)
s4 = reg4.groupby(["type", "owner"]).agg(breaks=("break_id", "size"), intervals=("n_intervals", "sum"),
                                        eur=("eur_impact", "sum")).reset_index()
display(s4)
both = pd.concat([reg2.assign(scope="2-way"), reg4.assign(scope="4-way")])
by_type = both.groupby("type")["abs_eur"].sum().sort_values(ascending=False)
viz.bars(by_type, "Open breaks by type, absolute EUR impact (12 months)", "EUR", fmt="{:,.0f}")
plt.show()
'''

CODE["scorecard"] = r'''
score = recon.scorecard(ops.truth_intervals, rec, x4)
display(score)
'''

CODE["bridge"] = r'''
vm = scenarios.variant_monthly(sets, P)
rp = portfolio.reserve_price_path(pd.date_range("2019-01-01", "2027-12-01", freq="MS"))
bt = scenarios.backtest(vm, pm, rp, "2024-12-31", "2026-09-01")
M = pd.Timestamp("2026-06-01")
op_assets = [a for a in ASSETS if a.cod_actual]

def bridge_for(month, origin):
    tab = bt[(bt["as_of"] == pd.Timestamp(origin)) & (bt["month"] == month)]
    parts, pairs = {}, {}
    for a in op_assets:
        b = tab[tab["asset_id"] == a.asset_id]["drivers"].iloc[0]
        row = pm[(pm["month"] == month) & (pm["asset_id"] == a.asset_id)]
        # not yet online (COD later than forecast): zero MW, unit drivers as budgeted -> pure COD-timing effect
        act = portfolio.drivers_row(row.iloc[0]) if len(row) else variance.Drivers(**{**vars(b), "mw": 0.0})
        parts[a.asset_id] = variance.bridge(b, act)
        pairs[a.asset_id] = (b, act)
    return variance.sum_bridges(list(parts.values())), parts, pairs

br_budget, parts_budget, pairs_budget = bridge_for(M, "2025-12-31")
br_latest, _, _ = bridge_for(M, "2026-05-31")
print(variance.ORDER_NOTE)
print("control - budget + effects - actual:", round(br_budget.attrs["check_diff"], 6), "EUR")
tab = br_budget[["step", "nature", "eur"]].rename(columns={"eur": "vs budget (Dec-2025 forecast)"})
tab["vs latest forecast (May-2026)"] = br_latest["eur"].to_numpy()
tab = variance.flag_material(tab.rename(columns={"vs budget (Dec-2025 forecast)": "eur"}), br_budget["eur"].iloc[0]) \
    .rename(columns={"eur": "vs budget (Dec-2025 forecast)"})
display(tab)
viz.waterfall(br_budget, "June 2026: budget (Dec-2025 rolling forecast) to actual net trading revenue")
plt.show()
'''

CODE["attribution"] = r'''
rows = []
for origin in sorted(bt["as_of"].unique()):
    month = pd.Timestamp(origin) + pd.Timedelta(days=1)
    if month > pd.Timestamp("2026-09-01"):
        continue
    br_, _, _ = bridge_for(month, origin)
    rows.append(br_.set_index("step")["eur"].rename(month))
att = pd.DataFrame(rows)
mid = att.columns[1:-1]
summary_att = pd.DataFrame({"sum over months (EUR)": att[mid].sum(), "mean |effect| (EUR)": att[mid].abs().mean(),
                            "months actual > forecast": (att[mid] > 0.5).sum(), "months actual < forecast": (att[mid] < -0.5).sum()})
summary_att["share of total |error|"] = summary_att["mean |effect| (EUR)"] / summary_att["mean |effect| (EUR)"].sum()
print(f"One-month-ahead forecast errors, {len(att)} months, decomposed with the same bridge:")
display(summary_att.sort_values("mean |effect| (EUR)", ascending=False))
'''

CODE["backtest"] = r'''
acc = scenarios.accuracy_by_horizon(bt)
display(acc)
pt = bt.groupby(["as_of", "month", "h"])[["Net trading revenue", "actual"]].sum().reset_index()
act_m = pt.groupby("month")["actual"].first()
h1 = pt[pt["h"] == 1].set_index("month")["Net trading revenue"]
h6 = pt[pt["h"] == 6].set_index("month")["Net trading revenue"]
comp = pd.DataFrame({"actual": act_m, "forecast 1 month ahead": h1, "forecast 6 months ahead": h6}).dropna(subset=["forecast 1 month ahead"])
viz.lines(comp, list(comp.columns), "Rolling forecast vs actual, portfolio net trading revenue", "EUR per month", scale="k")
plt.show()
'''

CODE["season_test"] = r'''
# model choice is tested, not assumed: seasonal index from the mean vs the median of yearly profiles
res = {}
for agg in ["mean", "median"]:
    scenarios.SEASONAL_AGG = agg
    a_ = scenarios.accuracy_by_horizon(scenarios.backtest(vm, pm, rp, "2024-12-31", "2026-09-01"))
    res[f"seasonality = {agg}"] = a_.loc[[1, 3, 6, 12], "MAPE %"].rename(lambda h: f"MAPE h={h}")
scenarios.SEASONAL_AGG = "median"
display(pd.DataFrame(res).T)
'''

CODE["scenarios"] = r'''
AS_OF = "2026-09-30"
inp = scenarios.build_inputs(vm, pm, rp, AS_OF)
ratio = scenarios.spread_ratio_distribution(vm)
print(f"Spread ratio (next 12m / trailing 12m), n={len(ratio)} monthly origins since 2019, centred on its median "
      f"(raw median {ratio.attrs['raw_median']:.2f}):")
display(ratio.quantile([0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]).to_frame("ratio").T.round(2))
sc_sets = scenarios.scenario_sets(ratio)
display(pd.DataFrame({k: vars(v) for k, v in sc_sets.items()}).T)
scen = scenarios.run_scenarios(inp, sc_sets)
display(scen.round(0))
'''

CODE["tornado"] = r'''
tor_l1 = scenarios.tornado(inp, ratio, "layer1_net")
tor_co = scenarios.tornado(inp, ratio, "company_ebitda")
fig, axs = plt.subplots(2, 1, figsize=(11, 8.4))  # each panel sorted by its own range, so each keeps its own labels
viz.tornado(tor_l1, f"Layer 1 net trading revenue (base {tor_l1.attrs['base'] / 1e6:,.1f}M EUR)", scale="M", ax=axs[0])
viz.tornado(tor_co, f"Company EBITDA, Layer 2 module (base {tor_co.attrs['base'] / 1e3:,.0f}k EUR)", scale="k", ax=axs[1])
plt.tight_layout(); plt.show()
display(pd.concat({"Layer 1": tor_l1.set_index("driver"), "company EBITDA": tor_co.set_index("driver")}, axis=1).round(0))
'''

CODE["mc"] = r'''
mc = scenarios.monte_carlo(inp, ratio, n=config.MC_DRAWS)
pct = pd.DataFrame({c: scenarios.percentiles(mc[c]) for c in ["layer1_net", "company_ebitda"]}).T
display(pct.round(2))
p = scenarios.percentiles(mc["layer1_net"])
viz.histogram(mc["layer1_net"], {"P50": p["P50"], "P90 empirical": p["P90 (empirical)"], "P90 normal approx": p["P90 (normal approx)"]},
              f"Monte Carlo, 12-month Layer 1 net trading revenue ({len(mc):,} draws)", scale="M")
plt.show()
print(f"share of draws in which the C2 floor pays out: {(mc['floor_topup'] > 0).mean():.1%}; "
      f"mean top-up when it does: {eur(mc.loc[mc['floor_topup'] > 0, 'floor_topup'].mean())}")
'''

CODE["profit"] = r'''
PNL_START, PNL_END = "2025-10-01", "2026-09-01"
prod, cust, attr = profitability.product_pnl(pm, PNL_START, PNL_END)
cols = ["revenue", "contribution_l1", "company_revenue", "direct_costs", "CM1", "joint_by_revenue", "joint_by_activity",
        "net_by_revenue_key", "net_by_activity_key"]
display(prod[cols].round(0))
viz.grouped_bars(prod[["CM1", "net_by_revenue_key", "net_by_activity_key"]].rename(columns={
    "CM1": "CM1 (before joint costs)", "net_by_revenue_key": "net, revenue key", "net_by_activity_key": "net, activity key"}),
    ["CM1 (before joint costs)", "net, revenue key", "net, activity key"],
    "Product contribution, Oct-2025 to Sep-2026 (company view)", "EUR")
plt.show()
'''

CODE["drop"] = r'''
loser = prod["net_by_activity_key"].idxmin()
dt = profitability.drop_test(prod, loser)
print(f"Drop test for '{loser}' (looks loss-making under the activity key):")
display(dt.round(0))
print(f"company result before: {eur(dt.attrs['company_result_before'])} | after dropping: {eur(dt.attrs['company_result_after'])} "
      f"| change = -CM1 of the dropped product: {eur(-dt.attrs['dropped_cm1'])}")
display(cust.round(0))
'''

CODE["excel"] = r'''
rd_l = ops.ledger.assign(local_date=ops.ledger["ts_utc"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d")) \
    .groupby(["asset_id", "local_date"]).agg(ledger_rows=("net_mwh", "size"), ledger_mwh=("net_mwh", "sum"), ledger_eur=("amount_eur", "sum"))
rd_e = ep.assign(local_date=ep["ts_utc"].dt.tz_convert(TZ).dt.strftime("%Y-%m-%d"), net=ep["sell_mwh"] - ep["buy_mwh"]) \
    .groupby(["asset_id", "local_date"]).agg(ecc_rows=("net", "size"), ecc_mwh=("net", "sum"), ecc_eur=("amount_eur", "sum"))
recon_daily = rd_l.join(rd_e, how="outer").fillna(0).reset_index()[
    ["asset_id", "local_date", "ledger_rows", "ecc_rows", "ledger_mwh", "ecc_mwh", "ledger_eur", "ecc_eur"]]
reg_cols = ["break_id", "scope", "asset_id", "local_date", "type", "n_intervals", "mwh_diff", "eur_impact", "severity",
            "owner", "root_cause", "next_step", "age_days", "status"]
register = pd.concat([reg2.assign(scope="2-way", break_id="R2-" + reg2["break_id"]),
                      reg4.assign(scope="4-way", break_id="R4-" + reg4["break_id"])])[reg_cols]
register["severity"] = register["severity"].astype(str)
R = {
    "assumptions": config.assumptions_table(), "pm": pm,
    "bridge_label": "June 2026, budget = Dec-2025 rolling forecast", "bridge_order_note": variance.ORDER_NOTE,
    "bridge_parts": pairs_budget, "bridge_steps": {k: v for k, v in parts_budget.items()},
    "interval_check": ic, "recon_daily": recon_daily, "register": register,
    "accuracy": acc, "scenarios": scen, "tornado_l1": tor_l1, "tornado_co": tor_co, "mc_pct": pct,
    "prod": prod, "cust": cust, "pnl_window": "Oct-2025 to Sep-2026", "pnl_months": prod.attrs["months"],
    "pnl_start": pd.Timestamp(PNL_START), "pnl_end": pd.Timestamp(PNL_END),
    "joint_costs": config.JOINT_COSTS_EUR_MONTH, "alloc_activity": config.ALLOC_ACTIVITY,
    "direct_costs": config.DIRECT_COSTS_EUR_MONTH,
    "change_log": pd.DataFrame([
        {"date": pd.Timestamp("2026-10-20"), "version": "1.0", "change": "First version: data audit, reconciliation, bridge, scenarios, product P&L."},
    ]),
}
xlsx = OUT / "flex_controlling_case.xlsx"
try:
    info = excel_out.write_workbook(str(xlsx), R)
    print("written:", xlsx.relative_to(ROOT) if xlsx.is_relative_to(ROOT) else xlsx)
    for i, c in enumerate(info["checks"], 1):
        print(f"  check {i:2d}: {c}")
except ImportError as exc:                       # e.g. a fresh environment without xlsxwriter
    print("Excel export skipped:", exc, "-> pip install xlsxwriter")
'''

CODE["summary"] = r'''
kpi = pd.Series({
    "price intervals audited": f"{audit['intervals']:,}",
    "forecast MAE ML vs naive (EUR/MWh)": f"{fm.loc['gradient boosting (walk-forward)', 'MAE']:.1f} vs {fm.loc['naive (D-1 / D-7 weekday rule)', 'MAE']:.1f}",
    "capture rate ML vs naive (2025)": f"{cap.loc[2025, 'capture ML %']:.1f}% vs {cap.loc[2025, 'capture naive %']:.1f}%",
    "intervals reconciled (12 m)": f"{summ['intervals_compared']:,}",
    "within tolerance": f"{summ['within_tolerance_%']:.1f}%",
    "injected break types caught": f"{(score['recall_%'] == 100).sum()} of {len(score)} at 100% recall",
    "rolling forecast bias (1 month ahead)": f"{acc.loc[1, 'bias % (forecast-actual)']:.1f}%",
    "12m Layer 1 net, base / P90 empirical": f"{scen.loc['Base', 'layer1_net'] / 1e6:.1f}M / {pct.loc['layer1_net', 'P90 (empirical)'] / 1e6:.1f}M EUR",
    "company EBITDA stress / base / upside": " / ".join(f"{scen.loc[k, 'company_ebitda'] / 1e3:,.0f}k" for k in ["Stress", "Base", "Upside"]),
}, name="value")
display(kpi.to_frame())
'''

ORDER = [
    ("md", "title"), ("code", "setup"), ("md", "provenance"), ("code", "assumptions"),
    ("md", "s1"), ("code", "load"), ("md", "s1_audit"), ("code", "hf_audit"), ("md", "s1_audit_after"),
    ("md", "s2"), ("code", "forecast"), ("md", "s2_metrics"), ("code", "dispatch"), ("md", "s2_dispatch"),
    ("md", "s3"), ("code", "portfolio"), ("md", "s3_after"),
    ("md", "s4"), ("code", "ops"), ("md", "s4_parse"), ("code", "parse"), ("md", "s4_two_way"), ("code", "two_way"),
    ("md", "s4_two_way_after"), ("code", "fees"), ("md", "s4_deep"), ("code", "deep_dive"), ("md", "s4_four"),
    ("code", "four_way"), ("md", "s4_score"), ("code", "scorecard"),
    ("md", "s5"), ("code", "bridge"), ("md", "s5_after"), ("code", "attribution"), ("md", "s5_attr"),
    ("md", "s6"), ("code", "backtest"), ("md", "s6_backtest"), ("code", "season_test"), ("md", "s6_season"),
    ("code", "scenarios"), ("md", "s6_scen"),
    ("code", "tornado"), ("md", "s6_tornado"), ("code", "mc"), ("md", "s6_mc"),
    ("md", "s7"), ("code", "profit"), ("md", "s7_after"), ("code", "drop"), ("md", "s7_drop"),
    ("md", "s8"), ("code", "excel"), ("md", "s9"), ("code", "summary"), ("md", "limits"),
]


def build() -> nbf.NotebookNode:
    text = load_narrative()
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    for kind, key in ORDER:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(text.get(key, f"*(missing narrative: {key})*")))
        else:
            nb.cells.append(nbf.v4.new_code_cell(CODE[key].strip("\n")))
    return nb


if __name__ == "__main__":
    nb = build()
    NB.parent.mkdir(parents=True, exist_ok=True)
    if "--execute" in sys.argv:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}).execute()
    nbf.write(nb, NB)
    print("written", NB)
