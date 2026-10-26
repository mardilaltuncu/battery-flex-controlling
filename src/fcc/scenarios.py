"""Rolling forecast, scenario sets, tornado and Monte Carlo (P50 / P90).

Rolling forecast: at every month-end the next 12 months are re-forecast from trailing data:
  * theoretical spread  = trailing-12m level x seasonal index (seasonality from all complete years of history)
  * cycles (MWh per MW) = trailing-12m level x seasonal index
  * capture rate, fee rate, imbalance volume / price = trailing 12 months
  * reserve capacity price = last 3 months, flat (a deliberate naive choice that the backtest then tests)
  * MW online = known COD if it already happened, otherwise the planned COD probability-weighted with typical delays
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from . import contracts
from .config import (ASSETS, AVAILABILITY, COD_DELAY_PROBS, COST_TO_SERVE_EUR_MONTH, DEGRADATION_EUR_MWH,
                     DIRECT_COSTS_EUR_MONTH, JOINT_COSTS_EUR_MONTH, MC_DRAWS, RANDOM_SEED, Asset)
from .portfolio import daily_per_mw
from .variance import Drivers


# --------------------------------------------------------------------------------------------------
# Market history per asset variant (per MW)
# --------------------------------------------------------------------------------------------------
def variant_monthly(sets: dict, price: pd.Series) -> pd.DataFrame:
    rows = []
    for key, s in sets.items():
        pf_d = daily_per_mw(s["pf"], price)
        fc = s["fc"].dropna()
        fc_d = daily_per_mw(fc, price)
        d = pd.DataFrame({"pf_gross": pf_d["gross_eur"], "pf_dis": pf_d["discharge_mwh"], "pf_net": pf_d["net_eur"]})
        d = d.join(pd.DataFrame({"fc_gross": fc_d["gross_eur"], "fc_dis": fc_d["discharge_mwh"],
                                 "fc_net": fc_d["net_eur"]}), how="left")
        d["gap"] = (d["pf_net"] - d["fc_net"]).clip(lower=0)
        d["days"] = 1
        m = d.groupby(d.index.to_period("M")).sum(min_count=1)
        m["fc_days"] = d["fc_dis"].notna().groupby(d.index.to_period("M")).sum()
        m.index = m.index.to_timestamp()
        m["variant"] = [key] * len(m)
        rows.append(m.reset_index(names="month"))
    out = pd.concat(rows, ignore_index=True)
    out["theo"] = out["pf_gross"] / out["pf_dis"]
    out["q_pf_day"] = out["pf_dis"] / out["days"]
    return out


SEASONAL_AGG = "median"      # robust to crisis years (2021-22 winters); "mean" kept for comparison


def seasonal_index(vm: pd.DataFrame, variant, col: str, before: pd.Timestamp, agg: str | None = None) -> pd.Series:
    v = vm[(vm["variant"] == variant) & (vm["month"] < pd.Timestamp(before.year, 1, 1))].copy()
    v["y"] = v["month"].dt.year
    full = v.groupby("y")["month"].transform("size") == 12
    v = v[full]
    v["rel"] = v[col] / v.groupby("y")[col].transform("mean")
    s = v.groupby(v["month"].dt.month)["rel"].agg(agg or SEASONAL_AGG)
    return s / s.mean()


# --------------------------------------------------------------------------------------------------
# Scenario levers
# --------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Levers:
    theo_mult: float = 1.0
    cap_add: float = 0.0
    q_mult: float = 1.0
    res_mult: float = 1.0
    fee_mult: float = 1.0
    imb_rate_mult: float = 1.0
    cod_mode: str = "weighted"          # plan | weighted | delay
    cod_delay_m: int = 0
    share_add: float = 0.0              # Layer-2 only


@dataclass
class Inputs:
    as_of: pd.Timestamp
    theo_level: dict
    theo_season: dict
    q_level: dict
    q_season: dict
    cap: dict
    traded_per_dis: float
    fee_rate: float
    imb_vol_per_dis: float
    imb_rate: float
    res_price_mw_h: float
    extra: dict = field(default_factory=dict)


def build_inputs(vm: pd.DataFrame, pm: pd.DataFrame, reserve_px: pd.Series, as_of: str | pd.Timestamp) -> Inputs:
    """Everything the forecast may know at month-end ``as_of``."""
    as_of = pd.Timestamp(as_of)                                                     # a month-end
    first = (as_of + pd.Timedelta(days=1)).to_period("M").to_timestamp()            # first forecast month
    win = (vm["month"] >= first - pd.DateOffset(months=12)) & (vm["month"] < first)
    tl, ts, ql, qs, cap = {}, {}, {}, {}, {}
    for key in vm["variant"].unique():
        v = vm[win & (vm["variant"] == key)]
        s_t = seasonal_index(vm, key, "theo", first)
        s_q = seasonal_index(vm, key, "q_pf_day", first)
        months_k = v["month"].dt.month
        tl[key] = v["theo"].mean() / s_t.reindex(months_k).mean()
        ts[key] = s_t
        fc_dis_day = (v["fc_dis"] / v["fc_days"]).mean()
        ql[key] = fc_dis_day / s_q.reindex(months_k).mean()
        qs[key] = s_q
        realised = (v["fc_gross"].sum() + 0.0) / v["fc_dis"].sum()
        cap[key] = realised / (v["pf_gross"].sum() / v["pf_dis"].sum())
    p = pm[(pm["month"] >= first - pd.DateOffset(months=12)) & (pm["month"] < first)]
    # intraday recovery is part of capture: add it at portfolio level (share of DA gross)
    id_uplift = p["id_rev"].sum() / max(p["gross_da"].sum(), 1.0)
    cap = {k: c * (1 + id_uplift) for k, c in cap.items()}
    last3 = reserve_px[(reserve_px.index >= first - pd.DateOffset(months=3)) & (reserve_px.index < first)].mean()
    return Inputs(as_of=as_of, theo_level=tl, theo_season=ts, q_level=ql, q_season=qs, cap=cap,
                  traded_per_dis=p["traded_mwh"].sum() / p["dis_mwh"].sum(),
                  fee_rate=p["fees"].sum() / p["traded_mwh"].sum(),
                  imb_vol_per_dis=p["imb_mwh"].sum() / p["dis_mwh"].sum(),
                  imb_rate=p["imb_cost"].sum() / p["imb_mwh"].sum(),
                  res_price_mw_h=float(last3), extra={"first_month": first, "id_uplift": id_uplift})


def online_fraction(asset: Asset, month: pd.Timestamp, as_of: pd.Timestamp, lev: Levers) -> float:
    days = pd.Timestamp(month).days_in_month
    end = month + pd.DateOffset(months=1)

    def frac(cod: pd.Timestamp) -> float:
        if cod >= end:
            return 0.0
        if cod <= month:
            return 1.0
        return (end - cod).days / days

    if asset.cod_actual and pd.Timestamp(asset.cod_actual) <= as_of:
        return frac(pd.Timestamp(asset.cod_actual))
    plan = pd.Timestamp(asset.cod_plan)
    if lev.cod_mode == "plan":
        return frac(plan)
    if lev.cod_mode == "delay":
        return frac(plan + pd.DateOffset(months=lev.cod_delay_m))
    return sum(p * frac(plan + pd.DateOffset(months=d)) for d, p in COD_DELAY_PROBS.items())


def forecast_drivers(asset: Asset, month: pd.Timestamp, inp: Inputs, lev: Levers) -> Drivers:
    key = (asset.duration_h, asset.grid_cap_frac)
    k = month.month
    days = month.days_in_month
    theo = inp.theo_level[key] * inp.theo_season[key].get(k, 1.0) * lev.theo_mult
    q = inp.q_level[key] * inp.q_season[key].get(k, 1.0) * days * lev.q_mult
    res_price = inp.res_price_mw_h * 24 * days * AVAILABILITY * lev.res_mult
    return Drivers(mw=asset.mw * online_fraction(asset, month, inp.as_of, lev), res_share=asset.reserve_share, q=q,
                   theo=theo, cap=max(inp.cap[key] + lev.cap_add, 0.0), res_price=res_price,
                   traded_per_dis=inp.traded_per_dis, fee_rate=inp.fee_rate * lev.fee_mult, deg_rate=DEGRADATION_EUR_MWH,
                   imb_vol_per_dis=inp.imb_vol_per_dis, imb_rate=inp.imb_rate * lev.imb_rate_mult)


def forecast_table(inp: Inputs, lev: Levers = Levers(), horizon: int = 12, assets: list[Asset] | None = None,
                   last_month: pd.Timestamp | None = None) -> pd.DataFrame:
    rows = []
    first = inp.extra["first_month"]
    for h in range(1, horizon + 1):
        m = first + pd.DateOffset(months=h - 1)
        if last_month is not None and m > last_month:
            break
        for a in assets or ASSETS:
            d = forecast_drivers(a, m, inp, lev)
            rows.append({"as_of": inp.as_of, "month": m, "h": h, "asset_id": a.asset_id, "drivers": d, **d.lines()})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------------
# Backtest of the rolling forecast
# --------------------------------------------------------------------------------------------------
def backtest(vm, pm, reserve_px, first_origin: str, last_actual_month: str) -> pd.DataFrame:
    last = pd.Timestamp(last_actual_month)
    origins = pd.date_range(first_origin, last - pd.DateOffset(days=1), freq="ME")
    act = pm.set_index(["month", "asset_id"])["net_rev"]
    out = []
    for o in origins:
        inp = build_inputs(vm, pm, reserve_px, o)
        ft = forecast_table(inp, Levers(), assets=[a for a in ASSETS if a.cod_actual], last_month=last)
        ft["actual"] = [act.get((m, a), 0.0) for m, a in zip(ft["month"], ft["asset_id"])]
        out.append(ft)
    return pd.concat(out, ignore_index=True)


def accuracy_by_horizon(bt: pd.DataFrame) -> pd.DataFrame:
    p = bt.groupby(["as_of", "month", "h"])[["Net trading revenue", "actual"]].sum().reset_index()
    p["err"] = p["Net trading revenue"] - p["actual"]
    p["ape"] = p["err"].abs() / p["actual"].abs()
    g = p.groupby("h")
    return pd.DataFrame({"n": g.size(), "MAPE %": 100 * g["ape"].mean(),
                         "bias % (forecast-actual)": 100 * g["err"].sum() / g["actual"].sum()}).round(1)


# --------------------------------------------------------------------------------------------------
# Forward view: scenarios, tornado, Monte Carlo
# --------------------------------------------------------------------------------------------------
def spread_ratio_distribution(vm: pd.DataFrame, variant=(2.0, 1.0)) -> pd.Series:
    """Next-12m / trailing-12m theoretical spread, for every month-end in history. Both are 12-month windows, so
    seasonality cancels; the result is an empirical, skewed distribution of how much spreads move in a year."""
    v = vm[vm["variant"] == variant].set_index("month").sort_index()
    g = v["pf_gross"]
    dsum = v["pf_dis"]
    trail = g.rolling(12).sum() / dsum.rolling(12).sum()
    nxt = g[::-1].rolling(12).sum()[::-1].shift(-1) / dsum[::-1].rolling(12).sum()[::-1].shift(-1)
    r = (nxt / trail).dropna()
    # centre on the median: the base case already carries the current level, the distribution adds only dispersion
    out = (r / r.median()).rename("spread_ratio")
    out.attrs["raw_median"] = float(r.median())
    return out


def company_view(ft: pd.DataFrame, lev: Levers, assets: list[Asset] | None = None) -> dict:
    """12-month Layer-1 net, Layer-2 company revenue and company EBITDA for a forecast table."""
    assets = assets or ASSETS
    out = {"layer1_net": 0.0, "company_revenue": 0.0, "floor_topup": 0.0, "cost_to_serve": 0.0}
    for a in assets:
        g = ft[ft["asset_id"] == a.asset_id]
        net = g["Net trading revenue"].sum()
        months_on = sum(d.mw / a.mw for d in g["drivers"]) if a.mw else 0.0
        cs = contracts.company_share(a, net, months=months_on, share_add=lev.share_add)
        out["layer1_net"] += net
        out["company_revenue"] += cs["company"]
        out["floor_topup"] += cs["topup"]
        out["cost_to_serve"] += COST_TO_SERVE_EUR_MONTH[a.customer] * months_on
    n_m = ft["month"].nunique()
    out["direct_costs"] = sum(DIRECT_COSTS_EUR_MONTH.values()) * n_m
    out["joint_costs"] = sum(JOINT_COSTS_EUR_MONTH.values()) * n_m
    out["company_ebitda"] = out["company_revenue"] - out["direct_costs"] - out["joint_costs"] - out["cost_to_serve"]
    return out


def scenario_sets(ratio: pd.Series) -> dict[str, Levers]:
    """Internally consistent driver sets. Spread levels come from the empirical ratio distribution:
    downside / upside = P25 / P75, stress = P10 with every other lever on its adverse side."""
    p10, p25, p75 = ratio.quantile(0.10), ratio.quantile(0.25), ratio.quantile(0.75)
    return {
        "Stress": Levers(theo_mult=p10, cap_add=-0.06, q_mult=0.95, res_mult=0.55, imb_rate_mult=1.40,
                         cod_mode="delay", cod_delay_m=9),
        "Downside": Levers(theo_mult=p25, cap_add=-0.03, q_mult=0.97, res_mult=0.70, imb_rate_mult=1.20,
                           cod_mode="delay", cod_delay_m=6),
        "Base": Levers(),
        "Upside": Levers(theo_mult=p75, cap_add=0.02, q_mult=1.0, res_mult=1.10, imb_rate_mult=0.90, cod_mode="plan"),
    }


def run_scenarios(inp: Inputs, sets: dict[str, Levers]) -> pd.DataFrame:
    rows = {}
    for name, lev in sets.items():
        ft = forecast_table(inp, lev)
        rows[name] = company_view(ft, lev)
    return pd.DataFrame(rows).T


def tornado(inp: Inputs, ratio: pd.Series, metric: str) -> pd.DataFrame:
    base = Levers()
    p10, p90 = ratio.quantile(0.10), ratio.quantile(0.90)
    tests = {
        "Market spread level (P10 / P90)": (replace(base, theo_mult=p10), replace(base, theo_mult=p90)),
        "Capture rate (-5 / +5 pp)": (replace(base, cap_add=-0.05), replace(base, cap_add=0.05)),
        "Cycles per MW (-10 / +10 %)": (replace(base, q_mult=0.9), replace(base, q_mult=1.1)),
        "Reserve capacity price (-35 / +15 %)": (replace(base, res_mult=0.65), replace(base, res_mult=1.15)),
        "COD of pre-COD asset (+9 m / on plan)": (replace(base, cod_mode="delay", cod_delay_m=9), replace(base, cod_mode="plan")),
        "Imbalance price (+30 / -30 %)": (replace(base, imb_rate_mult=1.3), replace(base, imb_rate_mult=0.7)),
        "Exchange fees (+25 / -25 %)": (replace(base, fee_mult=1.25), replace(base, fee_mult=0.75)),
        "Revenue share rate (-3 / +3 pp)": (replace(base, share_add=-0.03), replace(base, share_add=0.03)),
    }
    b = company_view(forecast_table(inp, base), base)[metric]
    rows = []
    for name, (lo, hi) in tests.items():
        v_lo = company_view(forecast_table(inp, lo), lo)[metric]
        v_hi = company_view(forecast_table(inp, hi), hi)[metric]
        rows.append({"driver": name, "low": v_lo - b, "high": v_hi - b, "range": abs(v_hi - v_lo)})
    out = pd.DataFrame(rows).sort_values("range", ascending=False).reset_index(drop=True)
    out.attrs["base"] = b
    return out


def monte_carlo(inp: Inputs, ratio: pd.Series, n: int = MC_DRAWS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    delays = np.array(list(COD_DELAY_PROBS.keys()))
    probs = np.array(list(COD_DELAY_PROBS.values()))
    rows = []
    r = ratio.to_numpy()
    for _ in range(n):
        lev = Levers(theo_mult=float(rng.choice(r)), cap_add=float(np.clip(rng.normal(0, 0.03), -0.15, 0.10)),
                     q_mult=float(rng.normal(1, 0.04)), res_mult=float(rng.lognormal(-0.10, 0.25)),
                     imb_rate_mult=float(rng.lognormal(0, 0.30)), cod_mode="delay",
                     cod_delay_m=int(rng.choice(delays, p=probs)))
        rows.append(company_view(forecast_table(inp, lev), lev))
    return pd.DataFrame(rows)


def percentiles(x: pd.Series) -> dict:
    """P50 / P90 / P99 (P90 = exceeded with 90 % probability) - empirical and under a normal assumption."""
    mu, sd = x.mean(), x.std()
    return {"mean": mu, "P50": x.quantile(0.50), "P90 (empirical)": x.quantile(0.10), "P90 (normal approx)": mu - 1.2816 * sd,
            "P99 (empirical)": x.quantile(0.01), "P99 (normal approx)": mu - 2.3263 * sd, "skewness": x.skew()}
