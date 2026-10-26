"""Battery day-ahead dispatch as a linear programme (per MW, so results scale linearly with asset size).

The day-ahead auction closes at 12:00 CET on D-1 for the whole local delivery day D, so each local day (92 / 96 / 100
intervals) is optimised independently. Net spread follows ``P_sell - P_buy / RTE``; degradation and exchange fees are
part of the objective, so a trade only happens when it is economic after those costs.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from scipy.optimize import linprog

from .config import (DEGRADATION_EUR_MWH, DT_H, FEE_DA_EUR_MWH, MAX_CYCLES_PER_DAY, RTE, SOC_START_FRAC, TZ)


@lru_cache(maxsize=8)
def _tril(t: int) -> np.ndarray:
    return np.tril(np.ones((t, t)))


def solve_day(price: np.ndarray, duration_h: float, rte: float = RTE, deg: float = DEGRADATION_EUR_MWH,
              fee: float = FEE_DA_EUR_MWH, max_cycles: float = MAX_CYCLES_PER_DAY, soc0: float = SOC_START_FRAC,
              dis_cap: np.ndarray | None = None, dt: float = DT_H) -> tuple[np.ndarray, np.ndarray]:
    """Optimal charge / discharge power (MW per MW of rated power) for one local day."""
    t = len(price)
    eta = np.sqrt(rte)
    e_cap = duration_h                      # MWh per MW
    e0 = soc0 * e_cap
    lo = _tril(t)
    a = dt * np.hstack([eta * lo, -lo / eta])          # state of charge change after each interval
    c_cost = dt * (price + fee)
    d_cost = dt * (-price + fee + deg)
    obj = np.concatenate([c_cost, d_cost])
    # rows: SoC <= capacity | SoC >= 0 | end SoC >= start SoC | daily discharge <= cycle limit
    a_ub = np.vstack([a, -a, -a[-1:], dt * np.hstack([np.zeros(t), np.ones(t)])[None, :]])
    b_ub = np.concatenate([np.full(t, e_cap - e0), np.full(t, e0), [0.0], [max_cycles * e_cap]])
    d_up = np.ones(t) if dis_cap is None else dis_cap
    bounds = [(0, 1)] * t + [(0, float(u)) for u in d_up]
    res = linprog(obj, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success:                                # defensive: idle if the solver fails
        return np.zeros(t), np.zeros(t)
    return res.x[:t], res.x[t:]


def grid_cap_vector(index: pd.DatetimeIndex, frac: float) -> np.ndarray | None:
    """Co-located asset: discharge is capped to ``frac`` of rated power in PV hours (10:00-16:00 local, Apr-Sep)."""
    if frac >= 1.0:
        return None
    loc = index.tz_convert(TZ)
    pv = (loc.month >= 4) & (loc.month <= 9) & (loc.hour >= 10) & (loc.hour < 16)
    return np.where(pv, frac, 1.0)


def local_day_groups(index: pd.DatetimeIndex) -> list[np.ndarray]:
    loc = pd.Series(index.tz_convert(TZ).date, index=np.arange(len(index)))
    return [g.index.to_numpy() for _, g in loc.groupby(loc.to_numpy(), sort=True)]


def run_dispatch(price_used: pd.Series, duration_h: float, grid_cap_frac: float = 1.0) -> pd.DataFrame:
    """Optimise every local day of ``price_used`` (the prices the optimiser *believes*).

    Returns per-MW charge / discharge power on the same index.
    """
    idx = price_used.index
    pv = price_used.to_numpy(dtype=float)
    cap_all = grid_cap_vector(idx, grid_cap_frac)
    c_out = np.zeros(len(idx))
    d_out = np.zeros(len(idx))
    for pos in local_day_groups(idx):
        p = pv[pos]
        if np.isnan(p).any():
            continue
        cap = None if cap_all is None else cap_all[pos]
        c, d = solve_day(p, duration_h, dis_cap=cap)
        c_out[pos], d_out[pos] = c, d
    return pd.DataFrame({"charge_mw": c_out, "discharge_mw": d_out}, index=idx)


def settle(sched: pd.DataFrame, price_actual: pd.Series, fee: float = FEE_DA_EUR_MWH,
           deg: float = DEGRADATION_EUR_MWH) -> pd.DataFrame:
    """Value a (per-MW) schedule at actual prices. Columns are EUR per interval and MWh per interval."""
    dis = sched["discharge_mw"] * DT_H
    chg = sched["charge_mw"] * DT_H
    out = pd.DataFrame({
        "discharge_mwh": dis,
        "charge_mwh": chg,
        "gross_eur": (dis - chg) * price_actual,
        "fees_eur": (dis + chg) * fee,
        "degradation_eur": dis * deg,
    }, index=sched.index)
    out["net_eur"] = out["gross_eur"] - out["fees_eur"] - out["degradation_eur"]
    return out


def monthly(settled: pd.DataFrame) -> pd.DataFrame:
    m = settled.groupby(settled.index.tz_convert(TZ).tz_localize(None).to_period("M")).sum()
    m.index = m.index.to_timestamp()
    return m
