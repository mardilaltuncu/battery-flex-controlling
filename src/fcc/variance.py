"""Price-volume-mix variance bridge on a MWh basis (budget / forecast -> actual net trading revenue).

Every revenue and cost line is generated from a small set of *drivers*, so the bridge is exact by construction and the
control ``sum of effects == actual - budget`` can be asserted. Effects are computed in a fixed order (volume, mix,
price; within the arbitrage price effect: cycles, market spread, capture rate) and the order is stated in the output.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import MATERIALITY_EUR, MATERIALITY_PCT


@dataclass
class Drivers:
    """Unit economics of one asset in one month (all money in EUR)."""
    mw: float                 # MW online (month average)
    res_share: float          # share of MW committed to reserve capacity
    q: float                  # MWh discharged per arbitrage-MW in the month
    theo: float               # theoretical (perfect-foresight) gross spread, EUR per MWh discharged
    cap: float                # capture rate = realised spread / theoretical spread
    res_price: float          # reserve capacity revenue, EUR per reserved MW in the month
    traded_per_dis: float     # traded MWh (charge + discharge, DA + intraday) per MWh discharged
    fee_rate: float           # blended exchange fee, EUR per traded MWh
    deg_rate: float           # degradation cost, EUR per MWh discharged
    imb_vol_per_dis: float    # absolute imbalance MWh per MWh discharged
    imb_rate: float           # imbalance cost, EUR per imbalance MWh

    # ---- derived quantities
    @property
    def mw_arb(self) -> float:
        return self.mw * (1 - self.res_share)

    @property
    def mw_res(self) -> float:
        return self.mw * self.res_share

    @property
    def dis_mwh(self) -> float:
        return self.mw_arb * self.q

    @property
    def arb_rev(self) -> float:
        return self.dis_mwh * self.theo * self.cap

    @property
    def res_rev(self) -> float:
        return self.mw_res * self.res_price

    @property
    def traded_mwh(self) -> float:
        return self.dis_mwh * self.traded_per_dis

    @property
    def fees(self) -> float:
        return self.traded_mwh * self.fee_rate

    @property
    def deg(self) -> float:
        return self.dis_mwh * self.deg_rate

    @property
    def imb_mwh(self) -> float:
        return self.dis_mwh * self.imb_vol_per_dis

    @property
    def imb_cost(self) -> float:
        return self.imb_mwh * self.imb_rate

    @property
    def gross_rev(self) -> float:
        return self.arb_rev + self.res_rev

    @property
    def net_rev(self) -> float:
        return self.gross_rev - self.fees - self.deg - self.imb_cost

    def lines(self) -> dict:
        return {"Arbitrage (DA + intraday)": self.arb_rev, "Reserve capacity": self.res_rev, "Exchange fees": -self.fees,
                "Degradation": -self.deg, "Imbalance": -self.imb_cost, "Net trading revenue": self.net_rev}


def drivers_from_actuals(mw: float, res_share: float, dis_mwh: float, arb_rev: float, pf_gross: float, pf_dis: float,
                         res_rev: float, traded_mwh: float, fees: float, deg: float, imb_mwh: float,
                         imb_cost: float) -> Drivers:
    """Back out drivers from realised monthly sums."""
    mw_arb, mw_res = mw * (1 - res_share), mw * res_share
    safe = lambda a, b: a / b if abs(b) > 1e-12 else 0.0
    theo = safe(pf_gross, pf_dis)
    s = safe(arb_rev, dis_mwh)
    return Drivers(mw=mw, res_share=res_share, q=safe(dis_mwh, mw_arb), theo=theo, cap=safe(s, theo),
                   res_price=safe(res_rev, mw_res), traded_per_dis=safe(traded_mwh, dis_mwh),
                   fee_rate=safe(fees, traded_mwh), deg_rate=safe(deg, dis_mwh),
                   imb_vol_per_dis=safe(imb_mwh, dis_mwh), imb_rate=safe(imb_cost, imb_mwh))


ORDER_NOTE = ("Order of effects: volume (MW online) -> mix (arbitrage vs reserve) -> price; arbitrage price effect is split "
              "into cycles -> market spread -> capture rate; costs: volume then rate. Same order every month.")


def bridge(b: Drivers, a: Drivers) -> pd.DataFrame:
    """Budget -> actual net trading revenue. Returns steps with the EUR effect and a controllability tag."""
    mb_arb, mb_res = 1 - b.res_share, b.res_share
    ma_arb, ma_res = 1 - a.res_share, a.res_share
    p_arb_b = b.q * b.theo * b.cap                      # EUR per arbitrage-MW (budget)
    p_arb_a = a.q * a.theo * a.cap
    unit_b = mb_arb * p_arb_b + mb_res * b.res_price    # budget gross revenue per MW

    vol = (a.mw - b.mw) * unit_b
    mix = a.mw * ((ma_arb - mb_arb) * p_arb_b + (ma_res - mb_res) * b.res_price)
    w_arb = a.mw * ma_arb
    cycles = w_arb * (a.q - b.q) * b.theo * b.cap
    market = w_arb * a.q * (a.theo - b.theo) * b.cap
    capture = w_arb * a.q * a.theo * (a.cap - b.cap)
    res_price = a.mw * ma_res * (a.res_price - b.res_price)

    fee_vol = -b.fee_rate * (a.traded_mwh - b.traded_mwh)
    fee_tar = -a.traded_mwh * (a.fee_rate - b.fee_rate)
    deg_vol = -b.deg_rate * (a.dis_mwh - b.dis_mwh)
    deg_rate = -a.dis_mwh * (a.deg_rate - b.deg_rate)
    imb_vol = -b.imb_rate * (a.imb_mwh - b.imb_mwh)
    imb_rate = -a.imb_mwh * (a.imb_rate - b.imb_rate)

    steps = [
        ("Budget net trading revenue", b.net_rev, "start"),
        ("Volume: MW online (COD timing)", vol, "timing"),
        ("Mix: arbitrage vs reserve", mix, "controllable"),
        ("Cycles: MWh per MW", cycles, "controllable"),
        ("Market spread (theoretical)", market, "market"),
        ("Capture rate (algorithm)", capture, "controllable"),
        ("Reserve capacity price", res_price, "market"),
        ("Fees: activity", fee_vol, "driven by volume"),
        ("Fees: tariff", fee_tar, "external"),
        ("Degradation: volume", deg_vol, "driven by volume"),
        ("Degradation: cost rate", deg_rate, "assumption"),
        ("Imbalance: volume", imb_vol, "controllable"),
        ("Imbalance: price", imb_rate, "market"),
    ]
    out = pd.DataFrame(steps, columns=["step", "eur", "nature"])
    total = out.loc[1:, "eur"].sum()
    out.loc[len(out)] = ["Actual net trading revenue", a.net_rev, "end"]
    out.attrs["check_diff"] = float(b.net_rev + total - a.net_rev)
    out.attrs["order_note"] = ORDER_NOTE
    return out


def flag_material(df: pd.DataFrame, budget_net: float) -> pd.DataFrame:
    """Explain-required flag: both the % and the EUR threshold must be exceeded."""
    out = df.copy()
    mid = ~out["step"].isin(["Budget net trading revenue", "Actual net trading revenue"])
    pct = out["eur"].abs() / max(abs(budget_net), 1.0)
    out["explain"] = mid & (pct >= MATERIALITY_PCT) & (out["eur"].abs() >= MATERIALITY_EUR)
    return out


def sum_bridges(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Portfolio bridge = sum of asset bridges (effects add up because each bridge is exact)."""
    base = parts[0][["step", "nature"]].copy()
    base["eur"] = sum(p["eur"].to_numpy() for p in parts)
    base.attrs["check_diff"] = float(sum(p.attrs["check_diff"] for p in parts))
    base.attrs["order_note"] = ORDER_NOTE
    return base


def drivers_frame(d: Drivers) -> pd.Series:
    return pd.Series(asdict(d))
