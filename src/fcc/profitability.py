"""Product- and customer-level profitability for the CFO.

Product = market product (day-ahead, intraday, reserve capacity). Each asset's net trading revenue is split into
product contributions; company revenue (Layer-2 module) is attributed to products in proportion to their contribution.
Joint costs are allocated twice - by revenue and by activity - to show that the key, not the product, often decides
which product "looks" unprofitable. Drop decisions use avoidable costs only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import contracts
from .config import ALLOC_ACTIVITY, ASSETS, COST_TO_SERVE_EUR_MONTH, DIRECT_COSTS_EUR_MONTH, JOINT_COSTS_EUR_MONTH

PRODUCTS = ["Day-ahead", "Intraday", "Reserve capacity"]


def product_lines(pm: pd.DataFrame) -> pd.DataFrame:
    """Layer-1 contribution by asset and product (sums back to net trading revenue)."""
    d = pm.copy()
    w_id = d["traded_id"] / (d["traded_da"] + d["traded_id"])
    rows = []
    for _, r in d.iterrows():
        rows += [
            {"month": r["month"], "asset_id": r["asset_id"], "product": "Day-ahead",
             "revenue": r["gross_da"], "market_costs": r["fees_da"] + r["deg"] + r["imb_cost"] * (1 - w_id[_])},
            {"month": r["month"], "asset_id": r["asset_id"], "product": "Intraday",
             "revenue": r["id_rev"], "market_costs": r["fees_id"] + r["imb_cost"] * w_id[_]},
            {"month": r["month"], "asset_id": r["asset_id"], "product": "Reserve capacity",
             "revenue": r["res_rev"], "market_costs": 0.0},
        ]
    out = pd.DataFrame(rows)
    out["contribution_l1"] = out["revenue"] - out["market_costs"]
    return out


def product_pnl(pm: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Trailing-period product P&L, customer P&L and the attribution detail."""
    w = pm[(pm["month"] >= start) & (pm["month"] <= end)]
    n_m = w["month"].nunique()
    pl = product_lines(w)
    by = pl.groupby(["asset_id", "product"])[["revenue", "market_costs", "contribution_l1"]].sum().reset_index()

    cust_rows, attrib = [], []
    for a in [x for x in ASSETS if x.cod_actual]:
        g = by[by["asset_id"] == a.asset_id]
        if g.empty:
            continue
        net = g["contribution_l1"].sum()
        months_on = w.loc[w["asset_id"] == a.asset_id, "days_online"].sum() / (365.25 / 12)
        cs = contracts.company_share(a, net, months=months_on)
        pos = g["contribution_l1"].clip(lower=0)
        wts = pos / pos.sum() if pos.sum() > 0 else pd.Series(1 / len(g), index=g.index)
        for i, row in g.iterrows():
            attrib.append({"asset_id": a.asset_id, "product": row["product"], "company_revenue": cs["company"] * wts[i],
                           "floor_topup": -cs["topup"] * wts[i]})
        cts = COST_TO_SERVE_EUR_MONTH[a.customer] * months_on
        cust_rows.append({"customer": a.customer, "asset_id": a.asset_id, "type": a.kind, "MW": a.mw,
                          "contract": a.contract, "layer1_net": net, "company_revenue": cs["company"],
                          "of_which_floor_topup": -cs["topup"], "cost_to_serve": cts,
                          "contribution_after_cts": cs["company"] - cts})
    at = pd.DataFrame(attrib)
    prod = by.groupby("product")[["revenue", "market_costs", "contribution_l1"]].sum()
    prod = prod.join(at.groupby("product")[["company_revenue", "floor_topup"]].sum())
    prod["direct_costs"] = [DIRECT_COSTS_EUR_MONTH[p] * n_m for p in prod.index]
    prod["CM1"] = prod["company_revenue"] - prod["direct_costs"]
    joint_total = sum(JOINT_COSTS_EUR_MONTH.values()) * n_m
    rev_key = prod["company_revenue"].clip(lower=0) / prod["company_revenue"].clip(lower=0).sum()
    prod["joint_by_revenue"] = rev_key * joint_total
    act = pd.Series(0.0, index=prod.index)
    for pool, keys in ALLOC_ACTIVITY.items():
        for p, s in keys.items():
            act[p] += JOINT_COSTS_EUR_MONTH[pool] * n_m * s
    prod["joint_by_activity"] = act
    prod["net_by_revenue_key"] = prod["CM1"] - prod["joint_by_revenue"]
    prod["net_by_activity_key"] = prod["CM1"] - prod["joint_by_activity"]
    prod = prod.reindex(PRODUCTS)
    prod.attrs["months"] = n_m
    prod.attrs["joint_total"] = joint_total
    cust = pd.DataFrame(cust_rows)
    cust["company_rev_per_MW"] = cust["company_revenue"] / cust["MW"]
    cust["cts_per_MW"] = cust["cost_to_serve"] / cust["MW"]
    return prod, cust, at


def drop_test(prod: pd.DataFrame, product: str, key: str = "net_by_activity_key") -> pd.DataFrame:
    """'Death spiral' check: what happens if the product that looks loss-making is dropped.

    Joint costs do not disappear, they move to the remaining products. The company result changes only by the
    dropped product's avoidable result (company revenue - direct costs = CM1)."""
    joint_col = "joint_by_activity" if key == "net_by_activity_key" else "joint_by_revenue"
    keep = prod.drop(index=product).copy()
    share = keep[joint_col] / keep[joint_col].sum()
    keep["joint_after_drop"] = share * prod.attrs["joint_total"]
    keep["net_after_drop"] = keep["CM1"] - keep["joint_after_drop"]
    before = prod["CM1"].sum() - prod.attrs["joint_total"]
    after = keep["CM1"].sum() - prod.attrs["joint_total"]
    out = pd.DataFrame({"net before drop": prod[key].drop(index=product), "net after drop": keep["net_after_drop"]})
    out.attrs["company_result_before"] = before
    out.attrs["company_result_after"] = after
    out.attrs["dropped_cm1"] = float(prod.loc[product, "CM1"])
    return out
