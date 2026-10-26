"""Optional Layer-2 module: how much of the asset's net trading revenue becomes *company* revenue.

This assumes a revenue-share style business model (one of several possible models, e.g. tolling or route-to-market).
It is kept separate on purpose: replace the terms below with the company's actual contracts before using any number.
Contract terms are evaluated on an annual basis; floors and tiers make company revenue non-linear in market outcomes.
"""
from __future__ import annotations

from .config import (FLOOR_EUR_MW_YR, MIN_FEE_C3_EUR_MONTH, SHARE_C1, SHARE_C2_HIGH, SHARE_C2_LOW, SHARE_C3, SHARE_C4,
                     TIER_THRESHOLD_EUR_MW_YR, Asset)


def company_share(asset: Asset, net_rev: float, months: float = 12.0, share_add: float = 0.0) -> dict:
    """Company revenue, customer payout and floor top-up for ``months`` of net trading revenue (Layer 1)."""
    yrs = months / 12.0
    if months <= 0:
        return {"company": 0.0, "customer": 0.0, "topup": 0.0}
    if asset.contract == "net_fixed":
        rate = (SHARE_C1 if asset.customer == "C1" else SHARE_C4) + share_add
        comp = rate * max(net_rev, 0.0)
        return {"company": comp, "customer": net_rev - comp, "topup": 0.0}
    if asset.contract == "tiered_floor":
        thr = TIER_THRESHOLD_EUR_MW_YR * asset.mw * yrs
        base = max(net_rev, 0.0)
        comp = (SHARE_C2_LOW + share_add) * min(base, thr) + (SHARE_C2_HIGH + share_add) * max(base - thr, 0.0)
        customer = net_rev - comp
        floor = FLOOR_EUR_MW_YR * asset.mw * yrs
        topup = max(0.0, floor - customer)                 # the floor is a put option written by the company
        return {"company": comp - topup, "customer": customer + topup, "topup": topup}
    if asset.contract == "fixed_min_fee":
        comp = max((SHARE_C3 + share_add) * max(net_rev, 0.0), MIN_FEE_C3_EUR_MONTH * months)
        return {"company": comp, "customer": net_rev - comp, "topup": 0.0}
    raise ValueError(asset.contract)
