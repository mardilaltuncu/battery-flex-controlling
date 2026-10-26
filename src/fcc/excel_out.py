"""Excel deliverable with live formulas and a check sheet.

Design rules (from the controlling primer): inputs, calculations and outputs on separate sheets; input cells coloured;
no hard-coded numbers inside formulas (every assumption is an input cell); one check sheet with a master flag; a change
log. Python writes the data and the formulas; Excel recalculates. Cached values are written alongside so the file
also reads correctly in viewers that do not recalculate.
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import xlsxwriter
from xlsxwriter.utility import xl_col_to_name, xl_rowcol_to_cell as rc

from .variance import Drivers

DRV = ["mw", "res_share", "q", "theo", "cap", "res_price", "traded_per_dis", "fee_rate", "deg_rate",
       "imb_vol_per_dis", "imb_rate"]


class _Fmt:
    def __init__(self, wb: xlsxwriter.Workbook):
        base = {"font_name": "Segoe UI", "font_size": 10}
        self.title = wb.add_format({**base, "bold": True, "font_size": 14})
        self.sub = wb.add_format({**base, "italic": True, "font_color": "#52514e"})
        self.h = wb.add_format({**base, "bold": True, "bottom": 1, "bg_color": "#f0efec", "text_wrap": True, "valign": "top"})
        self.txt = wb.add_format(base)
        self.wrap = wb.add_format({**base, "text_wrap": True, "valign": "top"})
        self.inp = wb.add_format({**base, "font_color": "#1c5cab", "bg_color": "#fff6d6", "num_format": "#,##0.00##"})
        self.inp_pct = wb.add_format({**base, "font_color": "#1c5cab", "bg_color": "#fff6d6", "num_format": "0.0%"})
        self.inp_txt = wb.add_format({**base, "font_color": "#1c5cab", "bg_color": "#fff6d6"})
        self.eur = wb.add_format({**base, "num_format": "#,##0;[Red]-#,##0"})
        self.eur2 = wb.add_format({**base, "num_format": "#,##0.00;[Red]-#,##0.00"})
        self.num = wb.add_format({**base, "num_format": "#,##0.000"})
        self.pct = wb.add_format({**base, "num_format": "0.0%"})
        self.date = wb.add_format({**base, "num_format": "yyyy-mm"})
        self.day = wb.add_format({**base, "num_format": "yyyy-mm-dd"})
        self.tot = wb.add_format({**base, "bold": True, "top": 1, "num_format": "#,##0;[Red]-#,##0"})
        self.ok = wb.add_format({**base, "bold": True, "font_color": "#006300"})
        self.err = wb.add_format({**base, "bold": True, "font_color": "#d03b3b", "bg_color": "#fbe3e3"})
        self.flag = wb.add_format({**base, "bg_color": "#fbe3e3"})


def _cf_okerr(ws, rng: str, f: _Fmt) -> None:
    ws.conditional_format(rng, {"type": "cell", "criteria": "==", "value": '"OK"', "format": f.ok})
    ws.conditional_format(rng, {"type": "cell", "criteria": "==", "value": '"ERROR"', "format": f.err})


def _table(ws, r0: int, c0: int, df: pd.DataFrame, f: _Fmt, fmts: dict | None = None, header_fmt=None) -> tuple[int, int]:
    fmts = fmts or {}
    for j, col in enumerate(df.columns):
        ws.write(r0, c0 + j, str(col), header_fmt or f.h)
    for i, row in enumerate(df.itertuples(index=False)):
        for j, v in enumerate(row):
            col = df.columns[j]
            fm = fmts.get(col)
            if isinstance(v, (pd.Timestamp, date)) and not isinstance(v, str):
                ws.write_datetime(r0 + 1 + i, c0 + j, pd.Timestamp(v).to_pydatetime(), fm or f.day)
            elif isinstance(v, (float, np.floating)) and (np.isnan(v) or np.isinf(v)):
                ws.write_blank(r0 + 1 + i, c0 + j, None)
            elif isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool):
                ws.write_number(r0 + 1 + i, c0 + j, float(v), fm or f.txt)
            else:
                ws.write(r0 + 1 + i, c0 + j, "" if v is None else str(v), fm or f.txt)
    return r0 + 1, r0 + len(df)          # first and last data row (0-based)


# --------------------------------------------------------------------------------------------------
def write_workbook(path: str, R: dict) -> dict:
    """Write the workbook. ``R`` holds the computed results (see the notebook for the keys)."""
    wb = xlsxwriter.Workbook(path)
    f = _Fmt(wb)
    checks: list[tuple[str, str, str]] = []          # (description, formula returning TRUE/FALSE, sheet)
    ws_checks = wb.add_worksheet("Checks")           # first tab; filled at the end once all checks are known

    # ------------------------------------------------------------------------------------------ README
    ws = wb.add_worksheet("README")
    ws.set_column("A:A", 120)
    ws.write("A1", "Flex-optimizer controlling case - Excel deliverable", f.title)
    lines = [
        "Purpose: interval reconciliation, variance bridge, rolling-forecast scenarios and product profitability for an",
        "illustrative battery portfolio. Day-ahead prices are real (Energy-Charts / SMARD, CC BY 4.0); assets, contracts,",
        "ledger, exchange statements, schedules, meters, reserve and imbalance prices are synthetic and labelled as such.",
        "",
        "Sheets: Inputs (all assumptions, blue cells = editable) | Monthly_Actuals | Bridge (live PVM formulas) |",
        "Interval_Count (DST-aware day structure) | Recon_Daily | Break_Register | Scenarios | Product_PnL (allocation",
        "key switch) | Checks (master flag) | Change_Log.",
        "",
        "Rule: do not publish any number while Checks!B2 shows ERROR.",
        "Not in scope: accounting treatment, tax, project finance, margining.",
    ]
    for i, t in enumerate(lines):
        ws.write(2 + i, 0, t, f.txt)

    # ------------------------------------------------------------------------------------------ Inputs
    ws = wb.add_worksheet("Inputs")
    ws.write("A1", "Inputs and assumptions (provenance per row)", f.title)
    ws.write("A2", "Blue cells are inputs. 'illustrative assumption' rows must be replaced with company figures.", f.sub)
    a = R["assumptions"].copy()
    a["value"] = [v if isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool) else str(v)
                  for v in a["value"]]
    ws.set_column("A:A", 44)
    ws.set_column("B:B", 18)
    ws.set_column("C:D", 22)
    ws.set_column("E:E", 90)
    r1, r2 = _table(ws, 3, 0, a[["parameter", "value", "unit", "source", "note"]], f, {"value": f.inp})
    names = {}
    for i, p in enumerate(a["parameter"]):
        names[p] = f"Inputs!$B${r1 + i + 1}"
    for nm, p in [("rte", "round_trip_efficiency"), ("mat_pct", "variance_materiality_pct"),
                  ("mat_eur", "variance_materiality_eur")]:
        wb.define_name(nm, "=" + names[p])
    checks.append(("Round-trip efficiency is plausible (0.80-0.95)", "AND(rte>0.8,rte<0.95)", "Inputs"))

    # ------------------------------------------------------------------------------------------ Monthly_Actuals
    ws = wb.add_worksheet("Monthly_Actuals")
    ws.write("A1", "Monthly actuals per asset (Layer 1: asset net trading revenue, EUR)", f.title)
    pm = R["pm"].copy()
    cols = ["month", "asset_id", "mw", "dis_mwh", "traded_mwh", "arb_rev", "res_rev", "fees", "deg", "imb_cost"]
    df = pm[cols].copy()
    ws.set_column("A:A", 10)
    ws.set_column("B:B", 9)
    ws.set_column("C:M", 14)
    r1, r2 = _table(ws, 2, 0, df, f, {"month": f.date, "mw": f.num, "dis_mwh": f.eur, "traded_mwh": f.eur,
                                      "arb_rev": f.eur, "res_rev": f.eur, "fees": f.eur, "deg": f.eur, "imb_cost": f.eur})
    ws.write(2, 10, "net_rev (formula)", f.h)
    ws.write(2, 11, "net_rev (python)", f.h)
    ws.write(2, 12, "match", f.h)
    for i in range(r1, r2 + 1):
        x = i + 1
        ws.write_formula(i, 10, f"=F{x}+G{x}-H{x}-I{x}-J{x}", f.eur, float(pm["net_rev"].iloc[i - r1]))
        ws.write_number(i, 11, float(pm["net_rev"].iloc[i - r1]), f.eur)
        ws.write_formula(i, 12, f"=ABS(K{x}-L{x})<0.01", f.txt, True)
    tr = r2 + 1
    ws.write(tr, 0, "Total", f.tot)
    for c in range(3, 12):
        L = xl_col_to_name(c)
        ws.write_formula(tr, c, f"=SUM({L}{r1 + 1}:{L}{r2 + 1})", f.tot)
    ma = f"Monthly_Actuals!$M${r1 + 1}:$M${r2 + 1}"
    checks.append(("Monthly net revenue = sum of its lines (every row)", f"COUNTIF({ma},FALSE)=0", "Monthly_Actuals"))
    checks.append(("Cost lines are entered as positive amounts (sign convention)",
                   f"COUNTIF(Monthly_Actuals!$H${r1 + 1}:$I${r2 + 1},\"<0\")=0", "Monthly_Actuals"))
    ma_month = f"Monthly_Actuals!$A${r1 + 1}:$A${r2 + 1}"
    ma_net = f"Monthly_Actuals!$K${r1 + 1}:$K${r2 + 1}"

    # ------------------------------------------------------------------------------------------ Bridge
    ws = wb.add_worksheet("Bridge")
    ws.write("A1", f"Variance bridge {R['bridge_label']} - live formulas", f.title)
    ws.write("A2", R["bridge_order_note"], f.sub)
    ws.set_column("A:A", 34)
    ws.set_column("B:N", 13)
    assets = list(R["bridge_parts"].keys())
    blocks = {}
    row = 3
    for tag, idx in [("Budget drivers", 0), ("Actual drivers", 1)]:
        ws.write(row, 0, tag, f.h)
        for j, d in enumerate(DRV):
            ws.write(row, 1 + j, d, f.h)
        ws.write(row, 12, "net revenue", f.h)
        for k, aid in enumerate(assets):
            r = row + 1 + k
            d: Drivers = R["bridge_parts"][aid][idx]
            ws.write(r, 0, aid, f.txt)
            for j, name in enumerate(DRV):
                ws.write_number(r, 1 + j, float(getattr(d, name)), f.inp if idx == 0 else f.num)
            x = r + 1
            ws.write_formula(r, 12, f"=B{x}*(1-C{x})*D{x}*(E{x}*F{x}-H{x}*I{x}-J{x}-K{x}*L{x})+B{x}*C{x}*G{x}", f.eur,
                             float(d.net_rev))
        blocks[tag] = row + 1
        row += len(assets) + 3

    steps = ["Volume: MW online (COD timing)", "Mix: arbitrage vs reserve", "Cycles: MWh per MW",
             "Market spread (theoretical)", "Capture rate (algorithm)", "Reserve capacity price", "Fees: activity",
             "Fees: tariff", "Degradation: volume", "Degradation: cost rate", "Imbalance: volume", "Imbalance: price"]
    ws.write(row, 0, "Effect (EUR)", f.h)
    for k, aid in enumerate(assets):
        ws.write(row, 1 + k, aid, f.h)
    pcol = 1 + len(assets)
    ws.write(row, pcol, "Portfolio", f.h)
    ws.write(row, pcol + 1, "explain?", f.h)
    eff0 = row + 1

    def cell(tag, k, drv):
        return f"{xl_col_to_name(1 + DRV.index(drv))}{blocks[tag] + k + 1}"

    for s_i, s in enumerate(steps):
        r = eff0 + s_i
        ws.write(r, 0, s, f.txt)
        for k, aid in enumerate(assets):
            B = lambda d: cell("Budget drivers", k, d)
            A = lambda d: cell("Actual drivers", k, d)
            arb_b = f"(1-{B('res_share')})*{B('q')}*{B('theo')}*{B('cap')}"
            dis_a = f"{A('mw')}*(1-{A('res_share')})*{A('q')}"
            dis_b = f"{B('mw')}*(1-{B('res_share')})*{B('q')}"
            formula = {
                0: f"=({A('mw')}-{B('mw')})*({arb_b}+{B('res_share')}*{B('res_price')})",
                1: f"={A('mw')}*(((1-{A('res_share')})-(1-{B('res_share')}))*{B('q')}*{B('theo')}*{B('cap')}"
                   f"+({A('res_share')}-{B('res_share')})*{B('res_price')})",
                2: f"={A('mw')}*(1-{A('res_share')})*({A('q')}-{B('q')})*{B('theo')}*{B('cap')}",
                3: f"={A('mw')}*(1-{A('res_share')})*{A('q')}*({A('theo')}-{B('theo')})*{B('cap')}",
                4: f"={A('mw')}*(1-{A('res_share')})*{A('q')}*{A('theo')}*({A('cap')}-{B('cap')})",
                5: f"={A('mw')}*{A('res_share')}*({A('res_price')}-{B('res_price')})",
                6: f"=-{B('fee_rate')}*({dis_a}*{A('traded_per_dis')}-{dis_b}*{B('traded_per_dis')})",
                7: f"=-{dis_a}*{A('traded_per_dis')}*({A('fee_rate')}-{B('fee_rate')})",
                8: f"=-{B('deg_rate')}*({dis_a}-{dis_b})",
                9: f"=-{dis_a}*({A('deg_rate')}-{B('deg_rate')})",
                10: f"=-{B('imb_rate')}*({dis_a}*{A('imb_vol_per_dis')}-{dis_b}*{B('imb_vol_per_dis')})",
                11: f"=-{dis_a}*{A('imb_vol_per_dis')}*({A('imb_rate')}-{B('imb_rate')})",
            }[s_i]
            val = float(R["bridge_steps"][aid].iloc[s_i + 1]["eur"])
            ws.write_formula(r, 1 + k, formula, f.eur, val)
        L0, L1 = xl_col_to_name(1), xl_col_to_name(len(assets))
        ws.write_formula(r, pcol, f"=SUM({L0}{r + 1}:{L1}{r + 1})", f.eur)
        P = xl_col_to_name(pcol)
        bud_tot = f"SUM(M{blocks['Budget drivers'] + 1}:M{blocks['Budget drivers'] + len(assets)})"
        ws.write_formula(r, pcol + 1, f"=AND(ABS({P}{r + 1})>=mat_eur,ABS({P}{r + 1})>=mat_pct*ABS({bud_tot}))", f.txt)
    ws.conditional_format(eff0, pcol + 1, eff0 + len(steps) - 1, pcol + 1,
                          {"type": "cell", "criteria": "==", "value": "TRUE", "format": f.flag})
    r = eff0 + len(steps) + 1
    bb, ab = blocks["Budget drivers"], blocks["Actual drivers"]
    P = xl_col_to_name(pcol)
    ws.write(r, 0, "Budget net (portfolio)", f.tot)
    ws.write_formula(r, pcol, f"=SUM(M{bb + 1}:M{bb + len(assets)})", f.tot)
    ws.write(r + 1, 0, "+ sum of effects", f.txt)
    ws.write_formula(r + 1, pcol, f"=SUM({P}{eff0 + 1}:{P}{eff0 + len(steps)})", f.eur)
    ws.write(r + 2, 0, "Actual net (portfolio)", f.tot)
    ws.write_formula(r + 2, pcol, f"=SUM(M{ab + 1}:M{ab + len(assets)})", f.tot)
    ws.write(r + 3, 0, "Check: budget + effects - actual", f.txt)
    ws.write_formula(r + 3, pcol, f"={P}{r + 1}+{P}{r + 2}-{P}{r + 3}", f.eur2)
    checks.append(("Bridge: budget + effects = actual (no unexplained residual)", f"ABS(Bridge!{P}{r + 4})<0.01", "Bridge"))
    L0, L1 = xl_col_to_name(1), xl_col_to_name(len(assets))
    checks.append(("Bridge: asset columns cross-foot to the portfolio column",
                   f"ABS(SUM(Bridge!{L0}{eff0 + 1}:{L1}{eff0 + len(steps)})-SUM(Bridge!{P}{eff0 + 1}:{P}{eff0 + len(steps)}))<0.01",
                   "Bridge"))

    # ------------------------------------------------------------------------------------------ Interval_Count
    ws = wb.add_worksheet("Interval_Count")
    ws.write("A1", "Intervals per local delivery day: only DST days may differ from 96 (92 in March, 100 in October)", f.title)
    ic = R["interval_check"].reset_index()[["local_date", "n_expected", "n_found"]]
    ws.set_column("A:D", 14)
    r1, r2 = _table(ws, 2, 0, ic, f, {"local_date": f.day})
    ws.write(2, 3, "gap", f.h)
    for i in range(r1, r2 + 1):
        ws.write_formula(i, 3, f"=C{i + 1}-B{i + 1}", f.txt, 0)
    rng = f"Interval_Count!$D${r1 + 1}:$D${r2 + 1}"
    checks.append(("Price series complete on every local day incl. DST days", f"COUNTIF({rng},\"<>0\")=0", "Interval_Count"))
    exp = f"Interval_Count!$B${r1 + 1}:$B${r2 + 1}"
    fnd = f"Interval_Count!$C${r1 + 1}:$C${r2 + 1}"
    checks.append(("Intervals in the price file = intervals implied by the calendar (cross-foot)",
                   f"SUM({fnd})=SUM({exp})", "Interval_Count"))

    # ------------------------------------------------------------------------------------------ Recon_Daily
    ws = wb.add_worksheet("Recon_Daily")
    ws.write("A1", "Ledger vs exchange statement per asset and local day (EUR)", f.title)
    rd = R["recon_daily"]
    ws.set_column("A:H", 15)
    r1, r2 = _table(ws, 2, 0, rd, f, {"ledger_mwh": f.num, "ecc_mwh": f.num, "ledger_eur": f.eur2, "ecc_eur": f.eur2})
    dcol = len(rd.columns)
    ws.write(2, dcol, "diff_eur", f.h)
    ci = {c: xl_col_to_name(j) for j, c in enumerate(rd.columns)}
    for i in range(r1, r2 + 1):
        ws.write_formula(i, dcol, f"={ci['ecc_eur']}{i + 1}-{ci['ledger_eur']}{i + 1}", f.eur2,
                         float(rd["ecc_eur"].iloc[i - r1] - rd["ledger_eur"].iloc[i - r1]))
    D = xl_col_to_name(dcol)
    tr = r2 + 1
    ws.write(tr, 0, "Total", f.tot)
    for c in ["ledger_eur", "ecc_eur"]:
        ws.write_formula(tr, rd.columns.get_loc(c), f"=SUM({ci[c]}{r1 + 1}:{ci[c]}{r2 + 1})", f.tot)
    ws.write_formula(tr, dcol, f"=SUM({D}{r1 + 1}:{D}{r2 + 1})", f.tot)
    checks.append(("Recon: daily differences cross-foot to total statement - total ledger",
                   f"ABS(Recon_Daily!{D}{tr + 1}-(Recon_Daily!{ci['ecc_eur']}{tr + 1}-Recon_Daily!{ci['ledger_eur']}{tr + 1}))<0.01",
                   "Recon_Daily"))
    ws.autofilter(2, 0, r2, dcol)

    # ------------------------------------------------------------------------------------------ Break_Register
    ws = wb.add_worksheet("Break_Register")
    ws.write("A1", "Break register (two-way core + four-way extension) - status column is an input", f.title)
    br = R["register"]
    ws.set_column("A:A", 11)
    ws.set_column("B:F", 13)
    ws.set_column("G:H", 13)
    ws.set_column("I:L", 26)
    r1, r2 = _table(ws, 2, 0, br, f, {"eur_impact": f.eur, "mwh_diff": f.num, "status": f.inp_txt})
    sc = br.columns.get_loc("status")
    ws.data_validation(r1, sc, r2, sc, {"validate": "list", "source": ["open", "in progress", "resolved", "accepted"]})
    ws.autofilter(2, 0, r2, len(br.columns) - 1)
    ws.freeze_panes(3, 0)

    # ------------------------------------------------------------------------------------------ Scenarios
    ws = wb.add_worksheet("Scenarios")
    ws.write("A1", "Rolling forecast accuracy, scenarios, tornado and Monte Carlo (12 months forward)", f.title)
    ws.set_column("A:A", 40)
    ws.set_column("B:H", 15)
    r = 2
    for title, df_, fm in [("Backtest: accuracy by horizon (portfolio, monthly)", R["accuracy"].reset_index(), None),
                           ("Scenario sets (EUR, 12 months)", R["scenarios"].reset_index(names="scenario"), f.eur),
                           ("Tornado - Layer 1 net trading revenue (change vs base, EUR)", R["tornado_l1"], f.eur),
                           ("Tornado - company EBITDA (change vs base, EUR)", R["tornado_co"], f.eur),
                           ("Monte Carlo percentiles (EUR)", R["mc_pct"].reset_index(names="metric"), f.eur)]:
        ws.write(r, 0, title, f.sub)
        fmts = {c: fm for c in df_.columns if fm is not None and df_[c].dtype.kind in "fi"}
        _, last = _table(ws, r + 1, 0, df_, f, fmts)
        r = last + 3

    # ------------------------------------------------------------------------------------------ Product_PnL
    ws = wb.add_worksheet("Product_PnL")
    ws.write("A1", f"Product profitability, {R['pnl_window']} (company view, EUR) - change the key in B3", f.title)
    ws.write("A3", "Allocation key", f.h)
    ws.write("B3", "Activity", f.inp_txt)
    ws.data_validation("B3", {"validate": "list", "source": ["Activity", "Revenue"]})
    ws.write("A4", "Months in window", f.h)
    ws.write_number("B4", R["pnl_months"], f.inp)
    ws.set_column("A:A", 30)
    ws.set_column("B:K", 15)
    prods = list(R["prod"].index)
    pools = list(R["joint_costs"].keys())
    # joint cost pools and activity keys (inputs)
    r0 = 6
    ws.write(r0, 0, "Joint cost pool (EUR / month)", f.h)
    ws.write(r0, 1, "monthly cost", f.h)
    for j, p in enumerate(prods):
        ws.write(r0, 2 + j, f"key: {p}", f.h)
    ws.write(r0, 2 + len(prods), "key sum", f.h)
    for i, pool in enumerate(pools):
        r = r0 + 1 + i
        ws.write(r, 0, pool, f.txt)
        ws.write_number(r, 1, R["joint_costs"][pool], f.inp)
        for j, p in enumerate(prods):
            ws.write_number(r, 2 + j, R["alloc_activity"][pool][p], f.inp_pct)
        c0, c1 = xl_col_to_name(2), xl_col_to_name(1 + len(prods))
        ws.write_formula(r, 2 + len(prods), f"=SUM({c0}{r + 1}:{c1}{r + 1})", f.pct, 1.0)
    pool_first, pool_last = r0 + 2, r0 + 1 + len(pools)
    ks = xl_col_to_name(2 + len(prods))
    checks.append(("Activity keys sum to 100 % for every cost pool",
                   f"SUMPRODUCT(--(ABS(Product_PnL!{ks}{pool_first}:{ks}{pool_last}-1)<0.0001))={len(pools)}", "Product_PnL"))
    ws.write(pool_last, 0, "Joint costs in window", f.tot)
    ws.write_formula(pool_last, 1, f"=SUM(B{pool_first}:B{pool_last})*$B$4", f.tot)
    joint_tot = f"$B${pool_last + 1}"

    r0 = pool_last + 2
    hdr = ["product", "L1 contribution", "company revenue", "direct costs / month", "direct costs", "CM1",
           "joint (revenue key)", "joint (activity key)", "joint (selected)", "net contribution"]
    for j, h in enumerate(hdr):
        ws.write(r0, j, h, f.h)
    P = R["prod"]
    for i, p in enumerate(prods):
        r = r0 + 1 + i
        x = r + 1
        ws.write(r, 0, p, f.txt)
        ws.write_number(r, 1, float(P.loc[p, "contribution_l1"]), f.eur)
        ws.write_number(r, 2, float(P.loc[p, "company_revenue"]), f.eur)
        ws.write_number(r, 3, float(R["direct_costs"][p]), f.inp)
        ws.write_formula(r, 4, f"=D{x}*$B$4", f.eur)
        ws.write_formula(r, 5, f"=C{x}-E{x}", f.eur)
        first, last = r0 + 2, r0 + 1 + len(prods)
        ws.write_formula(r, 6, f"=MAX(C{x},0)/SUMPRODUCT((C${first}:C${last}>0)*C${first}:C${last})*{joint_tot}", f.eur)
        kc = xl_col_to_name(2 + i)
        ws.write_formula(r, 7, f"=SUMPRODUCT($B${pool_first}:$B${pool_last},{kc}${pool_first}:{kc}${pool_last})*$B$4", f.eur)
        ws.write_formula(r, 8, f"=IF($B$3=\"Revenue\",G{x},H{x})", f.eur)
        ws.write_formula(r, 9, f"=F{x}-I{x}", f.eur)
    first, last = r0 + 2, r0 + 1 + len(prods)
    tr = last
    ws.write(tr, 0, "Total", f.tot)
    for c in range(1, 10):
        if c == 3:
            continue
        L = xl_col_to_name(c)
        ws.write_formula(tr, c, f"=SUM({L}{first}:{L}{last})", f.tot)
    ws.conditional_format(first - 1, 9, last - 1, 9, {"type": "cell", "criteria": "<", "value": 0, "format": f.flag})
    checks.append(("Joint costs fully allocated - revenue key", f"ABS(Product_PnL!G{tr + 1}-Product_PnL!{joint_tot})<0.01", "Product_PnL"))
    checks.append(("Joint costs fully allocated - activity key", f"ABS(Product_PnL!H{tr + 1}-Product_PnL!{joint_tot})<0.01", "Product_PnL"))
    s, e = R["pnl_start"], R["pnl_end"]
    checks.append(("Product L1 contributions = portfolio net revenue in the window (Monthly_Actuals)",
                   f"ABS(Product_PnL!B{tr + 1}-SUMIFS({ma_net},{ma_month},\">=\"&DATE({s.year},{s.month},1),"
                   f"{ma_month},\"<=\"&DATE({e.year},{e.month},1)))<1", "Product_PnL"))
    # customer table
    r0 = tr + 3
    cust = R["cust"][["customer", "asset_id", "type", "MW", "contract", "company_revenue", "of_which_floor_topup",
                      "cost_to_serve"]]
    _, last_c = _table(ws, r0, 0, cust, f, {"company_revenue": f.eur, "of_which_floor_topup": f.eur, "cost_to_serve": f.eur})
    ws.write(r0, 8, "after cost to serve", f.h)
    ws.write(r0, 9, "cost to serve / MW", f.h)
    for i in range(r0 + 1, last_c + 1):
        ws.write_formula(i, 8, f"=F{i + 1}-H{i + 1}", f.eur)
        ws.write_formula(i, 9, f"=H{i + 1}/D{i + 1}", f.eur)

    # ------------------------------------------------------------------------------------------ Change_Log
    ws = wb.add_worksheet("Change_Log")
    ws.write("A1", "Change log", f.title)
    ws.set_column("A:A", 12)
    ws.set_column("B:B", 12)
    ws.set_column("C:C", 100)
    _table(ws, 2, 0, R["change_log"], f, {"date": f.day})

    # ------------------------------------------------------------------------------------------ Checks
    ws = ws_checks
    ws.activate()
    ws.set_column("A:A", 80)
    ws.set_column("B:C", 16)
    ws.write("A1", "Check sheet", f.title)
    ws.write("A2", "MASTER FLAG (publish only if OK)", f.h)
    ws.write("A4", "check", f.h)
    ws.write("B4", "result", f.h)
    ws.write("C4", "sheet", f.h)
    for i, (desc, formula, sheet) in enumerate(checks):
        r = 4 + i
        ws.write(r, 0, f"{i + 1}. {desc}", f.wrap)
        ws.write_formula(r, 1, f"=IF({formula},\"OK\",\"ERROR\")", f.txt, "OK")
        ws.write(r, 2, sheet, f.txt)
    last = 4 + len(checks)
    ws.write_formula("B2", f"=IF(COUNTIF(B5:B{last},\"ERROR\")=0,\"OK\",\"ERROR\")", f.txt, "OK")
    _cf_okerr(ws, f"B2:B{last}", f)
    wb.close()
    return {"checks": [c[0] for c in checks], "check_rows": (5, last)}
