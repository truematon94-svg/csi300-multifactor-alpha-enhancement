# -*- coding: utf-8 -*-
"""P2 基础面板构建：把原始长表整理成因子计算所需的中间数据。

输入 data/raw/：
  - stock_daily_YYYY.csv（长表 date,code,field,value）
  - universe.csv（code,date,universe，含 csi300/csi800）
  - status_financial_*.csv（财务五张，月频点状时点）
  - status_marketcap.csv（半年频市值）
  - data/processed/industry_sw_l1_map.csv（静态行业映射）

输出 data/processed/：
  - close.parquet         （date × code 前复权收盘价）
  - volume.parquet        （date × code 成交量）
  - financial.parquet     （长表 code,date,table,field,value）
  - industry_map.parquet  （code → industry_sw_l1）
  - market_cap.parquet    （date × code，forward-fill 到日频）
  - universe_csi300.parquet （ym,code 每月成员记录）
"""

import os
import pandas as pd

RAW = "data/raw"
PROC = "data/processed"
YEARS = range(2008, 2023)


def load_price_field(field):
    """行情字段长表 -> 宽表 (date × code)。"""
    frames = []
    for y in YEARS:
        f = os.path.join(RAW, "stock_daily_%d.csv" % y)
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, usecols=["date", "code", "field", "value"], dtype={"code": str})
        df = df[df["field"] == field]
        if df.empty:
            continue
        frames.append(df.pivot_table(index="date", columns="code", values="value", aggfunc="last"))
    out = pd.concat(frames, axis=0)
    out.index = pd.to_datetime(out.index)
    return out.sort_index()


def load_universe():
    """沪深300时点成分：返回 {(ym, code)} 每月成员集合。"""
    uni = pd.read_csv(os.path.join(RAW, "universe.csv"), dtype=str)
    uni = uni[uni["universe"] == "csi300"].copy()
    uni["ym"] = uni["date"].str[:7]
    return set(zip(uni["ym"], uni["code"]))


def load_financial():
    """五张财务表合并成一张长表 (code, date, [pubDate], table, field, value)。"""
    tables = {
        "valuation": "status_financial_valuation.csv",
        "indicator": "status_financial_indicator.csv",
        "balance": "status_financial_balance.csv",
        "income": "status_financial_income.csv",
        "cashflow": "status_financial_cashflow.csv",
    }
    frames = []
    for tbl, fname in tables.items():
        f = os.path.join(RAW, fname)
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, dtype={"code": str})
        # pubDate(字符串) 不参与 melt，避免和数值字段混成 object 列
        id_vars = ["code", "date"]
        if "pubDate" in df.columns:
            id_vars.append("pubDate")
        df = df.melt(id_vars=id_vars, var_name="field", value_name="value")
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["table"] = tbl
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_industry():
    """静态行业映射 code -> 申万一级。"""
    f = os.path.join(PROC, "industry_sw_l1_map.csv")
    df = pd.read_csv(f, dtype=str, usecols=["code", "industry_sw_l1"])
    return df.set_index("code")["industry_sw_l1"]


def load_marketcap(calendar):
    """市值（半年频 -> 日频 forward-fill），返回 date × code。"""
    df = pd.read_csv(os.path.join(RAW, "status_marketcap.csv"), dtype={"code": str})
    df["date"] = pd.to_datetime(df["date"])
    piv = df.pivot_table(index="date", columns="code", values="market_cap", aggfunc="last")
    piv = piv.sort_index()
    piv = piv.reindex(calendar, method="ffill")
    return piv


def build_base():
    """构建并保存基础中间数据。"""
    os.makedirs(PROC, exist_ok=True)

    close_path = os.path.join(PROC, "close.parquet")
    if os.path.exists(close_path):
        close = pd.read_parquet(close_path)
        print("close 已存在，跳过: %s" % str(close.shape))
    else:
        print("pivot close ...")
        close = load_price_field("close")
        close.to_parquet(close_path)
        print("  close: %s" % str(close.shape))

    volume_path = os.path.join(PROC, "volume.parquet")
    if os.path.exists(volume_path):
        volume = pd.read_parquet(volume_path)
        print("volume 已存在，跳过: %s" % str(volume.shape))
    else:
        print("pivot volume ...")
        volume = load_price_field("volume")
        volume.to_parquet(volume_path)
        print("  volume: %s" % str(volume.shape))

    print("load financial ...")
    fin = load_financial()
    fin.to_parquet(os.path.join(PROC, "financial.parquet"), index=False)
    print("  financial: %s" % str(fin.shape))

    print("load industry ...")
    ind = load_industry()
    ind.to_frame("industry_sw_l1").to_parquet(os.path.join(PROC, "industry_map.parquet"))
    print("  industry: %d 只" % len(ind))

    print("load marketcap (forward-fill to daily) ...")
    mc = load_marketcap(close.index)
    mc.to_parquet(os.path.join(PROC, "market_cap.parquet"))
    print("  market_cap: %s" % str(mc.shape))

    print("save universe ...")
    uni = load_universe()
    pd.DataFrame(list(uni), columns=["ym", "code"]).to_parquet(
        os.path.join(PROC, "universe_csi300.parquet"), index=False)
    print("  universe: %d 条 (ym, code)" % len(uni))

    print("完成")


if __name__ == "__main__":
    build_base()
