# -*- coding: utf-8 -*-
"""9 个候选因子计算（月度调仓，点状时点对齐）。

价量类（只用行情，信号日收盘前已知）：
  momentum_12_1  过去12月收益(剔除最近1月)，sign +1
  reversal_1m    最近1月收益，sign -1
  lowvol_idio    过去60日特异波动率(残差std)，sign -1
  liquidity_turnover  近20日日均换手率，sign -1

基本面类（财务按 pubDate 公告日对齐，只用到 T 日已公告的最新报表）：
  value_ep        净利润 / 总市值，sign +1
  quality_roe     净利润 / 净资产，sign +1
  quality_accruals (净利润-经营现金流) / 总资产，sign -1
  growth_revenue  营收同比增速，sign +1
  quality_leverage 净资产 / 总资产，sign +1

统一约定：最终因子值 = sign * raw，值越大预期收益越高。
输出 data/processed/factor_panel.parquet（长表 date,code,factor,value）。
"""

import pandas as pd
import numpy as np

PROC = "data/processed"
RAW = "data/raw"


def _load():
    close = pd.read_parquet(PROC + "/close.parquet")
    volume = pd.read_parquet(PROC + "/volume.parquet")
    fin = pd.read_parquet(PROC + "/financial.parquet")
    mc = pd.read_parquet(PROC + "/market_cap.parquet")
    return close, volume, fin, mc


def rebalance_dates(close):
    """每月最后一个交易日。"""
    s = pd.Series(close.index, index=close.index)
    return list(s.groupby(close.index.strftime("%Y-%m")).max().values)


def _index_returns(close):
    """沪深300指数日收益（用于特异波动率回归）。"""
    idx = pd.read_csv(RAW + "/index_daily.csv")
    idx = idx[idx["index_code"] == "000300.XSHG"].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date")["close"].sort_index()
    return idx.pct_change()


def _financial_wide(fin, table, field):
    """某财务表某字段 -> date × code 宽表。"""
    sub = fin[(fin["table"] == table) & (fin["field"] == field)].copy()
    sub["date"] = pd.to_datetime(sub["date"])
    return sub.pivot_table(index="date", columns="code", values="value", aggfunc="last").sort_index()


def price_factors(close, volume, fin, mc, dates):
    """价量类因子（date × code 宽表，已按 sign 定方向）。"""
    ret = close.pct_change()
    mom = close.shift(21) / close.shift(252) - 1                      # 12-1 动量
    rev = close / close.shift(21) - 1                                  # 1月反转

    # 特异波动率：60日窗口，ret = alpha + beta * mkt + eps，取 eps 的 std
    mkt = _index_returns(close)
    mkt = mkt.reindex(close.index)
    cov_rm = ret.rolling(60).cov(mkt)
    var_m = mkt.rolling(60).var()
    beta = cov_rm.div(var_m, axis=0)          # axis=0 按行广播，避免按列对齐错位
    resid = ret.sub(beta.mul(mkt, axis=0))
    idio = resid.rolling(60).std()                                     # 特异波动

    # 换手率：20日均成交额 / 流通市值（流通市值 = circulating_market_cap 月频 ffill）
    circ = _financial_wide(fin, "valuation", "circulating_market_cap")
    circ = circ.reindex(close.index, method="ffill")
    turn = (volume * close).rolling(20).mean() / circ                  # 换手率近似

    codes = close.columns
    out = []
    for d in dates:
        block = pd.DataFrame(index=codes)
        block["momentum_12_1"] = mom.loc[d].reindex(codes)
        block["reversal_1m"] = -rev.loc[d].reindex(codes)
        block["lowvol_idio"] = -idio.loc[d].reindex(codes)
        block["liquidity_turnover"] = -turn.loc[d].reindex(codes)
        block.index.name = "code"
        block = block.reset_index()
        block["date"] = d
        out.append(block)
    return pd.concat(out, ignore_index=True)


def financial_factors(fin, mc, dates):
    """基本面类因子（点状时点，date × code）。"""
    # 各字段 point-in-time 宽表（date = 财务快照月，已是"截至该日已公告"）
    np_ = _financial_wide(fin, "income", "net_profit")
    rev_ = _financial_wide(fin, "income", "total_operating_revenue")
    ocf = _financial_wide(fin, "cashflow", "net_operate_cash_flow")
    ta = _financial_wide(fin, "balance", "total_assets")
    te = _financial_wide(fin, "balance", "total_owner_equities")

    # 财务月频快照 forward-fill 到日频，再取调仓日
    def _at(wide, d):
        w = wide.reindex(close_index, method="ffill")
        return w.reindex([d]).iloc[0]   # 日期越界返回 NaN

    close = pd.read_parquet(PROC + "/close.parquet")
    close_index = close.index
    codes = close.columns

    out = pd.DataFrame()
    for d in dates:
        ep = _at(np_, d) / _at(mc, d)                                   # EP = 净利/市值
        roe = _at(np_, d) / _at(te, d)                                  # ROE
        acc = (_at(np_, d) - _at(ocf, d)) / _at(ta, d)                  # 应计
        lev = _at(te, d) / _at(ta, d)                                   # 杠杆(权益比)

        # 营收增速：T 与 T-12月 点状时点快照对比
        d_prev = d - pd.offsets.DateOffset(months=12)
        growth = _at(rev_, d) / _at(rev_, d_prev) - 1
        growth = growth.replace([np.inf, -np.inf], np.nan)   # 除零/除近零 -> NaN

        block = pd.DataFrame(index=codes)
        block["value_ep"] = ep.reindex(codes)
        block["quality_roe"] = roe.reindex(codes)
        block["quality_accruals"] = -acc.reindex(codes)
        block["growth_revenue"] = growth.reindex(codes)
        block["quality_leverage"] = lev.reindex(codes)
        block.index.name = "code"
        block = block.reset_index()
        block["date"] = d
        out = pd.concat([out, block], ignore_index=True)
    return out


def compute_factors():
    close, volume, fin, mc = _load()
    dates = rebalance_dates(close)
    print("调仓日数量:", len(dates), "| 起止:", dates[0], "->", dates[-1])

    print("价量因子 ...")
    pf = price_factors(close, volume, fin, mc, dates)
    print("基本面因子 ...")
    ff = financial_factors(fin, mc, dates)

    panel = pd.concat([pf, ff], ignore_index=True)
    panel = panel.melt(id_vars=["date", "code"], var_name="factor", value_name="value")
    panel["value"] = panel["value"].replace([np.inf, -np.inf], np.nan)
    panel = panel.dropna(subset=["value"])
    panel.to_parquet(PROC + "/factor_panel.parquet", index=False)
    print("factor_panel:", panel.shape)
    print("因子数量:", panel["factor"].nunique())
    print(panel.groupby("factor")["value"].count().to_string())
    return panel


if __name__ == "__main__":
    compute_factors()
