# -*- coding: utf-8 -*-
"""
沪深300 多因子 Alpha 指数增强 —— 聚宽回测版（与本地口径完全一致）

与本地 P0~P7 pipeline 一致：
  - 股票池：沪深300 时点成分
  - 月度调仓（每月最后一个交易日，次日开盘成交）
  - 6 因子等权合成：价值EP、质量ROE、成长营收同比、反转、低波特异波动、流动性换手
  - 预处理：MAD 3倍缩尾 → z-score → 行业+市值 双中性化
  - 组合：约束型优化（scipy linprog 复现本地 cvxpy 优化器）
"""

import numpy as np
import pandas as pd
import datetime
import io
from scipy.optimize import linprog

LAMBDA_TURNOVER = 0.05
MAX_WEIGHT = 0.10
MAX_ACTIVE = 0.02
MAX_IND_DEV = 0.05


def initialize(context):
    set_benchmark('000300.XSHG')
    set_option('use_real_price', True)
    set_option('avoid_future_data', True)   # 无未来函数：财务用 get_fundamentals(date=d) 点状时点，行情用 history 历史数据
    set_slippage(FixedSlippage(0.001))
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                             open_commission=0.00025, close_commission=0.00025,
                             close_today_commission=0, min_commission=5), type='stock')

    g.factor_names = ['value_ep', 'quality_roe', 'growth_revenue',
                      'reversal_1m', 'lowvol_idio', 'liquidity_turnover']
    g.ret_window = 252
    g.idio_window = 60
    g.w_prev = None

    # 行业映射：从研究环境读取上传的文件（文件夹名按你的研究环境）
    content = read_file('04_multi_alpha_and_index_enhancement/industry_sw_l1_map.csv')
    if isinstance(content, bytes):
        content = content.decode('utf-8-sig')
    g.industry = pd.read_csv(io.StringIO(content), dtype=str).set_index('code')['industry_sw_l1']

    g.last_month = None   # 上月月份，用于检测月切换（每月首个交易日调仓）


def handle_data(context, data):
    # 每月第一个交易日调仓（月末信号 -> 次月首日成交，即 t+1 成交，无未来函数）
    if context.current_dt.month == g.last_month:
        return
    g.last_month = context.current_dt.month
    w = compute_weights(context)
    if w is None:
        return
    # 卖出不在目标的
    for stock in list(context.portfolio.positions.keys()):
        if stock not in w:
            order_target_value(stock, 0)
    # 按目标权重买入
    for stock, wt in w.items():
        if wt > 1e-6:
            order_target_value(stock, context.portfolio.total_value * wt)


def compute_weights(context):
    # 信号用上一交易日(=上月末)数据，t+1 成交，无未来函数
    d = context.previous_date
    d_str = str(d)
    codes = get_index_stocks('000300.XSHG', d_str)
    if not codes:
        return None

    # ---- 财务（回测中不传 date 自动对齐到上一交易日；营收同比单独 query，避免复用 query 状态异常）----
    q_val = query(valuation.code, valuation.market_cap,
                  valuation.circulating_market_cap).filter(valuation.code.in_(codes))
    val = get_fundamentals(q_val).set_index('code')

    q_inc = query(income.code, income.net_profit,
                  income.total_operating_revenue).filter(income.code.in_(codes))
    inc = get_fundamentals(q_inc).set_index('code')

    q_inc_prev = query(income.code, income.total_operating_revenue).filter(income.code.in_(codes))
    prev_date = (d - datetime.timedelta(days=365)).strftime('%Y-%m-%d')
    inc_prev = get_fundamentals(q_inc_prev, date=prev_date).set_index('code')

    q_bal = query(balance.code, balance.total_owner_equities).filter(balance.code.in_(codes))
    bal = get_fundamentals(q_bal).set_index('code')

    # ---- 价量 ----
    price = history(g.ret_window, '1d', 'close', codes, df=True, skip_paused=False, fq='pre')
    if price is None or price.empty:
        return None
    if isinstance(price.columns, pd.MultiIndex):
        price = price['close']
    if len(price) < 21:
        return None
    reversal = price.iloc[-1] / price.iloc[-21] - 1.0

    ret = price.pct_change().iloc[-g.idio_window:]
    bench = history(g.idio_window + 1, '1d', 'close', '000300.XSHG', df=True, skip_paused=False, fq='pre')
    if isinstance(bench, pd.DataFrame):
        if isinstance(bench.columns, pd.MultiIndex):
            bench = bench['close']
        bench_ret = bench.iloc[:, 0].pct_change()
    else:
        bench_ret = pd.Series(bench).pct_change()
    idio = pd.Series(index=codes, dtype=float)
    for c in codes:
        r = ret[c].dropna()
        if len(r) < 30 or bench_ret.std() == 0:
            idio[c] = np.nan
            continue
        b = bench_ret.reindex(r.index)
        beta = np.cov(r, b)[0, 1] / np.var(b)
        idio[c] = (r - beta * b).std()

    money = history(20, '1d', 'money', codes, df=True, skip_paused=False, fq='pre')
    if money is not None and not money.empty:
        if isinstance(money.columns, pd.MultiIndex):
            money = money['money']
        turnover = money.mean() / val['circulating_market_cap']
    else:
        turnover = pd.Series(np.nan, index=codes)

    # ---- 拼因子（与本地 factors.py 相同公式）----
    df = pd.DataFrame(index=codes)
    df['value_ep'] = inc['net_profit'] / val['market_cap']
    df['quality_roe'] = inc['net_profit'] / bal['total_owner_equities']
    df['growth_revenue'] = inc['total_operating_revenue'] / inc_prev['total_operating_revenue'] - 1.0
    df['reversal_1m'] = -reversal
    df['lowvol_idio'] = -idio
    df['liquidity_turnover'] = -turnover
    df['log_mcap'] = np.log(val['market_cap'])
    df['industry'] = g.industry.reindex(df.index)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=g.factor_names + ['log_mcap'])
    if len(df) < 30:
        return None

    # ---- 预处理：MAD → z-score → 行业+市值 中性化 ----
    for f in g.factor_names:
        df[f] = winsorize_mad(df[f])
        df[f] = (df[f] - df[f].mean()) / df[f].std()
        df[f] = neutralize(df[f], df['industry'], df['log_mcap'])

    # ---- 等权合成 alpha ----
    alpha = df[g.factor_names].mean(axis=1)
    alpha = (alpha - alpha.mean()) / alpha.std()

    # ---- 约束优化 ----
    w_bench = (val.loc[df.index, 'market_cap'] / val.loc[df.index, 'market_cap'].sum()).clip(lower=0).values
    w_prev = np.zeros(len(df))
    if g.w_prev is not None:
        for i, c in enumerate(df.index):
            w_prev[i] = g.w_prev.get(c, 0.0)
    w_opt = optimize_weights(alpha, w_bench, w_prev, df['industry'])
    if w_opt is None:
        return None
    g.w_prev = {c: w_opt[i] for i, c in enumerate(df.index)}
    return {c: max(0.0, w_opt[i]) for i, c in enumerate(df.index)}


def optimize_weights(alpha, w_bench, w_prev, ind_series):
    """scipy linprog 复现本地 cvxpy 约束优化。"""
    n = len(alpha)
    c = np.concatenate([-alpha.values, LAMBDA_TURNOVER * np.ones(n)])

    bounds = []
    for i in range(n):
        lo = max(0.0, w_bench[i] - MAX_ACTIVE)
        hi = min(MAX_WEIGHT, w_bench[i] + MAX_ACTIVE)
        bounds.append((lo, hi))
    bounds += [(0.0, None)] * n

    A_eq = np.zeros((1, 2 * n)); A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])

    A_ub, b_ub = [], []
    for i in range(n):
        row = np.zeros(2 * n); row[i] = 1.0; row[n + i] = -1.0
        A_ub.append(row); b_ub.append(w_prev[i])
        row = np.zeros(2 * n); row[i] = -1.0; row[n + i] = -1.0
        A_ub.append(row); b_ub.append(-w_prev[i])
    for ind in ind_series.dropna().unique():
        mask = (ind_series.fillna('缺失') == ind).values.astype(float)
        row = np.zeros(2 * n); row[:n] = mask
        A_ub.append(row); b_ub.append(MAX_IND_DEV + mask @ w_bench)
        row = np.zeros(2 * n); row[:n] = -mask
        A_ub.append(row); b_ub.append(MAX_IND_DEV - mask @ w_bench)

    try:
        res = linprog(c, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='interior-point')
        if res.success:
            return res.x[:n]
    except Exception:
        pass
    return None


def winsorize_mad(s, n=3.0):
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if mad == 0 or np.isnan(mad):
        return s
    return s.clip(med - n * mad, med + n * mad)

def neutralize(y, ind, log_mcap):
    df = pd.DataFrame({'y': y, 'ind': ind.fillna('缺失'), 'lm': log_mcap}).dropna(subset=['y'])
    X = pd.get_dummies(df['ind'], prefix='ind', drop_first=True).astype(float)
    X['lm'] = df['lm']
    X = X.dropna()
    if len(X) < 20:
        return y - y.mean()
    idx2 = X.index
    Xm = np.column_stack([np.ones(len(X)), X.values])
    beta, *_ = np.linalg.lstsq(Xm, df.loc[idx2, 'y'].values, rcond=None)
    resid = df.loc[idx2, 'y'].values - Xm @ beta
    out = pd.Series(np.nan, index=y.index)
    out.loc[idx2] = resid
    return out
