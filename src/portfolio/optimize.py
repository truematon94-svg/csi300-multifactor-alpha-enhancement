# -*- coding: utf-8 -*-
"""P5 约束型指数增强（cvxpy）。

每月求解：
  maximize  alpha'·w  -  λ·‖w - w_prev‖₁
  s.t.      Σw = 1, 0 ≤ w ≤ 0.10,  |w - w_bench| ≤ 0.02,
            行业主动偏离 ≤ 0.05
其中 w_bench 用市值权重近似沪深300基准权重。

输出 data/processed/enhanced_returns.parquet（date, ret）。
"""

import numpy as np
import pandas as pd
import cvxpy as cp

PROC = "data/processed"

MAX_WEIGHT = 0.10
MAX_ACTIVE = 0.02
MAX_IND_DEV = 0.05
LAMBDA_TURNOVER = 0.05


def rebalance_dates(close):
    s = pd.Series(close.index, index=close.index)
    return list(s.groupby(close.index.strftime("%Y-%m")).max().values)


def optimize_once(alpha, w_bench, w_prev, ind_series):
    n = len(alpha)
    w = cp.Variable(n)
    turnover = cp.norm1(w - w_prev)
    objective = cp.Maximize(alpha.values @ w - LAMBDA_TURNOVER * turnover)
    constraints = [
        cp.sum(w) == 1,
        w >= 0,
        w <= MAX_WEIGHT,
        cp.abs(w - w_bench) <= MAX_ACTIVE,
    ]
    for ind in ind_series.dropna().unique():
        m = (ind_series.fillna("缺失") == ind).values.astype(float)
        constraints.append(cp.abs(m @ w - m @ w_bench) <= MAX_IND_DEV)
    prob = cp.Problem(objective, constraints)
    try:
        prob.solve(solver=cp.CLARABEL)
        if w.value is None:
            return None
        return w.value
    except Exception:
        return None


def run_enhancement():
    alpha = pd.read_parquet(PROC + "/alpha_panel.parquet")
    uni = pd.read_parquet(PROC + "/universe_csi300.parquet")
    close = pd.read_parquet(PROC + "/close.parquet")
    mcap = pd.read_parquet(PROC + "/market_cap.parquet")
    ind = pd.read_parquet(PROC + "/industry_map.parquet")["industry_sw_l1"]

    dates = rebalance_dates(close)
    fwd_ret = close.loc[dates].pct_change(fill_method=None).shift(-1)
    uni_map = uni.groupby("ym")["code"].apply(set).to_dict()

    w_prev = None
    recs = []
    for d in dates:
        d = pd.Timestamp(d)
        ym = d.strftime("%Y-%m")
        members = uni_map.get(ym, set())
        if not members:
            continue
        a = alpha[alpha["date"] == d].set_index("code")["equal"].dropna()
        r = fwd_ret.loc[d].dropna()
        mc = mcap.loc[d].dropna() if d in mcap.index else pd.Series(dtype=float)
        common = sorted(set(a.index) & set(r.index) & set(mc.index) & members)
        if len(common) < 50:
            continue
        a_c = a.loc[common]
        mc_c = mc.loc[common]
        w_bench = (mc_c / mc_c.sum()).clip(lower=0).values
        ind_c = ind.reindex(common)
        w_prev_c = np.zeros(len(common))
        if w_prev is not None:
            for i, c in enumerate(common):
                w_prev_c[i] = w_prev.get(c, 0.0)

        w_opt = optimize_once(a_c, w_bench, w_prev_c, ind_c)
        if w_opt is None:
            # 回退：Top-K 等权
            k = min(50, len(common))
            top = a_c.nlargest(k).index
            w_opt = np.zeros(len(common))
            for i, c in enumerate(common):
                if c in top:
                    w_opt[i] = 1.0 / k
        w_opt = np.clip(w_opt, 0, None)
        w_opt = w_opt / w_opt.sum()
        w_prev = {c: w_opt[i] for i, c in enumerate(common)}

        ret = float((r.loc[common].values * w_opt).sum())
        recs.append({"date": d, "ret": ret})

    df = pd.DataFrame(recs)
    df.to_parquet(PROC + "/enhanced_returns.parquet", index=False)

    r = df.set_index("date")["ret"]
    ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol
    cum = (1 + r).prod() - 1
    mdd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
    print("=== 约束型指数增强（未计成本）===")
    print("月份数:", len(r))
    print("年化收益: %.2f%%" % (ann_ret * 100))
    print("年化波动: %.2f%%" % (ann_vol * 100))
    print("Sharpe: %.3f" % sharpe)
    print("累计收益: %.2f%%" % (cum * 100))
    print("最大回撤: %.2f%%" % (mdd * 100))
    return df


if __name__ == "__main__":
    run_enhancement()
