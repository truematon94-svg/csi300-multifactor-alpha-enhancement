# -*- coding: utf-8 -*-
"""P7 收益归因 + 稳健性。

1. 年度收益分解（基准 / Top-K / 增强 各年收益与超额）；
2. 风格暴露（Top-K 组合相对全样本的因子偏移，即押注了哪些因子）；
3. 稳健性（不同 Top-K 数量 K 的净收益）。

输出 data/processed/attribution_*.parquet。
"""

import numpy as np
import pandas as pd

PROC = "data/processed"


def annual_returns():
    df = pd.read_parquet(PROC + "/backtest_net.parquet")
    df.index = pd.to_datetime(df.index)
    cols = ["benchmark", "topk_net", "enhanced_net"]
    annual = df[cols].groupby(df.index.year).apply(lambda g: (1 + g).prod() - 1)
    annual["excess_topk"] = annual["topk_net"] - annual["benchmark"]
    annual["excess_enhanced"] = annual["enhanced_net"] - annual["benchmark"]
    annual.to_parquet(PROC + "/attribution_annual.parquet")
    print("=== 年度收益（%）===")
    print((annual * 100).round(2).to_string())
    print("\n正超额年数: Top-K %d/%d，增强 %d/%d" % (
        (annual.excess_topk > 0).sum(), len(annual),
        (annual.excess_enhanced > 0).sum(), len(annual)))
    return annual


def factor_tilt(K=50):
    alpha = pd.read_parquet(PROC + "/alpha_panel.parquet")
    fp = pd.read_parquet(PROC + "/factor_processed.parquet")
    close = pd.read_parquet(PROC + "/close.parquet")
    s = pd.Series(close.index, index=close.index)
    dates = [pd.Timestamp(d) for d in s.groupby(close.index.strftime("%Y-%m")).max().values]

    tilts = []
    for d in dates:
        a = alpha[alpha["date"] == d].set_index("code")["equal"].dropna()
        if len(a) < K:
            continue
        top = set(a.nlargest(K).index)
        f = fp[fp["date"] == d].pivot(index="code", columns="factor", values="value")
        common = f.index.intersection(a.index)
        if len(common) < K:
            continue
        f_c = f.loc[common]
        in_top = f_c.index.isin(top)
        tilt = f_c[in_top].mean() - f_c.mean()
        tilt.name = d
        tilts.append(tilt)
    tilt_df = pd.DataFrame(tilts)
    tilt_df.to_parquet(PROC + "/attribution_tilt.parquet")
    print("\n=== 风格暴露（Top-K 组合相对全样本的因子偏移，z-score 单位）===")
    print(tilt_df.mean().round(3).to_string())
    return tilt_df


def robustness(ks=(30, 50, 80)):
    alpha = pd.read_parquet(PROC + "/alpha_panel.parquet")
    uni = pd.read_parquet(PROC + "/universe_csi300.parquet")
    close = pd.read_parquet(PROC + "/close.parquet")
    s = pd.Series(close.index, index=close.index)
    dates = [pd.Timestamp(d) for d in s.groupby(close.index.strftime("%Y-%m")).max().values]
    close_dates = close.loc[dates]
    close_t1 = close.shift(-1).loc[dates]  # T+1 收盘价（t+1 成交，去收盘价前视）
    fwd_ret = close_dates.shift(-1) / close_t1 - 1
    uni_map = uni.groupby("ym")["code"].apply(set).to_dict()
    ROUND_TRIP = 0.0035

    print("\n=== 稳健性：不同 Top-K 数量（含成本）===")
    for K in ks:
        w_prev = None
        nets = []
        turns = []
        for d in dates:
            ym = d.strftime("%Y-%m")
            members = uni_map.get(ym, set())
            a = alpha[alpha["date"] == d].set_index("code")["equal"].dropna()
            r = fwd_ret.loc[d].dropna()
            common = sorted(set(a.index) & set(r.index) & members)
            if len(common) < K:
                continue
            top = a.loc[common].nlargest(K).index
            w = np.zeros(len(common))
            w[[i for i, c in enumerate(common) if c in top]] = 1.0 / K
            w_prev_arr = np.zeros(len(common))
            if w_prev is not None:
                for i, c in enumerate(common):
                    w_prev_arr[i] = w_prev.get(c, 0.0)
            turn = 0.5 * np.abs(w - w_prev_arr).sum()
            nets.append(float((r.loc[common].values * w).sum()) - turn * ROUND_TRIP)
            turns.append(turn)
            w_prev = {c: w[i] for i, c in enumerate(common)}
        r = pd.Series(nets)
        ann = (1 + r).prod() ** (12 / len(r)) - 1
        shp = ann / (r.std() * np.sqrt(12))
        print("K=%-3d 年化%.2f%%  Sharpe %.3f  平均换手%.1f%%" % (K, ann * 100, shp, np.mean(turns) * 100))


if __name__ == "__main__":
    annual_returns()
    factor_tilt()
    robustness()
