# -*- coding: utf-8 -*-
"""P3 分组收益 + 单调性 + 多空 + IC 衰减。

分组：每月按因子值分 5 组（等权），算各组下月收益，训练期(2008-2019)汇总。
衰减：IC 在 1/5/10/20 交易日前向收益上的表现。

输出 data/processed/group_returns.parquet 与 ic_decay.parquet。
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROC = "data/processed"
TRAIN = ("2008-01-01", "2019-12-31")


def rebalance_dates(close):
    s = pd.Series(close.index, index=close.index)
    return list(s.groupby(close.index.strftime("%Y-%m")).max().values)


def group_returns():
    fp = pd.read_parquet(PROC + "/factor_processed.parquet")
    close = pd.read_parquet(PROC + "/close.parquet")
    dates = rebalance_dates(close)
    close_dates = close.loc[dates]
    close_t1 = close.shift(-1).loc[dates]  # T+1 收盘价（t+1 成交，去收盘价前视）
    fwd_ret = close_dates.shift(-1) / close_t1 - 1

    rows = []
    for factor, grp in fp.groupby("factor"):
        fwide = grp.pivot(index="date", columns="code", values="value").reindex(dates)
        qrets = {g: [] for g in range(5)}
        for d in dates:
            if not (TRAIN[0] <= str(d) <= TRAIN[1]):
                continue
            f = fwide.loc[d].dropna()
            r = fwd_ret.loc[d].dropna()
            common = f.index.intersection(r.index)
            if len(common) < 50:
                continue
            q = pd.qcut(f.loc[common], 5, labels=False, duplicates="drop")
            for g in range(5):
                vals = r.loc[common][q == g]
                if len(vals):
                    qrets[g].append(vals.mean())
        if not qrets[0]:
            continue
        monthly = {g: float(np.mean(qrets[g])) for g in range(5)}
        ls = monthly[4] - monthly[0]
        # 单调性：组序号与组收益的 Spearman 相关
        mono = spearmanr(list(range(5)), [monthly[g] for g in range(5)])[0]
        rows.append({
            "factor": factor,
            "Q1": monthly[0], "Q2": monthly[1], "Q3": monthly[2],
            "Q4": monthly[3], "Q5": monthly[4],
            "long_short": ls, "monotonicity": mono,
        })

    df = pd.DataFrame(rows).set_index("factor")
    df.to_parquet(PROC + "/group_returns.parquet")
    print("=== 分组月均收益（训练期，等权，Q1最低→Q5最高）===")
    print((df * 100).round(3).to_string())
    print("\n注：单位为月收益 %；long_short = Q5 - Q1；monotonicity 越接近 1 越单调。")
    return df


def ic_decay():
    fp = pd.read_parquet(PROC + "/factor_processed.parquet")
    close = pd.read_parquet(PROC + "/close.parquet")
    dates = rebalance_dates(close)

    horizons = [1, 5, 10, 20]
    rows = []
    for factor, grp in fp.groupby("factor"):
        fwide = grp.pivot(index="date", columns="code", values="value").reindex(dates)
        for h in horizons:
            fwd = close.pct_change(fill_method=None).shift(-h).cumsum(axis=0) * 0  # 占位
            # 用 h 日前向收益：close.shift(-h)/close - 1
            fwd = close.shift(-h) / close - 1
            ics = []
            for d in dates:
                if not (TRAIN[0] <= str(d) <= TRAIN[1]):
                    continue
                f = fwide.loc[d].dropna()
                r = fwd.loc[d].dropna()
                common = f.index.intersection(r.index)
                if len(common) < 50:
                    continue
                ics.append(spearmanr(f.loc[common], r.loc[common])[0])
            rows.append({"factor": factor, "horizon": h,
                         "RankIC_mean": float(np.mean(ics)) if ics else np.nan})

    df = pd.DataFrame(rows).pivot(index="factor", columns="horizon", values="RankIC_mean")
    df.to_parquet(PROC + "/ic_decay.parquet")
    print("\n=== IC 衰减（RankIC，训练期，1/5/10/20 交易日前向收益）===")
    print(df.round(4).to_string())
    return df


if __name__ == "__main__":
    group_returns()
    ic_decay()
