# -*- coding: utf-8 -*-
"""P3 单因子检验：IC / RankIC / ICIR / RankICIR / 正IC比例。

对每个因子、每个调仓日，计算因子值与下月收益的截面相关系数。
分训练期(2008-2019)与测试期(2020-2022)汇总。

输出 data/processed/ic_series.parquet（逐日 IC）与 ic_summary.parquet（汇总）。
"""

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROC = "data/processed"
TRAIN = ("2008-01-01", "2019-12-31")
TEST = ("2020-01-01", "2022-12-31")


def rebalance_dates(close):
    s = pd.Series(close.index, index=close.index)
    return list(s.groupby(close.index.strftime("%Y-%m")).max().values)


def compute_ic():
    fp = pd.read_parquet(PROC + "/factor_processed.parquet")
    close = pd.read_parquet(PROC + "/close.parquet")

    dates = rebalance_dates(close)
    close_dates = close.loc[dates]
    fwd_ret = close_dates.pct_change(fill_method=None).shift(-1)  # 前向1月收益

    rows = []
    for factor, grp in fp.groupby("factor"):
        fwide = grp.pivot(index="date", columns="code", values="value").reindex(dates)
        for d in dates:
            f = fwide.loc[d].dropna()
            r = fwd_ret.loc[d].dropna()
            common = f.index.intersection(r.index)
            if len(common) < 30:
                continue
            fv, rv = f.loc[common].values, r.loc[common].values
            rows.append({
                "date": d, "factor": factor,
                "IC": pearsonr(fv, rv)[0],
                "RankIC": spearmanr(fv, rv)[0],
            })

    ic_df = pd.DataFrame(rows)
    ic_df.to_parquet(PROC + "/ic_series.parquet", index=False)

    summary = []
    for factor, grp in ic_df.groupby("factor"):
        for split, (a, b) in [("train", TRAIN), ("test", TEST)]:
            sub = grp[(grp["date"] >= a) & (grp["date"] <= b)]
            if sub.empty:
                continue
            summary.append({
                "factor": factor, "split": split, "n": len(sub),
                "IC_mean": sub["IC"].mean(),
                "ICIR": sub["IC"].mean() / sub["IC"].std(),
                "RankIC_mean": sub["RankIC"].mean(),
                "RankICIR": sub["RankIC"].mean() / sub["RankIC"].std(),
                "IC_pos_ratio": (sub["IC"] > 0).mean(),
            })

    sdf = pd.DataFrame(summary)
    sdf.to_parquet(PROC + "/ic_summary.parquet", index=False)

    print("=== IC 汇总（train=2008-2019, test=2020-2022）===")
    piv = sdf.pivot(index="factor", columns="split",
                    values=["IC_mean", "ICIR", "RankICIR", "IC_pos_ratio"]).round(4)
    print(piv.to_string())
    return sdf


if __name__ == "__main__":
    compute_ic()
