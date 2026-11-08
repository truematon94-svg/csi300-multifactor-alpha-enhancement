# -*- coding: utf-8 -*-
"""P5 组合构建：基线组合 + Top-K 组合的月度收益。

基线：
  - benchmark：沪深300指数
  - constituent_equal：沪深300成分等权
  - topk_equal：综合 alpha 前 K 名等权（K 默认 50）

输出 data/processed/portfolio_returns.parquet（date, name, ret）。
"""

import numpy as np
import pandas as pd

PROC = "data/processed"
RAW = "data/raw"


def rebalance_dates(close):
    s = pd.Series(close.index, index=close.index)
    return list(s.groupby(close.index.strftime("%Y-%m")).max().values)


def _benchmark_ret(dates):
    idx = pd.read_csv(RAW + "/index_daily.csv")
    idx = idx[idx["index_code"] == "000300.XSHG"].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    idx = idx.set_index("date")["close"].sort_index()
    idx_dates = idx.loc[dates]
    return idx_dates.pct_change(fill_method=None).shift(-1)


def build_portfolios(K=50):
    alpha = pd.read_parquet(PROC + "/alpha_panel.parquet")
    uni = pd.read_parquet(PROC + "/universe_csi300.parquet")
    close = pd.read_parquet(PROC + "/close.parquet")

    dates = rebalance_dates(close)
    fwd_ret = close.loc[dates].pct_change(fill_method=None).shift(-1)
    bench_ret = _benchmark_ret(dates)
    uni_map = uni.groupby("ym")["code"].apply(set).to_dict()

    recs = []
    for d in dates:
        d = pd.Timestamp(d)
        ym = d.strftime("%Y-%m")
        members = uni_map.get(ym, set())
        if not members:
            continue
        a = alpha[alpha["date"] == d].set_index("code")["equal"].dropna()
        r = fwd_ret.loc[d].dropna()
        common = a.index.intersection(r.index).intersection(members)
        if len(common) < K:
            continue
        top_k = a.loc[common].nlargest(K).index

        recs.append({"date": d, "name": "benchmark", "ret": float(bench_ret.loc[d])})
        recs.append({"date": d, "name": "constituent_equal", "ret": float(r.loc[list(common)].mean())})
        recs.append({"date": d, "name": "topk_equal", "ret": float(r.loc[top_k].mean())})

    df = pd.DataFrame(recs)
    df.to_parquet(PROC + "/portfolio_returns.parquet", index=False)

    # 汇总指标
    print("=== 组合月收益汇总（未计成本，2008-2022 全样本）===")
    summary = []
    for name, g in df.groupby("name"):
        r = g.set_index("date")["ret"]
        ann_ret = (1 + r).prod() ** (12 / len(r)) - 1
        ann_vol = r.std() * np.sqrt(12)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
        cum = (1 + r).prod() - 1
        mdd = ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()
        summary.append({"name": name, "年化收益": ann_ret, "年化波动": ann_vol,
                        "Sharpe": sharpe, "累计收益": cum, "最大回撤": mdd})
    sdf = pd.DataFrame(summary).set_index("name")
    print((sdf * 100).round(2).to_string())
    print("\n(注：年化收益/波动/累计/回撤单位为 %；未计交易成本，P6 加成本。)")
    return df


if __name__ == "__main__":
    build_portfolios()
