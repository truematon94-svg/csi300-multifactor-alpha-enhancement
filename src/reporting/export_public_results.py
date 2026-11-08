# -*- coding: utf-8 -*-
"""从 data/processed 中间数据复算并导出公开结果小表（results/*.csv）。

原始大数据在 data/ 下已 gitignore，这里只生成可提交的小型结果表。
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROC = "data/processed"
RES = "results"
TRAIN = ("2008-01-01", "2019-12-31")
TEST = ("2020-01-01", "2022-12-31")
ROUND_TRIP = 2 * (0.00025 + 0.001) + 0.001  # 佣金+滑点(双边)+印花税 ≈ 0.35%


def rebalance_dates(close):
    s = pd.Series(close.index, index=close.index)
    return [pd.Timestamp(d) for d in s.groupby(close.index.strftime("%Y-%m")).max().values]


def forward_ret(close, dates):
    """t+1 成交前向月收益：fwd[T] = close[T_next] / close[T+1] - 1。"""
    cd = close.loc[dates]
    t1 = close.shift(-1).loc[dates]
    return cd.shift(-1) / t1 - 1


def export_factor_summary():
    ic = pd.read_parquet(PROC + "/ic_summary.parquet")
    cols = ["factor", "split", "n", "IC_mean", "ICIR", "RankIC_mean", "RankICIR", "IC_pos_ratio"]
    out = ic[cols].sort_values(["split", "factor"])
    out.to_csv(RES + "/factor_tests/factor_summary.csv", index=False)
    return out


def export_alpha_comparison():
    alpha = pd.read_parquet(PROC + "/alpha_panel.parquet")
    close = pd.read_parquet(PROC + "/close.parquet")
    dates = rebalance_dates(close)
    fwd = forward_ret(close, dates)
    rows = []
    for method in ["equal", "ic", "risk"]:
        awide = alpha.pivot(index="date", columns="code", values=method).reindex(dates)
        for split, (a, b) in [("train", TRAIN), ("test", TEST)]:
            ics = []
            for d in dates:
                if not (a <= str(d) <= b):
                    continue
                f = awide.loc[d].dropna()
                r = fwd.loc[d].dropna()
                common = f.index.intersection(r.index)
                if len(common) < 50:
                    continue
                ics.append(spearmanr(f.loc[common], r.loc[common])[0])
            if ics:
                m = np.mean(ics)
                rows.append({"method": method, "split": split,
                             "RankIC_mean": m, "RankICIR": m / np.std(ics)})
    out = pd.DataFrame(rows)
    out.to_csv(RES + "/factor_tests/alpha_comparison.csv", index=False)
    return out


def export_backtest_summary():
    df = pd.read_parquet(PROC + "/backtest_net.parquet")
    df.index = pd.to_datetime(df.index)
    rows = []
    for name, col in [("benchmark", "benchmark"), ("topk", "topk_net"), ("enhanced", "enhanced_net")]:
        r = df[col]
        ann = (1 + r).prod() ** (12 / len(r)) - 1
        vol = r.std() * np.sqrt(12)
        rows.append({"name": name, "annual_return": ann, "annual_vol": vol,
                     "sharpe": ann / vol, "cumulative_return": (1 + r).prod() - 1,
                     "max_drawdown": ((1 + r).cumprod() / (1 + r).cumprod().cummax() - 1).min()})
    for name, col in [("topk", "topk_turnover"), ("enhanced", "enhanced_turnover")]:
        turn = df[col].mean()
        for row in rows:
            if row["name"] == name:
                row["monthly_turnover"] = turn
                row["monthly_cost"] = turn * ROUND_TRIP
    out = pd.DataFrame(rows)
    out.to_csv(RES + "/backtests/backtest_summary.csv", index=False)
    return out


def export_annual_returns():
    df = pd.read_parquet(PROC + "/backtest_net.parquet")
    df.index = pd.to_datetime(df.index)
    annual = df[["benchmark", "topk_net", "enhanced_net"]].groupby(df.index.year).apply(lambda g: (1 + g).prod() - 1)
    annual["excess_topk"] = annual["topk_net"] - annual["benchmark"]
    annual["excess_enhanced"] = annual["enhanced_net"] - annual["benchmark"]
    annual.index.name = "year"
    annual.to_csv(RES + "/attribution/annual_returns.csv")
    return annual


def export_factor_tilts():
    tilt = pd.read_parquet(PROC + "/attribution_tilt.parquet")
    out = tilt.mean().to_frame("mean_tilt")
    out.index.name = "factor"
    out.to_csv(RES + "/attribution/factor_tilts.csv")
    return out


def export_robustness(ks=(30, 50, 80)):
    alpha = pd.read_parquet(PROC + "/alpha_panel.parquet")
    uni = pd.read_parquet(PROC + "/universe_csi300.parquet")
    close = pd.read_parquet(PROC + "/close.parquet")
    mc = pd.read_parquet(PROC + "/market_cap.parquet")
    dates = rebalance_dates(close)
    fwd = forward_ret(close, dates)
    uni_map = uni.groupby("ym")["code"].apply(set).to_dict()
    rows = []
    for K in ks:
        w_prev = None
        nets, turns = [], []
        for d in dates:
            ym = d.strftime("%Y-%m")
            members = uni_map.get(ym, set())
            a = alpha[alpha["date"] == d].set_index("code")["equal"].dropna()
            r = fwd.loc[d].dropna()
            mc_d = mc.loc[d].dropna() if d in mc.index else pd.Series(dtype=float)
            common = sorted(set(a.index) & set(r.index) & set(mc_d.index) & members)
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
        rows.append({"K": K, "annual_return": ann,
                     "sharpe": ann / (r.std() * np.sqrt(12)),
                     "monthly_turnover": np.mean(turns)})
    out = pd.DataFrame(rows)
    out.to_csv(RES + "/backtests/robustness_topk.csv", index=False)
    return out


def export_grouping_summary():
    gr = pd.read_parquet(PROC + "/group_returns.parquet")
    gr.index.name = "factor"
    gr.to_csv(RES + "/factor_tests/grouping_summary.csv")
    return gr


def export_factor_annual_ic():
    ic = pd.read_parquet(PROC + "/ic_series.parquet")
    ic["year"] = pd.to_datetime(ic["date"]).dt.year
    annual = (ic.groupby(["factor", "year"])
                .agg(IC_mean=("IC", "mean"), RankIC_mean=("RankIC", "mean"))
                .reset_index())
    annual.to_csv(RES + "/factor_tests/factor_annual_ic.csv", index=False)
    return annual


def main():
    for sub in ["factor_tests", "backtests", "attribution"]:
        os.makedirs(RES + "/" + sub, exist_ok=True)
    export_factor_summary()
    export_alpha_comparison()
    export_backtest_summary()
    export_annual_returns()
    export_factor_tilts()
    export_robustness()
    export_grouping_summary()
    export_factor_annual_ic()
    print("公开结果表已导出到 results/ 下")


if __name__ == "__main__":
    main()
