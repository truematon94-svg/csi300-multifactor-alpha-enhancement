# -*- coding: utf-8 -*-
"""P6 可执行回测：换手 + 交易成本 + 净收益，并生成可视化图表。

成本模型：佣金万2.5(双边) + 印花税千1(卖出) + 滑点10bp(双边)。
净收益 = 毛收益 - 换手率 * 往返成本。

输出 data/processed/backtest_net.parquet 与 results/figures/*.png。
"""

import os
import numpy as np
import pandas as pd
import cvxpy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROC = "data/processed"
RAW = "data/raw"
FIG = "results/figures"

COMMISSION = 0.00025
STAMP_TAX = 0.001
SLIPPAGE = 0.001
ROUND_TRIP = 2 * (COMMISSION + SLIPPAGE) + STAMP_TAX   # 单边换手对应的往返成本 ≈ 0.35%

MAX_WEIGHT = 0.10
MAX_ACTIVE = 0.02
MAX_IND_DEV = 0.05
LAMBDA_TURNOVER = 0.05

# 配色（验证过的 categorical palette 前3槽）
C = {"benchmark": "#2a78d6", "topk": "#eb6834", "enhanced": "#1baf7a"}
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"


def rebalance_dates(close):
    s = pd.Series(close.index, index=close.index)
    return list(s.groupby(close.index.strftime("%Y-%m")).max().values)


def optimize_once(alpha, w_bench, w_prev, ind_series):
    n = len(alpha)
    w = cp.Variable(n)
    obj = cp.Maximize(alpha.values @ w - LAMBDA_TURNOVER * cp.norm1(w - w_prev))
    cons = [cp.sum(w) == 1, w >= 0, w <= MAX_WEIGHT, cp.abs(w - w_bench) <= MAX_ACTIVE]
    for ind in ind_series.dropna().unique():
        m = (ind_series.fillna("缺失") == ind).values.astype(float)
        cons.append(cp.abs(m @ w - m @ w_bench) <= MAX_IND_DEV)
    try:
        cp.Problem(obj, cons).solve(solver=cp.CLARABEL)
        return w.value
    except Exception:
        return None


def run_backtest(K=50):
    alpha = pd.read_parquet(PROC + "/alpha_panel.parquet")
    uni = pd.read_parquet(PROC + "/universe_csi300.parquet")
    close = pd.read_parquet(PROC + "/close.parquet")
    mcap = pd.read_parquet(PROC + "/market_cap.parquet")
    ind = pd.read_parquet(PROC + "/industry_map.parquet")["industry_sw_l1"]

    dates = rebalance_dates(close)
    close_dates = close.loc[dates]
    close_t1 = close.shift(-1).loc[dates]  # T+1 收盘价（t+1 成交，去收盘价前视）
    fwd_ret = close_dates.shift(-1) / close_t1 - 1
    uni_map = uni.groupby("ym")["code"].apply(set).to_dict()

    idx = pd.read_csv(RAW + "/index_daily.csv")
    idx = idx[idx["index_code"] == "000300.XSHG"].copy()
    idx["date"] = pd.to_datetime(idx["date"])
    bench = idx.set_index("date")["close"].sort_index().loc[dates].pct_change(fill_method=None).shift(-1)

    w_topk_prev, w_enh_prev = None, None
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
        if len(common) < K:
            continue
        a_c = a.loc[common]
        mc_c = mc.loc[common]
        r_c = r.loc[common].values

        # --- Top-K 等权 ---
        top = a_c.nlargest(K).index
        w_topk = np.zeros(len(common))
        w_topk[[i for i, c in enumerate(common) if c in top]] = 1.0 / K
        w_topk_prev_arr = np.zeros(len(common))
        if w_topk_prev is not None:
            for i, c in enumerate(common):
                w_topk_prev_arr[i] = w_topk_prev.get(c, 0.0)
        turn_topk = 0.5 * np.abs(w_topk - w_topk_prev_arr).sum()
        gross_topk = float((r_c * w_topk).sum())
        net_topk = gross_topk - turn_topk * ROUND_TRIP
        w_topk_prev = {c: w_topk[i] for i, c in enumerate(common)}

        # --- 约束增强 ---
        w_bench = (mc_c / mc_c.sum()).clip(lower=0).values
        ind_c = ind.reindex(common)
        w_prev_arr = np.zeros(len(common))
        if w_enh_prev is not None:
            for i, c in enumerate(common):
                w_prev_arr[i] = w_enh_prev.get(c, 0.0)
        w_enh = optimize_once(a_c, w_bench, w_prev_arr, ind_c)
        if w_enh is None:
            w_enh = w_topk.copy()
        w_enh = np.clip(w_enh, 0, None)
        w_enh = w_enh / w_enh.sum()
        turn_enh = 0.5 * np.abs(w_enh - w_prev_arr).sum()
        gross_enh = float((r_c * w_enh).sum())
        net_enh = gross_enh - turn_enh * ROUND_TRIP
        w_enh_prev = {c: w_enh[i] for i, c in enumerate(common)}

        recs.append({"date": d, "benchmark": float(bench.loc[d]),
                     "topk_gross": gross_topk, "topk_net": net_topk, "topk_turnover": turn_topk,
                     "enhanced_gross": gross_enh, "enhanced_net": net_enh, "enhanced_turnover": turn_enh})

    df = pd.DataFrame(recs).set_index("date")
    df.to_parquet(PROC + "/backtest_net.parquet")

    # --- 汇总 ---
    print("=== P6 回测（含成本，2008-2022）===")
    print("平均月换手率: Top-K=%.1f%%  约束增强=%.1f%%" % (df.topk_turnover.mean()*100, df.enhanced_turnover.mean()*100))
    print("平均月成本:   Top-K=%.2f%%  约束增强=%.2f%%" % (df.topk_turnover.mean()*ROUND_TRIP*100, df.enhanced_turnover.mean()*ROUND_TRIP*100))
    for col, name in [("benchmark","沪深300基准"), ("topk_net","Top-K(净)"), ("enhanced_net","约束增强(净)")]:
        r = df[col]
        ann = (1+r).prod()**(12/len(r))-1
        vol = r.std()*np.sqrt(12)
        shp = ann/vol
        mdd = ((1+r).cumprod()/(1+r).cumprod().cummax()-1).min()
        print("%-12s 年化%.2f%%  波动%.2f%%  Sharpe %.3f  累计%.2f%%  回撤%.2f%%" % (name, ann*100, vol*100, shp, ((1+r).prod()-1)*100, mdd*100))

    # --- 绘图 ---
    os.makedirs(FIG, exist_ok=True)
    plot_results(df)
    return df


def plot_results(df):
    plt.rcParams.update({"font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
                         "axes.unicode_minus": False, "figure.dpi": 120})

    nav = (1 + df[["benchmark", "topk_net", "enhanced_net"]]).cumprod()

    # 1) 累计净值
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for col, label in [("benchmark", "CSI300 基准"), ("topk_net", "Top-K 等权"), ("enhanced_net", "约束型增强")]:
        ax.plot(nav.index, nav[col], color=C[col.split("_")[0]], lw=1.8, label=label)
    ax.set_title("累计净值（含成本，2008-2022）", color=INK, fontsize=13, loc="left")
    ax.set_ylabel("净值（初始=1）", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG + "/nav.png", bbox_inches="tight"); plt.close(fig)

    # 2) 回撤
    dd = nav / nav.cummax() - 1
    fig, ax = plt.subplots(figsize=(10, 5))
    for col, label in [("benchmark", "CSI300 基准"), ("topk_net", "Top-K 等权"), ("enhanced_net", "约束型增强")]:
        ax.fill_between(dd.index, dd[col], 0, color=C[col.split("_")[0]], alpha=0.25, lw=0)
        ax.plot(dd.index, dd[col], color=C[col.split("_")[0]], lw=1.5, label=label)
    ax.set_title("回撤", color=INK, fontsize=13, loc="left")
    ax.set_ylabel("回撤", color=MUTED)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.spines["left"].set_color(MUTED); ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG + "/drawdown.png", bbox_inches="tight"); plt.close(fig)

    print("\n图表已保存: results/figures/nav.png, results/figures/drawdown.png")


if __name__ == "__main__":
    run_backtest()
