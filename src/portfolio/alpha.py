# -*- coding: utf-8 -*-
"""P4 因子筛选 + Alpha 合成。

1. 按训练期 RankICIR 阈值筛选因子；
2. 三种合成：等权 / 训练期 IC 加权 / 风险调整(最大化 ICIR: w ∝ Σ^{-1}·IC)；
3. 权重只用训练期确定，应用到全样本；并检验合成 alpha 的 IC。

输出 data/processed/alpha_panel.parquet（date, code, equal, ic, risk）。
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROC = "data/processed"
TRAIN = ("2008-01-01", "2019-12-31")
TEST = ("2020-01-01", "2022-12-31")
RANKICIR_THRESHOLD = 0.15


def rebalance_dates(close):
    s = pd.Series(close.index, index=close.index)
    return list(s.groupby(close.index.strftime("%Y-%m")).max().values)


def select_factors(ic_summary):
    train = ic_summary[ic_summary["split"] == "train"]
    return train[train["RankICIR"] >= RANKICIR_THRESHOLD]["factor"].tolist()


def compute_weights(ic_series, factors):
    train = ic_series[(ic_series["date"] >= TRAIN[0]) & (ic_series["date"] <= TRAIN[1])]
    ic_wide = train.pivot(index="date", columns="factor", values="IC")[factors]
    mean_ic = ic_wide.mean()
    cov_ic = ic_wide.cov()

    w_equal = pd.Series(1.0 / len(factors), index=factors)
    w_ic = mean_ic / mean_ic.sum()
    w_risk = pd.Series(np.linalg.pinv(cov_ic.values) @ mean_ic.values, index=factors)
    w_risk = w_risk / w_risk.sum()
    return {"equal": w_equal, "ic": w_ic, "risk": w_risk}


def combine():
    fp = pd.read_parquet(PROC + "/factor_processed.parquet")
    ic_summary = pd.read_parquet(PROC + "/ic_summary.parquet")
    ic_series = pd.read_parquet(PROC + "/ic_series.parquet")

    factors = select_factors(ic_summary)
    print("保留因子(%d个，训练期RankICIR>=%.2f):" % (len(factors), RANKICIR_THRESHOLD), factors)

    weights = compute_weights(ic_series, factors)
    print("\n三种合成权重:")
    for name, w in weights.items():
        print("  %-5s: %s" % (name, {k: round(v, 3) for k, v in w.items()}))

    fwide = fp.pivot_table(index=["date", "code"], columns="factor", values="value")[factors]
    alpha = pd.DataFrame(index=fwide.index)
    for name, w in weights.items():
        alpha[name] = (fwide * w).sum(axis=1)
    alpha = alpha.reset_index()
    alpha.to_parquet(PROC + "/alpha_panel.parquet", index=False)

    # 检验合成 alpha 的 RankIC
    close = pd.read_parquet(PROC + "/close.parquet")
    dates = rebalance_dates(close)
    fwd_ret = close.loc[dates].pct_change(fill_method=None).shift(-1)

    print("\n合成 alpha 的 RankIC:")
    for name in ["equal", "ic", "risk"]:
        awide = alpha.pivot(index="date", columns="code", values=name).reindex(dates)
        for split, (a, b) in [("train", TRAIN), ("test", TEST)]:
            ics = []
            for d in dates:
                if not (a <= str(d) <= b):
                    continue
                f = awide.loc[d].dropna()
                r = fwd_ret.loc[d].dropna()
                common = f.index.intersection(r.index)
                if len(common) < 50:
                    continue
                ics.append(spearmanr(f.loc[common], r.loc[common])[0])
            if ics:
                ic_mean = np.mean(ics)
                icir = ic_mean / np.std(ics)
                print("  %-5s %-5s: RankIC_mean=%.4f  RankICIR=%.3f" % (name, split, ic_mean, icir))
    return alpha


if __name__ == "__main__":
    combine()
