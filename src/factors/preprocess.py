# -*- coding: utf-8 -*-
"""P2 预处理：截面 MAD 去极值 + z-score + 行业/市值中性化。

对每个调仓日（截面）：
  1. 每个因子 MAD 3 倍缩尾（median ± 3*1.4826*MAD 截断）；
  2. 截面 z-score 标准化；
  3. 对行业哑变量 + log(市值) 做 OLS 回归取残差（中性化）。

输出 data/processed/factor_processed.parquet（长表 date,code,factor,value）。
"""

import numpy as np
import pandas as pd

PROC = "data/processed"


def winsorize_mad(s, n=3.0):
    """MAD 缩尾：超出 median ± n*1.4826*MAD 的值截断到边界。"""
    med = s.median()
    mad = (s - med).abs().median() * 1.4826
    if mad == 0 or np.isnan(mad):
        return s
    lo, hi = med - n * mad, med + n * mad
    return s.clip(lo, hi)


def _neutralize(y, ind, log_mcap):
    """把 y 对行业哑变量 + log市值 回归，返回残差（NaN 处保持 NaN）。"""
    df = pd.DataFrame({"y": y, "ind": ind, "lm": log_mcap})
    df = df.dropna(subset=["y"])
    df["ind"] = df["ind"].fillna("缺失")
    X = pd.get_dummies(df["ind"], prefix="ind", drop_first=True).astype(float)
    X["lm"] = df["lm"]
    X = X.dropna()
    if X.shape[0] < 20 or X.shape[1] == 0:
        return y - y.mean()
    idx = X.index
    X = np.column_stack([np.ones(len(X)), X.values])
    yy = df.loc[idx, "y"].values
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid_vals = yy - X @ beta
    out = pd.Series(np.nan, index=y.index)
    out.loc[idx] = resid_vals
    return out


def preprocess():
    panel = pd.read_parquet(PROC + "/factor_panel.parquet")
    ind = pd.read_parquet(PROC + "/industry_map.parquet")["industry_sw_l1"]
    mcap = pd.read_parquet(PROC + "/market_cap.parquet")

    out = []
    for d, grp in panel.groupby("date"):
        wide = grp.pivot(index="code", columns="factor", values="value")

        # 1. MAD 去极值
        wide = wide.apply(winsorize_mad)

        # 2. z-score（截面）
        wide = (wide - wide.mean()) / wide.std()

        # 3. 中性化
        if d in mcap.index:
            lm = np.log(mcap.loc[d]).reindex(wide.index)
        else:
            lm = pd.Series(np.nan, index=wide.index)
        ind_d = ind.reindex(wide.index)
        for col in wide.columns:
            wide[col] = _neutralize(wide[col], ind_d, lm)

        long = wide.reset_index().melt(id_vars="code", var_name="factor", value_name="value")
        long["date"] = d
        out.append(long)

    result = pd.concat(out, ignore_index=True)
    result.to_parquet(PROC + "/factor_processed.parquet", index=False)
    print("factor_processed:", result.shape)
    print("各因子处理后 count:")
    print(result.groupby("factor")["value"].count().to_string())
    print("各因子处理后 mean/std:")
    print(result.groupby("factor")["value"].agg(mean="mean", std="std").round(4).to_string())
    return result


if __name__ == "__main__":
    preprocess()
