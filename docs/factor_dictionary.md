# 因子字典

统一约定：每个因子经过符号调整（sign），使「值越大、预期收益越高」。

## 最终采用因子（6 个）

| 因子 | 英文名 | 公式 | 方向 | 数据来源 | 经济含义 |
|---|---|---|---|---|---|
| 价值 | value_ep | 净利润 / 总市值 | + | income.net_profit / valuation.market_cap | 买便宜的（盈利收益率） |
| 质量 | quality_roe | 净利润 / 净资产 | + | income.net_profit / balance.total_owner_equities | 买赚钱能力强的 |
| 成长 | growth_revenue | 营业收入同比增速 | + | income.total_operating_revenue（本期 / 去年同期 − 1） | 买增长快的 |
| 反转 | reversal_1m | −(近1月收益) | − | close[-1] / close[-21] − 1 | 买近期超跌的（短期反转） |
| 低波 | lowvol_idio | −(60日特异波动率) | − | 残差收益 60 日标准差 | 买波动小的 |
| 流动性 | liquidity_turnover | −(近20日日均换手率) | − | 成交额 / 流通市值 | 买交易清淡的 |

## 被淘汰的候选因子（训练期 IC 不显著）

| 因子 | 公式 | 淘汰原因 |
|---|---|---|
| momentum_12_1 | 过去12月收益(剔除最近1月) | 训练期 RankICIR ≈ −0.03（A股大盘 12 月动量失效） |
| quality_accruals | (净利润 − 经营现金流) / 总资产 | 训练期 RankICIR ≈ −0.04（应计异象在大盘股很弱） |
| quality_leverage | 净资产 / 总资产 | 训练期 RankICIR ≈ 0.12（偏弱） |

## 预处理

对每个因子、每个调仓日截面依次执行：

1. **去极值**：MAD 3 倍缩尾（超出 median ± 3×1.4826×MAD 截断到边界）；
2. **标准化**：截面 z-score（均值 0、标准差 1）；
3. **中性化**：对行业哑变量 + log(市值) 做截面回归，取残差，剥离行业与市值效应。

## Alpha 合成

6 个中性化因子**等权平均**后再次 z-score，得到综合 alpha。
