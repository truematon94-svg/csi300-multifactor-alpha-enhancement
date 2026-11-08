# 数据字典

## 数据来源与范围

- 数据源：聚宽（JQData），研究环境下载。
- 样本区间：2008-01-01 ~ 2022-12-31。
- 频率：日频行情，月度调仓。

## 原始数据（data/raw/）

| 文件 | 字段 | 说明 |
|---|---|---|
| stock_daily_YYYY.csv | date, code, field, value | 日频行情长表，field ∈ {open, high, low, close, volume, money, factor} |
| index_daily.csv | date, close, volume, money, index_code | 指数日行情（含 000300/000905/000906） |
| universe.csv | code, date, universe | 时点成分（csi300 / csi800） |
| status_listing.csv | code, start_date | 上市日期 |
| status_st.csv | code, date, is_st | ST 状态（半年频快照） |
| status_marketcap.csv | code, market_cap, circulating_market_cap, date | 市值（半年频快照） |
| status_financial_valuation.csv | code, pe_ratio, pb_ratio, market_cap, circulating_market_cap, date | 估值（月频点状时点） |
| status_financial_indicator.csv | code, roe, eps, gross_profit_margin, net_profit_margin, date | 财务指标（月频点状时点） |
| status_financial_balance.csv | code, total_assets, total_liability, total_owner_equities, pubDate, date | 资产负债表（月频点状时点） |
| status_financial_income.csv | code, total_operating_revenue, net_profit, pubDate, date | 利润表（月频点状时点） |
| status_financial_cashflow.csv | code, net_operate_cash_flow, pubDate, date | 现金流量表（月频点状时点） |

## 中间数据（data/processed/，由脚本生成）

| 文件 | 说明 |
|---|---|
| close.parquet / volume.parquet | 日频行情宽表（date × code） |
| financial.parquet | 财务长表（code, date, table, field, value） |
| industry_sw_l1_map.csv | 静态申万一级行业映射（code → industry_sw_l1，5208 只） |
| market_cap.parquet | 市值（半年频 → 日频 forward-fill） |
| universe_csi300.parquet | 沪深300 时点成分（ym, code） |
| factor_panel.parquet | 原始因子值（长表） |
| factor_processed.parquet | 中性化后因子（长表） |
| ic_series.parquet / ic_summary.parquet | 逐日 IC 与汇总 |
| group_returns.parquet | 分组收益 |
| alpha_panel.parquet | 合成 alpha（等权/IC加权/风险调整） |
| portfolio_returns.parquet | 基线组合收益 |
| backtest_net.parquet | 回测净收益（含成本） |

## 点状时点（时点正确）纪律

- 历史成分：每月时点成分，不含当前成分回填，保留退市股。
- 财务数据：`get_fundamentals(date=)` 只返回「截至该日已公告」的最新报表（按 pubDate 对齐）。
- 行情：前复权（fq='pre'），因子只用信号日及以前的数据。
