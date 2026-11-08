# P0 审计与数据接口清单

> 记录 2026-11-08 数据可得性审计结果与已冻结决策。方法借鉴、非复现。

## 1. 方法借鉴框架（已冻结）

- 主借鉴：华泰证券多因子系列（单因子测试口径 / 因子合成 / 结构化风险模型）
- 组合构建：长江证券 沪深300多因子增强策略
- 组合优化：光大证券 单因子组合优化在指数增强中的应用
- 归因：渤海证券 沪深300指数增强组合业绩归因分析
- 仅借口径，不逐项复制数值。

## 2. JQData 能力核验

本地无 `jqdatasdk`，数据由聚宽研究环境 notebook 下载（项目三 download_data.py 模板）。

已确认可用：`get_price`(日频 OHLCV+factor)、`get_index_stocks`(历史成分)、`get_security_info`(上市日)、`get_extras('is_st')`、`get_fundamentals`(valuation 市值)。

三个缺口（P1 补）：
1. 申万行业 sw_l1：`get_industry` 返回空（免费账户不可得，申万/证监会/聚宽均无），已确认降级为仅市值中性化、去行业偏离约束；
2. 停牌/涨跌停：`get_price` 未返回 `high_limit/low_limit/paused`，P6 可执行回测受限；
3. 财务+公告日：未下载，需 `get_fundamentals(query(balance/income/cashflow...))` + `pubDate`。

补数据脚本：`src/data/download_missing.py`（先跑 `probe()` 锁定字段名）。

## 3. 泄漏风险清单

1. `data/processed` 原有 `eligible_core/paper_csi300.parquet` 是项目三产物（依赖项目三 label 与 paused/limit 字段），口径不符，**已删除**；项目四需用自己的数据/定义重建。
2. 项目三"论文对齐"的停牌/涨跌停过滤实为空操作（字段未下下来），勿搬其结论。
3. 财务因子必须按公告日(`pubDate`)对齐；`get_fundamentals(date=)` 本身按公告日取数。
4. 行业必须时点正确，不能拿当前行业回填历史。
5. ST/市值是半年频快照 forward-fill，半年内状态变化不反映。
6. 复权：`get_price(fq='pre')` 返回价已前复权，`factor` 为复权因子，构造因子不能重复复权。

## 4. 数据字段清单

见 `configs/data.yaml`（价格 / 可交易 / 截面 / 财务四类字段）。

## 5. 待办

- P1：聚宽跑 `download_missing.py` 的 `probe()` → 贴回输出 → 锁定字段名 → 跑 `download_*()`。
- P2：9 因子构造 + 截面相关性去冗余（收敛 6~8）+ 中性化。
- 小样本抽查验收：本轮按用户决定跳过，时点正确性由 P1 数据审计 + P2 因子错位检查兜底。
