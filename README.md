# 沪深300多因子 Alpha 与指数增强

以历史时点真实的沪深300成分股为研究对象，建立从单因子检验、多因子 Alpha 合成，到带基准主动权重、行业偏离和换手约束的指数增强组合构建的完整研究链路。

## 研究结论

- 6 个候选因子（价值 EP、质量 ROE、成长营收同比、反转、低波特异波动、流动性换手）在训练期（2008–2019）具有稳定的横截面预测能力（RankIC 0.03~0.05，RankICIR 0.38~0.56）。
- 等权合成后综合 alpha 的训练期 RankICIR 达 0.94，显著高于任一单因子。
- 约束型指数增强在 2008–2022 取得正超额（本地回测年化 4.86%、超额 2.58%/年；聚宽独立回测信息比率 0.53），并在 2023–2025 全新样本外继续有效（超额约 5%/年）。
- 价值/质量因子在 2021–2022 存在明显失效期。

## 研究链路

```text
历史沪深300成分股与时点正确数据
→ 因子构造与质量检查
→ IC / RankIC / 分组收益 / 单调性 / 衰减
→ 因子相关性、冗余处理与中性化
→ 等权 / IC加权 / 风险调整的 Alpha 合成
→ 简单 Top-K 组合与约束型指数增强组合
→ 可执行回测、交易成本与换手分析
→ 风格暴露、行业偏离与收益归因
→ 样本外稳健性和正式报告
```

## 目录结构

```
├── configs/          研究配置（因子、参数、切分）
├── data/             原始数据与中间数据（gitignore，不提交）
├── docs/             研究协议、因子字典、数据字典、研究报告
├── src/              代码
│   ├── data/         数据下载与面板构建
│   ├── factors/      因子计算、预处理、单因子检验
│   ├── portfolio/    Alpha 合成、组合构建、约束优化
│   ├── backtest/     回测与聚宽回测版
│   └── attribution/  收益归因与稳健性
├── results/          结果表与图表
├── README.md
└── WORKFLOW.md
```

## 复现

```bash
# 1. 基础面板（行情/财务/行业/市值/成分 -> parquet）
python src/data/build_base.py

# 2. 因子计算 + 预处理（中性化）
python src/factors/factors.py
python src/factors/preprocess.py

# 3. 单因子检验
python src/factors/ic_test.py
python src/factors/grouping_test.py

# 4. Alpha 合成
python src/portfolio/alpha.py

# 5. 组合构建与回测
python src/portfolio/portfolio.py
python src/portfolio/optimize.py
python src/backtest/backtest.py

# 6. 收益归因
python src/attribution/attribution.py
```

聚宽回测版见 `src/backtest/joinquant_backtest.py`。

## 数据与因子

- 因子定义：`docs/factor_dictionary.md`
- 数据字段：`docs/data_dictionary.md`
- 研究协议：`docs/research_protocol.md`

## 免责声明

本项目仅用于量化研究方法展示，不构成任何投资建议。
