# -*- coding: utf-8 -*-
"""
项目四 P1 补数据下载脚本 —— 在聚宽研究环境运行（整段粘贴到 notebook 跑）。

补齐 configs/data.yaml 里缺失的三类数据（区间与现有行情一致 2008-01 ~ 2022-12）：
  1. 财务数据（估值 + 三大报表 + 财务指标），点状时点 → 价值/质量/成长因子；
  2. 申万一级行业（时点正确）→ 行业中性化 + 行业主动偏离约束；
  3. 停牌 / 涨跌停（可交易状态）→ P6 可执行回测。

【直接下载】__main__ 依次下载财务/行业/停涨跌停。若某张表报错或全空，
按报错信息修正对应 download_*() 字段名后重跑该函数即可（断点续跑会跳过已完成表）。
probe() 保留作诊断用，非必须。

产出 CSV（下载到本地 data/raw/）：
  status_financial_valuation.csv  status_financial_indicator.csv
  status_financial_balance.csv    status_financial_income.csv
  status_financial_cashflow.csv
  status_industry.csv（覆盖原来全空的那份）
  status_tradability.csv

断点续跑：对应 CSV 已存在则跳过。
"""

import time
import pandas as pd

SAMPLE_START = "2008-01-01"
SAMPLE_END = "2022-12-31"
PRIMARY = "000300.XSHG"

BATCH = 300          # 每批股票数（控制接口压力）
SLEEP = 0.3          # 批次间隔（秒）


# ============ 0. 快照日 ============
def snapshot_dates(freq="month"):
    """返回快照日期列表。财务用月末（月频调仓对应），行业/停涨跌用半年末。"""
    bench = get_price(PRIMARY, SAMPLE_START, SAMPLE_END, 'daily',
                      ['close'], fq='pre', panel=False)
    cal = pd.to_datetime(bench['close'].sort_index().index)
    if freq == "month":
        last = {}
        for d in cal:
            last[d.strftime('%Y-%m')] = d
        return [last[k].strftime('%Y-%m-%d') for k in sorted(last)]
    # 半年末最后一个交易日
    out = []
    for y in range(2008, 2023):
        for m in (6, 12):
            month_end = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
            cand = cal[cal <= month_end]
            if len(cand):
                out.append(cand[-1].strftime('%Y-%m-%d'))
    return out




def ensure_universe():
    """聚宽里生成 universe.csv（历史成分并集）；已存在则跳过。"""
    if _exists("universe.csv"):
        return
    bench = get_price(PRIMARY, SAMPLE_START, SAMPLE_END, 'daily', ['close'], fq='pre', panel=False)
    cal = pd.to_datetime(bench['close'].sort_index().index)
    rows = []
    for y in range(2008, 2023):
        for m in range(1, 13):
            month_end = pd.Timestamp(y, m, 1) + pd.offsets.MonthEnd(0)
            cand = cal[cal <= month_end]
            if len(cand):
                d = cand[-1].strftime('%Y-%m-%d')
                for c in get_index_stocks(PRIMARY, d):
                    rows.append({'code': c, 'date': d, 'universe': 'csi300'})
    pd.DataFrame(rows).to_csv("universe.csv", index=False)
    print("universe.csv 已生成：%d 行" % len(rows))

# ============ 1. 探针：先搞清账户能返回什么 ============
def probe():
    print("== 探针1: get_price 停牌/涨跌停字段 ==")
    try:
        df = get_price("000001.XSHE", "2021-06-01", "2021-06-05", 'daily',
                       ['open', 'high', 'low', 'close', 'volume', 'money',
                        'factor', 'high_limit', 'low_limit', 'paused'],
                       fq='pre', panel=False)
        print("  返回列:", list(df.columns))
        print(df.head(2).to_string())
    except Exception as e:
        print("  报错:", e)

    print("\n== 探针2: get_industry 返回结构 ==")
    for flag in (False, True):
        try:
            r = get_industry("000001.XSHE", "2021-06-30", df=flag)
            print("  df=%s -> %r" % (flag, r))
        except Exception as e:
            print("  df=%s 报错: %s" % (flag, e))

    print("\n== 探针3: 财务表字段（估值/指标/三表 + pubDate 公告日）==")
    for name, q in [
        ("valuation", query(valuation.code, valuation.pe_ratio, valuation.pb_ratio,
                            valuation.market_cap, valuation.circulating_market_cap)),
        ("indicator", query(indicator.code, indicator.roe, indicator.eps,
                            indicator.gross_profit_margin)),
        ("balance",   query(balance.code, balance.total_assets, balance.total_liability,
                            balance.total_owner_equities, balance.pubDate)),
        ("income",    query(income.code, income.total_operating_revenue,
                            income.net_profit, income.pubDate)),
        ("cash_flow", query(cash_flow.code, cash_flow.net_operate_cash_flow,
                            cash_flow.pubDate)),
    ]:
        try:
            df = get_fundamentals(q, date="2021-06-30")
            print("  %-10s 列: %s" % (name, list(df.columns)))
        except Exception as e:
            print("  %-10s 报错: %s" % (name, e))

    print("\n>>> 把上面探针输出贴回来，据此锁定字段名，再跑 download_*()。")


# ============ 2. 财务数据（点状时点） ============
# 说明：get_fundamentals(query, date=X) 返回「截至 X 日已公告」的最新报表，
# 天然满足"财务因子只能在公告日之后进入"的时点纪律；pubDate 用于核对公告日。
# 月频快照覆盖调仓日，因子构造时按 date 对齐即可。
def _exists(f):
    import os
    return os.path.exists(f)


def _download_financial_table(name, q_builder, out_file):
    if _exists(out_file):
        print("  %s 已存在，跳过" % out_file)
        return
    dates = snapshot_dates("month")
    all_codes = sorted(pd.read_csv("universe.csv")['code'].unique())
    rows = []
    for d in dates:
        for i in range(0, len(all_codes), BATCH):
            sub = all_codes[i:i + BATCH]
            try:
                q = q_builder(sub)
                df = get_fundamentals(q, date=d)
                if df is not None and len(df):
                    df = df.copy()
                    df["date"] = d
                    rows.append(df)
            except Exception as e:
                print("    %s %s 批 %d 报错: %s" % (name, d, i, e))
            time.sleep(SLEEP)
        print("  %s %s 完成，累计 %d 行" % (name, d, sum(len(r) for r in rows)))
    if rows:
        pd.concat(rows, ignore_index=True).to_csv(out_file, index=False)
    else:
        pd.DataFrame().to_csv(out_file, index=False)
    print("  %s 已保存 -> %s" % (name, out_file))


def download_financial():
    print("下载财务数据（估值/指标/三表，月频点状时点）...")
    _download_financial_table("valuation", lambda codes: query(
        valuation.code, valuation.pe_ratio, valuation.pb_ratio,
        valuation.market_cap, valuation.circulating_market_cap,
    ).filter(valuation.code.in_(codes)), "status_financial_valuation.csv")

    _download_financial_table("indicator", lambda codes: query(
        indicator.code, indicator.roe, indicator.eps,
        indicator.gross_profit_margin, indicator.net_profit_margin,
    ).filter(indicator.code.in_(codes)), "status_financial_indicator.csv")

    _download_financial_table("balance", lambda codes: query(
        balance.code, balance.total_assets, balance.total_liability,
        balance.total_owner_equities, balance.pubDate,
    ).filter(balance.code.in_(codes)), "status_financial_balance.csv")

    _download_financial_table("income", lambda codes: query(
        income.code, income.total_operating_revenue, income.net_profit,
        income.pubDate,
    ).filter(income.code.in_(codes)), "status_financial_income.csv")

    _download_financial_table("cash_flow", lambda codes: query(
        cash_flow.code, cash_flow.net_operate_cash_flow, cash_flow.pubDate,
    ).filter(cash_flow.code.in_(codes)), "status_financial_cashflow.csv")


# ============ 3. 申万一级行业（时点正确） ============
# 说明：申万行业会调整，需按日期取时点行业。半年末快照 + 逐票 get_industry。
# 多种取法依次尝试，返回第一个非空；若仍全空，说明 get_industry 不可用，需换接口。
def _get_sw_l1(c, d):
    # dict 形式：遍历所有行业分类，优先 sw_l1 -> zjw -> jq_l1 -> 任意非空
    try:
        r = get_industry(c, d)
        if isinstance(r, dict):
            for key in ('sw_l1', 'zjw', 'jq_l1', 'sw_l2'):
                v = r.get(key)
                if isinstance(v, dict) and v.get('industry_name'):
                    return v['industry_name']
            for v in r.values():
                if isinstance(v, dict) and v.get('industry_name'):
                    return v['industry_name']
    except Exception:
        pass
    # df 形式
    try:
        r = get_industry(c, d, df=True)
        if hasattr(r, 'to_dict') and len(r):
            rec = r.to_dict('records')[0]
            for k in ('industry_name', 'name', 'sw_l1', 'industry', 'index_name'):
                if k in rec and rec[k]:
                    return rec[k]
    except Exception:
        pass
    return None


def download_industry():
    out_file = "status_industry.csv"
    if _exists(out_file):
        print("  %s 已存在，跳过" % out_file)
        return
    dates = snapshot_dates("half")
    all_codes = sorted(pd.read_csv("universe.csv")['code'].unique())
    rows = []
    nonempty = 0
    for d in dates:
        for c in all_codes:
            sw = _get_sw_l1(c, d)
            if sw:
                nonempty += 1
            rows.append({'date': d, 'code': c, 'industry_sw_l1': sw})
            time.sleep(0.05)  # 逐票调用，间隔小；若被限流则加大
        print("  industry %s 完成，累计非空 %d" % (d, nonempty))
    pd.DataFrame(rows).to_csv(out_file, index=False)
    print("  行业下载完成：共 %d 行，非空 %d（若非空率过低，get_industry 仍不可用，需换接口）"
          % (len(rows), nonempty))


# ============ 4. 停牌 / 涨跌停（可交易状态） ============
# 说明：若 probe 显示 get_price 不返回 high_limit/low_limit/paused，
# 则本函数只落盘能拿到的字段，并在注释里记录缺口（供 P6 决策是否降级）。
def download_tradability():
    out_file = "status_tradability.csv"
    if _exists(out_file):
        print("  %s 已存在，跳过" % out_file)
        return
    fields = ['high_limit', 'low_limit', 'paused']
    # 先探一只，确认字段是否可用
    try:
        t = get_price("000001.XSHE", "2021-06-01", "2021-06-03", 'daily',
                      fields, fq='pre', panel=False)
        avail = [f for f in fields if f in t.columns]
        print("  可交易字段实际可用: %s" % avail)
    except Exception as e:
        print("  get_price 可交易字段探测报错: %s" % e)
        avail = []
    if not avail:
        print("  >>> 免费账户拿不到停牌/涨跌停字段，本文件留空；P6 需改用其他口径或降级。")
        pd.DataFrame(columns=['date', 'code', 'field', 'value']).to_csv(out_file, index=False)
        return

    years = range(2008, 2023)
    all_codes = sorted(pd.read_csv("universe.csv")['code'].unique())
    first = True
    for y in years:
        start, end = "%d-01-01" % y, "%d-12-31" % y
        for i in range(0, len(all_codes), BATCH):
            sub = all_codes[i:i + BATCH]
            try:
                raw = get_price(sub, start, end, 'daily', avail, fq='pre', panel=False)
                if raw is None or len(raw) == 0:
                    continue
                if 'code' not in raw.columns:
                    continue
                date_col = next((c for c in ('time', 'date', 'datetime') if c in raw.columns), None)
                if date_col is None:
                    raw = raw.reset_index()
                    date_col = next((c for c in ('time', 'date', 'datetime') if c in raw.columns),
                                    raw.columns[0])
                long = raw.melt(id_vars=[date_col, 'code'], value_vars=avail,
                                var_name='field', value_name='value')
                long = long.rename(columns={date_col: 'date'})
                long.to_csv(out_file, mode='a', header=first, index=False)
                first = False
            except Exception as e:
                print("    tradability %d 批 %d 报错: %s" % (y, i, e))
            time.sleep(SLEEP)
        print("  tradability %d 完成" % y)


# 直接下载
ensure_universe()
download_financial()
download_industry()
download_tradability()
print("全部下载完成")
