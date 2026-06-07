"""
回测日历：从 stock_data.csv 的真实交易日切分自然周，并为每周计算预测基准日。

设计要点：
- "一周"按自然周（ISO 周一~周五）划分；遇到节假日短周就按该周实际交易日。
- 每周的 base_date = 严格早于该周首个交易日的最后一个交易日。
  模型在 base_date 决策、d1 开盘买入、dk 开盘卖出，与 score_self.py 的
  "首日开盘买、末日开盘卖" 严格对齐（详见架构审查结论）。
- 只保留 5 日预测窗口完整落在数据范围内的周（最后一周若 5 日窗口越界则丢弃）。
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def load_trading_days():
    """返回排序去重后的全部交易日（pd.Timestamp 列表）。"""
    df = pd.read_csv(config.STOCK_DATA_PATH, usecols=['日期'])
    days = pd.to_datetime(df['日期']).dt.normalize().drop_duplicates().sort_values()
    return list(days)


def build_weeks(year=None, include_short=True):
    """切分指定年份的自然周，返回回测周清单。

    每个周是一个 dict：
      {
        'week_idx': 1-based 连续序号,
        'iso_year', 'iso_week': ISO 年/周,
        'trading_days': [d1, ..., dk] 该周交易日(Timestamp),
        'd1', 'dk': 首/末交易日,
        'base_date': 预测基准日(该周首日前一交易日),
        'n_days': 该周交易日数,
      }
    """
    if year is None:
        year = config.BACKTEST_YEAR
    all_days = load_trading_days()
    if not all_days:
        return []

    # 交易日 -> 在全序列中的位置，便于查 base_date
    pos = {d: i for i, d in enumerate(all_days)}

    # 只取目标年份的交易日，按 ISO (year, week) 分组
    s = pd.Series(all_days)
    iso = s.dt.isocalendar()
    frame = pd.DataFrame({'date': s.values, 'iso_year': iso['year'].values,
                          'iso_week': iso['week'].values})
    frame = frame[frame['date'].dt.year == year]

    weeks = []
    idx = 0
    for (iso_year, iso_week), grp in frame.groupby(['iso_year', 'iso_week'], sort=True):
        tdays = sorted(grp['date'].tolist())
        n_days = len(tdays)
        if n_days == 0:
            continue
        if not include_short and n_days < 5:
            continue

        d1 = tdays[0]
        dk = tdays[-1]

        # base_date：d1 在全序列中的前一个交易日
        d1_pos = pos[d1]
        if d1_pos == 0:
            # 没有更早的交易日，无法构造历史窗口，跳过
            continue
        base_date = all_days[d1_pos - 1]

        idx += 1
        weeks.append({
            'week_idx': idx,
            'iso_year': int(iso_year),
            'iso_week': int(iso_week),
            'trading_days': tdays,
            'd1': d1,
            'dk': dk,
            'base_date': base_date,
            'n_days': n_days,
        })

    return weeks


def filter_weeks(weeks, selected=None):
    """按 config.WEEKS（或显式 selected）过滤周清单。selected/WEEKS 是 1-based 序号列表。"""
    sel = selected if selected is not None else config.WEEKS
    if sel is None:
        return weeks
    sel = set(sel)
    return [w for w in weeks if w['week_idx'] in sel]


def fmt(d):
    """Timestamp -> 'YYYY-MM-DD'。"""
    return pd.Timestamp(d).strftime('%Y-%m-%d')


def main():
    weeks = build_weeks(include_short=config.INCLUDE_SHORT_WEEKS)
    print(f"{config.BACKTEST_YEAR} 年共 {len(weeks)} 个回测周\n")
    print(f"{'周序':>4}  {'ISO周':>8}  {'基准日':>12}  {'起始':>12}  {'结束':>12}  {'交易日':>6}")
    print('-' * 64)
    for w in weeks:
        print(f"{w['week_idx']:>4}  {w['iso_year']}-W{w['iso_week']:<3}  "
              f"{fmt(w['base_date']):>12}  {fmt(w['d1']):>12}  {fmt(w['dk']):>12}  "
              f"{w['n_days']:>4}天")


if __name__ == '__main__':
    main()
