"""合并所有数据源：基础量价+估值 + 行业 + 季频基本面 + 资金流向 + 龙虎榜 + 宏观/情绪"""
import pandas as pd
import numpy as np
import os
from pathlib import Path


DATA_DIR = Path("data")


def load_base():
    """加载增强版基础量价数据（含PE/PB）"""
    path = DATA_DIR / "stock_data.csv"
    if not path.exists():
        raise FileNotFoundError(f"请先运行 get_stock_data.py: {path}")
    df = pd.read_csv(path)
    df['日期'] = pd.to_datetime(df['日期'])
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    print(f"基础数据: {len(df)} 行, {df['股票代码'].nunique()} 只股票")
    print(f"  列: {list(df.columns)}")
    return df


def load_and_join_industry(base):
    """加载行业分类并合并"""
    path = DATA_DIR / "stock_industry.csv"
    if not path.exists():
        print("  行业数据不存在，跳过")
        base['行业名称'] = '未知'
        return base

    industry = pd.read_csv(path)
    industry['股票代码'] = industry['股票代码'].astype(str).str.zfill(6)
    base = base.merge(industry[['股票代码', '行业名称']], on='股票代码', how='left')
    base['行业名称'] = base['行业名称'].fillna('未知')
    print(f"  合并行业: {base['行业名称'].nunique()} 个行业")
    return base


def load_and_join_fundamental(base):
    """加载季频基本面，前向填充到日频"""
    path = DATA_DIR / "stock_fundamental.csv"
    if not path.exists():
        print("  季频基本面数据不存在，跳过")
        return base

    fund = pd.read_csv(path)
    fund['股票代码'] = fund['股票代码'].astype(str).str.zfill(6)
    fund['统计日期'] = pd.to_datetime(fund['统计日期'])
    fund = fund.sort_values(['股票代码', '统计日期'])

    # 数值列
    val_cols = [c for c in ['ROE_avg', '净利率', 'EPS_TTM', '净利同比', '权益同比', '资产负债率']
                if c in fund.columns]
    if not val_cols:
        print("  无有效基本面板列")
        return base

    # 为每只股票构建日频映射
    all_dates = base['日期'].unique()
    all_dates = pd.to_datetime(all_dates)
    all_dates = np.sort(all_dates)

    merged_rows = []
    for code, group in base.groupby('股票代码'):
        stock_fund = fund[fund['股票代码'] == code]
        if stock_fund.empty:
            group_fund_cols = {c: np.nan for c in val_cols}
        else:
            # 为每个交易日找最近一季度的数据 (statDate <= tradeDate)
            stock_fund = stock_fund.set_index('统计日期').sort_index()
            group_fund_cols = {}
            for col in val_cols:
                s = stock_fund[col].dropna()
                if len(s) == 0:
                    group_fund_cols[col] = np.nan
                else:
                    # 前向填充
                    s_reindexed = s.reindex(all_dates, method='ffill')
                    group_fund_cols[col] = s_reindexed

        for col, series in group_fund_cols.items():
            if isinstance(series, pd.Series):
                idx_map = {d: v for d, v in zip(all_dates, series.values)}
                group[col] = group['日期'].map(idx_map)
            else:
                group[col] = series

        merged_rows.append(group)

    result = pd.concat(merged_rows, ignore_index=True)
    # 对仍然缺失的值填0
    for c in val_cols:
        if c in result.columns:
            result[c] = result[c].fillna(0)
    print(f"  合并季频基本面: {val_cols}")
    return result


def load_and_join_market(base):
    """加载宏观/市场情绪数据"""
    path = DATA_DIR / "market_data.csv"
    if not path.exists():
        print("  市场数据不存在，跳过")
        return base

    market = pd.read_csv(path)
    market['日期'] = pd.to_datetime(market['日期'])

    base = base.merge(market, on='日期', how='left')
    mkt_cols = [c for c in market.columns if c != '日期']
    for c in mkt_cols:
        if c in base.columns:
            base[c] = base[c].fillna(method='ffill')  # 前向填充非交易日

    # 对仍然缺失的值填0
    for c in mkt_cols:
        if c in base.columns:
            base[c] = base[c].fillna(0)
    print(f"  合并宏观/情绪: {mkt_cols}")
    return base


def main():
    output_path = DATA_DIR / "stock_data_merged.csv"

    print("=" * 60)
    print("开始合并多维数据...")
    print("=" * 60)

    # 1. 加载基础数据
    base = load_base()

    # 2. 行业分类
    print("\n[1/3] 合并行业分类...")
    base = load_and_join_industry(base)

    # 3. 季频基本面
    print("\n[2/3] 合并季频基本面...")
    base = load_and_join_fundamental(base)

    # 4. 宏观/市场情绪
    print("\n[3/3] 合并宏观与市场情绪...")
    base = load_and_join_market(base)

    # 保存
    base.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n{'=' * 60}")
    print(f"合并完成: {output_path}")
    print(f"总行数: {len(base)}, 股票数: {base['股票代码'].nunique()}")
    print(f"总列数: {len(base.columns)}")
    print(f"日期范围: {base['日期'].min().date()} ~ {base['日期'].max().date()}")
    print(f"所有列: {list(base.columns)}")


if __name__ == "__main__":
    main()
