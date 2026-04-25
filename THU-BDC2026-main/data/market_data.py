"""爬取市场情绪数据（akshare）"""
import akshare as ak
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta


def fetch_macro_shibor():
    """Shibor 银行间拆借利率"""
    try:
        df = ak.macro_china_shibor_all()
        df = df.rename(columns={
            '日期': '日期',
            'O/N-定价': 'SHIBOR_ON',
            '1W-定价': 'SHIBOR_1W',
            '1M-定价': 'SHIBOR_1M',
        })
        df['日期'] = pd.to_datetime(df['日期'])
        cols = ['日期'] + [c for c in ['SHIBOR_ON', 'SHIBOR_1W', 'SHIBOR_1M'] if c in df.columns]
        return df[cols]
    except Exception as e:
        print(f"  Shibor获取失败: {e}")
        return None


def fetch_market_vix():
    """中国波指 QVIX (50ETF期权隐含波动率)"""
    try:
        df = ak.index_option_50etf_qvix()
        df = df.rename(columns={'date': '日期', 'close': 'QVIX'})
        df['日期'] = pd.to_datetime(df['日期'])
        return df[['日期', 'QVIX']]
    except Exception as e:
        print(f"  QVIX获取失败: {e}")
        return None


def fetch_limit_up_stats(start_date, end_date):
    """获取每日涨停股票数量统计"""
    date_range = pd.date_range(start=start_date, end=end_date, freq='B')
    records = []
    for d in date_range:
        try:
            d_str = d.strftime('%Y%m%d')
            zt_df = ak.stock_zt_pool_em(date=d_str)
            if zt_df is not None and len(zt_df) > 0:
                records.append({
                    '日期': d,
                    '涨停家数': len(zt_df),
                    '连板均值': zt_df['连板数'].astype(float).mean() if '连板数' in zt_df.columns else 0,
                    '封单资金均值': zt_df['封单资金'].astype(float).mean() if '封单资金' in zt_df.columns else 0,
                })
            else:
                records.append({'日期': d, '涨停家数': 0, '连板均值': 0, '封单资金均值': 0})
        except Exception:
            records.append({'日期': d, '涨停家数': 0, '连板均值': 0, '封单资金均值': 0})
    return pd.DataFrame(records)


def merge_all_market_data(start_date, end_date):
    """合并所有市场数据并按日期对齐"""
    dfs = {}

    print("获取 Shibor ...")
    dfs['shibor'] = fetch_macro_shibor()

    print("获取 QVIX ...")
    dfs['vix'] = fetch_market_vix()

    print(f"获取涨停数据 ({start_date} ~ {end_date}) ...")
    dfs['limit_up'] = fetch_limit_up_stats(start_date, end_date)

    # 从涨停数据获取完整日期范围
    date_col = dfs['limit_up'][['日期']].copy()
    date_col['日期'] = pd.to_datetime(date_col['日期'])

    for key, df in dfs.items():
        if df is not None and len(df) > 0:
            df['日期'] = pd.to_datetime(df['日期'])
            date_col = date_col.merge(df, on='日期', how='left')

    # 前向填充月度/季度数据
    date_col = date_col.sort_values('日期').ffill().bfill()

    print(f"合并后共 {len(date_col)} 个交易日, {len(date_col.columns)-1} 个特征")
    return date_col


def main():
    input_dir = "data"
    output_path = os.path.join(input_dir, "market_data.csv")

    # 从 stock_data.csv 确定日期范围
    stock_path = os.path.join(input_dir, "stock_data.csv")
    if os.path.exists(stock_path):
        stock_df = pd.read_csv(stock_path)
        stock_df['日期'] = pd.to_datetime(stock_df['日期'])
        start_date = stock_df['日期'].min().strftime('%Y-%m-%d')
        end_date = stock_df['日期'].max().strftime('%Y-%m-%d')
    else:
        start_date = "2024-01-01"
        end_date = "2026-04-20"

    print(f"数据日期范围: {start_date} ~ {end_date}")

    result = merge_all_market_data(start_date, end_date)
    result.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"市场数据已保存到: {output_path}")


if __name__ == "__main__":
    main()
