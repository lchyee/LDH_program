"""爬取沪深300成分股的季频基本面数据（ROE/成长性/偿债能力）"""
import baostock as bs
import pandas as pd
import numpy as np
import os
import time
from tqdm import tqdm


def login():
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(f"baostock登录失败: {lg.error_msg}")
    print("baostock登录成功")


def logout():
    bs.logout()


def query_to_df(rs):
    """将baostock查询结果转为DataFrame"""
    if rs.error_code != '0':
        return pd.DataFrame()
    data = []
    while (rs.error_code == '0') & rs.next():
        data.append(rs.get_row_data())
    if not data:
        return pd.DataFrame()
    return pd.DataFrame(data, columns=rs.fields)


def get_stock_fundamental(bs_code):
    """获取单只股票的所有季频基本面数据并合并"""
    # 盈利能力 (ROE, EPS, 净利润率等)
    profit = query_to_df(bs.query_profit_data(code=bs_code))
    # 成长能力 (净利同比增长等)
    growth = query_to_df(bs.query_growth_data(code=bs_code))
    # 偿债能力 (资产负债率等)
    balance = query_to_df(bs.query_balance_data(code=bs_code))

    if profit.empty:
        return pd.DataFrame()

    merged = profit.merge(growth, on='code', how='left', suffixes=('', '_growth')) \
                   .merge(balance, on='code', how='left', suffixes=('', '_balance'))

    # 统一提取需要的列
    cols_map = {
        'code': '股票代码',
        'pubDate': '发布日期',
        'statDate': '统计日期',
        'roeAvg': 'ROE_avg',
        'npMargin': '净利率',
        'epsTTM': 'EPS_TTM',
        'YOYNI': '净利同比',
        'YOYEquity': '权益同比',
        'liabilityToAsset': '资产负债率',
    }

    result = merged.rename(columns={k: v for k, v in cols_map.items() if k in merged.columns})
    needed = [v for k, v in cols_map.items() if v in result.columns]
    return result[needed] if needed else pd.DataFrame()


def main():
    input_path = os.path.join("data", "hs300_stock_list.csv")
    output_path = os.path.join("data", "stock_fundamental.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"请先运行 get_stock_data.py 获取成分股列表: {input_path}")

    hs300 = pd.read_csv(input_path)
    hs300['纯代码'] = hs300['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
    stock_codes = sorted(hs300['纯代码'].unique())

    login()
    try:
        all_data = []
        failed = []
        for code in tqdm(stock_codes, desc="获取季频基本面"):
            bs_code = f"sh.{code}" if code.startswith(('6', '9')) else f"sz.{code}"
            try:
                df = get_stock_fundamental(bs_code)
                if len(df) > 0:
                    # 确保股票代码为6位字符串
                    df['股票代码'] = df['股票代码'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
                    all_data.append(df)
                else:
                    failed.append(code)
            except Exception as e:
                print(f"  失败 {code}: {e}")
                failed.append(code)
            time.sleep(0.1)  # 避免请求过快

        if all_data:
            result = pd.concat(all_data, ignore_index=True)
            result.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"季频基本面已保存到: {output_path}")
            print(f"共 {len(result)} 条记录，覆盖 {result['股票代码'].nunique()} 只股票")
            print(f"统计日期范围: {result['统计日期'].min()} ~ {result['统计日期'].max()}")
        else:
            print("未获取到任何基本面数据")

        if failed:
            print(f"失败股票 ({len(failed)}): {failed[:10]}...")
    finally:
        logout()


if __name__ == "__main__":
    main()
