"""爬取沪深300成分股的申万行业分类"""
import baostock as bs
import pandas as pd
import os


def login():
    lg = bs.login()
    if lg.error_code != '0':
        raise Exception(f"baostock登录失败: {lg.error_msg}")
    print("baostock登录成功")


def logout():
    bs.logout()


def get_stock_industry(stock_code):
    """获取单只股票的行业分类"""
    rs = bs.query_stock_industry(code=stock_code)
    if rs.error_code != '0':
        return None
    data = []
    while (rs.error_code == '0') & rs.next():
        data.append(rs.get_row_data())
    if not data:
        return None
    return pd.DataFrame(data, columns=rs.fields)


def main():
    input_path = os.path.join("data", "hs300_stock_list.csv")
    output_path = os.path.join("data", "stock_industry.csv")

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"请先运行 get_stock_data.py 获取成分股列表: {input_path}")

    hs300 = pd.read_csv(input_path)
    # 提取纯数字代码
    hs300['纯代码'] = hs300['code'].str.replace('sh.', '').str.replace('sz.', '').str.zfill(6)
    stock_codes = hs300['纯代码'].unique()

    login()
    try:
        industries = []
        for code in stock_codes:
            # baostock需要 sh.600000 或 sz.000001 格式
            bs_code = f"sh.{code}" if code.startswith(('6', '9')) else f"sz.{code}"
            df = get_stock_industry(bs_code)
            if df is not None and len(df) > 0:
                row = df.iloc[-1]  # 取最新一条
                industries.append({
                    '股票代码': code,
                    '行业名称': row.get('industry', ''),
                    '行业分类标准': row.get('industryClassification', ''),
                })
            else:
                industries.append({
                    '股票代码': code,
                    '行业名称': '未知',
                    '行业分类标准': '',
                })

        result = pd.DataFrame(industries)
        result.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"行业分类已保存到: {output_path}")
        print(f"行业分布:\n{result['行业名称'].value_counts().to_string()}")
    finally:
        logout()


if __name__ == "__main__":
    main()
