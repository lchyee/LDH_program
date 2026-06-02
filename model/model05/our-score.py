"""
单模型自评分脚本。
- 自动检测自身所在目录，读取同级 output/result.csv 作为预测结果
- 计算预测股票平均收益 vs 沪深300平均收益的超额收益
- 输出每只预测股票在全部成分股中的收益排名
- 复制到任意 model 文件夹下即可直接使用，无需修改任何路径
"""
import sys
from pathlib import Path
import pandas as pd

# ===== 自动定位路径（基于脚本自身位置） =====
SCRIPT_DIR = Path(__file__).resolve().parent          # 当前模型目录（如 models/model01/）
PROJECT_ROOT = SCRIPT_DIR.parent.parent               # 项目根目录
OUTPUT_PATH = SCRIPT_DIR / 'output' / 'result.csv'    # 本模型的预测结果
TEST_DATA_PATH = PROJECT_ROOT / 'data' / 'test.csv'   # 共享测试数据
RESULT_SAVE_PATH = SCRIPT_DIR / 'output' / 'our_score.csv'  # 评分结果保存位置
TOP50_SAVE_PATH = SCRIPT_DIR / 'output' / 'our_top50_score.csv'  # 预测前50收益率与排名保存位置
STOCK_LIST_PATH = PROJECT_ROOT / 'data' / 'hs300_stock_list.csv'  # 成分股名称表

MODEL_NAME = SCRIPT_DIR.name  # 自动取文件夹名作为模型标识


def calculate_stock_return(test_data, stock_code):
    """计算单只股票在测试期的收益率：(末日开盘 - 首日开盘) / 首日开盘"""
    stock_df = test_data[test_data['股票代码'] == stock_code].sort_values('日期')
    if len(stock_df) < 2:
        return None
    start_open = stock_df.iloc[0]['开盘']
    end_open = stock_df.iloc[-1]['开盘']
    if start_open < 1e-6:
        return None
    return (end_open - start_open) / start_open


def write_aligned_table(f, headers, rows):
    """把 rows(dict 列表)按列宽对齐、用空格分隔写入文件 f。"""
    widths = [
        max(len(str(r[h])) for r in rows + [{h: h for h in headers}])
        for h in headers
    ]
    f.write('  '.join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip() + '\n')
    for row in rows:
        f.write('  '.join(str(row[h]).ljust(widths[i]) for i, h in enumerate(headers)).rstrip() + '\n')


def main():
    print(f'[{MODEL_NAME}] 评分开始')
    print(f'  预测文件: {OUTPUT_PATH}')
    print(f'  测试数据: {TEST_DATA_PATH}')

    if not OUTPUT_PATH.exists():
        print(f'[错误] 未找到预测结果: {OUTPUT_PATH}')
        sys.exit(1)
    if not TEST_DATA_PATH.exists():
        print(f'[错误] 未找到测试数据: {TEST_DATA_PATH}')
        sys.exit(1)

    test_data = pd.read_csv(TEST_DATA_PATH)
    output_data = pd.read_csv(OUTPUT_PATH)

    # 统一股票代码格式
    test_data['股票代码'] = test_data['股票代码'].astype(str).str.zfill(6)

    id_col = 'stock_id' if 'stock_id' in output_data.columns else '股票代码'
    output_data[id_col] = output_data[id_col].astype(str).str.zfill(6)
    our_stocks = output_data[id_col].head(5).tolist()
    our_top50 = output_data[id_col].head(50).tolist()  # 模型预测排名前50（供新文档使用）

    # 计算所有成分股的收益率
    all_stocks = sorted(test_data['股票代码'].unique())
    returns = {}
    for stock in all_stocks:
        ret = calculate_stock_return(test_data, stock)
        if ret is not None:
            returns[stock] = ret

    if not returns:
        print('[错误] 无法计算任何股票的收益率，请检查测试数据。')
        sys.exit(1)

    # 沪深300平均收益
    hs300_avg_return = sum(returns.values()) / len(returns)

    # 我们预测的股票的收益
    our_returns = {}
    for stock in our_stocks:
        if stock in returns:
            our_returns[stock] = returns[stock]
        else:
            print(f'  [警告] 股票 {stock} 在测试数据中无有效收益，跳过')

    if not our_returns:
        print('[错误] 预测的股票在测试数据中均无有效收益。')
        sys.exit(1)

    our_avg_return = sum(our_returns.values()) / len(our_returns)
    excess_return = our_avg_return - hs300_avg_return

    # 按收益排序，计算排名
    sorted_stocks = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    rank_map = {stock: rank + 1 for rank, (stock, _) in enumerate(sorted_stocks)}
    total_stocks = len(sorted_stocks)

    # ===== 输出结果 =====
    print('=' * 60)
    print(f'  模型: {MODEL_NAME}')
    print(f'  测试期成分股数量: {total_stocks}')
    print(f'  沪深300平均收益率: {hs300_avg_return * 100:.4f}%')
    print(f'  预测股票平均收益率: {our_avg_return * 100:.4f}%')
    print(f'  超额收益（跑赢沪深300）: {excess_return * 100:+.4f}%')
    print('=' * 60)
    print(f'\n  预测股票收益排名（共 {total_stocks} 只）:')
    print(f'  {"股票代码":<10} {"收益率":>10} {"排名":>8} {"百分位":>8}')
    print(f'  {"-" * 40}')
    for stock in our_stocks:
        if stock in returns:
            ret = returns[stock]
            rank = rank_map[stock]
            percentile = (1 - rank / total_stocks) * 100
            print(f'  {stock:<10} {ret * 100:>+9.4f}% {rank:>5}/{total_stocks} {percentile:>7.1f}%')
        else:
            print(f'  {stock:<10} {"N/A":>10} {"N/A":>8}')
    print()

    # 加载股票名称映射
    stock_name_map = {}
    if STOCK_LIST_PATH.exists():
        name_df = pd.read_csv(STOCK_LIST_PATH)
        # code 格式为 "sh.600000" 或 "sz.000001"，提取后6位
        name_df['股票代码'] = name_df['code'].str[-6:]
        stock_name_map = dict(zip(name_df['股票代码'], name_df['code_name']))

    # 保存结果
    rows = []
    for stock in our_stocks:
        if stock in returns:
            rows.append({
                '股票代码': stock,
                '股票名称': stock_name_map.get(stock, ''),
                '收益率': returns[stock],
                '排名': rank_map[stock],
                '总数': total_stocks,
                '百分位': round((1 - rank_map[stock] / total_stocks) * 100, 2),
            })
    result_df = pd.DataFrame(rows)

    # 收益排名前30只股票
    top30_rows = []
    for rank_idx, (stock, ret) in enumerate(sorted_stocks[:30]):
        top30_rows.append({
            '排名': rank_idx + 1,
            '股票代码': stock,
            '股票名称': stock_name_map.get(stock, ''),
            '收益率': ret,
        })
    top30_df = pd.DataFrame(top30_rows)

    summary = pd.DataFrame([{
        '指标': '模型',
        '值': MODEL_NAME,
    }, {
        '指标': '沪深300平均收益率',
        '值': f'{hs300_avg_return * 100:.4f}%',
    }, {
        '指标': '预测股票平均收益率',
        '值': f'{our_avg_return * 100:.4f}%',
    }, {
        '指标': '超额收益',
        '值': f'{excess_return * 100:+.4f}%',
    }])

    RESULT_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULT_SAVE_PATH, 'w', encoding='utf-8') as f:
        f.write('# 评分摘要\n')
        summary.to_csv(f, index=False)
        f.write('\n# 我们预测的股票明细\n')
        result_df.to_csv(f, index=False)
        f.write('\n# 沪深300收益排名前30\n')
        top30_df.to_csv(f, index=False)

    print(f'  结果已保存到: {RESULT_SAVE_PATH}')

    # ===== 新增：预测排名前50的实际收益率与排名（仅生成文件，对齐排版）=====
    top50_rows = []
    for stock in our_top50:
        if stock in returns:
            ret = returns[stock]
            actual_rank = rank_map[stock]
            top50_rows.append({
                '股票代码': stock,
                '收益率': f'{ret * 100:+.4f}%',
                '排名': f'{actual_rank}/{total_stocks}',
                '百分位': f'{(1 - actual_rank / total_stocks) * 100:.1f}%',
            })
        else:
            top50_rows.append({
                '股票代码': stock,
                '收益率': 'N/A',
                '排名': 'N/A',
                '百分位': 'N/A',
            })

    # ===== 功能1 & 2：预测前10/20/30/40/50名的分段位次统计 =====
    # 功能1：每个分段内预测股票的平均实际位次（全部有效股票）
    # 功能2：每个分段内，按实际位次排序取表现最好的80%（剔除最差20%误差），算平均实际位次
    seg1_rows = []  # 功能1：平均位次
    seg2_rows = []  # 功能2：前80%平均位次
    for n in [10, 20, 30, 40, 50]:
        seg_stocks = our_top50[:n]
        seg_ranks = [rank_map[s] for s in seg_stocks if s in returns]

        # 功能1：全部有效股票的平均实际位次
        if seg_ranks:
            avg_rank = sum(seg_ranks) / len(seg_ranks)
            seg1_rows.append({
                '预测前N名': f'前{n}',
                '有效数': f'{len(seg_ranks)}/{n}',
                '平均位次': f'{avg_rank:.1f}/{total_stocks}',
            })
        else:
            seg1_rows.append({'预测前N名': f'前{n}', '有效数': f'0/{n}', '平均位次': 'N/A'})

        # 功能2：实际位次升序，取最好的80%求平均（剔除最差20%）
        if seg_ranks:
            sorted_ranks = sorted(seg_ranks)
            keep = max(1, int(len(sorted_ranks) * 0.8))
            avg_rank_80 = sum(sorted_ranks[:keep]) / keep
            seg2_rows.append({
                '预测前N名': f'前{n}',
                '取样数': f'{keep}/{len(seg_ranks)}',
                '前80%平均位次': f'{avg_rank_80:.1f}/{total_stocks}',
            })
        else:
            seg2_rows.append({'预测前N名': f'前{n}', '取样数': f'0/{n}', '前80%平均位次': 'N/A'})

    # 以空格对齐写入独立文档
    seg1_headers = ['预测前N名', '有效数', '平均位次']
    seg2_headers = ['预测前N名', '取样数', '前80%平均位次']
    detail_headers = ['股票代码', '收益率', '排名', '百分位']
    with open(TOP50_SAVE_PATH, 'w', encoding='utf-8') as f:
        f.write('# 功能1：预测前N名的平均位次\n')
        write_aligned_table(f, seg1_headers, seg1_rows)
        f.write('\n# 功能2：预测前N名中表现最好的80%的平均位次（剔除最差20%误差）\n')
        write_aligned_table(f, seg2_headers, seg2_rows)
        f.write('\n# 预测前50明细\n')
        write_aligned_table(f, detail_headers, top50_rows)

    print(f'  前50结果已保存到: {TOP50_SAVE_PATH}')


if __name__ == '__main__':
    main()
