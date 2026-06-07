"""
逐模型逐周报表生成器（完全参照 model/model01/our-score.py 的输出格式）。

为每个模型新建一个文件夹，里面每个回测周一个 .txt 文件，内容包含：
  1) 评分摘要（our_score.csv 风格）：沪深300均收益、预测Top5均收益、超额收益、Top5明细
  2) 功能1：预测前N名(10/20/30/40/50)的平均实际位次
  3) 功能2：剔除最差20%后的前80%平均位次
  4) 预测前50明细：每只票的收益率、实际排名、百分位

收益/排名口径与 our-score.py 完全一致：
  单股收益 = (该周末日开盘 - 首日开盘) / 首日开盘；全市场按收益降序排名。

输出目录：backtest/results/per_model/<model>/week_NN_<isoweek>.txt
"""
import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

PER_MODEL_DIR = config.RESULTS_DIR / 'per_model'
ALL_MODELS = ['model01', 'model02', 'model03', 'model04', 'model05', 'model07']


def write_aligned_table(lines, headers, rows):
    """把 rows(dict 列表)按列宽对齐、用空格分隔，追加到 lines 列表。与 our-score.py 同款。"""
    widths = [max(len(str(r[h])) for r in rows + [{h: h for h in headers}]) for h in headers]
    lines.append('  '.join(h.ljust(widths[i]) for i, h in enumerate(headers)).rstrip())
    for row in rows:
        lines.append('  '.join(str(row[h]).ljust(widths[i]) for i, h in enumerate(headers)).rstrip())


def load_stock_data():
    df = pd.read_csv(config.STOCK_DATA_PATH)
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    return df


def load_stock_names():
    """从 hs300_stock_list.csv 读股票名称映射（code 形如 sh.600000，取后6位）。"""
    path = config.DATA_DIR / 'hs300_stock_list.csv'
    if not path.exists():
        return {}
    name_df = pd.read_csv(path)
    name_df['股票代码'] = name_df['code'].astype(str).str[-6:]
    return dict(zip(name_df['股票代码'], name_df['code_name']))


def week_returns_and_ranks(stock_df, d1, dk):
    """计算该周全市场每只股票收益 + 排名映射，与 our-score.py 一致。"""
    wk = stock_df[(stock_df['日期'] >= d1) & (stock_df['日期'] <= dk)]
    returns = {}
    for code, g in wk.groupby('股票代码'):
        g = g.sort_values('日期')
        if len(g) < 2:
            continue
        s_open = g.iloc[0]['开盘']
        e_open = g.iloc[-1]['开盘']
        if s_open < 1e-6:
            continue
        returns[code] = (e_open - s_open) / s_open
    sorted_stocks = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    rank_map = {c: i + 1 for i, (c, _) in enumerate(sorted_stocks)}
    return returns, rank_map, len(sorted_stocks)


def read_model_ranking(model, week_idx):
    """读取某模型某周的完整预测排序（按 rank 升序的 stock_id 列表）。"""
    csv = config.RESULTS_DIR / f'week_{week_idx:02d}' / 'rankings' / f'{model}_result.csv'
    if not csv.exists():
        return None
    df = pd.read_csv(csv, dtype={'stock_id': str})
    df['stock_id'] = df['stock_id'].str.zfill(6)
    if 'rank' in df.columns:
        df = df.sort_values('rank')
    return df['stock_id'].tolist()


def build_week_report(model, week, stock_df, name_map):
    """生成某模型某周的报表文本（our-score.py 风格）。"""
    d1, dk = week['d1'], week['dk']
    returns, rank_map, total = week_returns_and_ranks(stock_df, d1, dk)

    ranked_ids = read_model_ranking(model, week['week_idx'])
    if ranked_ids is None or not returns:
        return None

    our_top5 = ranked_ids[:5]
    our_top50 = ranked_ids[:50]

    # 基准 & Top5
    bench = sum(returns.values()) / len(returns)
    top5_rets = [returns[s] for s in our_top5 if s in returns]
    our_avg = (sum(top5_rets) / len(top5_rets)) if top5_rets else None
    excess = (our_avg - bench) if our_avg is not None else None

    lines = []
    # ===== 摘要（our_score.csv 风格）=====
    lines.append(f'# 模型 {model}  第{week["week_idx"]}周  {week["iso_year"]}-W{week["iso_week"]}')
    lines.append(f'# 持有区间 {d1.date()} ~ {dk.date()}（{week["n_days"]}个交易日）  决策日 {week["base_date"].date()}')
    lines.append('')
    lines.append('# 评分摘要')
    lines.append(f'沪深300平均收益率,{bench*100:.4f}%')
    lines.append(f'预测Top5平均收益率,{our_avg*100:.4f}%' if our_avg is not None else '预测Top5平均收益率,N/A')
    lines.append(f'超额收益,{excess*100:+.4f}%' if excess is not None else '超额收益,N/A')
    lines.append('')

    # ===== Top5 明细 =====
    lines.append('# 预测Top5明细')
    t5_rows = []
    for s in our_top5:
        if s in returns:
            t5_rows.append({
                '股票代码': s, '股票名称': name_map.get(s, ''),
                '收益率': f'{returns[s]*100:+.4f}%',
                '排名': f'{rank_map[s]}/{total}',
                '百分位': f'{(1-rank_map[s]/total)*100:.1f}%',
            })
        else:
            t5_rows.append({'股票代码': s, '股票名称': name_map.get(s, ''),
                            '收益率': 'N/A', '排名': 'N/A', '百分位': 'N/A'})
    write_aligned_table(lines, ['股票代码', '股票名称', '收益率', '排名', '百分位'], t5_rows)
    lines.append('')

    # ===== 功能1 & 2 =====
    seg1_rows, seg2_rows = [], []
    for n in [10, 20, 30, 40, 50]:
        seg = our_top50[:n]
        seg_ranks = [rank_map[s] for s in seg if s in returns]
        if seg_ranks:
            avg_rank = sum(seg_ranks) / len(seg_ranks)
            seg1_rows.append({'预测前N名': f'前{n}', '有效数': f'{len(seg_ranks)}/{n}',
                              '平均位次': f'{avg_rank:.1f}/{total}'})
            sr = sorted(seg_ranks)
            keep = max(1, int(len(sr) * 0.8))
            seg2_rows.append({'预测前N名': f'前{n}', '取样数': f'{keep}/{len(seg_ranks)}',
                              '前80%平均位次': f'{sum(sr[:keep])/keep:.1f}/{total}'})
        else:
            seg1_rows.append({'预测前N名': f'前{n}', '有效数': f'0/{n}', '平均位次': 'N/A'})
            seg2_rows.append({'预测前N名': f'前{n}', '取样数': f'0/{n}', '前80%平均位次': 'N/A'})

    lines.append('# 功能1：预测前N名的平均位次')
    write_aligned_table(lines, ['预测前N名', '有效数', '平均位次'], seg1_rows)
    lines.append('')
    lines.append('# 功能2：预测前N名中表现最好的80%的平均位次（剔除最差20%误差）')
    write_aligned_table(lines, ['预测前N名', '取样数', '前80%平均位次'], seg2_rows)
    lines.append('')

    # ===== 预测前50明细 =====
    lines.append('# 预测前50明细')
    t50_rows = []
    for s in our_top50:
        if s in returns:
            t50_rows.append({
                '股票代码': s, '股票名称': name_map.get(s, ''),
                '收益率': f'{returns[s]*100:+.4f}%',
                '排名': f'{rank_map[s]}/{total}',
                '百分位': f'{(1-rank_map[s]/total)*100:.1f}%',
            })
        else:
            t50_rows.append({'股票代码': s, '股票名称': name_map.get(s, ''),
                             '收益率': 'N/A', '排名': 'N/A', '百分位': 'N/A'})
    write_aligned_table(lines, ['股票代码', '股票名称', '收益率', '排名', '百分位'], t50_rows)

    return '\n'.join(lines)


def main():
    import build_calendar as cal
    weeks = cal.build_weeks(include_short=config.INCLUDE_SHORT_WEEKS)
    if not weeks:
        print('无回测周数据。')
        return 1

    stock_df = load_stock_data()
    name_map = load_stock_names()
    PER_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    total_files = 0
    for model in ALL_MODELS:
        model_dir = PER_MODEL_DIR / model
        model_dir.mkdir(exist_ok=True)
        n_written = 0
        for week in weeks:
            text = build_week_report(model, week, stock_df, name_map)
            if text is None:
                continue
            fname = f"week_{week['week_idx']:02d}_{week['iso_year']}-W{week['iso_week']:02d}.txt"
            (model_dir / fname).write_text(text, encoding='utf-8')
            n_written += 1
            total_files += 1
        print(f'  {model}: 生成 {n_written} 周报表 -> {model_dir}')

    print(f'\n完成。共生成 {total_files} 个报表文件，目录: {PER_MODEL_DIR}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
