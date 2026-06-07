"""
各模型 Top-N 分段表现分析（N = 5/10/20/30/40/50）。

动机：投票融合实际是按各模型"前50名"加权的，而之前的 alpha/beta 分析只看了 Top5。
本脚本对每个模型，分别取其预测的前 5/10/20/30/40/50 名，逐档计算：
  - 等权周收益（该档 N 只票等权持有的当周收益）
  - 超额收益（减基准）
  - 平均实际位次（该档预测票实际涨幅排名的均值，越小越好；满分=总股票数）
  - 命中率（该档预测票实际进入全市场前20%的比例）
20 周取平均，看"放宽到前N名"时选股质量如何变化。

口径与 our-score.py 一致：单股收益=(末日开盘-首日开盘)/首日开盘，全市场按收益降序排名。

输出：
  - backtest/results/summary/topN_segments.csv         逐模型×逐档 的汇总指标
  - backtest/results/summary/topN_return_curve.png     各档等权收益对比图
  - backtest/results/summary/topN_avgrank_curve.png    各档平均实际位次对比图
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import build_calendar as cal
from aggregate import setup_chinese_font

OUT_CSV = config.SUMMARY_DIR / 'topN_segments.csv'
OUT_RET_PNG = config.SUMMARY_DIR / 'topN_return_curve.png'
OUT_RANK_PNG = config.SUMMARY_DIR / 'topN_avgrank_curve.png'

ALL_MODELS = ['model01', 'model02', 'model03', 'model04', 'model05', 'model07']
SEGMENTS = [5, 10, 20, 30, 40, 50]


def load_stock_data():
    df = pd.read_csv(config.STOCK_DATA_PATH)
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    return df


def week_returns_ranks(stock_df, d1, dk):
    wk = stock_df[(stock_df['日期'] >= d1) & (stock_df['日期'] <= dk)]
    returns = {}
    for code, g in wk.groupby('股票代码'):
        g = g.sort_values('日期')
        if len(g) < 2:
            continue
        s_open = g.iloc[0]['开盘']
        if s_open < 1e-6:
            continue
        returns[code] = (g.iloc[-1]['开盘'] - s_open) / s_open
    ranked = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    rank_map = {c: i + 1 for i, (c, _) in enumerate(ranked)}
    return returns, rank_map, len(ranked)


def read_ranking(model, week_idx):
    csv = config.RESULTS_DIR / f'week_{week_idx:02d}' / 'rankings' / f'{model}_result.csv'
    if not csv.exists():
        return None
    df = pd.read_csv(csv, dtype={'stock_id': str})
    df['stock_id'] = df['stock_id'].str.zfill(6)
    if 'rank' in df.columns:
        df = df.sort_values('rank')
    return df['stock_id'].tolist()


def main():
    weeks = cal.build_weeks(include_short=config.INCLUDE_SHORT_WEEKS)
    if not weeks:
        print('无回测周数据。')
        return 1
    stock_df = load_stock_data()

    # 每周每模型每档：收益、超额、平均位次、命中数
    # 累计后对周取平均
    records = {m: {n: {'ret': [], 'excess': [], 'avgrank': [], 'hit': [], 'cnt': []}
                   for n in SEGMENTS} for m in ALL_MODELS}

    for week in weeks:
        returns, rank_map, total = week_returns_ranks(stock_df, week['d1'], week['dk'])
        if not returns:
            continue
        bench = sum(returns.values()) / len(returns)
        for m in ALL_MODELS:
            ranked = read_ranking(m, week['week_idx'])
            if not ranked:
                continue
            for n in SEGMENTS:
                seg = ranked[:n]
                rets = [returns[s] for s in seg if s in returns]
                ranks = [rank_map[s] for s in seg if s in returns]
                if not rets:
                    continue
                avg_ret = sum(rets) / len(rets)
                records[m][n]['ret'].append(avg_ret)
                records[m][n]['excess'].append(avg_ret - bench)
                records[m][n]['avgrank'].append(sum(ranks) / len(ranks))
                hit = sum(1 for r in ranks if r <= total * 0.2)  # 实际进前20%
                records[m][n]['hit'].append(hit)
                records[m][n]['cnt'].append(len(ranks))

    # 汇总成表
    rows = []
    for m in ALL_MODELS:
        for n in SEGMENTS:
            d = records[m][n]
            if not d['ret']:
                continue
            rows.append({
                'model': m,
                'topN': n,
                'avg_weekly_return_%': round(np.mean(d['ret']) * 100, 3),
                'avg_excess_%': round(np.mean(d['excess']) * 100, 3),
                'cumulative_return_%': round((np.prod([1 + x for x in d['ret']]) - 1) * 100, 2),
                'avg_actual_rank': round(np.mean(d['avgrank']), 1),
                'total_stocks': total,
                'hit_rate_top20%_%': round(sum(d['hit']) / sum(d['cnt']) * 100, 1),
            })
    res = pd.DataFrame(rows)
    config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    # ===== 画图 =====
    has_cn = setup_chinese_font()
    _plot(res, has_cn)

    # ===== 终端打印 =====
    print('========== 各模型 Top-N 分段表现（20周平均）==========')
    print('解读：放宽到前N名时，等权收益/超额是否下降、平均实际位次是否变差')
    for m in ALL_MODELS:
        sub = res[res['model'] == m]
        if sub.empty:
            continue
        print(f'\n[{m}]')
        print(f"  {'档位':<6}{'周均收益':>10}{'周均超额':>10}{'累计收益':>10}{'平均位次':>12}{'命中率':>8}")
        for _, r in sub.iterrows():
            print(f"  Top{int(r['topN']):<4}{r['avg_weekly_return_%']:>9.2f}%{r['avg_excess_%']:>+9.2f}%"
                  f"{r['cumulative_return_%']:>9.2f}%{r['avg_actual_rank']:>8.1f}/{int(r['total_stocks'])}"
                  f"{r['hit_rate_top20%_%']:>7.1f}%")
    print(f'\n汇总表已保存: {OUT_CSV}')
    return 0


def _plot(res, has_cn):
    import matplotlib.pyplot as plt

    def L(cn, en):
        return cn if has_cn else en

    # 图1：各档周均超额收益
    fig, ax = plt.subplots(figsize=(11, 7))
    for m in ALL_MODELS:
        sub = res[res['model'] == m].sort_values('topN')
        if sub.empty:
            continue
        ax.plot(sub['topN'], sub['avg_excess_%'], marker='o', label=m, linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.6)
    ax.set_xlabel(L('档位 (Top-N)', 'Segment (Top-N)'))
    ax.set_ylabel(L('周均超额收益 (%)', 'Avg weekly excess (%)'))
    ax.set_title(L('各模型 Top-N 分段：周均超额收益（越靠左上越好）',
                   'Top-N segments: avg weekly excess return'))
    ax.set_xticks(SEGMENTS)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_RET_PNG, dpi=150)
    plt.close(fig)

    # 图2：各档平均实际位次（越低越好）
    fig, ax = plt.subplots(figsize=(11, 7))
    for m in ALL_MODELS:
        sub = res[res['model'] == m].sort_values('topN')
        if sub.empty:
            continue
        ax.plot(sub['topN'], sub['avg_actual_rank'], marker='o', label=m, linewidth=1.5)
    total = res['total_stocks'].iloc[0] if not res.empty else 300
    ax.axhline(y=total / 2, color='gray', linestyle='--', alpha=0.6,
               label=L('随机水平', 'random level'))
    ax.set_xlabel(L('档位 (Top-N)', 'Segment (Top-N)'))
    ax.set_ylabel(L('预测票的平均实际位次（越低越好）', 'Avg actual rank (lower=better)'))
    ax.set_title(L('各模型 Top-N 分段：选股质量（平均实际位次）',
                   'Top-N segments: avg actual rank of picks'))
    ax.set_xticks(SEGMENTS)
    ax.invert_yaxis()  # 位次越小越好，反转Y轴让"好"在上面
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_RANK_PNG, dpi=150)
    plt.close(fig)
    print(f'分段图已保存: {OUT_RET_PNG.name}, {OUT_RANK_PNG.name}')


if __name__ == '__main__':
    sys.exit(main())
