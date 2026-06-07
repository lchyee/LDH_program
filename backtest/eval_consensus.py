"""
强共识选股实验：只买融合得分明显拔尖的股票，而非机械凑满5只。

评分口径：原版 1/(rank+2) + 模型等权（已验证最优）。
对比多种"选几只/怎么选"的策略，跑20周看收益/波动/夏普：
  - top1 / top2 / top3 / top5            固定只数
  - gap_ratio>=R                          只买得分≥第1名R倍的票（强共识断层）
  - score>=mean+k*std                     只买得分显著高于均值的票
每只在组合内按其融合得分占比加权。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import build_calendar as cal

MODELS = ['model01', 'model02', 'model03', 'model04', 'model05', 'model07']


def load_stock():
    df = pd.read_csv(config.STOCK_DATA_PATH)
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    return df


def week_returns(stock_df, d1, dk):
    wk = stock_df[(stock_df['日期'] >= d1) & (stock_df['日期'] <= dk)]
    out = {}
    for c, g in wk.groupby('股票代码'):
        g = g.sort_values('日期')
        if len(g) < 2:
            continue
        o = g.iloc[0]['开盘']
        if o > 1e-6:
            out[c] = (g.iloc[-1]['开盘'] - o) / o
    return out


def fuse(week_idx):
    """原版口径融合，返回按得分降序的 [(stock_id, score), ...]。"""
    agg = defaultdict(float)
    for m in MODELS:
        csv = config.RESULTS_DIR / f'week_{week_idx:02d}' / 'rankings' / f'{m}_result.csv'
        if not csv.exists():
            continue
        df = pd.read_csv(csv, dtype={'stock_id': str})
        df['stock_id'] = df['stock_id'].str.zfill(6)
        if 'rank' in df.columns:
            df = df.sort_values('rank')
        for i, sid in enumerate(df['stock_id'].tolist()):
            agg[sid] += 1.0 / (i + 1 + 2.0)
    return sorted(agg.items(), key=lambda x: x[1], reverse=True)


def select(ranked, mode, param):
    """按策略从融合排序里选股，返回 [(sid, weight)]，weight按得分占比。"""
    if not ranked:
        return []
    if mode == 'topN':
        chosen = ranked[:param]
    elif mode == 'gap':  # 得分 >= 第1名 * param
        thr = ranked[0][1] * param
        chosen = [(s, sc) for s, sc in ranked if sc >= thr][:5]  # 最多5只(合规)
    elif mode == 'zscore':  # 得分 >= mean + param*std
        scs = np.array([sc for _, sc in ranked])
        thr = scs.mean() + param * scs.std()
        chosen = [(s, sc) for s, sc in ranked if sc >= thr][:5]
    else:
        chosen = ranked[:5]
    if not chosen:
        chosen = ranked[:1]  # 至少买1只
    tot = sum(sc for _, sc in chosen)
    return [(s, sc / tot) for s, sc in chosen]


def run_strategy(weeks, stock_df, mode, param):
    weekly, bench_w, ncounts = [], [], []
    for wk in weeks:
        rets = week_returns(stock_df, wk['d1'], wk['dk'])
        if not rets:
            continue
        bench_w.append(sum(rets.values()) / len(rets))
        ranked = fuse(wk['week_idx'])
        picks = select(ranked, mode, param)
        ncounts.append(len(picks))
        pr = sum(w * rets[s] for s, w in picks if s in rets)
        weekly.append(pr)
    arr = np.array(weekly)
    b = np.array(bench_w)
    nav = np.prod(1 + arr)
    vol = arr.std()
    sharpe = arr.mean() / vol if vol > 0 else 0
    excess = arr.mean() - b.mean()
    wins = sum(1 for i in range(len(arr)) if arr[i] > b[i])
    return {
        'cum': nav - 1, 'mean': arr.mean(), 'excess': excess, 'vol': vol,
        'sharpe': sharpe, 'wins': wins, 'n': len(arr), 'avg_picks': np.mean(ncounts),
    }


def main():
    weeks = cal.build_weeks(include_short=config.INCLUDE_SHORT_WEEKS)
    stock_df = load_stock()

    strategies = [
        ('只买前1名', 'topN', 1),
        ('只买前2名', 'topN', 2),
        ('只买前3名', 'topN', 3),
        ('买满前5名(原版基准)', 'topN', 5),
        ('强共识:得分≥第1名70%', 'gap', 0.70),
        ('强共识:得分≥第1名60%', 'gap', 0.60),
        ('强共识:得分≥第1名50%', 'gap', 0.50),
        ('显著:得分≥均值+1.5std', 'zscore', 1.5),
        ('显著:得分≥均值+2.0std', 'zscore', 2.0),
    ]

    bench = np.array([week_returns(stock_df, w['d1'], w['dk']) for w in weeks])
    b = np.array([sum(r.values())/len(r) for r in bench if r])
    bnav = np.prod(1 + b)

    print('========== 强共识选股策略对比（20周，原版评分口径）==========')
    print(f"{'策略':<24}{'累计收益':>10}{'周均超额':>10}{'波动率':>9}{'夏普':>8}{'胜率':>8}{'平均持股':>9}")
    rows = []
    for name, mode, param in strategies:
        r = run_strategy(weeks, stock_df, mode, param)
        rows.append((name, r))
        print(f"{name:<24}{r['cum']*100:>9.2f}%{r['excess']*100:>+9.2f}%{r['vol']*100:>8.2f}%"
              f"{r['sharpe']:>8.2f}{r['wins']}/{r['n']:<3}{r['avg_picks']:>7.1f}只")
    print(f"{'沪深300基准':<22}{(bnav-1)*100:>9.2f}%{'0.00%':>10}{b.std()*100:>8.2f}%"
          f"{b.mean()/b.std():>8.2f}{'-':>8}{'-':>9}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
