"""
按手工分段评分，把每个模型当成独立策略回测：
  每周取该模型预测前50只，每只权重 = 它的分段得分 / 前50得分之和（归一化），
  加权持有，算该模型当周组合收益率；20周复利得累计收益。

目的：用各模型的实际加权收益，作为后续融合层分配模型权重的依据。
口径：单股周收益 = (末日开盘 - 首日开盘)/首日开盘（与 score_self.py 一致）。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import build_calendar as cal


def _lin(rank, r0, s0, r1, s1):
    """两端点线性插值：rank=r0 时=s0，rank=r1 时=s1。"""
    return s0 + (rank - r0) * (s1 - s0) / (r1 - r0)


def score_model01(r):
    if r <= 5:   return 5.0
    if r <= 10:  return 4.5
    if r <= 20:  return _lin(r, 11, 3.0, 20, 2.0)
    if r <= 50:  return _lin(r, 21, 2.0, 50, 1.0)
    return 0.0


def score_model02(r):
    if r <= 5:   return 3.0
    if r <= 20:  return 5.0
    if r <= 30:  return _lin(r, 21, 3.5, 30, 3.0)
    if r <= 50:  return 3.0
    return 0.0


def score_model03(r):
    if r <= 5:   return 5.0
    if r <= 50:  return _lin(r, 6, 5.0, 50, 3.0)
    return 0.0


def score_model04(r):
    if r <= 5:   return 5.0
    if r <= 20:  return 4.0
    if r <= 50:  return _lin(r, 21, 4.0, 50, 2.0)
    return 0.0


def score_model05(r):
    if r == 1:   return 5.0
    if r <= 50:  return 4.0
    return 0.0


def score_model07(r):
    if r <= 5:   return 2.5
    if r <= 10:  return 4.0
    if r <= 50:  return _lin(r, 11, 5.0, 50, 3.0)
    return 0.0


SCORERS = {
    'model01': score_model01, 'model02': score_model02, 'model03': score_model03,
    'model04': score_model04, 'model05': score_model05, 'model07': score_model07,
}


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
        if o < 1e-6:
            continue
        out[c] = (g.iloc[-1]['开盘'] - o) / o
    return out


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
    stock_df = load_stock()

    # 每模型每周的加权组合收益
    model_weekly = {m: [] for m in SCORERS}
    bench_weekly = []

    for wk in weeks:
        rets = week_returns(stock_df, wk['d1'], wk['dk'])
        if not rets:
            continue
        bench_weekly.append(sum(rets.values()) / len(rets))
        for m, scorer in SCORERS.items():
            ids = read_ranking(m, wk['week_idx'])
            if not ids:
                model_weekly[m].append(None)
                continue
            top = ids[:50]
            # 分数加权（仅对有收益数据的票）
            num, den = 0.0, 0.0
            for i, sid in enumerate(top):
                rank = i + 1
                sc = scorer(rank)
                if sc <= 0 or sid not in rets:
                    continue
                num += sc * rets[sid]
                den += sc
            model_weekly[m].append(num / den if den > 0 else None)

    # 汇总：周均、累计净值
    print('========== 各模型（新分段加权，前50持有）独立表现 ==========')
    print(f"{'模型':<10}{'周均收益':>10}{'周均超额':>10}{'累计收益':>12}{'波动率':>10}{'夏普(周)':>10}")
    bench = np.array(bench_weekly)

    results = {}
    for m in SCORERS:
        arr = np.array([x for x in model_weekly[m] if x is not None])
        if len(arr) == 0:
            continue
        nav = np.prod(1 + arr)
        vol = arr.std()
        sharpe = arr.mean() / vol if vol > 0 else 0
        # 对齐基准计算超额（按有效周）
        paired = [(model_weekly[m][i], bench_weekly[i]) for i in range(len(bench_weekly))
                  if model_weekly[m][i] is not None]
        excess = np.mean([a - b for a, b in paired])
        results[m] = {'cum': nav - 1, 'mean': arr.mean(), 'excess': excess, 'sharpe': sharpe}
        print(f"{m:<10}{arr.mean()*100:>9.2f}%{excess*100:>+9.2f}%{(nav-1)*100:>11.2f}%"
              f"{vol*100:>9.2f}%{sharpe:>10.2f}")

    nav_b = np.prod(1 + bench)
    print(f"{'基准':<10}{bench.mean()*100:>9.2f}%{'0.00%':>10}{(nav_b-1)*100:>11.2f}%"
          f"{bench.std()*100:>9.2f}%{bench.mean()/bench.std():>10.2f}")

    # 给出按"累计收益"和按"夏普"两种归一化的融合权重建议
    print('\n========== 融合权重参考（仅正收益模型参与）==========')
    pos = {m: max(results[m]['cum'], 0) for m in results}
    s = sum(pos.values())
    if s > 0:
        print('按累计收益归一化:')
        for m in SCORERS:
            if m in results:
                print(f'  {m}: {pos[m]/s*100:5.1f}%')
    pos2 = {m: max(results[m]['sharpe'], 0) for m in results}
    s2 = sum(pos2.values())
    if s2 > 0:
        print('按夏普比率归一化:')
        for m in SCORERS:
            if m in results:
                print(f'  {m}: {pos2[m]/s2*100:5.1f}%')
    return 0


if __name__ == '__main__':
    sys.exit(main())
