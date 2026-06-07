"""
单周评分模块。

口径与 test/score_self.py、model/model01/our-score.py 严格一致：
- 单股周收益 = (末日开盘 - 首日开盘) / 首日开盘，取该股在 test 期最后 5 个交易日
- 沪深300基准 = 全部成分股周收益的等权平均（our-score.py 的 hs300_avg_return）
- 集成收益 = sum(单股周收益 × vote权重)，权重原样读取（可能 <1，未投部分视为现金）
- 单模型收益 = Top-N 预测股票周收益的等权平均（our-score.py 的 our_avg_return）

另外参考 our-score.py，给出每个模型 Top-N 预测股票的实际收益排名与百分位。
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def _zfill6(series):
    return series.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(6)


def compute_stock_returns(test_df):
    """计算 test 期内每只股票的周收益率。

    与 score_self.py 一致：每只股票取最后 5 条记录，(末日开盘-首日开盘)/首日开盘。
    返回 dict: {股票代码: 收益率}。
    """
    df = test_df.copy()
    df['股票代码'] = _zfill6(df['股票代码'])
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期'])
    # 每只股票取最后 5 条（短周则取实际全部）
    df = df.groupby('股票代码').tail(5)

    returns = {}
    for code, g in df.groupby('股票代码'):
        g = g.sort_values('日期')
        if len(g) < 2:
            continue
        start_open = g.iloc[0]['开盘']
        end_open = g.iloc[-1]['开盘']
        if start_open is None or start_open < 1e-6:
            continue
        returns[code] = (end_open - start_open) / start_open
    return returns


def benchmark_return(returns):
    """沪深300基准 = 全部成分股周收益等权平均。"""
    if not returns:
        return None
    return sum(returns.values()) / len(returns)


def rank_map_from_returns(returns):
    """按收益降序生成 {股票代码: 排名(1-based)}，及总数。"""
    sorted_stocks = sorted(returns.items(), key=lambda x: x[1], reverse=True)
    rmap = {code: i + 1 for i, (code, _) in enumerate(sorted_stocks)}
    return rmap, len(sorted_stocks)


def read_predictions(result_csv, top_n=None):
    """读取某模型/集成的 result.csv，返回按 rank 升序排列的 DataFrame。

    兼容两种格式：
      - model01: stock_id, rank
      - model02-07: stock_id, score, rank
      - 集成 output/result.csv: stock_id, weight
    统一补出 'rank' 列并按 rank 升序；股票代码 zfill(6)。
    """
    df = pd.read_csv(result_csv)
    id_col = 'stock_id' if 'stock_id' in df.columns else '股票代码'
    df = df.rename(columns={id_col: 'stock_id'})
    df['stock_id'] = _zfill6(df['stock_id'])

    if 'rank' not in df.columns:
        if 'score' in df.columns:
            df['rank'] = df['score'].rank(method='min', ascending=False)
        else:
            # 集成结果(只有 weight)：按出现顺序即为排名
            df['rank'] = range(1, len(df) + 1)
    df = df.sort_values('rank').reset_index(drop=True)
    if top_n is not None:
        df = df.head(top_n)
    return df


def score_single_model(result_csv, returns, top_n):
    """单模型评分：Top-N 等权平均收益 + 每只票的实际排名/百分位。"""
    pred = read_predictions(result_csv, top_n=top_n)
    rmap, total = rank_map_from_returns(returns)

    rows = []
    valid_rets = []
    for _, r in pred.iterrows():
        code = r['stock_id']
        ret = returns.get(code)
        if ret is None:
            rows.append({'stock_id': code, 'return': None, 'rank': None,
                         'total': total, 'percentile': None})
            continue
        rank = rmap.get(code)
        rows.append({
            'stock_id': code,
            'return': ret,
            'rank': rank,
            'total': total,
            'percentile': round((1 - rank / total) * 100, 2) if rank else None,
        })
        valid_rets.append(ret)

    avg_return = sum(valid_rets) / len(valid_rets) if valid_rets else None
    return avg_return, rows


def score_ensemble(result_csv, returns):
    """集成评分：sum(单股收益 × 权重)。权重原样读取，不重新归一化。"""
    df = pd.read_csv(result_csv)
    id_col = 'stock_id' if 'stock_id' in df.columns else '股票代码'
    w_col = 'weight' if 'weight' in df.columns else '权重'
    df = df.rename(columns={id_col: 'stock_id', w_col: 'weight'})
    df['stock_id'] = _zfill6(df['stock_id'])

    rmap, total = rank_map_from_returns(returns)
    weighted = 0.0
    rows = []
    for _, r in df.iterrows():
        code = r['stock_id']
        w = float(r['weight'])
        ret = returns.get(code)
        contrib = (ret * w) if ret is not None else 0.0
        weighted += contrib
        rows.append({
            'stock_id': code,
            'weight': w,
            'return': ret,
            'rank': rmap.get(code),
            'total': total,
        })
    return weighted, rows
