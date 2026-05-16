"""
投票合并脚本。
读取 models/*/output/result.csv（每个模型输出的 Top-K 候选），按以下规则投票：
- 每只股票获得的票数 = 在多少个模型的 Top-K 名单中出现
- 平票时使用次级关键字打破：所有模型给该股票的排名之和（越小越好）
- 取票数最高的 5 只股票，每只权重固定 0.2
最终输出到 ./output/result.csv，仅包含 stock_id, weight 两列（符合赛事方评分规范）。
"""
import os
import sys
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
FINAL_OUTPUT_DIR = PROJECT_ROOT / 'output'
FINAL_OUTPUT_PATH = FINAL_OUTPUT_DIR / 'result.csv'
TOP_N = 5


def load_model_predictions():
    """读取所有 models/*/output/result.csv，返回 [(model_id, dataframe), ...]"""
    if not MODELS_DIR.exists():
        return []

    predictions = []
    for sub in sorted(MODELS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        result_csv = sub / 'output' / 'result.csv'
        if not result_csv.exists():
            print(f'[跳过] 未找到 {result_csv}')
            continue
        try:
            df = pd.read_csv(result_csv, dtype={'stock_id': str})
        except Exception as e:
            print(f'[警告] 读取 {result_csv} 失败: {e}')
            continue
        if 'stock_id' not in df.columns:
            print(f'[警告] {result_csv} 缺少 stock_id 列，跳过')
            continue
        # 兜底：没有 rank 列时按行号生成
        if 'rank' not in df.columns:
            df = df.copy()
            df['rank'] = range(1, len(df) + 1)
        df['stock_id'] = df['stock_id'].astype(str).str.zfill(6)
        predictions.append((sub.name, df))
    return predictions


def vote(predictions):
    """
    投票逻辑：
    - votes: 该股票在多少个模型的 Top-K 中出现
    - rank_sum: 该股票在所有模型中的排名之和（缺席的模型按惩罚值计算，越小越靠前）
    返回按 (votes 降序, rank_sum 升序) 排序后的 DataFrame
    """
    if not predictions:
        raise ValueError('没有任何模型的预测结果可用于投票')

    # 统计每只股票被多少模型投中、累计排名
    from collections import defaultdict
    vote_count = defaultdict(int)
    rank_sum = defaultdict(float)

    # 缺席模型的排名惩罚：用该模型 Top-K 名单长度 + 1（即比最后一名还差）
    for model_id, df in predictions:
        penalty_rank = len(df) + 1
        seen = set()
        for _, row in df.iterrows():
            sid = row['stock_id']
            rk = float(row['rank'])
            vote_count[sid] += 1
            rank_sum[sid] += rk
            seen.add(sid)
        # 对未出现在此模型 Top-K 中的股票，先不补——只在最终排序时按 vote_count 比较
        # 这里 rank_sum 只累计实际出现过的部分，避免引入未参与候选的股票

    candidates = []
    for sid, votes in vote_count.items():
        candidates.append({
            'stock_id': sid,
            'votes': votes,
            'rank_sum': rank_sum[sid],
            # 平均排名（越小越好）作为最终关键字
            'avg_rank': rank_sum[sid] / votes,
        })

    df = pd.DataFrame(candidates)
    df = df.sort_values(by=['votes', 'avg_rank'], ascending=[False, True]).reset_index(drop=True)
    return df


def main():
    predictions = load_model_predictions()
    if not predictions:
        print('没有任何模型预测结果可用于投票，请先运行各模型的 predict.py')
        return 1

    print(f'共读取到 {len(predictions)} 个模型的预测: {[m for m, _ in predictions]}')

    ranked = vote(predictions)
    print('\n投票结果（前10）:')
    print(ranked.head(10).to_string(index=False))

    if len(ranked) < TOP_N:
        raise ValueError(f'候选股票不足 {TOP_N} 只，仅有 {len(ranked)} 只')

    top = ranked.head(TOP_N).copy()
    final_df = pd.DataFrame({
        'stock_id': top['stock_id'].tolist(),
        'weight': [round(1.0 / TOP_N, 6)] * TOP_N,
    })

    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(FINAL_OUTPUT_PATH, index=False)
    print(f'\n最终 Top{TOP_N} 已写入: {FINAL_OUTPUT_PATH}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
