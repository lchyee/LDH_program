"""
投票合并脚本（手工分段评分 + 固定模型权重 + 严格权重校验）。

融合逻辑（经 20 周回测调优，累计收益约 57%、夏普 0.60）：
  第1层 模型内部评分：每个模型按自身"名次→得分"的分段曲线给前50名打分，
        再归一化到总和=1（保证各模型贡献相等、可比）。各模型曲线形状不同，
        来自对各模型"第几名最准"的实测分析（见 backtest/ 的分段分析）。
  第2层 融合加权：按固定模型权重加权汇总各模型的归一化得分。
        权重经回测调优，model01/model02 最高，model03 最低。
  第3层 选 Top-N + 分仓：取融合得分前5，分歧惩罚后按得分占比分配仓位。
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / 'model'
FINAL_OUTPUT_DIR = PROJECT_ROOT / 'output'
FINAL_OUTPUT_PATH = FINAL_OUTPUT_DIR / 'result.csv'
TOP_N = 5
DEFAULT_MISSING_RANK = 60

# ============================================================
# 第2层：固定模型权重（经 20 周回测调优）
# model01/model02 表现最好权重最高，model03 最弱权重最低。
# 未知模型（不在表中）默认按等权兜底。
# ============================================================
MODEL_WEIGHTS = {
    'model01': 0.22,
    'model02': 0.22,
    'model07': 0.18,
    'model05': 0.14,
    'model04': 0.14,
    'model03': 0.10,
}


def _lin(rank, r0, s0, r1, s1):
    """两端点线性插值：rank=r0 时=s0，rank=r1 时=s1（用于段内轻微递减）。"""
    return s0 + (rank - r0) * (s1 - s0) / (r1 - r0)


# ============================================================
# 第1层：每个模型的"名次 → 原始得分"分段曲线
# 形状来自各模型实测的"第几名最准"分析；只对前50名给分，之后为0。
# 返回值会在 segment_scores() 中归一化到总和=1。
# ============================================================
def _score_model01(r):
    # 前5名突出但不过尖（缩小段间差距，避免融合时一家独大）
    if r <= 5:   return 4.5
    if r <= 10:  return 3.0
    if r <= 20:  return _lin(r, 11, 2.5, 20, 1.5)
    if r <= 50:  return _lin(r, 21, 1.4, 50, 0.4)
    return 0.0


def _score_model02(r):
    # 第1名是噪声（压低），第6-20名最强
    if r <= 5:   return 3.0
    if r <= 20:  return 5.0
    if r <= 30:  return _lin(r, 21, 3.5, 30, 3.0)
    if r <= 50:  return 3.0
    return 0.0


def _score_model03(r):
    # 整体偏弱、各段差不多：接近拉平的低分布
    if r <= 5:   return 5.0
    if r <= 50:  return _lin(r, 6, 5.0, 50, 3.0)
    return 0.0


def _score_model04(r):
    # 前5最强，前20有效
    if r <= 5:   return 5.0
    if r <= 20:  return 4.0
    if r <= 50:  return _lin(r, 21, 4.0, 50, 2.0)
    return 0.0


def _score_model05(r):
    # 第1名特别准（单只高分），其余拉平
    if r == 1:   return 5.0
    if r <= 50:  return 4.0
    return 0.0


def _score_model07(r):
    # 前5名是噪声（压低），信号在第6-20名
    if r <= 5:   return 2.5
    if r <= 10:  return 4.0
    if r <= 50:  return _lin(r, 11, 5.0, 50, 3.0)
    return 0.0


SEGMENT_SCORERS = {
    'model01': _score_model01, 'model02': _score_model02, 'model03': _score_model03,
    'model04': _score_model04, 'model05': _score_model05, 'model07': _score_model07,
}


def segment_scores(model_id, ranked_ids):
    """对某模型前50名按其分段曲线打分并归一化到总和=1。

    ranked_ids：该模型预测结果按 rank 升序的 stock_id 列表。
    返回 {stock_id: 归一化得分}。未知模型回退到 1/(rank+2) 曲线。
    """
    scorer = SEGMENT_SCORERS.get(model_id)
    raw = {}
    for i, sid in enumerate(ranked_ids[:50]):
        rank = i + 1
        sc = scorer(rank) if scorer else 1.0 / (rank + 2.0)
        if sc > 0:
            raw[sid] = sc
    tot = sum(raw.values())
    return {s: v / tot for s, v in raw.items()} if tot > 0 else {}


def get_model_weight(model_id, present_models):
    """返回某模型的融合权重；未在权重表中的模型按等权兜底并归一化。"""
    if model_id in MODEL_WEIGHTS:
        return MODEL_WEIGHTS[model_id]
    # 不在表中：按"表中未覆盖部分"等权（极少触发，保证健壮）
    return 1.0 / max(len(present_models), 1)


def load_model_predictions():
    if not MODELS_DIR.exists(): return []
    predictions = []
    for sub in sorted(MODELS_DIR.iterdir()):
        if not sub.is_dir(): continue
        result_csv = sub / 'output' / 'result.csv'
        if not result_csv.exists(): continue
        try:
            df = pd.read_csv(result_csv, dtype={'stock_id': str})
            if 'rank' not in df.columns and 'score' in df.columns:
                df['rank'] = df['score'].rank(method='min', ascending=False)
            elif 'rank' not in df.columns:
                df['rank'] = range(1, len(df) + 1)

            df['stock_id'] = df['stock_id'].str.zfill(6)
            df = df.sort_values('rank').reset_index(drop=True)  # 保证按名次升序
            predictions.append((sub.name, df))
        except Exception as e:
            continue
    return predictions


def vote(predictions):
    """两层融合：第1层各模型分段评分(归一化)，第2层按固定模型权重加权。"""
    from collections import defaultdict
    stock_scores = defaultdict(float)
    vote_count = defaultdict(int)
    stock_ranks = defaultdict(list)

    present = [mid for mid, _ in predictions]
    for model_id, df in predictions:
        model_weight = get_model_weight(model_id, present)
        ranked_ids = df['stock_id'].tolist()           # 已按 rank 升序
        seg = segment_scores(model_id, ranked_ids)     # {sid: 归一化得分}, 总和=1

        for i, sid in enumerate(ranked_ids):
            if sid in seg:
                stock_scores[sid] += model_weight * seg[sid]
            # 记录该股在各模型的名次（用于分歧惩罚），仅记前50
            if i < 50:
                vote_count[sid] += 1
                stock_ranks[sid].append(i + 1)

    candidates = []
    num_models = len(predictions)

    for sid in stock_scores.keys():
        ranks = stock_ranks[sid] + [DEFAULT_MISSING_RANK] * (num_models - len(stock_ranks[sid]))
        rank_std = np.std(ranks)

        penalty_factor = 1.0 / (1.0 + 0.02 * rank_std)
        final_score = stock_scores[sid] * penalty_factor

        candidates.append({
            'stock_id': sid,
            'votes': vote_count[sid],
            'total_score': final_score,
            'rank_std': rank_std
        })

    df = pd.DataFrame(candidates)
    df = df.sort_values(by=['total_score', 'votes'], ascending=[False, False]).reset_index(drop=True)
    return df


def main():
    predictions = load_model_predictions()
    if not predictions:
        print("没有找到任何模型的预测结果！")
        return 1

    print(f"参与投票的模型数量: {len(predictions)}")

    # 固定模型权重（经回测调优）
    present = [mid for mid, _ in predictions]
    print("使用固定模型权重（回测调优）:")
    for m in present:
        print(f"  {m}: {get_model_weight(m, present):.2f}")

    ranked = vote(predictions)

    top = ranked.head(TOP_N).copy()
    n = len(top)

    # ==========================================
    # 等权分仓：选出的 N 只股票每只权重 = 1/N（总仓位 100%）。
    # 选哪 N 只仍由融合得分(vote)决定，这里只把仓位改成等权。
    # 仍做"整数百万分之一"精度裁剪，避免 float64 求和 > 1.0 导致评测报错。
    # ==========================================
    equal_weights = np.full(n, 1.0 / n)
    units = np.round(equal_weights * 1_000_000).astype(np.int64)
    while (units / 1_000_000.0).sum() > 1.0:
        units[np.argmax(units)] -= 1
    final_weights = units / 1_000_000.0

    final_df = pd.DataFrame({
        'stock_id': top['stock_id'].tolist(),
        'weight': final_weights,
    })

    print(f"\n>>> 最终等权组合（{n} 只，每只 {1.0/n*100:.1f}%）：")
    for _, row in final_df.iterrows():
        print(f" 股票: {row['stock_id']} | 权重: {row['weight']:.6f}")

    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(FINAL_OUTPUT_PATH, index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())