"""
投票合并脚本（Top-N 宽容融合版 + 动态模型权重 + 严格权重校验）。
"""
import os
import re
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


# 单个模型权重上限，避免任何一个模型”一家独大”
MODEL_WEIGHT_CAP = 0.30
# model07 (MICN) 表现较差，施加惩罚因子
MODEL07_PENALTY = 0.7
# 缺测时的兜底位次（满分 300，越大越差）
DEFAULT_TRIMMED_RANK = 150.0


def _read_text(path):
    """以多种编码稳健读取文本文件，失败返回 None。"""
    for enc in ('utf-8-sig', 'utf-8', 'gbk'):
        try:
            return path.read_text(encoding=enc)
        except Exception:
            continue
    return None


def _parse_excess_return(path):
    """从 our_score.csv 中解析“超额收益”（百分比数值）。

    我们各模型的 our_score.csv 由同一套评分模板生成，因此“超额收益”在模型之间
    是可比的，远比各自 final_score.txt 里口径不一的分数更适合作为权重依据。
    """
    text = _read_text(path)
    if text is None:
        return None
    for line in text.splitlines():
        if line.startswith('超额收益'):
            m = re.search(r'([-+]?\d+(?:\.\d+)?)\s*%', line)
            if m:
                return float(m.group(1))
    return None


def _parse_trimmed_top10_rank(path):
    """从 our_top50_score.csv 的“功能2”里解析“前10、剔除最差20%后的平均位次”。

    位次越小越好（满分 300）。功能2 已剔除最差 20% 的离群误差，比功能1 更能
    反映模型“好票”的成色，且同样由统一模板生成、模型间可比。
    """
    text = _read_text(path)
    if text is None:
        return None
    idx = text.find('功能2')
    section = text[idx:] if idx >= 0 else text
    m = re.search(r'前10\s+\S+\s+([\d.]+)\s*/\s*300', section)
    return float(m.group(1)) if m else None


def _apply_weight_cap(weights, cap=MODEL_WEIGHT_CAP):
    """把单模型权重压到 cap 以下，溢出部分按比例分给其余模型。"""
    names = list(weights.keys())
    w = np.array([weights[n] for n in names], dtype=float)
    s = w.sum()
    if s <= 0:
        return {n: 1.0 / len(names) for n in names}
    w = w / s
    for _ in range(20):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if not under.any():
            break
        w[under] += excess * (w[under] / w[under].sum())
    return {n: float(v) for n, v in zip(names, w)}


def load_model_weights():
    """基于“可比的实测表现”计算各模型权重。

    口径：对每个模型，读取统一模板生成的两项指标——
      1) our_score.csv 的“超额收益”（越大越好）；
      2) our_top50_score.csv 功能2 的“前10 剔除最差20% 平均位次”（越小越好）。
    分别做组内归一化后按 0.5 / 0.5 融合，再施加单模型权重上限。

    不再使用 final_score.txt：model01 与 model02-07 的 final_score 口径不一致
    （前者来自独立评分脚本），平方加权会错误地把 ~50% 仓位压给实测中等偏下的
    model01。
    """
    excess = {}
    rank = {}
    for sub in sorted(MODELS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        er = _parse_excess_return(sub / 'output' / 'our_score.csv')
        rk = _parse_trimmed_top10_rank(sub / 'output' / 'our_top50_score.csv')
        if er is None and rk is None:
            continue
        excess[sub.name] = er if er is not None else 0.0
        rank[sub.name] = rk if rk is not None else DEFAULT_TRIMMED_RANK

    names = sorted(set(excess) | set(rank))
    if not names:
        return None

    # 超额收益：截断到非负后组内归一化
    er_arr = np.array([max(excess.get(n, 0.0), 0.0) for n in names], dtype=float)
    er_sum = er_arr.sum()
    er_norm = er_arr / er_sum if er_sum > 0 else np.full(len(names), 1.0 / len(names))

    # 位次质量：用 (300 - 位次) 把“越小越好”转成“越大越好”，再组内归一化
    rq_arr = np.array([max(300.0 - rank.get(n, DEFAULT_TRIMMED_RANK), 0.0) for n in names], dtype=float)
    rq_sum = rq_arr.sum()
    rq_norm = rq_arr / rq_sum if rq_sum > 0 else np.full(len(names), 1.0 / len(names))

    blended = 0.5 * er_norm + 0.5 * rq_norm
    weights = {n: blended[i] for i, n in enumerate(names)}

    # 对 model07 施加惩罚因子
    if 'model07' in weights:
        weights['model07'] *= MODEL07_PENALTY

    return _apply_weight_cap(weights)


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
            predictions.append((sub.name, df))
        except Exception as e:
            continue
    return predictions


def vote(predictions, model_weights=None):
    from collections import defaultdict
    stock_scores = defaultdict(float)
    vote_count = defaultdict(int)
    stock_ranks = defaultdict(list)

    for model_id, df in predictions:
        # 动态权重：优先使用基于实测表现计算的权重（归一化到平均为 1）
        if model_weights and model_id in model_weights:
            model_weight = model_weights[model_id] * len(predictions)
        else:
            # 无实测权重时退回等权，不再人为加权特定模型
            model_weight = 1.0

        for _, row in df.iterrows():
            sid = row['stock_id']
            rk = float(row['rank'])
            score = (1.0 / (rk + 2.0)) * model_weight

            stock_scores[sid] += score
            vote_count[sid] += 1
            stock_ranks[sid].append(rk)

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

    # 加载动态模型权重
    model_weights = load_model_weights()
    if model_weights:
        print("使用动态模型权重（基于验证集得分）:")
        for m, w in sorted(model_weights.items()):
            print(f"  {m}: {w:.4f}")
    else:
        print("未找到验证集得分，使用默认权重")

    ranked = vote(predictions, model_weights)

    top = ranked.head(TOP_N).copy()

    max_possible_votes = len(predictions) * TOP_N
    actual_votes = top['votes'].sum()
    confidence_ratio = actual_votes / max_possible_votes

    total_top_score = top['total_score'].sum()
    dynamic_weights = (top['total_score'] / total_top_score).values

    position_scale = 1.0 if confidence_ratio >= 0.5 else (confidence_ratio * 1.5)
    position_scale = min(1.0, position_scale)
    dynamic_weights = (dynamic_weights * position_scale).round(6)

    # ==========================================
    # 【核心修复】：解决浮点数溢出导致的评测报错
    # 在“整数百万分之一”的精度上裁剪，避免 np.round 把扣减又加回去。
    # 即使各权重十进制相加恰好等于 1，float64 求和仍可能得到
    # 1.0000000000000002（> 1.0），故循环从最大权重逐 1e-6 扣减，
    # 直到 float64 下的实际求和确实 <= 1.0。
    # ==========================================
    units = np.round(dynamic_weights * 1_000_000).astype(np.int64)
    while (units / 1_000_000.0).sum() > 1.0:
        units[np.argmax(units)] -= 1
    dynamic_weights = units / 1_000_000.0

    final_df = pd.DataFrame({
        'stock_id': top['stock_id'].tolist(),
        'weight': dynamic_weights,
    })

    print(f"\n>>> 最终动态组合 (总仓位: {position_scale*100:.1f}%)：")
    for _, row in final_df.iterrows():
        print(f" 股票: {row['stock_id']} | 权重: {row['weight']:.6f}")

    FINAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(FINAL_OUTPUT_PATH, index=False)
    return 0


if __name__ == '__main__':
    sys.exit(main())