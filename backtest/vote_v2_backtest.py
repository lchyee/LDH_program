"""
vote_v2 完整融合回测：六模型手工分段评分 + 分档模型权重，跑20周看集成成绩。

第1层：每个模型对前50名按手工分段赋分，归一化到总和=1。
第2层：按模型权重加权汇总，每票最终得分 = Σ 模型权重 × 该模型归一化得分。
第3层：取最终得分前5，按得分占比分配组合权重，算每周组合收益；20周复利。

与原版 vote.py 的 1/(rank+2) + 等权 做对比。
口径：单股周收益 = (末日开盘 - 首日开盘)/首日开盘。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import build_calendar as cal
from eval_segment_scoring import SCORERS, load_stock, week_returns, read_ranking

# 第2层：分档模型权重（依据夏普，高低差6%）
MODEL_WEIGHTS = {
    'model02': 0.1967,
    'model07': 0.1667, 'model01': 0.1667, 'model04': 0.1667, 'model05': 0.1667,
    'model03': 0.1367,
}
TOP_N = 5


def normalized_scores(model, ranked_ids):
    """某模型前50名的归一化得分字典 {stock_id: 权重}，总和=1。"""
    scorer = SCORERS[model]
    raw = {}
    for i, sid in enumerate(ranked_ids[:50]):
        sc = scorer(i + 1)
        if sc > 0:
            raw[sid] = sc
    tot = sum(raw.values())
    return {s: v / tot for s, v in raw.items()} if tot > 0 else {}


def fuse_week(week_idx):
    """融合一周，返回 {stock_id: 融合得分}。"""
    agg = {}
    for m, w in MODEL_WEIGHTS.items():
        ids = read_ranking(m, week_idx)
        if not ids:
            continue
        ns = normalized_scores(m, ids)
        for sid, sc in ns.items():
            agg[sid] = agg.get(sid, 0.0) + w * sc
    return agg


def main():
    weeks = cal.build_weeks(include_short=config.INCLUDE_SHORT_WEEKS)
    stock_df = load_stock()

    ens_weekly, bench_weekly = [], []
    picks_log = []

    for wk in weeks:
        rets = week_returns(stock_df, wk['d1'], wk['dk'])
        if not rets:
            continue
        bench_weekly.append(sum(rets.values()) / len(rets))

        agg = fuse_week(wk['week_idx'])
        if not agg:
            ens_weekly.append(None)
            continue
        # 取融合得分前5，按得分占比分配组合权重
        top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
        wsum = sum(s for _, s in top)
        port_ret, valid_w = 0.0, 0.0
        names = []
        for sid, sc in top:
            w = sc / wsum
            names.append(f'{sid}({w*100:.0f}%)')
            if sid in rets:
                port_ret += w * rets[sid]
                valid_w += w
        # 缺失收益的票按0计（极少），组合收益用已分配权重
        ens_weekly.append(port_ret)
        picks_log.append((wk['week_idx'], port_ret, bench_weekly[-1], names))

    bench = np.array(bench_weekly)
    ens = np.array([x for x in ens_weekly if x is not None])

    def stats(arr, b):
        nav = np.prod(1 + arr)
        vol = arr.std()
        sharpe = arr.mean() / vol if vol > 0 else 0
        excess = arr.mean() - b.mean()
        wins = sum(1 for i in range(len(arr)) if arr[i] > b[i])
        return nav - 1, arr.mean(), excess, vol, sharpe, wins

    print('========== vote_v2（分段评分+分档权重）融合成绩 ==========')
    cum, mean, excess, vol, sharpe, wins = stats(ens, bench)
    nav_b = np.prod(1 + bench)
    print(f"{'':<14}{'累计收益':>10}{'周均收益':>10}{'周均超额':>10}{'波动率':>10}{'夏普':>8}{'胜率':>8}")
    print(f"{'集成v2':<14}{cum*100:>9.2f}%{mean*100:>9.2f}%{excess*100:>+9.2f}%"
          f"{vol*100:>9.2f}%{sharpe:>8.2f}{wins}/{len(ens)}")
    print(f"{'沪深300基准':<12}{(nav_b-1)*100:>9.2f}%{bench.mean()*100:>9.2f}%{'0.00%':>10}"
          f"{bench.std()*100:>9.2f}%{bench.mean()/bench.std():>8.2f}{'-':>8}")

    # 与原版对比（从已有 weekly_returns.csv 读原版集成）
    wk_csv = config.SUMMARY_DIR / 'weekly_returns.csv'
    if wk_csv.exists():
        old = pd.read_csv(wk_csv)['ensemble'].dropna().values
        o_nav = np.prod(1 + old)
        o_vol = old.std()
        print(f"{'集成v1(原版)':<11}{(o_nav-1)*100:>9.2f}%{old.mean()*100:>9.2f}%"
              f"{(old.mean()-bench.mean())*100:>+9.2f}%{o_vol*100:>9.2f}%"
              f"{old.mean()/o_vol:>8.2f}{sum(1 for i in range(len(old)) if old[i]>bench[i])}/{len(old)}")

    print('\n========== v2 每周选股与收益 ==========')
    for wk_idx, pr, br, names in picks_log:
        print(f"第{wk_idx:>2}周  组合{pr*100:>+6.2f}%  基准{br*100:>+6.2f}%  超额{(pr-br)*100:>+6.2f}%  | {' '.join(names)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
