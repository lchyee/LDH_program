"""
分段评分 + 分档权重的"加权力度"扫描实验。

在已定的六模型分段评分 / 分档权重基础上，调节两个力度旋钮，跑20周对比：
  - 内层陡度 steep：把各模型分段得分做幂变换 score**steep 再归一化。
      steep=1 原样；steep>1 放大前后差距(更陡)；steep<1 压平。
  - 外层权重差 spread：模型权重围绕等权(1/6)按"相对强弱"放大。
      spread=0 等权；spread 越大，强模型(按独立夏普/收益)越高、弱模型越低。

口径：单股周收益=(末日开盘-首日开盘)/首日开盘；选融合得分前5，按得分占比分仓。
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import build_calendar as cal
from eval_segment_scoring import SCORERS, load_stock, week_returns, read_ranking

# 各模型独立表现强弱（按之前独立回测的累计收益排序，用于外层权重放大方向）
# model02最强, model03最弱
STRENGTH = {
    'model02': 1.0, 'model07': 0.7, 'model05': 0.55,
    'model04': 0.45, 'model01': 0.30, 'model03': 0.0,
}
TOP_N = 5


def seg_scores(model, ranked_ids, steep):
    """某模型前50名得分(幂变换加陡)后归一化。"""
    scorer = SCORERS[model]
    raw = {}
    for i, sid in enumerate(ranked_ids[:50]):
        sc = scorer(i + 1)
        if sc > 0:
            raw[sid] = sc ** steep
    tot = sum(raw.values())
    return {s: v / tot for s, v in raw.items()} if tot > 0 else {}


def model_weights(spread):
    """围绕等权放大：w_m = (1/6) * (1 + spread*(strength_m - 0.5))，再归一化。"""
    base = 1.0 / len(STRENGTH)
    raw = {m: base * (1 + spread * (STRENGTH[m] - 0.5)) for m in STRENGTH}
    raw = {m: max(v, 1e-6) for m, v in raw.items()}
    tot = sum(raw.values())
    return {m: v / tot for m, v in raw.items()}


def run(weeks, stock_df, steep, spread):
    mw = model_weights(spread)
    weekly, bench_w = [], []
    for wk in weeks:
        rets = week_returns(stock_df, wk['d1'], wk['dk'])
        if not rets:
            continue
        bench_w.append(sum(rets.values()) / len(rets))
        agg = {}
        for m, w in mw.items():
            ids = read_ranking(m, wk['week_idx'])
            if not ids:
                continue
            for sid, sc in seg_scores(m, ids, steep).items():
                agg[sid] = agg.get(sid, 0.0) + w * sc
        if not agg:
            continue
        top = sorted(agg.items(), key=lambda x: x[1], reverse=True)[:TOP_N]
        wsum = sum(s for _, s in top)
        pr = sum((sc / wsum) * rets[sid] for sid, sc in top if sid in rets)
        weekly.append(pr)
    arr = np.array(weekly)
    b = np.array(bench_w)
    nav = np.prod(1 + arr)
    vol = arr.std()
    return {
        'cum': nav - 1, 'mean': arr.mean(), 'excess': arr.mean() - b.mean(),
        'vol': vol, 'sharpe': arr.mean() / vol if vol > 0 else 0,
        'wins': sum(1 for i in range(len(arr)) if arr[i] > b[i]), 'n': len(arr),
    }


def main():
    weeks = cal.build_weeks(include_short=config.INCLUDE_SHORT_WEEKS)
    stock_df = load_stock()

    # 基准：原版 1/(rank+2)+等权
    b = np.array([sum(week_returns(stock_df, w['d1'], w['dk']).values()) /
                  len(week_returns(stock_df, w['d1'], w['dk']))
                  for w in weeks if week_returns(stock_df, w['d1'], w['dk'])])

    combos = [
        ('v2原样(steep1,spread1.2)', 1.0, 1.2),
        ('内层加陡 steep1.5', 1.5, 1.2),
        ('内层更陡 steep2.0', 2.0, 1.2),
        ('内层很陡 steep3.0', 3.0, 1.2),
        ('外层加大 spread2.0', 1.0, 2.0),
        ('外层很大 spread3.0', 1.0, 3.0),
        ('双加 steep2.0+spread2.0', 2.0, 2.0),
        ('双加 steep3.0+spread3.0', 3.0, 3.0),
        ('强双加 steep4.0+spread3.0', 4.0, 3.0),
    ]
    print('========== 加权力度扫描（20周）==========')
    print('steep=内层陡度(越大越聚焦前排)  spread=模型权重差(越大强弱越悬殊)')
    print(f"{'组合':<28}{'累计收益':>10}{'周均超额':>10}{'波动率':>9}{'夏普':>8}{'胜率':>8}")
    for name, steep, spread in combos:
        r = run(weeks, stock_df, steep, spread)
        print(f"{name:<28}{r['cum']*100:>9.2f}%{r['excess']*100:>+9.2f}%{r['vol']*100:>8.2f}%"
              f"{r['sharpe']:>8.2f}{r['wins']}/{r['n']}")
    bnav = np.prod(1 + b)
    print(f"{'沪深300基准':<26}{(bnav-1)*100:>9.2f}%{'0.00%':>10}{b.std()*100:>8.2f}%{b.mean()/b.std():>8.2f}{'-':>8}")

    # 同口径的原版1/(rank+2)+等权做锚点
    from eval_consensus import fuse
    weekly = []
    for wk in weeks:
        rets = week_returns(stock_df, wk['d1'], wk['dk'])
        if not rets:
            continue
        ranked = fuse(wk['week_idx'])[:5]
        ws = sum(s for _, s in ranked)
        weekly.append(sum((sc/ws)*rets[sid] for sid, sc in ranked if sid in rets))
    a = np.array(weekly)
    anav = np.prod(1+a)
    print(f"{'原版1/(rank+2)+等权(锚点)':<24}{(anav-1)*100:>9.2f}%{(a.mean()-b.mean())*100:>+9.2f}%"
          f"{a.std()*100:>8.2f}%{a.mean()/a.std():>8.2f}{sum(1 for i in range(len(a)) if a[i]>b[i])}/{len(a)}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
