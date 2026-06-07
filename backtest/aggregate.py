"""
聚合模块：读取 backtest/results/week_NN/score.json，生成汇总数据与净值曲线图。

产出（backtest/results/summary/）：
  - weekly_returns.csv   每周收益：基准 / 集成 / 各单模型
  - cumulative_nav.csv   累计净值曲线（逐周复利，起点 1.0）
  - all_rankings.csv     所有周、所有模型的预测排序 + 实际收益/排名
  - nav_curve.png        净值折线图（沪深300 vs 集成 vs 各单模型）

累计净值 NAV_t = NAV_{t-1} × (1 + 周收益)，所有曲线起点 1.0。
"""
import sys
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def load_all_scores():
    """按周序读取所有 score.json。"""
    scores = []
    if not config.RESULTS_DIR.exists():
        return scores
    for d in sorted(config.RESULTS_DIR.glob('week_*')):
        sj = d / 'score.json'
        if sj.exists():
            with open(sj, encoding='utf-8') as f:
                scores.append(json.load(f))
    scores.sort(key=lambda s: s['week_idx'])
    return scores


def build_weekly_returns(scores):
    """每周收益表：列 = week_idx, iso_week, dk, benchmark, ensemble, <各模型>。"""
    all_models = sorted({m for s in scores for m in s.get('models', {})})
    rows = []
    for s in scores:
        row = {
            'week_idx': s['week_idx'],
            'iso_week': s['iso_week'],
            'end_date': s['dk'],
            'benchmark': s.get('benchmark_return'),
            'ensemble': s.get('ensemble_return'),
        }
        for m in all_models:
            md = s.get('models', {}).get(m)
            row[m] = md.get('top_n_return') if md else None
        rows.append(row)
    return pd.DataFrame(rows), all_models


def build_cumulative_nav(weekly_df, all_models):
    """逐周复利累计净值，起点 1.0。

    在第 0 周显式插入 NAV=1.0 的起点，使曲线从 1.0 出发（week_idx=0）。
    缺失收益（某周该模型未成功）按 0% 计入，但会在 main() 中显式告警。
    """
    cols = ['benchmark', 'ensemble'] + all_models
    # 起点行：week_idx=0，所有曲线 NAV=1.0
    nav = {'week_idx': [0] + weekly_df['week_idx'].tolist(),
           'end_date': ['start'] + weekly_df['end_date'].tolist()}
    for c in cols:
        series = weekly_df[c].fillna(0.0).tolist()
        cum = [1.0]
        v = 1.0
        for r in series:
            v = v * (1.0 + (r if r is not None else 0.0))
            cum.append(v)
        nav[c] = cum
    return pd.DataFrame(nav)


def build_all_rankings(scores):
    """汇总所有周、所有模型 Top-N 预测股票的明细 + 集成明细。"""
    rows = []
    for s in scores:
        wk = s['week_idx']
        iso = s['iso_week']
        for m, md in s.get('models', {}).items():
            for d in md.get('top_n_detail', []):
                rows.append({
                    'week_idx': wk, 'iso_week': iso, 'source': m,
                    'stock_id': d['stock_id'], 'pred_top': True,
                    'actual_return': d.get('return'),
                    'actual_rank': d.get('rank'),
                    'total': d.get('total'),
                    'percentile': d.get('percentile'),
                    'weight': None,
                })
        for d in s.get('ensemble_detail', []):
            rows.append({
                'week_idx': wk, 'iso_week': iso, 'source': 'ensemble',
                'stock_id': d['stock_id'], 'pred_top': True,
                'actual_return': d.get('return'),
                'actual_rank': d.get('rank'),
                'total': d.get('total'),
                'percentile': None,
                'weight': d.get('weight'),
            })
    return pd.DataFrame(rows)


def setup_chinese_font():
    """配置 matplotlib 中文字体，云端无中文字体时回退到英文标签。"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    candidates = ['Microsoft YaHei', 'SimHei', 'STHeiti', 'WenQuanYi Micro Hei',
                  'Noto Sans CJK SC', 'PingFang SC', 'Source Han Sans SC']
    from matplotlib.font_manager import findfont, FontProperties
    available = None
    for name in candidates:
        try:
            path = findfont(FontProperties(family=name), fallback_to_default=False)
            if path:
                available = name
                break
        except Exception:
            continue
    if available:
        plt.rcParams['font.sans-serif'] = [available]
        plt.rcParams['axes.unicode_minus'] = False
        return True
    return False


def plot_nav(nav_df, all_models, has_cn):
    import matplotlib.pyplot as plt

    def L(cn, en):
        return cn if has_cn else en

    fig, ax = plt.subplots(figsize=(12, 7))
    x = nav_df['week_idx']

    # 基准与集成用粗线突出
    ax.plot(x, nav_df['benchmark'], label=L('沪深300(等权)', 'HS300(EW)'),
            color='black', linewidth=2.5, linestyle='--')
    ax.plot(x, nav_df['ensemble'], label=L('集成模型', 'Ensemble'),
            color='red', linewidth=2.5)
    for m in all_models:
        ax.plot(x, nav_df[m], label=m, linewidth=1.2, alpha=0.7)

    ax.set_xlabel(L('回测周序', 'Week Index'))
    ax.set_ylabel(L('累计净值(起点1.0)', 'Cumulative NAV (start=1.0)'))
    ax.set_title(L('滚动重训回测：累计净值曲线', 'Walk-forward Backtest: Cumulative NAV'))
    ax.axhline(y=1.0, color='gray', linewidth=0.8, alpha=0.5)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = config.SUMMARY_DIR / 'nav_curve.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    scores = load_all_scores()
    if not scores:
        print("未找到任何 score.json，请先运行 run_backtest.py。")
        return 1

    config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    weekly_df, all_models = build_weekly_returns(scores)
    weekly_df.to_csv(config.SUMMARY_DIR / 'weekly_returns.csv', index=False, encoding='utf-8-sig')

    # 显式告警：哪些周/模型缺失收益（partial 周或失败模型），这些在净值里按 0% 计入
    partial = [s['week_idx'] for s in scores if s.get('status') == 'partial']
    if partial:
        print(f"[警告] 以下周为 partial（部分模型或集成缺失，净值中按0%计入）: {partial}")
        print("       建议删除这些周的 score.json 后重跑 run_backtest.py 补全。")
    missing = []
    for c in ['benchmark', 'ensemble'] + all_models:
        n = int(weekly_df[c].isna().sum())
        if n:
            missing.append(f"{c}:{n}周")
    if missing:
        print(f"[警告] 收益缺失统计（按0%计入净值）: {', '.join(missing)}")

    nav_df = build_cumulative_nav(weekly_df, all_models)
    nav_df.to_csv(config.SUMMARY_DIR / 'cumulative_nav.csv', index=False, encoding='utf-8-sig')

    rankings_df = build_all_rankings(scores)
    rankings_df.to_csv(config.SUMMARY_DIR / 'all_rankings.csv', index=False, encoding='utf-8-sig')

    # 画图
    try:
        has_cn = setup_chinese_font()
        out = plot_nav(nav_df, all_models, has_cn)
        print(f"净值曲线图已保存: {out}" + ("" if has_cn else "（未找到中文字体，用英文标签）"))
    except ImportError:
        print("[提示] 未安装 matplotlib，跳过画图。数据 CSV 已生成，可自行绘制。")

    # 打印汇总
    print("\n========== 回测汇总 ==========")
    print(f"回测周数: {len(scores)}")
    final = nav_df.iloc[-1]
    print(f"\n累计净值（起点 1.0）:")
    print(f"  沪深300(等权): {final['benchmark']:.4f}  (总收益 {(final['benchmark']-1)*100:+.2f}%)")
    print(f"  集成模型:      {final['ensemble']:.4f}  (总收益 {(final['ensemble']-1)*100:+.2f}%)")
    for m in all_models:
        print(f"  {m}:        {final[m]:.4f}  (总收益 {(final[m]-1)*100:+.2f}%)")

    print(f"\n汇总数据已保存到: {config.SUMMARY_DIR}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
