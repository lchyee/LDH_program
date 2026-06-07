"""
回测结果可读报表生成器。

参照 model/model01/our-score.py 的展示风格，把 20 周回测的 score.json 汇总成：
  1) 每周每模型的"超额收益 + Top5 明细（收益/实际排名/百分位）"——逐周可读报表
  2) 各模型 20 周累计表现汇总（总收益、平均超额、命中率）

输出：
  - backtest/results/summary/report_weekly.txt   逐周逐模型明细（人类可读，对齐排版）
  - backtest/results/summary/model_summary.csv    各模型整体表现汇总表
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


def load_scores():
    scores = []
    for d in sorted(config.RESULTS_DIR.glob('week_*')):
        sj = d / 'score.json'
        if sj.exists():
            scores.append(json.load(open(sj, encoding='utf-8')))
    scores.sort(key=lambda s: s['week_idx'])
    return scores


def pct(v, sign=False):
    if v is None:
        return 'N/A'
    return f'{v*100:+.2f}%' if sign else f'{v*100:.2f}%'


def write_weekly_report(scores, out_path):
    """逐周逐模型明细，参照 our-score.py 的"超额收益 + 排名明细"风格。"""
    lines = []
    model_order = ['model01', 'model02', 'model03', 'model04', 'model05', 'model07']
    for s in scores:
        lines.append('=' * 72)
        lines.append(f"第 {s['week_idx']} 周  ({s['iso_week']})   "
                     f"持有区间 {s['d1']} ~ {s['dk']}（{s['n_days']}个交易日）")
        lines.append(f"  基准日(决策日): {s['base_date']}   "
                     f"沪深300(等权)收益: {pct(s.get('benchmark_return'))}")
        lines.append('-' * 72)

        # 集成模型
        lines.append(f"  【集成模型】 加权收益 {pct(s.get('ensemble_return'), True)}   "
                     f"超额 {pct(s.get('ensemble_excess'), True)}")
        for d in s.get('ensemble_detail', []):
            rk = d.get('rank')
            tot = d.get('total')
            rk_s = f"{rk}/{tot}" if rk else 'N/A'
            lines.append(f"      {d['stock_id']}  权重{d.get('weight', 0)*100:5.1f}%  "
                         f"收益{pct(d.get('return'), True):>9}  实际排名 {rk_s:>9}")

        # 各单模型
        for m in model_order:
            md = s.get('models', {}).get(m)
            if not md:
                continue
            lines.append(f"  【{m}】 Top5等权收益 {pct(md.get('top_n_return'), True)}   "
                         f"超额 {pct(md.get('excess_return'), True)}")
            for d in md.get('top_n_detail', []):
                rk = d.get('rank')
                tot = d.get('total')
                rk_s = f"{rk}/{tot}" if rk else 'N/A'
                perc = d.get('percentile')
                perc_s = f"{perc:.1f}%" if perc is not None else 'N/A'
                ret_s = pct(d.get('return'), True)
                lines.append(f"      预测 {d['stock_id']}  "
                             f"收益{ret_s:>9}  实际排名 {rk_s:>9}  百分位 {perc_s:>7}")
        lines.append('')

    out_path.write_text('\n'.join(lines), encoding='utf-8')
    return len(scores)


def write_model_summary(scores, out_path):
    """各模型 20 周整体表现：累计收益、平均周超额、命中率（预测票实际进前20%的比例）。"""
    import csv
    model_order = ['model01', 'model02', 'model03', 'model04', 'model05', 'model07']

    rows = []
    # 基准累计
    bench_nav = 1.0
    for s in scores:
        bench_nav *= (1 + (s.get('benchmark_return') or 0))

    # 集成
    def agg(get_return, get_details):
        nav = 1.0
        excess_sum = 0.0
        excess_n = 0
        hit = 0
        total_picks = 0
        for s in scores:
            r = get_return(s)
            if r is not None:
                nav *= (1 + r)
            b = s.get('benchmark_return')
            if r is not None and b is not None:
                excess_sum += (r - b)
                excess_n += 1
            for d in get_details(s):
                perc = d.get('percentile')
                if perc is not None:
                    total_picks += 1
                    if perc >= 80:   # 实际收益进前20%
                        hit += 1
        avg_excess = (excess_sum / excess_n) if excess_n else None
        hit_rate = (hit / total_picks) if total_picks else None
        return nav, avg_excess, hit_rate

    ens_nav, ens_excess, _ = agg(
        lambda s: s.get('ensemble_return'),
        lambda s: [])  # 集成明细无 percentile，命中率不算
    rows.append(['ensemble', ens_nav, (ens_nav-1), ens_excess, None])

    for m in model_order:
        nav, avg_excess, hit_rate = agg(
            lambda s, mm=m: (s.get('models', {}).get(mm) or {}).get('top_n_return'),
            lambda s, mm=m: (s.get('models', {}).get(mm) or {}).get('top_n_detail', []))
        rows.append([m, nav, (nav-1), avg_excess, hit_rate])

    with open(out_path, 'w', newline='', encoding='utf-8-sig') as f:
        w = csv.writer(f)
        w.writerow(['模型', '累计净值', '总收益', '平均周超额', 'Top5命中率(实际进前20%)'])
        w.writerow(['沪深300基准', round(bench_nav, 4), f'{(bench_nav-1)*100:+.2f}%', '0.00%', '-'])
        for name, nav, tot, exc, hit in rows:
            w.writerow([
                name, round(nav, 4), f'{tot*100:+.2f}%',
                (f'{exc*100:+.2f}%' if exc is not None else '-'),
                (f'{hit*100:.1f}%' if hit is not None else '-'),
            ])
    return rows, bench_nav


def main():
    scores = load_scores()
    if not scores:
        print('未找到 score.json，请先运行回测。')
        return 1
    config.SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    report_path = config.SUMMARY_DIR / 'report_weekly.txt'
    n = write_weekly_report(scores, report_path)
    print(f'逐周明细报表已生成（{n} 周）: {report_path}')

    summary_path = config.SUMMARY_DIR / 'model_summary.csv'
    rows, bench_nav = write_model_summary(scores, summary_path)
    print(f'模型汇总表已生成: {summary_path}\n')

    # 终端打印汇总
    print('========== 各模型 20 周整体表现 ==========')
    print(f"{'模型':<12}{'总收益':>10}{'平均周超额':>12}{'命中率':>10}")
    print(f"{'沪深300基准':<12}{(bench_nav-1)*100:>9.2f}%{'0.00%':>12}{'-':>10}")
    for name, nav, tot, exc, hit in rows:
        exc_s = f'{exc*100:+.2f}%' if exc is not None else '-'
        hit_s = f'{hit*100:.1f}%' if hit is not None else '-'
        print(f"{name:<12}{tot*100:>+9.2f}%{exc_s:>12}{hit_s:>10}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
