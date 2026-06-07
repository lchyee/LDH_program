"""
每周涨幅折线图（非累计）。

横轴 = 回测周序，纵轴 = 当周收益率（%）。
每条线一个对象：沪深300基准、融合模型、6个单模型。
与累计净值图不同——这里每个点是该周独立的涨跌幅，不做复利叠加。

读取 backtest/results/summary/weekly_returns.csv（aggregate.py 已生成）。
输出 backtest/results/summary/weekly_returns_curve.png
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from aggregate import setup_chinese_font

WEEKLY_CSV = config.SUMMARY_DIR / 'weekly_returns.csv'
OUT_PNG = config.SUMMARY_DIR / 'weekly_returns_curve.png'

ALL_MODELS = ['model01', 'model02', 'model03', 'model04', 'model05', 'model07']


def main():
    if not WEEKLY_CSV.exists():
        print(f'未找到 {WEEKLY_CSV}，请先运行 aggregate.py。')
        return 1

    df = pd.read_csv(WEEKLY_CSV)
    has_cn = setup_chinese_font()
    import matplotlib.pyplot as plt

    def L(cn, en):
        return cn if has_cn else en

    fig, ax = plt.subplots(figsize=(14, 7))
    x = df['week_idx']
    # 收益率转百分比
    pct = lambda col: df[col] * 100

    # 基准与融合用粗线突出
    ax.plot(x, pct('benchmark'), label=L('沪深300(等权)', 'HS300(EW)'),
            color='black', linewidth=2.5, linestyle='--', marker='o', markersize=4)
    ax.plot(x, pct('ensemble'), label=L('融合模型', 'Ensemble'),
            color='red', linewidth=2.5, marker='o', markersize=4)
    for m in ALL_MODELS:
        if m in df.columns:
            ax.plot(x, pct(m), label=m, linewidth=1.2, alpha=0.7, marker='.', markersize=4)

    ax.axhline(y=0, color='gray', linewidth=1.0, alpha=0.6)  # 0轴：盈亏分界
    ax.set_xlabel(L('回测周序', 'Week Index'))
    ax.set_ylabel(L('当周收益率 (%)', 'Weekly Return (%)'))
    ax.set_title(L('每周涨幅对比（非累计）：各模型 / 融合 / 沪深300',
                   'Weekly Returns (non-cumulative): Models / Ensemble / HS300'))
    ax.set_xticks(x)
    ax.legend(loc='best', fontsize=9, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)

    print(f'每周涨幅折线图已保存: {OUT_PNG}' + ('' if has_cn else '（未找到中文字体，用英文标签）'))

    # 顺带打印每周各对象收益，便于核对
    print('\n每周收益率(%)：')
    cols = ['week_idx', 'benchmark', 'ensemble'] + ALL_MODELS
    disp = df[cols].copy()
    for c in cols[1:]:
        disp[c] = (disp[c] * 100).round(2)
    print(disp.to_string(index=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
