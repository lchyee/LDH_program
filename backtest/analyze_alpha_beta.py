"""
Alpha/Beta 选股能力分析。

核心问题：模型的超额收益是真选股能力(alpha)，还是只是放大市场波动(beta杠杆)？

方法：对每个对象，把"当周收益"对"基准当周收益"做线性回归
    r = alpha + beta * benchmark
  - beta ≈ 1 且 alpha ≈ 0  → 只是跟随/放大市场，无选股能力
  - beta 低 且 alpha 显著>0 → 收益来自选股本身（真 alpha）
配合：波动率倍数、夏普比率、跑赢基准胜率、超额收益 t 检验。

输出：
  - backtest/results/summary/alpha_beta.csv     各对象指标表
  - backtest/results/summary/alpha_beta_scatter.png   beta-alpha 散点图

注意：样本仅 20 周，t值>2 才算超额"不太像运气"；结论仅供参考，且未计交易成本。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from aggregate import setup_chinese_font

WEEKLY_CSV = config.SUMMARY_DIR / 'weekly_returns.csv'
OUT_CSV = config.SUMMARY_DIR / 'alpha_beta.csv'
OUT_PNG = config.SUMMARY_DIR / 'alpha_beta_scatter.png'

ALL_MODELS = ['ensemble', 'model01', 'model02', 'model03', 'model04', 'model05', 'model07']
WEEKS_PER_YEAR = 52


def analyze(df):
    """对每个对象计算 alpha/beta/波动/夏普/胜率/t值。返回 DataFrame。"""
    b = df['benchmark'].values
    n = len(b)
    rows = []
    for m in ALL_MODELS:
        if m not in df.columns:
            continue
        r = df[m].values
        beta, alpha = np.polyfit(b, r, 1)          # r = beta*b + alpha
        corr = np.corrcoef(b, r)[0, 1]
        vol = r.std()
        nav = np.prod(1 + r)
        sharpe = (r.mean() / vol) if vol > 0 else 0.0
        excess = r - b
        wins = int((excess > 0).sum())
        t = (excess.mean() / (excess.std() / np.sqrt(n))) if excess.std() > 0 else 0.0
        rows.append({
            'object': m,
            'beta': round(beta, 3),
            'alpha_weekly_%': round(alpha * 100, 3),
            'alpha_annual_%': round(alpha * WEEKS_PER_YEAR * 100, 1),
            'corr_with_mkt': round(corr, 3),
            'weekly_vol_%': round(vol * 100, 3),
            'vol_vs_benchmark': round(vol / b.std(), 2) if b.std() > 0 else None,
            'total_return_%': round((nav - 1) * 100, 2),
            'sharpe_weekly': round(sharpe, 3),
            'win_weeks': f'{wins}/{n}',
            'win_rate_%': round(wins / n * 100, 1),
            'excess_t': round(t, 2),
        })
    # 基准自身
    rows.append({
        'object': 'benchmark', 'beta': 1.0, 'alpha_weekly_%': 0.0, 'alpha_annual_%': 0.0,
        'corr_with_mkt': 1.0, 'weekly_vol_%': round(b.std() * 100, 3), 'vol_vs_benchmark': 1.0,
        'total_return_%': round((np.prod(1 + b) - 1) * 100, 2),
        'sharpe_weekly': round(b.mean() / b.std(), 3) if b.std() > 0 else 0.0,
        'win_weeks': '-', 'win_rate_%': None, 'excess_t': None,
    })
    return pd.DataFrame(rows)


def plot_scatter(res, has_cn):
    import matplotlib.pyplot as plt

    def L(cn, en):
        return cn if has_cn else en

    fig, ax = plt.subplots(figsize=(11, 8))
    m = res[res['object'] != 'benchmark']

    betas = m['beta'].values
    alphas = m['alpha_weekly_%'].values
    names = m['object'].values
    # 点大小随总收益，颜色随夏普
    sizes = 80 + np.clip(m['total_return_%'].values, 0, None) * 8
    sc = ax.scatter(betas, alphas, s=sizes, c=m['sharpe_weekly'].values,
                    cmap='RdYlGn', edgecolors='black', linewidths=1, zorder=3)
    for x, y, nm in zip(betas, alphas, names):
        ax.annotate(nm, (x, y), xytext=(6, 4), textcoords='offset points', fontsize=10)

    # 参考线：beta=1（市场暴露），alpha=0（无超额）
    ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.6)
    ax.axhline(y=0.0, color='gray', linestyle='--', alpha=0.6)

    # 区域注释
    ax.text(0.05, 0.95, L('低beta·正alpha\n(真选股能力)', 'low-beta·+alpha\n(true stock-picking)'),
            transform=ax.transAxes, va='top', ha='left', fontsize=9, color='green', alpha=0.8)
    ax.text(0.95, 0.95, L('高beta·正alpha\n(含放大波动)', 'high-beta·+alpha\n(leverage involved)'),
            transform=ax.transAxes, va='top', ha='right', fontsize=9, color='orange', alpha=0.8)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(L('夏普比率(周)', 'Sharpe (weekly)'))
    ax.set_xlabel(L('Beta（对市场的敏感度，>1=放大市场波动）', 'Beta (market sensitivity, >1 = amplifies market)'))
    ax.set_ylabel(L('Alpha（每周超额收益 %，>0=选股贡献）', 'Alpha (weekly excess %, >0 = stock-picking)'))
    ax.set_title(L('选股能力诊断：Beta-Alpha 散点（点大小=总收益，颜色=夏普）',
                   'Skill Diagnosis: Beta-Alpha (size=total return, color=Sharpe)'))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    plt.close(fig)


def main():
    if not WEEKLY_CSV.exists():
        print(f'未找到 {WEEKLY_CSV}，请先运行 aggregate.py。')
        return 1
    df = pd.read_csv(WEEKLY_CSV)

    res = analyze(df)
    res.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')

    has_cn = setup_chinese_font()
    try:
        plot_scatter(res, has_cn)
        print(f'Beta-Alpha 散点图已保存: {OUT_PNG}' + ('' if has_cn else '（无中文字体，用英文标签）'))
    except ImportError:
        print('[提示] 未安装 matplotlib，跳过散点图。')

    print(f'指标表已保存: {OUT_CSV}\n')
    print('========== 选股能力诊断 ==========')
    print('解读: beta低+alpha正+t值大 = 真选股; beta>1+alpha≈0 = 放大波动')
    cols = ['object', 'beta', 'alpha_weekly_%', 'sharpe_weekly', 'vol_vs_benchmark', 'win_rate_%', 'excess_t']
    print(res[cols].to_string(index=False))
    print('\n注意: 仅20周样本, t值>2才算超额不太像运气; 未计交易成本。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
