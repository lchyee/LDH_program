#!/usr/bin/env bash
# 云端一键回测脚本。
# 用法：在项目根目录执行  bash backtest/run_cloud.sh
set -e

echo "========== 股票选股滚动回测（云端） =========="

# 1. 进入项目根目录（脚本所在目录的上一级）
cd "$(dirname "$0")/.."
PROJECT_ROOT="$(pwd)"
echo "项目根目录: $PROJECT_ROOT"

# 2. 检查原始数据
if [ ! -f "data/stock_data.csv" ]; then
  echo "[错误] 缺少 data/stock_data.csv。请先上传原始数据或运行 get_stock_data.py。"
  exit 1
fi

# 3. 安装依赖（按需）
echo ">>> 安装依赖 ..."
pip install -q matplotlib pandas numpy joblib torch scikit-learn TA-Lib 2>/dev/null || \
  pip install -q matplotlib pandas numpy joblib scikit-learn || true

# 4. 打印回测日历
echo ">>> 回测日历："
python backtest/build_calendar.py

# 5. 运行主回测（断点续跑：中断后重跑会自动跳过已完成周）
echo ">>> 开始滚动回测（每周重训全部模型，耗时较长）..."
python backtest/run_backtest.py

# 6. 聚合 + 画净值曲线
echo ">>> 聚合结果并绘制净值曲线 ..."
python backtest/aggregate.py

# 6b. 每周涨幅折线图（非累计）
echo ">>> 绘制每周涨幅折线图 ..."
python backtest/plot_weekly_returns.py

# 6c. Alpha/Beta 选股能力诊断
echo ">>> 分析选股能力(Alpha/Beta) ..."
python backtest/analyze_alpha_beta.py

# 6d. Top-N 分段表现分析（贴近投票的前50口径）
echo ">>> 分析各模型 Top-N 分段表现 ..."
python backtest/analyze_topN_segments.py

# 7. 生成逐周逐模型可读报表（our-score.py 风格）
echo ">>> 生成汇总报表与逐模型逐周明细 ..."
python backtest/make_report.py
python backtest/make_per_model_report.py

echo "========== 回测完成 =========="
echo "结果目录: backtest/results/"
echo "  - summary/nav_curve.png             累计净值曲线图"
echo "  - summary/weekly_returns_curve.png  每周涨幅折线图(非累计)"
echo "  - summary/alpha_beta_scatter.png    选股能力诊断散点图"
echo "  - summary/alpha_beta.csv            Alpha/Beta指标表"
echo "  - summary/topN_segments.csv         各模型Top-N分段表现"
echo "  - summary/topN_return_curve.png     各档超额收益对比图"
echo "  - summary/topN_avgrank_curve.png    各档选股质量对比图"
echo "  - summary/weekly_returns.csv        每周收益"
echo "  - summary/model_summary.csv         各模型整体表现"
echo "  - summary/report_weekly.txt         逐周逐模型明细"
echo "  - per_model/<model>/week_NN.txt     每模型每周详细报表(our-score风格)"
