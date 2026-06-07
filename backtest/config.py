"""
回测系统配置。

本地小范围验证时，把 MODELS 改成 ['model01', 'model02']、WEEKS 改成 [1, 2]
跑通整个流程后，再改回全量（MODELS=None, WEEKS=None）放到云上跑。
"""
from pathlib import Path

# ===== 路径 =====
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / 'data'
STOCK_DATA_PATH = DATA_DIR / 'stock_data.csv'
MODEL_DIR = PROJECT_ROOT / 'model'
OUTPUT_DIR = PROJECT_ROOT / 'output'              # 集成结果输出目录
BACKTEST_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BACKTEST_DIR / 'results'            # 回测存档根目录
SUMMARY_DIR = RESULTS_DIR / 'summary'             # 聚合结果目录

# 顶层调度脚本（复用竞赛真实流程）
SPLIT_SCRIPT = DATA_DIR / 'split_train_test.py'
VOTE_SCRIPT = PROJECT_ROOT / 'code' / 'src' / 'vote.py'

# ===== 回测范围 =====
# 训练数据起始日（扩展窗口：每周训练数据从这天一直到 base_date）
TRAIN_START = '2024-01-02'

# 回测哪些周。None = 全部周；也可以指定列表，如 [1, 2] 只跑前两周（本地验证用）
# 周序号见 build_calendar.py 生成的清单（从 1 开始）
WEEKS = None

# 回测哪些模型。None = 自动发现 model/ 下全部模型；
# 也可以指定子集，如 ['model01', 'model02']（本地验证用）
MODELS = None

# 回测年份（只回测该年的自然周）
BACKTEST_YEAR = 2026

# ===== 行为开关 =====
# 断点续跑：若某周已有 score.json，则跳过该周（云端中断后可继续）
SKIP_COMPLETED = True

# 短周处理：交易日不足 5 天的周是否仍然回测（True=回测，按实际天数；False=跳过）
# 用户选择：短周按实际交易日预测（如放假只有4天就预测周一到周四）
INCLUDE_SHORT_WEEKS = True

# 单模型 Top-N（评分时取每个模型预测排名前 N 只等权，与 our-score.py 一致）
TOP_N = 5


def discover_models():
    """发现 model/ 下所有包含 src/train.py 和 src/predict.py 的子模型目录名。"""
    if MODELS is not None:
        return list(MODELS)
    found = []
    if not MODEL_DIR.exists():
        return found
    for sub in sorted(MODEL_DIR.iterdir()):
        if not sub.is_dir():
            continue
        if (sub / 'src' / 'train.py').exists() and (sub / 'src' / 'predict.py').exists():
            found.append(sub.name)
    return found
