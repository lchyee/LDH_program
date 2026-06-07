"""
回测主调度器（滚动重训，无未来泄露，支持断点续跑）。

每个回测周流程：
  1. 计算 base_date / d1 / dk
  2. 清理上一周残留产物（checkpoint、our_score.csv、result.csv）——防止前视偏差与旧权重泄露
  3. split_train_test.py 重建 train.csv(截到 base_date) + test.csv(该周)
  4. 逐个训练选中模型（检查返回码，失败的模型不参与）
  5. 逐个预测
  6. vote.py 投票（无评分文件→自动等权，无前视偏差）
  7. 评分并存档到 backtest/results/week_NN/

全部周完成后运行 aggregate.py 生成净值曲线与汇总。
所有 subprocess 以项目根目录为 cwd（model01 依赖相对路径）。
"""
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
import build_calendar as cal
import score_week as sw


def log(msg):
    print(msg, flush=True)


def run_subprocess(script_path, extra_args=None, add_src_to_path=True):
    """以项目根目录为 cwd 运行脚本，返回退出码。"""
    script_path = Path(script_path)
    cmd = [sys.executable, str(script_path)]
    if extra_args:
        cmd += [str(a) for a in extra_args]
    env = os.environ.copy()
    env['PYTHONIOENCODING'] = 'utf-8'
    if add_src_to_path:
        src_dir = script_path.parent
        existing = env.get('PYTHONPATH', '')
        env['PYTHONPATH'] = str(src_dir) + (os.pathsep + existing if existing else '')
    proc = subprocess.run(cmd, cwd=str(config.PROJECT_ROOT), env=env)
    return proc.returncode


def clean_stale_artifacts(models):
    """删除可能造成泄露/串期的残留：
    - 各模型 checkpoint 目录全部内容（旧权重）
    - 各模型 output/our_score.csv、our_top50_score.csv（旧的真实收益→vote前视偏差）
    - 各模型 output/result.csv（旧预测）
    - 集成 output/result.csv（旧集成结果）

    注意：必须清理 model/ 下【全部】模型目录，而不仅是本次选中的子集。
    因为 vote.py 会发现并读取所有模型目录的 result.csv；若只清理子集，
    未选中模型上一次（甚至全量训练期）的 result.csv 会被投票读入，
    既导致参与模型数错误，又引入未来数据泄露（旧预测基于更晚的数据）。
    """
    all_model_dirs = [d for d in config.MODEL_DIR.iterdir()
                      if d.is_dir() and (d / 'src').exists()] if config.MODEL_DIR.exists() else []
    for sub in all_model_dirs:
        ckpt = sub / 'checkpoint'
        if ckpt.exists():
            for f in ckpt.iterdir():
                if f.is_file():
                    f.unlink()
        out = sub / 'output'
        if out.exists():
            for name in ('our_score.csv', 'our_top50_score.csv', 'result.csv', 'result-all.csv'):
                p = out / name
                if p.exists():
                    p.unlink()
    ens = config.OUTPUT_DIR / 'result.csv'
    if ens.exists():
        ens.unlink()


def regenerate_data(week):
    """调用 split_train_test.py 重建 train.csv / test.csv。"""
    args = [
        '--input', str(config.STOCK_DATA_PATH),
        '--output-dir', str(config.DATA_DIR),
        '--train-start', config.TRAIN_START,
        '--train-end', cal.fmt(week['base_date']),
        '--test-start', cal.fmt(week['d1']),
        '--test-end', cal.fmt(week['dk']),
    ]
    rc = run_subprocess(config.SPLIT_SCRIPT, args, add_src_to_path=False)
    return rc == 0


def train_model(model_name):
    """训练单个模型，返回是否成功。"""
    train_py = config.MODEL_DIR / model_name / 'src' / 'train.py'
    if not train_py.exists():
        log(f"  [跳过] {model_name} 无 train.py")
        return False
    rc = run_subprocess(train_py)
    if rc != 0:
        log(f"  [失败] {model_name} 训练返回码 {rc}")
        return False
    return True


def predict_model(model_name):
    """预测单个模型，返回是否成功且生成了 result.csv。"""
    predict_py = config.MODEL_DIR / model_name / 'src' / 'predict.py'
    if not predict_py.exists():
        log(f"  [跳过] {model_name} 无 predict.py")
        return False
    rc = run_subprocess(predict_py)
    result_csv = config.MODEL_DIR / model_name / 'output' / 'result.csv'
    if rc != 0 or not result_csv.exists():
        log(f"  [失败] {model_name} 预测返回码 {rc}，result.csv 存在={result_csv.exists()}")
        return False
    return True


def run_vote():
    """投票合并，返回是否成功生成集成 result.csv。"""
    rc = run_subprocess(config.VOTE_SCRIPT, add_src_to_path=False)
    ens = config.OUTPUT_DIR / 'result.csv'
    if rc != 0 or not ens.exists():
        log(f"  [失败] 投票返回码 {rc}，集成 result.csv 存在={ens.exists()}")
        return False
    return True


def archive_week(week, succeeded_models):
    """评分并把当周所有产物存档到 backtest/results/week_NN/。"""
    week_dir = config.RESULTS_DIR / f"week_{week['week_idx']:02d}"
    week_dir.mkdir(parents=True, exist_ok=True)
    rankings_dir = week_dir / 'rankings'
    rankings_dir.mkdir(exist_ok=True)

    # 读取 test.csv 计算真实收益
    test_df = pd.read_csv(config.DATA_DIR / 'test.csv')
    returns = sw.compute_stock_returns(test_df)
    bench = sw.benchmark_return(returns)

    score = {
        'week_idx': week['week_idx'],
        'iso_week': f"{week['iso_year']}-W{week['iso_week']}",
        'base_date': cal.fmt(week['base_date']),
        'd1': cal.fmt(week['d1']),
        'dk': cal.fmt(week['dk']),
        'n_days': week['n_days'],
        'benchmark_return': bench,
        'models': {},
        'ensemble_return': None,
    }

    # 各单模型评分 + 存档排序
    for m in succeeded_models:
        src_csv = config.MODEL_DIR / m / 'output' / 'result.csv'
        if not src_csv.exists():
            continue
        shutil.copy(src_csv, rankings_dir / f'{m}_result.csv')
        avg_ret, rows = sw.score_single_model(src_csv, returns, config.TOP_N)
        score['models'][m] = {
            'top_n_return': avg_ret,
            'excess_return': (avg_ret - bench) if (avg_ret is not None and bench is not None) else None,
            'top_n_detail': rows,
        }

    # 集成评分 + 存档
    ens_csv = config.OUTPUT_DIR / 'result.csv'
    if ens_csv.exists():
        shutil.copy(ens_csv, rankings_dir / 'ensemble_result.csv')
        ens_ret, ens_rows = sw.score_ensemble(ens_csv, returns)
        score['ensemble_return'] = ens_ret
        score['ensemble_excess'] = (ens_ret - bench) if bench is not None else None
        score['ensemble_detail'] = ens_rows

    # 完成状态：所有选中模型都成功预测、且集成成功，才算 complete。
    # 否则标记 partial，断点续跑时不跳过，下次会重试该周（避免降级结果被冻结）。
    expected = set(week.get('expected_models', []))
    got = set(score['models'].keys())
    complete = bool(expected) and expected.issubset(got) and (score['ensemble_return'] is not None)
    score['status'] = 'complete' if complete else 'partial'

    with open(week_dir / 'score.json', 'w', encoding='utf-8') as f:
        json.dump(score, f, ensure_ascii=False, indent=2)

    # 同时存一份当周用的 train/test 范围说明
    with open(week_dir / 'meta.txt', 'w', encoding='utf-8') as f:
        f.write(f"week_idx: {week['week_idx']}\n")
        f.write(f"iso_week: {week['iso_year']}-W{week['iso_week']}\n")
        f.write(f"train: {config.TRAIN_START} ~ {cal.fmt(week['base_date'])}\n")
        f.write(f"test:  {cal.fmt(week['d1'])} ~ {cal.fmt(week['dk'])} ({week['n_days']}天)\n")
        f.write(f"models: {succeeded_models}\n")

    return score


def is_completed(week):
    """断点续跑判断：score.json 存在且 status == 'complete' 才视为完成。

    部分成功（partial）的周不算完成，重跑时会重试——避免降级结果被永久冻结、
    再被 aggregate 的 fillna(0) 无声扭曲净值曲线。
    """
    week_dir = config.RESULTS_DIR / f"week_{week['week_idx']:02d}"
    sj = week_dir / 'score.json'
    if not sj.exists():
        return False
    try:
        with open(sj, encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return False
    return data.get('status') == 'complete'


def run_week(week, models):
    """执行单个回测周的完整流程。返回 score dict 或 None。"""
    tag = f"周{week['week_idx']:02d} ({week['iso_year']}-W{week['iso_week']}, " \
          f"{cal.fmt(week['d1'])}~{cal.fmt(week['dk'])}, {week['n_days']}天)"
    log(f"\n{'='*70}\n开始回测 {tag}")
    log(f"  训练区间: {config.TRAIN_START} ~ {cal.fmt(week['base_date'])}（基准日）")

    # 1. 清理残留
    clean_stale_artifacts(models)
    # 记录本周预期模型集合，供 archive_week 判定 complete/partial
    week['expected_models'] = list(models)

    # 2. 重建数据
    if not regenerate_data(week):
        log("  [错误] 数据重建失败，跳过该周")
        return None

    # 3. 训练
    succeeded = []
    for m in models:
        log(f"  训练 {m} ...")
        if train_model(m):
            succeeded.append(m)
    if not succeeded:
        log("  [错误] 无任何模型训练成功，跳过该周")
        return None

    # 4. 预测
    predicted = []
    for m in succeeded:
        log(f"  预测 {m} ...")
        if predict_model(m):
            predicted.append(m)
    if not predicted:
        log("  [错误] 无任何模型预测成功，跳过该周")
        return None

    # 5. 投票
    log("  投票合并 ...")
    has_ensemble = run_vote()

    # 6. 评分存档
    score = archive_week(week, predicted)

    def pct(v):
        return 'N/A' if v is None else format(v, '.4%')
    log(f"  完成。基准={pct(score['benchmark_return'])} "
        f"集成={pct(score['ensemble_return'])}")
    return score


def main():
    models = config.discover_models()
    if not models:
        log("未发现任何模型，请检查 config.MODELS 或 model/ 目录。")
        return 1

    weeks = cal.build_weeks(include_short=config.INCLUDE_SHORT_WEEKS)
    weeks = cal.filter_weeks(weeks)
    if not weeks:
        log("没有可回测的周，请检查 config.WEEKS。")
        return 1

    log(f"回测配置：模型={models}")
    log(f"回测周数={len(weeks)}（{weeks[0]['week_idx']}~{weeks[-1]['week_idx']}）")
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for week in weeks:
        if config.SKIP_COMPLETED and is_completed(week):
            log(f"\n周{week['week_idx']:02d} 已完成，跳过（断点续跑）。")
            continue
        run_week(week, models)

    log("\n全部回测周处理完毕。运行 aggregate.py 生成汇总与净值曲线。")
    return 0


if __name__ == '__main__':
    sys.exit(main())
