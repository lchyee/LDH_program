"""
顶层预测调度脚本。
- 自动发现 models/ 下的所有子模型（每个子模型须包含 src/predict.py）
- 依次调用其推理入口，各模型将自己的 Top-K 结果写到 models/<name>/output/result.csv
- 全部模型预测完成后，调用 vote.py 进行投票汇总，最终输出 ./output/result.csv
"""
import os
import sys
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'
VOTE_SCRIPT = Path(__file__).resolve().parent / 'vote.py'


def discover_models():
    if not MODELS_DIR.exists():
        return []
    result = []
    for sub in sorted(MODELS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        predict_py = sub / 'src' / 'predict.py'
        if predict_py.exists():
            result.append((sub.name, predict_py))
    return result


def run_model_predict(model_name: str, predict_py: Path) -> int:
    print(f'\n========== 开始预测: {model_name} ==========')
    src_dir = predict_py.parent
    cmd = [sys.executable, str(predict_py)]
    env = os.environ.copy()
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(src_dir) + (os.pathsep + existing if existing else '')
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return proc.returncode


def run_vote() -> int:
    print('\n========== 开始投票合并 ==========')
    cmd = [sys.executable, str(VOTE_SCRIPT)]
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    return proc.returncode


def main():
    models = discover_models()
    if not models:
        print(f'未在 {MODELS_DIR} 下发现可预测模型，请检查目录结构。')
        return 1

    print(f'发现 {len(models)} 个模型: {[name for name, _ in models]}')

    succeeded = []
    for name, predict_py in models:
        rc = run_model_predict(name, predict_py)
        if rc != 0:
            print(f'[警告] 模型 {name} 预测返回非零退出码: {rc}')
        else:
            succeeded.append(name)

    if not succeeded:
        print('没有任何模型预测成功，终止流程。')
        return 1

    print(f'\n预测完成的模型: {succeeded}')

    rc = run_vote()
    if rc != 0:
        print(f'[错误] 投票脚本失败，返回码: {rc}')
        return rc
    return 0


if __name__ == '__main__':
    sys.exit(main())
