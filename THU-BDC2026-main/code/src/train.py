"""
顶层训练调度脚本。
- 自动发现 models/ 下的所有子模型（每个子模型须包含 src/train.py）
- 逐个调用其训练入口
- 任一模型训练失败不影响其它模型（仅打印错误并继续）
"""
import os
import sys
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / 'models'


def discover_models():
    if not MODELS_DIR.exists():
        return []
    result = []
    for sub in sorted(MODELS_DIR.iterdir()):
        if not sub.is_dir():
            continue
        train_py = sub / 'src' / 'train.py'
        if train_py.exists():
            result.append((sub.name, train_py))
    return result


def run_model_train(model_name: str, train_py: Path) -> int:
    print(f'\n========== 开始训练: {model_name} ==========')
    src_dir = train_py.parent
    # 以项目根目录作为工作目录运行，保证 config 中相对路径有效
    cmd = [sys.executable, str(train_py)]
    env = os.environ.copy()
    # 把模型自己的 src 目录加入 PYTHONPATH，方便它 import 同目录模块
    existing = env.get('PYTHONPATH', '')
    env['PYTHONPATH'] = str(src_dir) + (os.pathsep + existing if existing else '')
    proc = subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env)
    return proc.returncode


def main():
    models = discover_models()
    if not models:
        print(f'未在 {MODELS_DIR} 下发现可训练模型，请检查目录结构。')
        return 0

    print(f'发现 {len(models)} 个模型: {[name for name, _ in models]}')

    failed = []
    for name, train_py in models:
        rc = run_model_train(name, train_py)
        if rc != 0:
            print(f'[警告] 模型 {name} 训练返回非零退出码: {rc}')
            failed.append(name)

    if failed:
        print(f'\n训练完成，但以下模型失败: {failed}')
    else:
        print('\n全部模型训练完成。')
    return 0


if __name__ == '__main__':
    sys.exit(main())
