"""
TiDE 推理脚本 (Model05) - 全量排名 + 历史拼接增强版
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import multiprocessing as mp
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent.parent
SHARED_DIR = PROJECT_ROOT / 'shared_components'

if str(SRC_DIR) not in sys.path: sys.path.append(str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

# 导入 TiDE
from TiDE import Model
from train import Configs
from shared_components.feature_utils import engineer_features


def main():
    print("========== 开始预测: TiDE (Model05) ==========")
    MODEL_ROOT = SRC_DIR.parent
    TRAIN_PATH = PROJECT_ROOT / 'data' / 'train.csv'
    TEST_PATH = PROJECT_ROOT / 'data' / 'test.csv'
    CHECKPOINT_DIR = MODEL_ROOT / 'checkpoint'
    OUTPUT_DIR = MODEL_ROOT / 'output'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_PATH.exists():
        print(f"找不到训练文件: {TRAIN_PATH}")
        return 1

    model_path = CHECKPOINT_DIR / 'best_model.pth'
    scaler_path = CHECKPOINT_DIR / 'scaler.pkl'
    feature_cols_path = CHECKPOINT_DIR / 'feature_cols.pkl'

    # 【修复信息泄露】：仅使用训练集数据，截止日期为训练集最后一天
    print(">>> 正在加载历史数据（仅训练集，避免使用未来数据）...")
    train_df = pd.read_csv(TRAIN_PATH, dtype={'股票代码': str})
    raw_df = train_df.copy()
    raw_df['股票代码'] = raw_df['股票代码'].astype(str).str.zfill(6)
    raw_df['日期'] = pd.to_datetime(raw_df['日期'])
    raw_df = raw_df.sort_values(['股票代码', '日期']).reset_index(drop=True)
    latest_date = raw_df['日期'].max()

    raw_df = raw_df.groupby('股票代码').tail(100).reset_index(drop=True)

    print(">>> 正在进行特征工程计算...")
    groups = [group for _, group in raw_df.groupby('股票代码', sort=False)]
    with mp.Pool(processes=min(8, mp.cpu_count())) as pool:
        processed_list = pool.map(engineer_features, groups)
    raw_df = pd.concat(processed_list).reset_index(drop=True)

    feature_cols = joblib.load(feature_cols_path)

    raw_df[feature_cols] = raw_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    scaler = joblib.load(scaler_path)
    raw_df[feature_cols] = scaler.transform(raw_df[feature_cols])

    configs = Configs(enc_in=len(feature_cols))
    stock_ids = sorted(raw_df['股票代码'].unique())
    sequences, valid_stock_ids = [], []

    for stock_id in stock_ids:
        history = raw_df[(raw_df['股票代码'] == stock_id) & (raw_df['日期'] <= latest_date)]
        history = history.sort_values('日期').tail(configs.seq_len)
        if len(history) == configs.seq_len:
            sequences.append(history[feature_cols].values)
            valid_stock_ids.append(stock_id)

    if not sequences:
        print("有效序列长度不足，无法预测！请检查历史数据是否拼接成功。")
        return 1

    x_input = torch.tensor(np.array(sequences), dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(configs).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        x_input = x_input.to(device)

        # 【安全调用机制】：自适应接口
        if hasattr(model, 'classification'):
            try:
                scores = model.classification(x_input, None).squeeze(-1).cpu().numpy()
            except TypeError:
                scores = model.classification(x_input).squeeze(-1).cpu().numpy()
        else:
            scores = model(x_input).squeeze(-1).cpu().numpy()

    result_df = pd.DataFrame({
        'stock_id': valid_stock_ids,
        'score': scores
    })

    result_df['rank'] = result_df['score'].rank(method='min', ascending=False).astype(int)
    result_df = result_df.sort_values('rank').reset_index(drop=True)

    result_path = OUTPUT_DIR / 'result.csv'
    result_df.to_csv(result_path, index=False)

    print(f"✓ [Model05] 预测完成！共输出 {len(result_df)} 只股票的全景排名。")
    return 0


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    sys.exit(main())