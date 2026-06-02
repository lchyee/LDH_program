"""
MICN 训练脚本 (Model07) - 横截面排序进化版
"""
import os
import sys
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent.parent
SHARED_DIR = PROJECT_ROOT / 'shared_components'
if str(SRC_DIR) not in sys.path: sys.path.append(str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path: sys.path.insert(0, str(PROJECT_ROOT))

from MICN import Model
from shared_components.feature_utils import engineer_features, add_market_relative_features


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class Configs:
    def __init__(self, enc_in):
        self.task_name = 'classification'
        self.seq_len = 40
        self.pred_len = 0
        self.label_len = 0
        self.num_class = 1
        self.enc_in = enc_in
        self.embed = 'timeF'
        self.freq = 'd'
        self.d_model = 64
        self.d_ff = 128
        self.e_layers = 2
        self.d_layers = 1
        self.dropout = 0.1
        self.c_out = enc_in  # 【核心修复】：必须等于 enc_in，防止残差连接广播错位
        self.moving_avg = 25
        self.n_heads = 4


class ApproxNDCGLoss(nn.Module):
    """可微分的 NDCG 损失函数"""
    def __init__(self, k=5, temperature=1.0):
        super(ApproxNDCGLoss, self).__init__()
        self.k = k
        self.temperature = temperature

    def forward(self, y_pred, y_true):
        device = y_pred.device
        batch_size, num_items = y_pred.size()

        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        approx_ranks = 1.0 + torch.sum(torch.sigmoid(pred_diff / self.temperature), dim=2)

        y_true_min = y_true.min(dim=1, keepdim=True)[0]
        y_true_max = y_true.max(dim=1, keepdim=True)[0]
        relevance = (y_true - y_true_min) / (y_true_max - y_true_min + 1e-12)

        gains = torch.pow(2.0, relevance) - 1.0
        discounts = torch.log2(approx_ranks + 1.0)
        dcg = torch.sum(gains / discounts, dim=1)

        sorted_relevance, _ = torch.sort(relevance, dim=1, descending=True)
        ideal_ranks = torch.arange(1, num_items + 1, device=device).float().unsqueeze(0)
        ideal_discounts = torch.log2(ideal_ranks + 1.0)
        ideal_gains = torch.pow(2.0, sorted_relevance) - 1.0
        idcg = torch.sum(ideal_gains / ideal_discounts, dim=1)

        ndcg = dcg / (idcg + 1e-12)
        return (1.0 - ndcg).mean()


class WeightedRankingLoss(nn.Module):
    def __init__(self, temperature=1.0, k=50, weight_factor=2.0, pairwise_weight=1.0, ndcg_weight=0.5):
        super(WeightedRankingLoss, self).__init__()
        self.temperature = temperature
        self.k = k
        self.weight_factor = weight_factor
        self.pairwise_weight = pairwise_weight
        self.ndcg_weight = ndcg_weight
        self.ndcg_loss = ApproxNDCGLoss(k=k, temperature=temperature)

    def forward(self, y_pred, y_true):
        batch_size, num_items = y_true.size()
        k = min(self.k, num_items)
        _, top_indices = torch.topk(y_true, k, dim=1)

        weights = torch.ones_like(y_true)
        for i in range(batch_size):
            weights[i, top_indices[i]] = self.weight_factor

        pred_probs = F.softmax(y_pred / self.temperature, dim=1)
        target_probs = F.softmax(y_true / self.temperature, dim=1)
        listwise = -(target_probs * torch.log(pred_probs + 1e-12) * weights).sum(dim=1).mean()

        pred_diff = y_pred.unsqueeze(2) - y_pred.unsqueeze(1)
        true_diff = y_true.unsqueeze(2) - y_true.unsqueeze(1)
        mask = (true_diff != 0).float()
        pairwise_loss = torch.sigmoid(-pred_diff * torch.sign(true_diff)) * mask
        pairwise = (pairwise_loss.sum(dim=[1, 2]) / mask.sum(dim=[1, 2]).clamp(min=1)).mean()

        ndcg = self.ndcg_loss(y_pred, y_true)

        return listwise + self.pairwise_weight * pairwise + self.ndcg_weight * ndcg


def calculate_topk_return_score(y_pred, y_true, k=5):
    """与赛事评测对齐的 top-k 归一化收益分（单日）。

    final_score = (预测top5实际收益 - 随机收益) / (理论最大收益 - 随机收益)
    取值 1 表示完美选股，0 表示与随机无异，负数表示劣于随机。
    y_pred / y_true: 1D tensor，长度为当天股票数。
    """
    n = y_true.numel()
    if n < k:
        return None
    kk = min(k, n)
    _, pred_idx = torch.topk(y_pred, kk)
    pred_return = y_true[pred_idx].sum().item()
    max_return = torch.topk(y_true, kk)[0].sum().item()
    random_return = kk * y_true.mean().item()
    denom = max_return - random_return
    if abs(denom) < 1e-6:
        return 0.0
    return (pred_return - random_return) / (denom + 1e-12)


class CrossSectionalDataset(Dataset):
    def __init__(self, df, feature_cols, seq_len, min_date=None, max_date=None):
        """
        min_date: 日期下界（包含），用于划分验证集
        max_date: 日期上界（不包含），用于划分训练集
        """
        self.valid_dates = []
        self.date_to_data = {}
        date_groups = defaultdict(list)

        print(">>> [Model07 - MICN] 正在构造横截面时序切片...")
        for stock_id, group in df.groupby('股票代码'):
            group = group.sort_values('日期').reset_index(drop=True)
            feats = group[feature_cols].values.astype(np.float32)
            targets = group['target'].values.astype(np.float32)
            dates = group['日期'].values

            if len(group) <= seq_len: continue
            for i in range(len(group) - seq_len):
                seq = feats[i: i + seq_len]
                target = targets[i + seq_len - 1]
                date = dates[i + seq_len - 1]
                if not np.isnan(target):
                    date_groups[date].append((seq, target))

        for d in sorted(date_groups.keys()):
            if min_date is not None and d < min_date:
                continue
            if max_date is not None and d >= max_date:
                continue
            samples = date_groups[d]
            if len(samples) >= 30:
                X = np.stack([s[0] for s in samples])
                y = np.array([s[1] for s in samples])
                self.date_to_data[d] = (X, y)
                self.valid_dates.append(d)

    def __len__(self):
        return len(self.valid_dates)

    def __getitem__(self, idx):
        d = self.valid_dates[idx]
        X, y = self.date_to_data[d]
        return torch.tensor(X), torch.tensor(y)


def main():
    set_seed(42)
    print("========== 开始训练: MICN (Ranking Loss 进化版) ==========")
    MODEL_ROOT = SRC_DIR.parent
    DATA_PATH = MODEL_ROOT.parent.parent / 'data' / 'train.csv'
    CHECKPOINT_DIR = MODEL_ROOT / 'checkpoint'
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH, dtype={'股票代码': str})
    df['股票代码'] = df['股票代码'].astype(str).str.zfill(6)
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['股票代码', '日期']).reset_index(drop=True)

    print(">>> 并行提取特征...")
    groups = [group for _, group in df.groupby('股票代码', sort=False)]
    with mp.Pool(min(8, mp.cpu_count())) as pool:
        df = pd.concat(pool.map(engineer_features, groups)).reset_index(drop=True)

    # 添加市场相对特征
    print(">>> 添加市场相对特征...")
    df = add_market_relative_features(df, date_col='日期')

    # 【启用横截面特征】：保留市场相对特征作为模型输入（它们是横截面信号，且非未来数据）。
    # 仅丢弃标识列与会泄露 target 的列。
    drop_cols = ['股票代码', '日期', 'label', 'instrument', 'datetime', 'raw_return', 'market_return']
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # 【修复标签对齐】生成5日超额收益作为 target（与评测指标对齐：首日开盘→末日开盘）
    df['open_t1'] = df.groupby('股票代码')['开盘'].shift(-1)
    df['open_t5'] = df.groupby('股票代码')['开盘'].shift(-5)
    df['raw_return'] = (df['open_t5'] - df['open_t1']) / (df['open_t1'] + 1e-12)
    df['market_return'] = df.groupby('日期')['raw_return'].transform('mean')
    df['target'] = df['raw_return'] - df['market_return']
    df = df.dropna(subset=['target'])
    df.drop(columns=['open_t1', 'open_t5', 'raw_return', 'market_return'], inplace=True)

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 【修复数据泄露】按时间切分，只在训练集上 fit scaler
    all_dates = sorted(df['日期'].unique())
    val_start_idx = int(len(all_dates) * 0.9)  # 最后10%作为验证集
    val_start_date = all_dates[val_start_idx]

    train_mask = df['日期'] < val_start_date
    scaler = StandardScaler()
    df.loc[train_mask, feature_cols] = scaler.fit_transform(df.loc[train_mask, feature_cols])
    df.loc[~train_mask, feature_cols] = scaler.transform(df.loc[~train_mask, feature_cols])

    joblib.dump(scaler, CHECKPOINT_DIR / 'scaler.pkl')
    joblib.dump(feature_cols, CHECKPOINT_DIR / 'feature_cols.pkl')

    configs = Configs(enc_in=len(feature_cols))

    # 【修复数据泄露】训练集和验证集严格按时间分离
    train_dataset = CrossSectionalDataset(df, feature_cols, configs.seq_len, max_date=val_start_date)
    val_dataset = CrossSectionalDataset(df, feature_cols, configs.seq_len, min_date=val_start_date)

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    print(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Model(configs).to(device)

    NUM_EPOCHS = 40
    criterion = WeightedRankingLoss(k=50)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.2, total_iters=NUM_EPOCHS)

    # 【选模型标准】：用与赛事评测对齐的 top5 归一化收益分（越高越好），而非验证 loss。
    best_score = -float('inf')
    best_epoch = -1
    for epoch in range(NUM_EPOCHS):
        # 训练
        model.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.squeeze(0).to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            if hasattr(model, 'classification'):
                try:
                    outputs = model.classification(x_batch, None).squeeze(-1)
                except TypeError:
                    outputs = model.classification(x_batch).squeeze(-1)
            else:
                outputs = model(x_batch).squeeze(-1)

            outputs = outputs.unsqueeze(0)

            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # 验证：累计每个验证日的 top5 收益分
        model.eval()
        val_loss = 0.0
        score_list = []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.squeeze(0).to(device)
                y_batch = y_batch.to(device)
                if hasattr(model, 'classification'):
                    try:
                        outputs = model.classification(x_batch, None).squeeze(-1)
                    except TypeError:
                        outputs = model.classification(x_batch).squeeze(-1)
                else:
                    outputs = model(x_batch).squeeze(-1)
                outputs_b = outputs.unsqueeze(0)
                loss = criterion(outputs_b, y_batch)
                val_loss += loss.item()

                s = calculate_topk_return_score(outputs, y_batch.squeeze(0), k=5)
                if s is not None:
                    score_list.append(s)

        scheduler.step()
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        avg_val_score = sum(score_list) / len(score_list) if score_list else 0.0
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val final_score: {avg_val_score:.4f}")

        if avg_val_score > best_score:
            best_score = avg_val_score
            best_epoch = epoch + 1
            torch.save(model.state_dict(), CHECKPOINT_DIR / 'best_model.pth')

    print(f"✓ [Model07] 训练完成！最佳 epoch: {best_epoch}, 最佳验证 final_score: {best_score:.4f}")

    # 保存验证集得分供 vote.py 使用（与 model01 同口径：top5 归一化收益分）
    with open(CHECKPOINT_DIR / 'final_score.txt', 'w') as f:
        f.write(f"{best_score:.6f}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    sys.exit(main())