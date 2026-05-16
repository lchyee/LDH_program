# 配置参数
sequence_length = 60
feature_num = '158+39'
config = {
    # ===== 模型标识 =====
    'model_id': 'model01',

    # ===== 路径 =====
    'data_path': './data',                          # 共享数据目录
    'output_dir': './models/model01/checkpoint',    # 训练产物（模型权重、scaler等）
    'model_output_dir': './models/model01/output',  # 单模型预测结果输出目录

    # ===== 特征 =====
    'feature_num': feature_num,       # '39' 或 '158+39'
    'sequence_length': sequence_length,  # 输入序列长度（过去N个交易日）

    # ===== 模型结构 =====
    'd_model': 256,
    'nhead': 4,
    'num_layers': 3,
    'dim_feedforward': 512,
    'dropout': 0.1,

    # ===== 训练超参 =====
    'batch_size': 4,
    'num_epochs': 50,
    'learning_rate': 1e-5,
    'weight_decay': 1e-5,
    'max_grad_norm': 5.0,
    'seed': 42,

    # ===== 学习率调度 =====
    'scheduler_start_factor': 1.0,
    'scheduler_end_factor': 0.2,

    # ===== 损失函数 =====
    'pairwise_weight': 1,
    'base_weight': 1.0,
    'top5_weight': 2.0,
    'loss_temperature': 1.0,
    'loss_k': 5,

    # ===== 验证集划分 =====
    'val_months': 2,  # 用最后N个月作为验证集

    # ===== 多进程 =====
    'num_processes': 10,  # 特征工程并行进程数上限

    # ===== 预测 =====
    'top_k': 5,  # 单模型输出的 Top-K（投票来源）
}
