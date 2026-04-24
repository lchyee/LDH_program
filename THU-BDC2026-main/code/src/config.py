sequence_length = 45
feature_num = '158+39'

config = {
    'sequence_length': sequence_length,
    'd_model': 128,          # 小！快！强！
    'nhead': 4,              # 必须整除 d_model
    'num_layers': 2,         # 两层足够，不慢
    'dim_feedforward': 256,  # 小网络，速度起飞
    'batch_size': 4,         # 保持小batch，不爆显存
    'num_epochs': 50,
    'learning_rate': 5e-5,   # 比你原来大，学得快
    'dropout': 0.1,
    'feature_num': feature_num,
    'max_grad_norm': 5.0,

    'pairwise_weight': 2,    # 加强排序
    'base_weight': 1.0,
    'top5_weight': 4.0,      # 重点抓赚钱股票

    'output_dir': f'./model/{sequence_length}_{feature_num}',
    'data_path': './data',
}