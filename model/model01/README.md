# Model01 - StockTransformer 排序学习模型

## 模型简介

基于 Transformer 的排序学习（Learning-to-Rank）选股模型。输入沪深300成分股过去60个交易日的量价特征序列，输出每只股票的排序分数，取分数最高的 Top10 作为候选。

## 模型架构

StockTransformer，主要模块：
- 输入投影 + 正弦位置编码
- 时序 TransformerEncoder（3层，4头，d_model=256）：提取单股票历史序列表示
- FeatureAttention：对时间维做注意力聚合，将序列压缩为单向量
- CrossStockAttention：同一交易日内股票间交互，建模相对强弱
- 排序头（MLP）：输出标量排序分数

输入形状：`[batch, num_stocks, 60, 197]`
输出形状：`[batch, num_stocks]`

## 特征

共 197 维（158 + 39）：
- 158 个 Qlib 风格 Alpha 特征（K线形态、动量、波动率、回归、相关性、量价背离等）
- 39 个经典技术指标（SMA/EMA/MACD/RSI/KDJ/BOLL/ATR/OBV 等，依赖 TA-Lib）

## 标签

`label = (open_t5 - open_t1) / open_t1`

即未来第5个交易日开盘价相对未来第1个交易日开盘价的收益率。

## 损失函数

WeightedRankingLoss = Listwise CE + Pairwise Loss，对真实 Top5 样本加权 2.0。

## 训练配置

| 参数 | 值 |
|------|------|
| sequence_length | 60 |
| d_model | 256 |
| nhead | 4 |
| num_layers | 3 |
| batch_size | 4 |
| num_epochs | 50 |
| learning_rate | 1e-5 |
| optimizer | AdamW (weight_decay=1e-5) |
| scheduler | LinearLR (1.0 → 0.2) |

## 输出

预测时输出 Top10 股票到 `models/model01/output/result.csv`，包含 stock_id、rank、score、model_id 四列，供投票脚本使用。
