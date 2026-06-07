# 沪深300 多模型集成选股系统

面向沪深300成分股的**排序学习(learning-to-rank)选股**方案：每个交易日对全部成分股打分排序，
最终输出 Top5 股票（等权，各 0.2）。本项目采用**多模型集成**架构——6 个不同的深度学习模型
各自独立排序，再通过投票融合得到最终结果。

> 本项目从单模型(StockTransformer)演进为 6 模型集成 + 滚动回测系统。
> 旧版单模型 README 已废弃，本文档对应当前架构。

---

## 1. 整体架构

```
原始行情(stock_data.csv)
        │  split_train_test.py 按日期切分
        ▼
   train.csv  ──────────────┐
        │                   │
        ▼                   ▼
  6 个模型各自训练      6 个模型各自预测
  (train.py 调度)      (test.py 调度)
        │                   │
        ▼                   ▼
  各模型 result.csv ──► vote.py 投票融合 ──► output/result.csv (最终Top5)
```

**核心理念**：不同模型有不同的"选股性格"(有的擅长抓最强的几只、有的擅长前20名)，
集成投票综合它们的判断，比单一模型更稳健。

### 6 个模型

| 模型目录 | 算法 | 特点 |
|---|---|---|
| `model/model01` | StockTransformer(自定义) | 时序编码 + 股票间交互注意力，序列长度60 |
| `model/model02` | iTransformer | 倒置 Transformer，序列长度20 |
| `model/model03` | DLinear | 线性分解模型，轻量 |
| `model/model04` | TimesNet | 时序周期建模 |
| `model/model05` | TiDE | 长期依赖编码 |
| `model/model07` | MICN | 多尺度卷积，序列长度40 |

每个模型目录结构统一：
```
model/modelXX/
├── src/
│   ├── train.py      训练脚本（读 data/train.csv，输出到 checkpoint/）
│   ├── predict.py    预测脚本（读 data/train.csv，输出 output/result.csv）
│   └── <算法>.py     模型定义
├── checkpoint/       训练产物：best_model.pth, scaler.pkl, feature_cols.pkl
└── output/           预测产物：result.csv（该模型的全量排名）
```

---

## 2. 目录结构

```
THU-BDC2026-5/
├── data/
│   ├── stock_data.csv          原始行情（全部历史，get_stock_data.py 抓取）
│   ├── train.csv / test.csv    split_train_test.py 切分产物
│   ├── split_train_test.py     按日期切分训练/测试集
│   └── hs300_stock_list.csv    成分股代码↔名称对照
├── model/                      6 个模型（见上）
├── shared_components/          模型共享组件
│   ├── feature_utils.py        统一特征工程（158 Alpha + 39 技术指标 + 市场相对特征）
│   ├── layers/                 Transformer 等网络层
│   └── nn_utils/               训练辅助
├── code/src/
│   ├── train.py                训练调度器：发现 model/ 下所有模型并逐个训练
│   ├── test.py                 预测调度器：逐个预测 + 调 vote.py 融合
│   ├── vote.py                 投票融合（核心，见第4节）
│   └── featurework.py          特征工程入口
├── backtest/                   滚动重训回测系统（见 backtest/README.md）
├── test/                       赛事方评分脚本（score_self.py 等）
├── output/result.csv           最终融合结果（提交用）
├── train.sh / test.sh          一键训练 / 一键预测
├── Dockerfile / docker-compose.yml   容器化打包
└── get_stock_data.py           数据抓取（Baostock）
```

---

## 3. 完整运行流程

### 环境准备

```bash
# 方式1：uv（推荐）
uv sync && source .venv/bin/activate

# 方式2：conda/pip
pip install pandas numpy scikit-learn matplotlib joblib torch
conda install -c conda-forge ta-lib    # TA-Lib 需系统库，conda 最省事
```

### 标准流程

```bash
# 1. 抓取数据（首次/更新时）
python get_stock_data.py

# 2. 切分训练/测试集（决定"预测哪几天"——见下方说明）
python data/split_train_test.py --train-end 2026-05-29 --test-start 2026-06-01 --test-end 2026-06-05

# 3. 训练全部 6 个模型
bash train.sh        # = python code/src/train.py

# 4. 预测 + 投票融合
bash test.sh         # = python code/src/test.py，最终结果在 output/result.csv
```

### 关于"预测哪几天"

**只有 `split_train_test.py` 决定预测日期**，train.py/test.py 都只读 `train.csv`、不关心日期：
- `--train-end` = **预测基准日**：模型用截止到这天的数据，预测之后5个交易日
- `--test-start/--test-end` = 评测区间（仅本地用 score_self.py 算分，不影响预测）

例：`--train-end 2026-05-29` → 模型基于 5/29 及之前的数据，预测 6/1~6/5 的最优5只。

---

## 4. 投票融合机制（code/src/vote.py）

最终的 5 只股票由投票融合产生，分三层（经 20 周回测调优，累计约 53%、夏普 0.71）：

### 第1层：模型内部分段评分
每个模型对自己预测的前 50 名按"名次→得分"的**分段曲线**打分，再归一化到总和=1
（保证各模型贡献相等、可比）。各模型曲线形状不同，来自对该模型"第几名最准"的实测分析：

| 模型 | 分段特点（依据实测） |
|---|---|
| model01 | 前5名最高(4.5)、6-10名(3.0)、之后递减——前排最准 |
| model02 | 第1-5名压低(3.0)、6-20名最高(5.0)——好票分散在前20 |
| model03 | 整体平缓(5.0→3.0)——各档差不多 |
| model04 | 前5最高(5.0)、6-20名(4.0) |
| model05 | 第1名独高(5.0)、其余拉平(4.0)——最自信一只最准 |
| model07 | 前5名压低(2.5)、6名起升高——前5是噪声，信号在中段 |

### 第2层：固定模型权重
按各模型整体表现分配固定权重（回测调优）：

| 模型 | 权重 |
|---|---|
| model01 / model02 | 各 22% |
| model07 | 18% |
| model04 / model05 | 各 14% |
| model03 | 10% |

最终得分 = Σ(模型权重 × 该模型归一化分段得分)。

### 第3层：选 Top5 + 等权分仓
取融合得分最高的 5 只，**等权分配（各 0.2）**，输出到 `output/result.csv`。
（含分歧惩罚：某股在各模型间名次分歧过大时降分。）

> 注：分段曲线和权重是在**本项目这 6 个模型、特定时间段**上调出的，换一套模型需重新分析。

---

## 5. 特征与标签

**特征**（`shared_components/feature_utils.py` 统一生成）：
- 158 个 Alpha 类因子（量价衍生）
- 39 个 TA-Lib 技术指标（MACD、RSI、布林带等）
- 市场相对特征（个股 vs 当日市场均值的超额、排名等，用于横截面比较）

**标签**：未来 5 个交易日的**超额收益**
- `raw_return = (open_t5 - open_t1) / open_t1`（次日开盘买、第5日开盘卖）
- `target = raw_return - 当日市场平均`（学的是"跑赢市场"，不是绝对涨跌）
- 由 `shift(-5) + dropna` 构造，末尾需要未来数据的样本自动丢弃 → **天然防泄露**

**收益口径**与赛事方 `test/score_self.py` 一致：(末日开盘−首日开盘)/首日开盘，按权重加权。

---

## 6. 回测系统（backtest/）

独立的**滚动重训回测**：从某年第1周到最后一周，每周用"该周之前的全部数据"重新训练全部模型、
预测、融合、评分，与沪深300基准对比。核心保证**无未来数据泄露**，结果可信。

```bash
bash backtest/run_cloud.sh    # 一键：回测全部周 + 出图 + 出报表
```

产出：累计净值曲线、每周涨幅图、Alpha/Beta 选股能力诊断、逐周逐模型明细报表等，
全部在 `backtest/results/`。详见 [backtest/README.md](backtest/README.md)。

---

## 7. Docker 打包（赛事提交）

```bash
docker compose build          # 构建镜像
docker compose run --rm app   # 运行（训练+预测）
```

`Dockerfile` 已配置国内镜像源加速依赖安装。详见 [GUIDE.md](GUIDE.md)。

---

## 8. 常见问题

**TA-Lib 安装失败**：需先装系统层 ta-lib 库。Linux 源码安装：
```bash
wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
  tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib && \
  ./configure --prefix=/usr && make -j1 && make install && cd .. && \
  rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
```
或直接 `conda install -c conda-forge ta-lib`。

**某个模型训练/预测失败**：调度器会跳过该模型继续跑其余模型，不影响整体；日志会标注。

**GPU/CPU 选择**：自动按 CUDA → CPU 选择，无 GPU 也能跑（慢）。
5系新卡(Blackwell)需 CUDA 12.8+。

**多进程**：train.py/predict.py 用 spawn 模式，请通过脚本入口运行，勿在交互环境直接多进程调用。

**数据/模型权重不在 git 里**：`data/*.csv`、`checkpoint/*.pth` 等被 .gitignore 排除，
clone 后需重新抓数据+训练。

