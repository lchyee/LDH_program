# 股票选股 — 滚动重训回测系统

这是一套**滚动重训(walk-forward)回测系统**：从某年第1个交易周到最后一周，
**每周都用"该周之前的全部历史数据"重新训练全部模型**，预测当周 Top5 选股，
再与沪深300基准对比。核心保证：训练/预测时**绝不使用未来数据**，回测结果可信。

> 本系统不修改任何模型代码，全程通过子进程调用项目里现成的
> `split_train_test.py`、各模型 `train.py`/`predict.py`、`code/src/vote.py`，
> 复用竞赛的真实流程，最大程度保证回测=实盘。

---

## 一、快速开始

```bash
# 在项目根目录（不是 backtest/ 里）执行：
bash backtest/run_cloud.sh
```

它会依次完成：打印回测日历 → 滚动回测全部周 → 聚合画图 → 生成各类报表。
结果全部在 `backtest/results/`。

**先小范围验证再全量**：编辑 `backtest/config.py`，把
`WEEKS=[1,2]`、`MODELS=['model01','model02']` 跑通（几分钟），
确认无误后改回 `WEEKS=None`、`MODELS=None`（全量，数小时，建议云端 GPU + tmux）。

云端部署详见 [README_CLOUD.md](README_CLOUD.md)。

---

## 二、文件清单

### 核心流程（run_cloud.sh 按顺序调用）

| 文件 | 作用 |
|---|---|
| `config.py` | **配置中心**：路径、回测范围、模型/周筛选。所有脚本都读它 |
| `build_calendar.py` | 从 stock_data.csv 切自然周，算每周"预测基准日" |
| `run_backtest.py` | **主调度器**：逐周重建数据→训练→预测→投票→评分存档（支持断点续跑） |
| `score_week.py` | 单周评分（被 run_backtest 调用），口径对齐 score_self.py |
| `aggregate.py` | 聚合20周结果 → 累计净值曲线图 + 汇总 CSV |
| `plot_weekly_returns.py` | 每周涨幅折线图（非累计） |
| `analyze_alpha_beta.py` | 选股能力诊断：Beta/Alpha 散点图（判断是真选股还是放大波动） |
| `make_report.py` | 逐周逐模型明细 + 各模型整体表现汇总 |
| `make_per_model_report.py` | 为每个模型每周生成 our-score.py 风格的详细报表 |

### 实验脚本（一次性调参用，**队友可忽略/删除**）

这些是为**我们这套模型**调投票参数时用的，成果已固化进 `code/src/vote.py`。
它们里面**硬编码了我们6个模型的特性**，换一套模型不适用。

| 文件 | 当时用途 |
|---|---|
| `eval_segment_scoring.py` | 试各模型"名次→得分"的分段评分 |
| `eval_consensus.py` | 试"只买强共识股"策略 |
| `eval_weight_sweep.py` | 扫描融合权重力度 |
| `vote_v2_backtest.py` | 完整融合方案回测 |
| `analyze_topN_segments.py` | 各模型 Top-N 分段表现分析 |

### 产出目录 `results/`

```
results/
├── week_01/ ... week_NN/        每周存档
│   ├── score.json               该周完整评分（含 status: complete/partial）
│   ├── meta.txt                 该周训练/测试区间
│   └── rankings/                各模型 + 集成的完整预测排序 csv
├── summary/                     聚合结果
│   ├── nav_curve.png            累计净值曲线图
│   ├── weekly_returns_curve.png 每周涨幅图
│   ├── alpha_beta_scatter.png   选股能力诊断图
│   ├── weekly_returns.csv       每周收益
│   ├── cumulative_nav.csv       累计净值数值
│   ├── all_rankings.csv         所有周所有模型预测排序汇总
│   └── model_summary.csv        各模型整体表现
└── per_model/<model>/week_NN.txt  每模型每周详细报表
```

---

## 三、移植到另一套模型 —— 注意事项（给队友 / 队友的 AI 看）

把整个 `backtest/` 文件夹复制到另一个项目即可使用，**前提是那个项目的目录结构和接口约定与本项目一致**。下面是这套系统对宿主项目的全部假设，逐条核对。

### 1. 必须满足的目录结构

```
项目根/
├── data/
│   ├── stock_data.csv          原始行情（全部历史，回测从这里切）
│   ├── split_train_test.py     切分脚本（见下方接口要求）
│   └── hs300_stock_list.csv    成分股名称表（可选，用于报表显示股票名）
├── model/
│   ├── <模型名>/src/train.py   每个模型的训练脚本
│   └── <模型名>/src/predict.py 每个模型的预测脚本
├── code/src/vote.py            投票融合脚本
└── output/result.csv           vote.py 输出的最终结果
```

**模型名和数量不限**：`config.discover_models()` 会自动发现 `model/` 下所有
含 `src/train.py` + `src/predict.py` 的目录。模型叫 modelA/modelB 还是 m1/m2 都行，
有3个还是8个都行——**不需要改回测代码**。

### 2. 接口约定（最容易踩坑，逐条验证）

| 约定 | 要求 | 不满足会怎样 |
|---|---|---|
| **split_train_test.py 参数** | 须支持 `--input --output-dir --train-start --train-end --test-start --test-end`，按日期 `<=` 过滤切出 train.csv/test.csv | 数据重建失败，回测跑不起来 |
| **predict.py 数据源** | **只读 `data/train.csv`，绝不读 test.csv**；用 train.csv 最后一天作预测基准 | ⚠️ **未来数据泄露**，结果虚高失真 |
| **predict.py 输出** | 写 `model/<名>/output/result.csv`，含 `stock_id` + `rank`（或 `score`） | 评分读不到，该模型被跳过 |
| **vote.py 输出** | 写 `output/result.csv`，含 `stock_id,weight` 两列 | 集成评分失败 |
| **数据列名** | stock_data.csv 须有中文列名：`股票代码 日期 开盘 收盘`（至少这4个） | 收益算不出 |
| **收益口径** | "(末日开盘−首日开盘)/首日开盘"，需与 score_self.py 一致 | 与竞赛口径不符 |

**最关键的一条**：所有 `predict.py` 必须**只读 train.csv**。这是无泄露的根基——
回测每周把 train.csv 截到"基准日"，predict 只看 train.csv 就自动只用了历史数据。
若某个 predict.py 偷读了 test.csv 或 stock_data.csv，泄露立刻发生且隐蔽。
**移植后第一件事：grep 所有 predict.py 确认没有 `test.csv` / `stock_data.csv` 字样。**

### 3. 运行环境

- Python 3.9+，依赖：`pandas numpy scikit-learn matplotlib joblib` + 各模型自身依赖（torch、TA-Lib 等）
- 所有子进程以**项目根目录为 cwd** 运行（有的模型 config 用相对路径，必须如此）
- 中文图：云端无中文字体时 `aggregate.py` 自动回退英文标签，不影响数据

### 4. ⚠️ 关于 code/src/vote.py 的特别提醒

本项目的 `code/src/vote.py` 已被**针对我们这6个模型深度定制**：
- 每个模型有**手工调的"名次→得分"分段曲线**（`SEGMENT_SCORERS`）
- 有**固定模型权重**（`MODEL_WEIGHTS`，如 model01=22%…model03=10%）

这些参数是在**我们的20周数据上调出来的，只适合我们的模型**。队友有两种选择：

- **(推荐) 用队友自己的通用 vote.py**：回测会调用宿主项目的 `code/src/vote.py`，
  队友保留自己那份即可，回测照常工作，评估的就是他们的融合逻辑。
- **若想复用我们的分段+权重思路**：必须用 `eval_*` 实验脚本在**队友自己的数据**上
  重新分析每个模型"第几名最准"、重新调权重——直接套用我们的数字无效（那是过拟合我们模型的结果）。

### 5. 移植后的验证清单（建议队友的 AI 照做）

1. 确认目录结构符合第1点
2. `grep -rn "test.csv\|stock_data.csv" model/*/src/predict.py` —— 确认 predict 不读未来数据
3. 改 `config.py`：`WEEKS=[1,2]`、`MODELS=` 设成队友的2个模型名
4. `python backtest/build_calendar.py` —— 看日历正常（确认 `BACKTEST_YEAR` 对）
5. `python backtest/run_backtest.py` —— 跑通2周，看是否正常输出 `完成。基准=… 集成=…`
6. 验证投票模型数正确（应等于参与模型数，不能莫名多出来）
7. 全部正常后改回 `WEEKS=None, MODELS=None` 跑全量

### 6. config.py 里可能要改的参数

| 参数 | 说明 |
|---|---|
| `BACKTEST_YEAR` | 回测哪一年（默认 2026）。换数据年份要改这里 |
| `TRAIN_START` | 训练数据起始日（扩展窗口起点） |
| `TOP_N` | 选几只股票（默认5，与竞赛一致） |
| `INCLUDE_SHORT_WEEKS` | 节假日短周（交易日<5）是否回测 |
| `WEEKS` / `MODELS` | 回测范围筛选，全量设 None |

---

## 四、无数据泄露说明（系统的核心价值）

每个回测周 N：
1. `base_date` = 该周首个交易日的**前一个**交易日
2. train.csv 截到 base_date（含），test.csv = 该周（base_date 之后）
3. 模型只读 train.csv 训练/预测 → 决策时看不到 base_date 之后任何数据
4. 标签是"未来5日收益"，由 `shift(-5)+dropna` 构造，截断处自动丢弃需要未来数据的样本

四重保证下，训练、预测、投票全程不触及未来数据。评分用的 test.csv 只在**算完预测之后**
才被读取，且仅用于打分，不回流到任何模型。
