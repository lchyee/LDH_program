# 回测系统说明 & 云端部署指南

滚动重训回测系统：从 2026 年第 1 周到最后一个完整周，**每周用该周之前的全部历史数据重新训练全部 6 个模型**，预测 Top5 选股，并与沪深300基准对比。

## 这套系统在做什么

每个回测周（自然周，周一~周五；遇节假日短周按实际交易日）：

1. 计算 `base_date`（该周首个交易日的前一交易日）
2. **清理残留产物**（旧模型权重、旧评分文件）——防止数据泄露与前视偏差
3. 重建 `train.csv`（数据截止到 `base_date`）和 `test.csv`（该周）
4. 滚动重训全部模型（复用各模型 `train.py`，不改一行模型代码）
5. 各模型预测 → 投票集成
6. 评分并存档：集成 / 各单模型 / 沪深300 的当周收益与排序

**为什么无泄露**：模型标签是"未来 5 个交易日收益"，由 per-stock `shift(-5)` + `dropna` 构造。当 `train.csv` 截止到 `base_date` 时，需要未来数据的样本会变成 NaN 被丢弃，模型训练时根本看不到 `base_date` 之后的任何数据。预测基准日也正好是 `base_date`，买入日 = base_date 的下一个交易日 = 该周首日，与评分口径完全对齐。

## 产出

运行后在 `backtest/results/`：

- `week_NN/score.json` — 每周评分（集成、各模型收益、排序明细）
- `week_NN/rankings/*.csv` — 每周各模型、集成的完整预测排序
- `week_NN/meta.txt` — 该周训练/测试区间
- `summary/weekly_returns.csv` — 每周收益（基准/集成/各模型）
- `summary/cumulative_nav.csv` — 累计净值曲线数值
- `summary/all_rankings.csv` — 所有周所有模型预测排序汇总
- `summary/nav_curve.png` — **净值折线图**（沪深300 vs 集成 vs 各单模型）

## 收益口径（与 test/score_self.py 严格一致）

- 单股周收益 = (末日开盘 − 首日开盘) / 首日开盘
- 沪深300基准 = 全部成分股周收益等权平均
- 集成收益 = Σ(单股收益 × vote权重)，权重原样读取（可能 <1，未投部分视为现金）
- 单模型收益 = Top5 等权平均（每只 20%）
- 累计净值 = 逐周复利，起点 1.0

---

## 本地小范围验证（先跑通再上云）

编辑 `backtest/config.py`：

```python
WEEKS = [1, 2]                    # 只跑前 2 周
MODELS = ['model01', 'model02']   # 只跑 2 个模型
```

然后：

```powershell
# Windows PowerShell（项目根目录）
$env:PYTHONIOENCODING='utf-8'
python backtest/build_calendar.py   # 看日历对不对
python backtest/run_backtest.py     # 跑回测（会训练模型，约10-40分钟）
python backtest/aggregate.py        # 出汇总和图
```

确认 `backtest/results/summary/` 下有数据和图后，把 config.py 改回全量：

```python
WEEKS = None      # 全部 20 周
MODELS = None     # 全部 6 个模型
```

---

## 云端部署（通用 GPU 租用平台）

适用于 AutoDL、揽睿星舟、Featurize、矩池云等任何提供 Linux + GPU 的平台。

### 1. 准备代码和数据

```bash
# 方式 A：git 克隆（代码已推到 GitHub）
git clone <你的仓库地址>
cd <仓库名>

# 方式 B：直接上传整个项目文件夹到云主机
```

**数据文件不在 git 里**（被 .gitignore 排除），需要单独准备 `data/stock_data.csv`：

```bash
# 若云主机能联网取数：
python get_stock_data.py
# 或：从本地用 scp / 平台的文件上传功能把 data/stock_data.csv 传上去
```

### 2. 装依赖

```bash
pip install pandas numpy joblib scikit-learn matplotlib torch
# TA-Lib 需要系统库，conda 装最省事：
conda install -c conda-forge ta-lib
```

### 3. 一键运行

```bash
bash backtest/run_cloud.sh
```

脚本会：打印日历 → 跑全部回测 → 聚合画图。

### 4. 后台长跑（推荐，避免 SSH 断开中断）

回测要每周重训 6 个模型 × 20 周 ≈ **120 次训练**，可能跑数小时到十几小时。用 `tmux` 后台跑：

```bash
tmux new -s backtest
bash backtest/run_cloud.sh
# 按 Ctrl+B 然后 D 脱离；重连用 tmux attach -t backtest

# 或 nohup 方式
nohup bash backtest/run_cloud.sh > backtest_run.log 2>&1 &
tail -f backtest_run.log     # 实时看进度
```

### 5. 断点续跑

中途断了不用怕：**已完成的周会自动跳过**（靠 `week_NN/score.json` 判断）。重新运行 `run_cloud.sh` 即可从中断处继续。

### 6. 取回结果

```bash
tar czf backtest_results.tar.gz backtest/results/
# 再用平台的下载功能 / scp 取回本地
```

---

## 常见问题

**Q：云端中文图变方框？**
A：`aggregate.py` 会自动检测中文字体，找不到就回退英文标签，图仍正常生成。想要中文图就在云主机装字体：`apt install fonts-wqy-microhei` 或 `conda install -c conda-forge font-noto-cjk`。

**Q：某个模型某周训练失败怎么办？**
A：该模型当周不参与，不影响其他模型和后续周。日志里会标 `[失败]`。

**Q：能只重算某一周吗？**
A：删掉对应的 `backtest/results/week_NN/score.json`，再跑 `run_backtest.py`，它会只重算缺失的周。

**Q：耗时太长想先看部分结果？**
A：任何时候都能运行 `python backtest/aggregate.py`，它会聚合**当前已完成**的周，出阶段性曲线。
