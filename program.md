# autoresearch

这是 Safe-OS 最小实验的 autoresearch 控制文件。

> 过渡说明：根目录 `AGENTS.md` 当前保留为 OMX/Codex 入口；本文件仍然是本仓库项目特定规则的 source of truth，并通过根 `AGENTS.md` 显式桥接引用。若后续重新初始化 OMX 或迁移到新环境，应先恢复这层 bridge，再继续实验工作流。

## Setup

开始一个新实验时，和用户一起完成下面这些固定步骤：

1. **确定 run tag**
   - 用当天日期或阶段目标生成一个 tag，例如 `apr9-safeos-minimal`。
   - 分支 `autoresearch/<tag>` 不能已经存在。
2. **创建实验分支（强制）**
   - 从当前 `master` 拉出：`git checkout -b autoresearch/<tag>`
   - 正式实验不允许直接在 `master` 上执行。
   - 如果当前不在 `autoresearch/<tag>` 分支，先停下，不继续跑实验。
3. **确认工作区状态（强制）**
   - 运行：`git status --short`
   - 第一轮基线 run 之前，工作区必须是干净的。
   - 带未提交改动做的运行，只能算调试，不算正式实验轮次。
4. **完整阅读 in-scope 文件**
   - `README.md`
   - `prepare.py`
   - `train.py`
   - `最小实验运行和指标信息.md`
   - `最小实验的优化目标.md`
5. **验证外部输入并准备数据**
   - 先确认 `prepare.py` 里定义的外部路径都存在。
   - 然后运行：`python prepare.py`
6. **初始化 results.tsv**
   - 新建 `results.tsv`，只写表头。
7. **确认基线**
   - 第一轮必须先直接运行当前默认配置，建立基线，再开始改动。

## Git 版本管理（强制）

每一轮正式实验都必须满足下面这些要求：

1. **一轮实验对应一个明确 commit**
   - 先改代码，再提交，再跑实验。
   - 没有 commit 的改动，不允许作为正式实验结果记录。
2. **正式实验前必须记录 3 个 git 信息**
   - `git branch --show-current`
   - `git status --short`
   - `git rev-parse --short HEAD`
3. **禁止在 dirty worktree 上连续跑编号实验**
   - 如果要继续改，先提交一个新 commit，再开始下一轮。
   - 如果只是修环境、排查路径、验证网络，不记为正式轮次。
4. **`run_name` 必须能追溯到代码版本**
   - 推荐把阶段 tag、样本规模、或 `commit short hash` 放进 run 名称。
   - 不能让结果目录和代码状态脱钩。
5. **每轮实验结束后必须落盘到 `results.tsv`**
   - 一轮一行，不能事后补一堆模糊记录。
   - `status` 只能写 `keep` / `discard` / `crash`。

## Experimentation

每轮实验都通过：

```bash
python train.py
```

来启动 Safe-OS / SQUIRL 最小实验。

正式实验只能在“已提交 commit”的状态下运行：

- 如果修改了 `train.py` 或本地 wrapper 校准代码，先提交，再跑下一轮。
- 如果当前只是临时调试，不要给 run 编号，也不要写进 `results.tsv`。
- 任何被拿来比较指标的 run，都必须能回溯到一个具体 commit。
- `train.py` 默认应拒绝 `master` / 非 `autoresearch/*` 分支 / dirty worktree；只有显式调试时才允许绕过。

### 你可以做的事

- 优先修改 `train.py`
- 如果问题属于本仓库的本地兼容层 / 校准层，可以修改 `run_evolve_train.py`
- 可以改的内容包括：
  - 默认跑 `minimal` 还是 `full`
  - `max_samples`
  - `checkpoint_every`
  - run 命名策略
  - 输出摘要与对比逻辑
  - staged experiment 策略
  - 本地 wrapper 的 API 兼容、上下文传递、最小校准逻辑
  - 任何能让这个控制台更稳定地服务当前目标实验的逻辑

### 你不能做的事

- 不修改 `prepare.py`
- 不修改 `最小实验运行和指标信息.md`
- 不修改 `最小实验的优化目标.md`
- 不直接改外部源数据
- 不直接改 `/data/Agent_Defense/code/` 下的上游 SQUIRL / Agent-SafetyBench 代码
- 不新增依赖

## 优化目标

目标不是单一 metric，而是下面这个有顺序的目标函数：

1. 先过硬门槛：
   - `benign_learned > 0`
   - `true_negative > 0`
   - `unsafe recall` 相对基线下降不超过 `5%`
2. 在过硬门槛的前提下：
   - 优先减少 `false_positive`
   - 提升 `specificity`
3. 然后再看：
   - `skip_rate` 是否下降
   - 指标是否更稳定、更可解释

当前最小实验不是为了优先追求 cross-benchmark transfer，也不是为了先做完整成本曲线。

## 输出格式

`train.py` 结束后会打印统一摘要。重点关注这些字段：

```text
false_positive:
true_negative:
benign_learned:
recall:
specificity:
skip_rate:
hard_gate_pass:
```

可以直接用：

```bash
grep "^false_positive:\|^true_negative:\|^benign_learned:\|^recall:\|^specificity:\|^skip_rate:\|^hard_gate_pass:" run.log
```

来抽取关键结果。

## Logging results

`results.tsv` 用 tab 分隔，不要用逗号。它不是可选项，而是正式实验账本。建议表头如下：

```text
commit	run_name	processed	skipped	evaluated	tp	fp	fn	tn	benign_learned	precision	recall	specificity	accuracy	skip_rate	failure_rate	hard_gate_pass	status	description
```

字段含义：

1. `commit`：短 commit hash
2. `run_name`：对应的本地 run 目录名
3. `processed/skipped/evaluated`：样本流量概况
4. `tp/fp/fn/tn`：四格统计
5. `benign_learned`：良性样本学习次数
6. `precision/recall/specificity/accuracy/skip_rate/failure_rate`：派生指标
7. `hard_gate_pass`：`yes/no/unknown`
8. `status`：`keep` / `discard` / `crash`
9. `description`：这一轮改了什么

额外要求：

- `commit` 必须来自运行前的 `git rev-parse --short HEAD`
- `run_name` 必须和这一轮实验目录一致
- 没产出 `summary.json` / `progress.json` 的运行，记为 `crash`
- 不允许把多个 commit 的结果混记到同一行

## Experiment loop

循环执行：

1. 确认当前分支不是 `master`，并记录当前 `git status` / `commit`。
2. 实现一个清晰、单一的实验想法；改完以后先提交 commit。
3. 运行：`python train.py > run.log 2>&1`
4. 如果摘要没出来，读取 `run.log` 末尾，判断是路径问题、环境问题还是实验失败。
5. 立刻把结果记到 `results.tsv`。
6. 决定 keep / discard：
   - `hard_gate_pass=no`：直接 `discard`
   - 都过硬门槛时：优先看 `fp` 是否更低、`specificity` 是否更高
   - 如果主指标差不多，再看 `skip_rate`
7. 如果更好，就保留当前 commit 继续往前；如果更差或无效，就回退到上一个 `keep` commit，而不是在脏工作区上继续叠改动。

## First run

第一轮必须是当前默认 `train.py`，不要先改代码。

## Crashes

- 如果是显然的配置错误、路径错误、输出目录冲突，修掉后重跑。
- 如果实验没有产出 `progress.json`，这轮记为 `crash`。
- 如果连续几次都只是环境问题，不要假装这是实验进展。
