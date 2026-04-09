# autoresearch

这是 Safe-OS 最小实验的 autoresearch 控制文件。

## Setup

开始一个新实验时，和用户一起完成下面这些固定步骤：

1. **确定 run tag**
   - 用当天日期或阶段目标生成一个 tag，例如 `apr9-safeos-minimal`。
   - 分支 `autoresearch/<tag>` 不能已经存在。
2. **创建分支**
   - 从当前 `master` 拉出：`git checkout -b autoresearch/<tag>`
3. **完整阅读 in-scope 文件**
   - `README.md`
   - `prepare.py`
   - `train.py`
   - `最小实验运行和指标信息.md`
   - `最小实验的优化目标.md`
4. **验证外部输入并准备数据**
   - 先确认 `prepare.py` 里定义的外部路径都存在。
   - 然后运行：`python prepare.py`
5. **初始化 results.tsv**
   - 新建 `results.tsv`，只写表头。
6. **确认基线**
   - 第一轮必须先直接运行当前默认配置，建立基线，再开始改动。

## Experimentation

每轮实验都通过：

```bash
python train.py
```

来启动 Safe-OS / SQUIRL 最小实验。

### 你可以做的事

- 只修改 `train.py`
- 可以改的内容包括：
  - 默认跑 `minimal` 还是 `full`
  - `max_samples`
  - `checkpoint_every`
  - run 命名策略
  - 输出摘要与对比逻辑
  - staged experiment 策略
  - 任何能让这个控制台更稳定地服务当前目标实验的逻辑

### 你不能做的事

- 不修改 `prepare.py`
- 不修改 `最小实验运行和指标信息.md`
- 不修改 `最小实验的优化目标.md`
- 不直接改外部源数据
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

`results.tsv` 用 tab 分隔，不要用逗号。建议表头如下：

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

## Experiment loop

循环执行：

1. 看当前 git 状态和当前 commit。
2. 只在 `train.py` 里实现一个实验想法。
3. 提交 commit。
4. 运行：`python train.py > run.log 2>&1`
5. 如果摘要没出来，读取 `run.log` 末尾，判断是路径问题、环境问题还是实验失败。
6. 把结果记到 `results.tsv`。
7. 决定 keep / discard：
   - `hard_gate_pass=no`：直接 `discard`
   - 都过硬门槛时：优先看 `fp` 是否更低、`specificity` 是否更高
   - 如果主指标差不多，再看 `skip_rate`
8. 如果更好，就保留当前 commit 继续往前；如果更差或无效，就回退。

## First run

第一轮必须是当前默认 `train.py`，不要先改代码。

## Crashes

- 如果是显然的配置错误、路径错误、输出目录冲突，修掉后重跑。
- 如果实验没有产出 `progress.json`，这轮记为 `crash`。
- 如果连续几次都只是环境问题，不要假装这是实验进展。
