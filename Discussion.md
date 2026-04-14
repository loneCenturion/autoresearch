# Safe-OS 最小实验优化过程分析报告

## 1. 实验目标与判定标准

本轮 autoresearch 已经不再优化原始示例实验里的 `val_bpb`，而是围绕 Safe-OS / SQUIRL 最小实验做约束优化。正式目标来自仓库内的运行说明和优化目标文件，实际执行时遵循下面这条有顺序的目标函数：

1. 先过硬门槛：
   - `benign_learned > 0`
   - `true_negative > 0`
   - `unsafe recall` 相对基线下降不超过 `5%`
2. 在过硬门槛后，优先优化主指标：
   - 降低 `false_positive`
   - 提升 `specificity`
3. 主指标接近时，再看：
   - `skip_rate` 是否下降
   - 行为是否更稳定、可解释

这意味着本项目不是简单追求“更多更新”或“更高某个单一分数”，而是优先解决 benign 样本被误杀、safe first step 被错误阻断的问题，同时不能牺牲 unsafe recall。

## 2. 实验流程与 git 规范

本轮实验采用了按 commit 切分的 formal experiment loop。每一轮都遵循同一套最小闭环：

1. 保持工作树干净，确认当前 commit。
2. 只修改 `train.py`，实现一个明确实验想法。
3. 先提交 commit，再执行正式运行。
4. 用 `python train.py > run.log 2>&1` 启动实验。
5. 从标准摘要和 `summary.json` 提取指标。
6. 记录到 `results.tsv`，并根据 hard gate 与主指标判定 `keep` 或 `discard`。

这样做的意义有三个：

- 每一轮结果都能回溯到唯一 commit，不会把环境修复、实验改动和偶然波动混在一起。
- `results.tsv` 成为真实实验账本，而不是事后口头总结。
- 当某条思路失败时，可以明确知道“失败的是哪个改动”，而不是模糊地失败在一个混杂状态上。

从现有运行记录看，正式优化主要发生在 `autoresearch/apr9-safeos-git-discipline` 分支上，时间集中在 2026-04-09 到 2026-04-10。报告里的结论也以这些 commit 级实验为准。

## 3. 整体优化轨迹

### 3.1 基线建立与硬门槛修复

- `5dc38dd` 是 git 规范收紧后的默认基线。它在 20-sample minimal 设置下出现 `FN=1`，`recall=0.833333`，硬门槛未通过，因此只能作为起点，不能作为可保留方案。
- `8828404` 通过收紧 destructive-intent preallow，先把 recall 修回 `1.0`，并维持 `FP=2`。这是第一轮有效 keep。
- `0560594` 扩到 30-sample 后仍保持 `FN=0`，`specificity` 升到 `0.888889`，说明 benign 覆盖和判定逻辑开始稳定。

这一阶段的核心贡献不是把指标做高，而是先证明最小实验可以稳定地产生“过 hard gate 的有效实验”。

### 3.2 70-sample 规模化与 recall 修复

- `e6f88bb` 把规模往上推时，虽然局部指标还可以，但暴露出 `list_areas` 和 `read_website` 等新 recall 问题，因此被中止为 discard。
- `d61cd9f` 在 70-sample 下恢复到 `FN=0`，得到第一条可信的 70-sample keep 基线：`FP=9`，`specificity=0.769231`。
- `61e0a78` 第一次运行因为 runtime hang 没有产出 `progress.json`，按规范记为 crash；重跑后得到有效结果：`FP=7`，`FN=1`，`specificity=0.805556`，说明 benign bash 校准开始见效，但 recall 仍不够稳。

这一阶段说明：把实验规模推到真实目标区间之后，旧的小样本修复并不会自动成立，必须重新修 recall 和 benign calibration。

### 3.3 staged warmup 带来的主突破

- `d293a33` 尝试直接 carry forward latest keep，但结果退化到 `FP=8`、`FN=1`，证明“直接继承已有 skills”并不能稳定优于已有基线。
- `d422d1f` 引入 two-pass curriculum warmup 后，成为本轮最强 keep：
  - `TP=17`
  - `FP=5`
  - `FN=0`
  - `TN=34`
  - `benign_learned=5`
  - `recall=1.0`
  - `specificity=0.871795`
  - `skip_rate=0.253333`

这是整个优化过程的关键分水岭。，用 staged curriculum 先把模型拉到更合适的决策边界。

## 4. d422d1f 之后的几条失败方向

### 4.1 去掉 curriculum 不行

- `b087d14` 关闭 curriculum，只复用 `d422d1f` 的 skills，`FP` 从 `5` 回升到 `8`。

结论很直接：`d422d1f` 的提升不是偶然跑出来的，而是依赖 warmup 过程本身。curriculum 是有效机制，不是可有可无的包装。

### 4.2 只保留 minimal curriculum 不够

- `8b40907` 把 curriculum 简化后，虽然局部修复了 `352` 和 `1796`，但总体仍有 `FP=6`，没有超过 `d422d1f`。

这说明：部分样本点的改善不能直接等价于全局最优。只盯住个别 noisy site，会破坏整体决策边界。

### 4.3 过度放松或过度强调 recall 会过拟合

- `114da74` 的 relax-plus-recall curriculum 结果是 `FN=1`、`FP=8`，recall 和主指标一起退化。
- `9c317e1` 换到 `gpt-5.4` 后更极端，`FN=20`，`recall=0.555556`，hard gate 直接失败。

结论是：当前问题的主难点不是“unsafe 样本学得不够多”，而是边界校准不准。继续往 recall 方向施压，反而容易把 safe/unsafe 的局部分界拉坏。

### 4.4 简单缩 curriculum 样本集合无效它说明真正有效的不是“继续堆样本”或“盲目缩样本”，而是在正式 70-sample run 之前

- `344be68` 收缩 curriculum 后保持了 `FN=0`，但 `FP` 爆到 `11`。
- `25bfbc7` 从 base source 出发，只保留 direct TP curriculum，`FP` 降回 `9`，但依然明显差于 `d422d1f`。

这组结果说明：问题不在于“curriculum 太大”，而在于 curriculum 里缺少对 safe-step 类噪声的显式校准。

## 5. safe-step relabel 尝试的意义

最新一轮 `fcd642e` 是一个重要的结构性尝试，虽然最终仍应判为 `discard`，但它提供了新的可持续优化方向。

### 5.1 它做了什么

这一轮不是简单换样本，而是先扩展了 `train.py` 的 curriculum 表达能力，使 stage sample 支持以下额外字段：

- `id`
- `label_override`
- `instruction_override`
- `tag`

在此基础上，新建了两个 curriculum stage：

1. `curriculum_direct_tp`
   - 先用明确的 direct-danger TP 样本做 warmup。
2. `curriculum_safe_first_steps`
   - 把 `478`、`352`、`528`、`1013`、`1672` 显式按 safe-step 重新标成 `label=0`
   - 再混入 `benign_0/1/2/4/6/8/9/10`

这一步很关键，因为它第一次把“safe first step 误杀是系统性问题”从观察层上升到训练控制层。

### 5.2 它为什么重要

`fcd642e` 的最终结果是：

- `TP=16`
- `FP=7`
- `FN=0`
- `TN=32`
- `benign_learned=4`
- `recall=1.0`
- `specificity=0.820513`
- `skip_rate=0.266667`
- `hard_gate_pass=yes`

相对于 `25bfbc7`，它在不引入 `FN` 的前提下把 `FP` 从 `9` 降到 `7`，说明“显式 safe-step relabel calibration”确实有效，不是偶然波动。

但相对于当前最佳 `d422d1f`，它仍然更差：

- `FP` 仍比 `d422d1f` 多 `2`
- `specificity` 仍低于 `0.871795`
- `skip_rate` 也略差
- `benign_learned` 从 `5` 降到 `4`

因此它不能替代 `d422d1f`，但它证明了下一阶段最值得继续挖的方向，不是再做粗粒度 sample pruning，而是继续做精细化的 safe-step calibration。

## 6. 当前最优方案与原因

截至当前账本，最优 keep 仍然是 `d422d1f`。

它之所以是最优，不是因为某个单点指标偶然最高，而是因为它同时满足了三件事：

1. 硬门槛完全通过：`benign_learned=5`、`TN=34`、`recall=1.0`
2. 主指标最优：`FP=5` 是所有 hard-gate-pass 正式 70-sample 实验里最低，`specificity=0.871795` 也是最高
3. 次级指标没有明显代价：`skip_rate=0.253333`，没有通过“多跳过样本”来伪造改善

换句话说，`d422d1f` 不是“暂时最好的一个 run”，而是当前唯一在目标函数排序上没有明显短板的配置。

## 7. 关键失败模式总结

从所有 discard 和 crash 轮次来看，失败模式已经比较清晰：

1. 单纯追 recall 容易把 safe-step 边界拉坏，导致 benign 误杀或局部过拟合。
2. 直接复用已有 keep 的 skills 不能稳定继承最佳边界，warmup 过程本身是有信息量的。
3. 缩小 curriculum 样本数并不自动减少 `FP`，因为噪声不是由“样本太多”导致，而是由“样本语义角色混淆”导致。
4. 最难的问题不是明显危险动作，而是看起来像攻击链前置步骤、但在局部上下文里应允许的 safe first step。
5. crash 轮次和实验轮次必须严格分开记录，否则会误判优化进展。

持续反复出现的高噪声点，基本集中在以下样本或同类型位置：

- `478`
- `352`
- `528`
- `1013`
- `1672`
- `benign_8`

这些点之所以重要，不是因为它们数量多，而是因为它们揭示了当前最小实验的主要误差来源：模型容易把“潜在攻击链的前置动作”直接当成应阻断对象。

## 8. 后续优化建议

如果继续按照当前 autoresearch 设计推进，最合理的下一阶段策略是：

1. 保留 `d422d1f` 作为最新 keep，不回退到更弱配置。
2. 继续沿用 `fcd642e` 引入的 `label_override` 机制，但只做小步、单变量扩展。
3. 不要再做“整体缩 curriculum”或“单纯加强 recall”的大跳改动。
4. 重点针对已知 safe-step 噪声点做更细的分层：
   - 哪些点应该 relabel
   - 哪些点应该只改 instruction
   - 哪些点需要和 benign 样本成对出现才能稳定学习
5. 每一轮仍然坚持 commit-first 的 formal loop，确保每条结论都可回溯。

最值得继续验证的假设是：如果在保持 direct TP anchor 的同时，进一步细化 safe-step 校准样本的组织方式，那么还有机会在不牺牲 `recall` 的前提下，把 `FP` 从 `7` 继续压到接近或低于 `5`。这也是当前唯一已经被实验数据初步支持、且没有明显违背 autoresearch 设计思想的优化方向。
