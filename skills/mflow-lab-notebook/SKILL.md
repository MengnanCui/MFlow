---
name: mflow-lab-notebook
description: Keep a research notebook and a lessons file for an MLIP project run through MFlow. Use at the start of any multi-step campaign (training, active learning, property study), after every command whose result changes a decision, and whenever a failure is diagnosed — so the next session (human or agent) can pick up without re-deriving anything.
---

# 实验记录与经验沉淀

两个文件，放在项目（或活动）根目录。模板在 `references/`。

| 文件 | 写什么 | 规则 |
|---|---|---|
| `NOTEBOOK.md` | 这个活动里发生了什么：问题、假设、命令、产物路径、结论 | **只追加**；每条带日期；数字必须指向产物（JSON 字段或文件） |
| `LESSONS.md` | 以后还会用到的经验 | **通过门槛才写**；一条一个教训；旧条目就地更新，不重复 |

## NOTEBOOK.md：每条记录

```
## 2026-10-10 14:05 · <一句话标题>
问题：要回答什么
做法：mflow … （完整命令）
产物：<dir>/<file>.json → status=…；关键字段 … = …
结论：…（只写证据支持的）
下一步：…
```

- 失败也记：失败的命令、第一条 fail、怎么查的（见 `mflow-failure-triage`）。
- 有竞争假设时并列写出，并写下哪个试验能区分它们。
- 用户做的决定（选哪个模式、接受哪个 warn）原话记下。

## LESSONS.md：门槛

只有同时满足才写：
1. **不显然，且从代码、配置、文档里找不回来**——反事实检验：这条记录如果不存在，别人会不会重犯同样的错，或重新花大量时间排查？不会就不写；
2. **已验证、已解决**——有对照试验或明确证据，不是猜测；
3. **可操作**——能写成"遇到 X 时做 Y"。

花的时间多、改动大、"终于跑通了"本身都不是写进去的理由。不满足就不写，并在 NOTEBOOK 里说明为什么不写。

格式（每条一个）：

```
### <短标题>
触发：什么时候会遇到（现象、报告里的 check 名）
做法：具体怎么做（命令、参数）
原因：为什么（一句话）
如何确认：看哪个字段 / 哪个闸门
证据：NOTEBOOK.md 的日期条目、产物路径
```

排错类经验可以按"问题 / 现象 / 没用的尝试 / 解决 / 为什么有效 / 如何预防"来写；知识类按"背景 / 做法 / 为什么重要 / 何时适用"。可选加一行"何时作废"（例如 mace-torch 升到 1.0 之后）。

写之前先搜 `LESSONS.md`：已有相近条目就**更新**它（补证据、改做法），不要新增。
和 MFlow 自身阈值有关的经验（"这个阈值对我的体系太严"），写进 `mflow_thresholds.yaml` 并在 LESSONS 里记一条，代码和经验引用同一个数。

## 清单

- [ ] 活动开始：建 `NOTEBOOK.md`，写目标、数据来源、DFT 设置、基础模型
- [ ] 每个改变决定的命令后：一条记录
- [ ] 活动结束或交接：在 NOTEBOOK 末尾写"现状 + 未完成 + 下一步"；检查是否有经验过了门槛
- [ ] 写"完成""通过"之前：重新运行能证明它的命令（或读刚生成的报告），看完整输出，再写；记录当时已经看过哪些结果

出处与改写说明见 `references/SOURCES.md`。
