# 讲义导读

这十篇讲义按「一个部件 → 一个可验证性质 → 一个系统后果」组织。建议**按顺序读**，因为后面的关卡默认你理解前面的形状和接口。

> English overview: [`README.en.md`](../README.en.md) ｜ 术语表：[`glossary.md`](glossary.md) ｜ 常见问题：[`faq.md`](faq.md)

## 依赖关系

```text
L0 Tokenizer 压缩
      │
L1 多头部哈希
      │
L2 检索与内存布局
      │
L3 上下文门控
      │
L4 ShortConv 与融合
      │
L5 Transformer 与训练 ──────┐
      │                     │
L6 稀疏分配与缩放律          │
      │                     │
L7 推理系统与 offload ───────┤
      │                     │
L8 机制分析 ─────────────────┘
      │
L9 端到端复现
```

L6、L7、L8 之间没有硬依赖，可以并行；L9 需要 L5 的模型和 L8 的分析工具。

## 时间预算

| 关卡 | 读讲义 | 写代码 | 调测试 | 合计（参考） |
|---|---|---|---|---|
| L0 | 20 min | 30 min | 10 min | 1 h |
| L1 | 30 min | 45 min | 20 min | 1.5 h |
| L2 | 20 min | 20 min | 10 min | 1 h |
| L3 | 30 min | 30 min | 20 min | 1.5 h |
| L4 | 30 min | 45 min | 30 min | 2 h |
| L5 | 40 min | 90 min | 40 min | 3 h |
| L6 | 30 min | 30 min | 20 min | 1.5 h |
| L7 | 45 min | 60 min | 30 min | 2.5 h |
| L8 | 30 min | 45 min | 20 min | 1.5 h |
| L9 | 20 min | 60 min | 60 min | 2.5 h |

总计约 18 小时。如果只想过一遍概念，可以只读 L0–L4 + L9，跳过实现。

## 三层目标：怎么判断「学会了」

每关开头的卡片都给出 **必做 / 进阶 / 挑战**，对应三种不同的理解深度：

| 层级 | 标志 | 验收方式 |
|---|---|---|
| **L1 理解**（必做） | 能用自己的话解释公式里每个符号，能通过该关测试 | `pytest tests/test_lX_*.py` 全绿 |
| **L2 迁移**（进阶） | 改一个设计/超参，能预测并解释结果 | 跑一组对照实验，写出 3–5 行结论 |
| **L3 研究**（挑战） | 能提出可证伪的问题，设计实验并诚实报告 | 1 页报告，含设置、数字、局限 |

只通过测试只是 L1；世界级课程的区别在于能完成 L2/L3。每关的思考题已经按
`[理解] → [推导] → [设计] → [开放]` 标注，建议至少做一道 `[推导]` 和一道 `[设计]`。

## 三种用法

### 1. 做题模式（推荐）

```bash
pytest tests/test_l0_tokenizer.py -x -q
# 读讲义 -> 补 TODO -> 测试变绿 -> 下一关
```

### 2. 阅读模式

不想写代码时，直接读 `solutions/` 和讲义，跑：

```bash
TINYENGRAM_IMPL=solutions pytest -q
TINYENGRAM_IMPL=solutions python experiments/exp2_train_tiny.py
```

### 3. 教学/分享模式

每关的「思考题」可以直接做讨论题；`scripts/grade.py` 适合课堂进度展示；`perf/bench.py` 适合讲系统权衡。

## 前置知识

- PyTorch：`nn.Module`、广播、`backward`、`state_dict`；
- Transformer：attention、residual、RMSNorm、RoPE（不熟也能做，讲义会解释）；
- 基础概率：期望、KL、余弦相似度；
- 不需要：CUDA 编程、分布式训练、MoE 实现经验。

## 记号约定

| 记号 | 含义 |
|---|---|
| `B, T, D` | batch、序列长度、hidden size |
| `M` | 多分支数（`hc_mult`） |
| `N` | 最大 n-gram order（默认 3） |
| `K` | 每个 order 的 hash head 数（默认 8） |
| `d_mem` | 拼接后的记忆宽度 |
| `d_head` | 单个 head 的宽度 |
| `P_tot/P_act/P_sparse` | 总/激活/稀疏参数 |
| `rho` | 稀疏预算中 MoE 的占比 |

## 卡住时的升级路径

1. 重读本关「常见坑」和「提示」；
2. 用 `pytest -x` 只看第一个失败，读断言里的数学性质；
3. 在 `solutions/` 里找对应函数，**只看 10 行**，然后关掉自己写；
4. 仍然卡住，把「我以为什么」和「测试说什么」写成两行，再对照讲义；
5. 最后才整体复制参考实现——并确保能解释每一行。

## 术语与常见问题

- 术语、记号、易错点：[`glossary.md`](glossary.md)
- 环境、测试、KV cache、性能：[`faq.md`](faq.md)

如果测试和讲义冲突，以测试为准并提 issue——那通常意味着讲义需要更新。`python scripts/check_docs.py`
会检查链接、关卡卡片和测试文件引用是否完整。
