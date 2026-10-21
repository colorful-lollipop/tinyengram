# 术语与记号表

本表用于消除跨关卡阅读时的歧义。每个词条给出「一句话定义 + 首次出现的关卡」。

## 记忆与稀疏性

| 术语 | 定义 | 首次出现 |
|---|---|---|
| **条件计算**（conditional computation） | 根据输入只激活部分参数，例如 MoE 的 top-k 专家。 | L6 |
| **条件记忆**（conditional memory） | 根据输入只检索部分静态参数，例如 Engram 的 n-gram 查表。 | L0 |
| **静态知识** | 不随上下文改变、可以被局部模式索引的知识（实体、搭配、公式模板）。 | L0 |
| **动态推理** | 需要多步组合、依赖上下文、无法用固定查表完成的计算。 | L6 |
| **P_tot / P_act / P_sparse** | 总参数 / 每 token 激活参数 / 两者之差（稀疏容量）。 | L6 |
| **rho** | 稀疏预算中分给 MoE 的比例；`rho=1` 是纯 MoE，`rho=0` 是纯记忆。 | L6 |
| **iso-param / iso-FLOPs** | 等总参数 / 等每 token 计算量；论文比较 Engram 与 MoE 的控制变量。 | L6 |
| **有效深度**（effective depth） | 通过 LogitLens/CKA 推断出的「等效层数」，不是真实层数。 | L8 |

## Tokenizer 与哈希

| 术语 | 定义 | 首次出现 |
|---|---|---|
| **n-gram** | 连续 `n` 个 token 组成的片段。 | L0 |
| **后缀 n-gram**（suffix n-gram） | 以当前位置 `t` 结尾的 n-gram，只能看见 `<= t` 的 token。 | L0 |
| **order** | n-gram 的 `n`；Engram 默认用 order 2 和 3。 | L0 |
| **满射投影**（surjective projection） | `P: V -> V'`，每个新 id 至少被一个旧 id 命中；多个旧 id 可折叠成一个。 | L0 |
| **NFKC** | Unicode 兼容分解 + 组合，常用于全角/连字归一。 | L0 |
| **hash head** | 一个独立的哈希函数 + 一张独立的 embedding 表。 | L1 |
| **乘法-XOR 哈希** | `x_t*m_0 XOR x_{t-1}*m_1 XOR ...`，无进位、GPU 友好。 | L1 |
| **素数模数** | 每个 head 的表大小取素数，且全局互不相同。 | L1 |
| **碰撞**（collision） | 不同 n-gram 映射到同一个 slot；多 head 降低其影响。 | L1 |
| **tokenizer 压缩** | 把文本等价的 token 折叠成一个规范 id；128k 词表约缩 23%。 | L0 |

## 形状与模块

| 记号 | 定义 | 首次出现 |
|---|---|---|
| `B, T, D` | batch、序列长度、hidden size。 | L2 |
| `M` | 多分支 residual stream 的数量（`hc_mult`）。 | L4 |
| `N` | 最大 n-gram order（默认 3）。 | L0 |
| `K` | 每个 order 的 hash head 数（默认 8）。 | L1 |
| `d_mem` | 所有 head 拼接后的记忆宽度；27B 为 1280。 | L2 |
| `d_head` | 单个 head 的宽度；`d_head = d_mem / ((N-1)*K)`；27B 为 80。 | L2 |
| **Engram Vocab Size** | 论文 Table 5 中**每个 head、每个 order** 的 slot 数（27B: 2,262,400）。 | L2 |
| **slot** | 表的一行，存 `d_head` 个数；一次查表取一个 slot。 | L2 |

> 易错点：`2,262,400 × 1280 × 2 layers = 5.79B` 之所以成立，是因为
> `1280 = 16 heads × 80 dims`；不要把 `2,262,400` 当成「所有 head 的总行数」。

## 门控与融合

| 术语 | 定义 | 首次出现 |
|---|---|---|
| **RMSNorm** | 只除以均方根、不减均值的归一化。 | L3 |
| **signed sqrt** | `sign(x) * sqrt(|x|)`；压缩大 logit、保留符号、缓解 sigmoid 饱和。 | L3 |
| **上下文门控**（context-aware gating） | 用 hidden 作 query、记忆作 key/value，算标量 gate。 | L3 |
| **ShortConv** | 深度可分离、因果、带 dilation 的短卷积。 | L4 |
| **零初始化恒等** | conv 权重初始为 0，使 `Y = u + 0 = u`。 | L4 |
| **多分支 / mHC** | 多条并行 residual stream；Engram 共享 `W_V`、分支独立 `W_K`。 | L4 |
| **融合**（fusion） | 把记忆增量加回 residual stream：`H <- H + Y`。 | L4 |

## 系统

| 术语 | 定义 | 首次出现 |
|---|---|---|
| **确定性寻址** | 索引只依赖 token id，不依赖 hidden state，因此可提前计算。 | L7 |
| **预取**（prefetch） | 在对应层计算前，把记忆行从 host DRAM 搬到 GPU。 | L7 |
| **offload** | 把大表放在 CPU/NVMe，按需传输，突破 GPU 显存限制。 | L7 |
| **Zipf 分布** | `P(rank r) ∝ 1/r^s`，少量高频 n-gram 占大多数访问。 | L7 |
| **LRU** | 最近最少使用缓存淘汰策略。 | L7 |
| **All-to-All** | 分布式训练中把索引/梯度按 owner 发给所有卡的通信模式。 | L7 |
| **分片 gather** | 每张卡只存一部分表行，按 owner 路由后拼回完整记忆。 | L7 |
| **KV cache** | 缓存历史 K/V，解码时只算新 token；Engram 还需 `context_ids` 补 n-gram 历史。 | L5 |
| **stall** | 预取传输时间超过计算窗口后暴露出的等待时间。 | L7 |

## 分析与实验

| 术语 | 定义 | 首次出现 |
|---|---|---|
| **LogitLens** | 把每层 hidden 过最终 norm + LM head，观察预测如何逐层收敛。 | L8 |
| **KL 散度** | `sum P log(P/Q)`；这里衡量中间层与最终层分布的差距。 | L8 |
| **CKA** | Centered Kernel Alignment，衡量两组表示的结构相似度。 | L8 |
| **软对齐** | 对每个 Engram 层，取最相似 top-k 层的加权平均索引 `a_j`。 | L8 |
| **敏感性分析** | 推理时关掉 Engram，测量各任务保留率。 | L8 |
| **负面结果** | 与预期相反但可解释的测量，例如 Triton gather 只快 3%。 | L7 |

## 相关方法

| 方法 | 与 Engram 的关系 |
|---|---|
| **OverEncoding** | 也用 n-gram embedding，但在 MoE 上没有明显收益；Engram 强调 iso-param/iso-FLOPs 与系统协同。 |
| **SCONE** | 辅助 f-gram 模型，偏推理；Engram 是端到端训练的模块。 |
| **BLT** | byte-level 哈希 n-gram；Engram 在 token 级做多 head 哈希。 |
| **PEER / PKM** | key-value 记忆，key 不是 n-gram。 |
| **RAG / RETRO** | 检索外部文档，非参数化、需要检索器；Engram 是参数化记忆。 |

---

返回：[讲义导读](README.md) ｜ [项目 README](../README.md)
