# Engram 精读笔记

> 论文：*Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*
> 作者：Xin Cheng, Wangding Zeng, Damai Dai, ... Huishuai Zhang, Dongyan Zhao, Wenfeng Liang
> 机构：Peking University / DeepSeek-AI
> 代码：<https://github.com/deepseek-ai/Engram>（Apache 2.0）
> PDF：官方仓库中的 `Engram_paper.pdf`

---

## 一页纸总结

**问题**：Transformer 没有原生的「知识查找」原语，只能用量计算去模拟检索。一个多 token 实体（如 "Diana, Princess of Wales"）要消耗 5–6 层 attention/FFN 才能被完整识别。这些深度本可以用于推理。

**主张**：除了 MoE 的「条件计算」，还应该有「条件记忆」这条稀疏轴。静态、局部、模式化的知识用 O(1) 查表；动态、组合、上下文相关的逻辑用计算。

**方法**：Engram = 现代化 n-gram embedding：
1. tokenizer 压缩（128k 词表缩 23.43%）；
2. 多 head 哈希（每个 n-gram order 8 个 head，素数表，乘法-XOR）；
3. 上下文门控（hidden 作 query，记忆作 key/value，signed sqrt + sigmoid）；
4. 深度可分离因果卷积（kernel 4，dilation 3，零初始化）；
5. 多分支集成（共享 W_V，分支独立 W_K）。

**发现**：
- 稀疏分配 U 形曲线，最优 `rho ≈ 0.75–0.80`；
- iso-param/iso-FLOPs 下，Engram-27B 全面超过 MoE-27B（MMLU +3.4, BBH +5.0, HumanEval +3.0）；
- 无限记忆 regime 下 loss 对 log(slots) 线性；
- 机制上等效于增加有效深度（CKA 上移、LogitLens 更早收敛）；
- 长上下文检索大幅提升（Multi-Query NIAH 84.2 → 97.0）；
- 100B 表 offload 到 host DRAM，吞吐只降 2.8%。

---

## 1. 问题设定：语言建模的两个子任务

论文把语言建模拆成：

| 子任务 | 特征 | 适合的 primitive |
|---|---|---|
| 组合推理 | 动态、上下文相关、需要多步 | 条件计算（MoE/FFN） |
| 知识检索 | 静态、局部、高度模式化 | 条件记忆（查表） |

经典 n-gram 模型（Shannon, 1948; Brants et al., 2007）证明了局部依赖可以被便宜地建模。但现代 Transformer 把一切都塞进计算，导致：

- 早期层在做「运行时重建静态查找表」；
- attention 被局部依赖占用，长上下文能力受损；
- 参数效率低：同样的实体要重复存储在多个 FFN 里。

---

## 2. 架构

### 2.1 检索

压缩后的 token 序列 `x'`，后缀 n-gram：

```
g_{t,n} = (x'_{t-n+1}, ..., x'_t)
```

哈希：

```
z_{t,n,k} = phi_{n,k}(g_{t,n}) mod M_{n,k}
e_{t,n,k} = E_{n,k}[z_{t,n,k}]
e_t = concat_{n=2..N} concat_{k=1..K} e_{t,n,k}
```

实现细节：

- `phi` 是乘法-XOR：`x_t*m_0 XOR x_{t-1}*m_1 XOR ...`；
- `m_j` 是随机奇数，每个 layer 独立；
- `M_{n,k}` 是素数，全局互不相同；
- `K=8`, `N=3`, `d_mem=1280`, `d_head=80`。

### 2.2 门控

```
k_t = W_K e_t
v_t = W_V e_t
alpha_t = sigmoid( signed_sqrt( <RMSNorm(h_t), RMSNorm(k_t)> / sqrt(d) ) )
u_t = alpha_t * v_t
```

- Query 来自 hidden（已聚合全局上下文）；
- 如果记忆与上下文矛盾，`alpha → 0`；
- signed sqrt 防止 sigmoid 饱和。

### 2.3 融合

```
Y = SiLU( Conv1D( RMSNorm(u) ) ) + u
H^(l) <- H^(l) + Y
```

- depthwise causal conv，kernel 4，dilation 3；
- 零初始化 → 初始恒等；
- Engram 只加在部分层（27B: layer 2, 15）。

### 2.4 多分支

mHC backbone（M=4）：

- 共享：embedding 表、`W_V`；
- 独立：`W_K^(m)`、gate、conv 的 channel group；
- `W_V` 和所有 `W_K` 拼成一个 FP8 GEMM。

---

## 3. 稀疏分配

定义：

```
P_sparse = P_tot - P_act
P_moe = rho * P_sparse
P_engram = (1 - rho) * P_sparse
```

实验：

- 2e20 FLOPs: `P_tot=5.7B`, `P_act=568M`，baseline 106 experts；
- 6e20 FLOPs: `P_tot=9.9B`, `P_act=993M`，baseline 99 experts；
- 固定 sparsity `P_tot/P_act ≈ 10`。

结果：

- U 形，最优 `rho ≈ 0.75–0.80`；
- 10B regime: loss 1.7248 (rho=1) → 1.7109 (rho≈0.8)，Δ=0.0139；
- 即使 MoE 只保留 40%，也能接近纯 MoE baseline。

解释：

- rho→1：缺少静态记忆，用深度重建 n-gram；
- rho→0：缺少条件计算，动态推理受损。

### 无限记忆

固定 3B MoE backbone（P_act=568M），扫 slots `2.58e5 → 1e7`：

- loss 对 `log10(slots)` 严格线性；
- 到 1e7 未饱和；
- OverEncoding 同样受益，但 Engram 斜率更陡。

---

## 4. 大模型实验

四个模型，都是 262B tokens、3.8B 激活参数：

| | Dense-4B | MoE-27B | Engram-27B | Engram-40B |
|---|---|---|---|---|
| 总参数 | 4.1B | 26.7B | 26.7B | 39.5B |
| 激活 | 3.8B | 3.8B | 3.8B | 3.8B |
| routed experts | - | 72 | 55 | 55 |
| Engram 参数 | - | - | 5.7B | 18.5B |

关键分数（vs MoE-27B）：

| 任务 | Δ |
|---|---|
| MMLU | +3.0（正文摘要写 +3.4，表格 60.4 vs 57.4 = +3.0；以表格为准） |
| CMMLU | +4.0 |
| BBH | +5.0 |
| ARC-Challenge | +3.7 |
| DROP | +3.3 |
| HumanEval | +3.0 |
| MATH | +2.4 |

> 注意摘要与 Table 1 有 0.4 的出入（MMLU）。写报告时引用表格并注明。

### 长上下文

YaRN 扩展到 32k，5000 步：

| 模型 | Multi-Query NIAH | VT |
|---|---|---|
| MoE-27B (50k) | 84.2 | 77.0 |
| Engram-27B (46k, iso-loss) | 97.0 | 87.2 |
| Engram-27B (50k, iso-FLOPs) | 97.0 | 89.0 |

即使 Engram 只训练 41k 步（82% FLOPs），也能匹配 baseline 的 LongPPL，并在 RULER 上更好。

---

## 5. 机制分析

### 5.1 有效深度

- **LogitLens KL**：Engram 早期层 KL 更低 → 更早「准备好」预测；
- **CKA**：Engram layer 5 最像 MoE layer 12 → 浅层做了深层的事；
- 软对齐 `a_j > j` 在很宽的范围成立。

解释：Engram 把「特征组合」的前几个阶段卸载到查表，等效于增加深度。

### 5.2 消融

固定 1.6B 记忆预算，3B MoE backbone（12 层，0.56B 激活），100B tokens：

- 参考配置：2/3-gram，layer 2+6，val loss 1.768（baseline 1.808）；
- 单模块 layer sweep：layer 2 最优（1.770），越深越差；
- 去掉 multi-branch / gating / tokenizer compression → 最大回退；
- 去掉 short conv → 轻微；
- 4-gram → 略差（容量稀释）。

### 5.3 敏感性

推理时关掉 Engram：

- 事实知识保留 29–44%；
- 阅读理解保留 81–93%。

说明 Engram 是事实知识的主要载体，而阅读理解主要靠 attention。

### 5.4 门控可视化

高激活出现在：

- "Alexander the Great" 的 "Great"；
- "the Milky Way" 的 "Way"；
- "Princess of Wales" 的 "Wales"；
- 中文 "四大发明"、"张仲景"。

因为后缀 3-gram 在短语最后一个 token 上才完整。

---

## 6. 系统

### 训练

- 表按行切到多卡；
- 前向 All-to-All gather，反向 dispatch 梯度；
- 容量随 accelerator 数线性增长。

### 推理

- 索引只依赖 token id，可提前预取；
- 100B 表放 host DRAM，PCIe 预取与前一层计算重叠；
- Table 4：4B dense 9031 → 8858 tok/s（-1.9%）；8B 6315 → 6140（-2.8%）；
- 传输量 ∝ 激活 slot 数，不 ∝ 表大小；
- Zipf → 多级缓存（HBM/DRAM/SSD）。

---

## 7. 批判性阅读

### 优点

1. **问题选得好**：把「记忆 vs 计算」提升为第一类架构问题，而不是工程 trick。
2. **实验严格**：iso-param + iso-FLOPs，控制变量；长上下文还做了 iso-loss。
3. **机制证据链完整**：LogitLens + CKA + 门控可视化 + 关掉模块的敏感性，互相印证。
4. **系统-算法协同**：确定性寻址 → 预取 → offload，不是事后优化。
5. **代码开源**：demo 可运行，降低复现门槛。

### 局限 / 可追问

1. **n-gram 键的语义上限**：只能捕捉局部、连续的模式。同义改写、跨句指代、长距离依赖仍然靠 attention。论文的 "conditional memory" 目前等价于 "n-gram memory"。
2. **碰撞不可解释**：哈希碰撞的槽位会混合语义，论文没有量化碰撞率对 loss 的影响。
3. **tokenizer 依赖**：23% 压缩率是特定 tokenizer 的结果；换 tokenizer/多语言/代码，压缩率和最优 n 可能变化。
4. **27B 只给了架构和分数，没有公开权重**；复现只能到小规模。
5. **baseline 的公平性**：MoE-27B 的 expert 数从 72 减到 55，但 shared expert、load balance、训练细节是否完全一致？论文说 identical pipeline，但未给所有超参。
6. **rho 最优值的普适性**：只在 sparsity≈10、两个 FLOPs 预算下验证；更大 sparsity、不同模态未知。
7. **长上下文的因果归因**：Engram 模型多训练了 context extension，且 base loss 不同；iso-loss 设置缓解了但没完全消除混淆。
8. **推理收益的边界**：100B 表只测了 dense backbone，没有测 MoE+Engram 的真实组合；PCIe 带宽是 H800 服务器级，消费级 GPU 不同。

### 和相邻工作的关系

| 工作 | 记忆形式 | 训练 FLOPs | 备注 |
|---|---|---|---|
| OverEncoding | n-gram embedding 平均 | 同 | MoE 上无明显收益 |
| SCONE | 辅助 f-gram 模型 | 额外 | 推理导向 |
| BLT | byte-level hash n-gram | 同 | 字节级 |
| PEER/PKM | key-value memory | 同 | 非 n-gram |
| RAG/RETRO | 文档检索 | 额外 | 非参数化 |

Engram 的差异：把 n-gram 记忆当**第一类 primitive**，在 iso-param/iso-FLOPs 下验证，并做系统协同设计。

---

## 8. 复现清单

- [ ] 复算 5.79B / 18.53B（`scripts/paper_math.py`）
- [ ] 复现 U 形（解析/小规模）
- [ ] 复现 log-linear slots（玩具）
- [ ] 训练 baseline vs Engram（合成）
- [ ] LogitLens KL 曲线
- [ ] CKA 软对齐
- [ ] 门控 top token
- [ ] offload 模拟 + 缓存
- [ ] Triton gather 基准（可选）
- [ ] 真实数据小规模（可选）

---

## 9. 术语表

| 术语 | 含义 |
|---|---|
| conditional computation | 按输入激活部分参数（MoE） |
| conditional memory | 按输入查表取静态参数（Engram） |
| P_tot / P_act / P_sparse | 总参数 / 激活参数 / 稀疏参数 |
| rho | 稀疏预算中分给 MoE 的比例 |
| iso-param / iso-FLOPs | 等参数 / 等计算量 |
| suffix n-gram | 以当前位置结尾的 n-gram |
| multi-head hash | 多个独立哈希头降低碰撞影响 |
| signed sqrt | `sign(x)*sqrt(|x|)`，压缩 logit |
| effective depth | 通过 CKA/LogitLens 推断的等效深度 |
| prefetch | 在计算前异步加载记忆行 |
