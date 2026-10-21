# Conditional Memory via Scalable Lookup

**A New Axis of Sparsity for Large Language Models**

- **作者**：Xin Cheng, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, Zhenda Xie, Kezhao Huang, Xingkai Yu, Zhewen Hao, Yukun Li, Han Zhang, Huishuai Zhang, Dongyan Zhao, Wenfeng Liang
- **机构**：Peking University；DeepSeek-AI
- **年份**：2026
- **代码**：<https://github.com/deepseek-ai/Engram>
- **PDF**：官方仓库 `Engram_paper.pdf`
- **精读笔记**：[`notes.md`](notes.md)

> 本文件只摘录与课程实现直接相关的段落，并附中文翻译。版权归原作者，引用请以原文为准。

---

## Abstract（摘要）

> While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack a native primitive for knowledge lookup, forcing them to inefficiently simulate retrieval through computation. To address this, we introduce conditional memory as a complementary sparsity axis, instantiated via Engram, a module that modernizes classic N-gram embedding for O(1) lookup. By formulating the Sparsity Allocation problem, we uncover a U-shaped scaling law that optimizes the trade-off between neural computation (MoE) and static memory (Engram). Guided by this law, we scale Engram to 27B parameters, achieving superior performance over a strictly iso-parameter and iso-FLOPs MoE baseline.

**中文**：MoE 通过条件计算扩展容量，但 Transformer 缺少原生的知识查找原语，只能低效地用量计算模拟检索。为此，我们把**条件记忆**作为一条互补的稀疏轴，用 Engram 实例化——它现代化了经典 n-gram embedding，实现 O(1) 查找。通过形式化「稀疏分配」问题，我们发现一条 U 形 scaling law，用来优化神经计算（MoE）与静态记忆（Engram）的权衡。据此把 Engram 扩展到 27B，在严格等参数、等 FLOPs 的 MoE baseline 上取得更优性能。

> Most notably, while the memory module is expected to aid knowledge retrieval (e.g., MMLU +3.4; CMMLU +4.0), we observe even larger gains in general reasoning (e.g., BBH +5.0; ARC-Challenge +3.7) and code/math domains (HumanEval +3.0; MATH +2.4).

**中文**：最值得注意的是，记忆模块直观上应该帮助知识检索（MMLU +3.4；CMMLU +4.0），但我们在通用推理（BBH +5.0；ARC-Challenge +3.7）和代码/数学（HumanEval +3.0；MATH +2.4）上观察到更大的收益。

> Mechanistic analyses reveal that Engram relieves the backbone's early layers from static reconstruction, effectively deepening the network for complex reasoning. Furthermore, by delegating local dependencies to lookups, it frees up attention capacity for global context, substantially boosting long-context retrieval (e.g., Multi-Query NIAH: 84.2 → 97.0). Finally, Engram establishes infrastructure-aware efficiency: its deterministic addressing enables runtime prefetching from host memory, incurring negligible overhead.

**中文**：机制分析表明，Engram 把 backbone 的早期层从静态重建中解放出来，等效于加深网络以做复杂推理。进一步地，把局部依赖交给查表，释放了 attention 处理全局上下文的能力，显著提升长上下文检索（Multi-Query NIAH 84.2 → 97.0）。最后，Engram 建立了「基础设施感知」的效率：确定性寻址支持从 host memory 运行时预取，开销可忽略。

---

## §1 Introduction（引言）

> To align model architecture with this linguistic duality, we advocate for a complementary axis of sparsity: conditional memory. Whereas conditional computation sparsely activates parameters to process dynamic logic (Bengio et al., 2013; Shazeer et al., 2017), conditional memory relies on sparse lookup operations to retrieve static embeddings for fixed knowledge.

**中文**：为了让模型架构与语言的这种二元性对齐，我们主张一条互补的稀疏轴：条件记忆。条件计算稀疏地激活参数来处理动态逻辑；条件记忆则依赖稀疏查找操作，为固定知识检索静态 embedding。

> As a preliminary exploration of this paradigm, we revisit N-gram embeddings (Bojanowski et al., 2017) as a canonical instantiation: local context serves as a key to index a massive embedding table via constant-time O(1) lookups.

**中文**：作为该范式的初步探索，我们重新审视 n-gram embedding：局部上下文作为 key，通过 O(1) 查找索引一张巨大的 embedding 表。

> Our investigation reveals that, perhaps surprisingly, this static retrieval mechanism can serve as an ideal complement to modern MoE architecture—but only if it is properly designed.

**中文**：我们的研究表明，这个静态检索机制可以成为现代 MoE 架构的理想补充——但前提是设计得当。

---

## §2.2 Sparse Retrieval via Hashed N-grams（哈希 n-gram 检索）

> To maximize semantic density, we implement a vocabulary projection layer. Specifically, we pre-compute a surjective function P : V → V′ that collapses raw token IDs into canonical identifiers based on normalized textual equivalence (using NFKC, lowercasing, etc.). In practice, this process achieves a 23% reduction in the effective vocabulary size for a 128k tokenizer (see Appendix C).

**中文**：为最大化语义密度，我们实现一个词表投影层：预计算满射 `P: V → V'`，基于规范化文本等价（NFKC、小写等）把原始 token id 折叠成规范 id。实践中，128k 词表缩小 23%。

> Following Tito Svenstrup et al. (2017), we adopt a hashing-based approach. To mitigate collisions, we employ K distinct hash heads for each N-gram order n. Each head k maps the compressed context to an index within an embedding table E_{n,k} (of prime size M_{n,k}) via a deterministic function phi_{n,k}.

**中文**：沿用 hash embedding 的思路。为缓解碰撞，每个 n-gram order 用 K 个独立 hash head；每个 head 通过确定性函数把压缩上下文映射到素数大小的表 `E_{n,k}` 中的索引。

---

## §2.3 Context-aware Gating（上下文门控）

> Being static, however, they inherently lack contextual adaptability and may suffer from noise due to hash collisions or polysemy. To enhance expressivity and resolve this ambiguity, we employ a context-aware gating mechanism inspired by Attention.

**中文**：静态记忆天生缺乏上下文适应性，且可能因哈希碰撞或一词多义引入噪声。为增强表达、消解歧义，我们采用受 attention 启发的上下文门控。

> To ensure gradient stability, we apply RMSNorm to the Query and Key before computing the scalar gate alpha_t in (0,1).

**中文**：为保证梯度稳定，在计算 0–1 标量门之前对 Query/Key 做 RMSNorm。

> This design enforces semantic alignment: if the retrieved memory e_t contradicts the current context h_t, the gate alpha_t tends toward zero, effectively suppressing the noise.

**中文**：该设计强制语义对齐：若检索到的记忆与当前上下文矛盾，gate 趋近 0，从而抑制噪声。

---

## §2.5 System Efficiency（系统效率）

> Unlike MoE, which relies on runtime hidden states for dynamic routing, Engram's retrieval indices depend solely on the input token sequence. This predictability facilitates specialized optimization strategies for both training and inference.

**中文**：与依赖运行时 hidden state 做动态路由的 MoE 不同，Engram 的检索索引只依赖输入 token 序列。这种可预测性为训练和推理都带来了专门优化。

> During inference, this deterministic nature enables a prefetch-and-overlap strategy. Since memory indices are known prior to the forward pass, the system can asynchronously retrieve embeddings from abundant host memory via PCIe.

**中文**：推理时，这种确定性支持「预取 + 重叠」：索引在 forward 之前已知，系统可以通过 PCIe 从充裕的 host memory 异步取 embedding。

> Furthermore, natural language N-grams inherently follow a Zipfian distribution, where a small fraction of patterns accounts for the vast majority of memory accesses. This statistical property motivates a Multi-Level Cache Hierarchy.

**中文**：自然语言 n-gram 服从 Zipf 分布，少量模式占绝大多数访问。这一统计性质驱动了多级缓存层次。

---

## §3.1 Optimal Allocation Ratio（最优分配比例）

> We keep P_tot and P_act fixed within each FLOPs budget, so that models have the same number of parameters and the same per-token FLOPs.

**中文**：在每个 FLOPs 预算内固定 `P_tot` 和 `P_act`，使模型具有相同参数量和相同每 token FLOPs。

> We define the allocation ratio rho in [0,1] as the fraction of the inactive-parameter budget assigned to MoE expert capacity.

**中文**：定义分配比例 `rho ∈ [0,1]` 为非激活参数预算中分给 MoE 专家容量的比例。

> Figure 3 (left) reveals a consistent U-shaped relationship between validation loss and the allocation ratio rho. ... reallocating roughly 20%–25% of the sparse parameter budget to Engram yields the best performance.

**中文**：验证 loss 与 `rho` 呈稳定 U 形；把约 20%–25% 的稀疏预算分给 Engram 效果最好（即 `rho ≈ 0.75–0.80`）。

> The location of this optimum is stable across regimes (rho ≈ 75%–80%), suggesting a robust allocation preference across the examined scales (under fixed sparsity).

**中文**：最优位置在两个预算下稳定在 75%–80%，说明在固定稀疏度下该偏好对规模不敏感。

---

## §3.2 Infinite Memory Regime（无限记忆）

> Across the explored range, the curve follows a strict power law (linear in log-space), indicating that Engram provides a predictable scaling knob: larger memory continues to pay off without requiring additional computation.

**中文**：在探索范围内，曲线严格服从幂律（log 空间线性），说明 Engram 提供了一个可预测的 scaling knob：更大的记忆持续带来收益，而不需要额外计算。

---

## §6.1 Effective Depth（有效深度）

> We posit that by equipping the model with an explicit knowledge lookup capability, Engram effectively mimics an increase in model depth by relieving the model of the early stages of feature composition.

**中文**：我们假设，通过赋予模型显式的知识查找能力，Engram 免除了早期特征组合阶段，从而等效于增加模型深度。

> We observe a distinct upward shift from the diagonal, meaning that a_j > j for a wide range of layers. For instance, the representations formed at layer 5 of Engram-27B align most closely with those at approximately layer 12 of the MoE baseline.

**中文**：观察到相对对角线的明显上移，即很多层满足 `a_j > j`。例如 Engram-27B 第 5 层的表示最接近 MoE baseline 第 12 层。

---

## §6.2 Structural Ablation（结构消融）

> We find that three components yield the most significant gains: (i) branch-specific fusion within the multi-branch backbone, (ii) context-aware gating, and (iii) tokenizer compression. Removing any of these causes the largest regressions in validation loss.

**中文**：三个组件贡献最大：多分支 backbone 内的分支特定融合、上下文门控、tokenizer 压缩。去掉任何一个都会造成最大的验证 loss 回退。

> Allocating capacity to 4-grams is slightly suboptimal under a fixed 1.6B budget—likely because it dilutes capacity from the more frequent 2/3-gram patterns.

**中文**：固定 1.6B 预算下，把容量分给 4-gram 略差，可能是因为稀释了更高频的 2/3-gram 模式。

---

## §6.3 Sensitivity（敏感性）

> Factual knowledge benchmarks suffer a catastrophic collapse, retaining only 29–44% of the original performance (e.g., TriviaQA at 29%), confirming that the Engram module acts as the primary repository for parametric knowledge. Conversely, reading comprehension tasks are remarkably resilient, retaining 81–93%.

**中文**：事实知识基准灾难性下降，只保留 29–44%（TriviaQA 29%），说明 Engram 是参数化知识的主要仓库；阅读理解则保留 81–93%，主要依赖 attention。

---

## §6.4 System Efficiency（系统效率）

> We insert a massive 100B-parameter Engram layer into the second Transformer block, with the entire embedding table resident in host DRAM. During inference, the system prefetches embeddings for the Engram layer asynchronously, overlapping the PCIe transfer with the computation of the first block.

**中文**：把 100B 参数的 Engram 层插入第二个 block，整张表放在 host DRAM。推理时异步预取，把 PCIe 传输与第一个 block 的计算重叠。

> Offloading a 100B-parameter embedding table incurs a negligible throughput penalty, peaking at only 2.8% on the 8B backbone.

**中文**：offload 100B 参数表只带来可忽略的吞吐损失，8B backbone 上最高 2.8%。

> Crucially, the effective communication volume per step scales with the number of activated slots rather than the total embedding table size.

**中文**：关键是，每步有效通信量随激活 slot 数增长，而不是总表大小。

---

## Table 1（节选）

| Benchmark | Dense-4B | MoE-27B | Engram-27B | Engram-40B |
|---|---|---|---|---|
| Total Params | 4.1B | 26.7B | 26.7B | 39.5B |
| Activated | 3.8B | 3.8B | 3.8B | 3.8B |
| MMLU | 48.6 | 57.4 | 60.4 | 60.6 |
| CMMLU | 47.9 | 57.9 | 61.9 | 63.4 |
| BBH | 42.8 | 50.9 | 55.9 | 57.5 |
| ARC-C | 59.3 | 70.1 | 73.8 | 76.4 |
| DROP | 41.6 | 55.7 | 59.0 | 60.7 |
| HumanEval | 26.8 | 37.8 | 40.8 | 38.4 |
| GSM8K | 35.5 | 58.4 | 60.6 | 62.6 |
| MATH | 15.2 | 28.3 | 30.7 | 30.6 |

## Table 5（超参）

| 项 | Engram-27B | Engram-40B |
|---|---|---|
| Total / Active | 26.7B / 3.8B | 39.5B / 3.8B |
| Layers | 30 | 30 |
| Hidden | 2560 | 2560 |
| Routed experts | 55 | 55 |
| Active experts | 6 | 6 |
| Shared experts | 2 | 2 |
| Engram Dim `d_mem` | 1280 | 1280 |
| Engram Vocab Size（每 head/order） | 2,262,400 | 7,239,680 |
| Engram Heads | 8 | 8 |
| Engram Layers | [2, 15] | [2, 15] |
| Engram N-gram | [2, 3] | [2, 3] |
| Conv Zero Init | True | True |
| LR Multiplier | ×5 | ×5 |
| Weight Decay | 0.0 | 0.0 |

---

## 引用格式

```bibtex
@article{cheng2026engram,
  title   = {Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models},
  author  = {Cheng, Xin and Zeng, Wangding and Dai, Damai and Chen, Qinyu and Wang, Bingxuan and
             Xie, Zhenda and Huang, Kezhao and Yu, Xingkai and Hao, Zhewen and Li, Yukun and
             Zhang, Han and Zhang, Huishuai and Zhao, Dongyan and Liang, Wenfeng},
  year    = {2026},
  note    = {DeepSeek-AI and Peking University},
  url     = {https://github.com/deepseek-ai/Engram}
}
```
