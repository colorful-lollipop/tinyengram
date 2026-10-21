# Lab 9：端到端复现 —— 从机制到一份可复现的实验报告

> 对应代码：`experiments/`、`tests/test_l9_reproduction.py`  
> 论文：§4、§5、§6

> **前置**：L0–L8 ｜ **预计**：2.5 h ｜ **验收**：`pytest tests/test_l9_*.py`  
> **必做**：通过 4 个 slow 测试，跑通 `make experiments`。  
> **进阶**：每个配置 3 个 seed，报告 mean±std；做插入层 sweep。  
> **挑战**：换真实数据/更大模型，写一篇 capstone 级别的复现报告。

---

## 9.1 目标

前 8 关每关只验证一个部件。这一关把它们串起来，回答三个问题：

1. 在小规模合成任务上，Engram 是否真的降低 loss / 提升 held-out 准确率？
2. 训练后的门控是否变得有选择性？
3. 能否跑通「训练 → 生成 → LogitLens → CKA → 门控可视化」的完整链路？

**不要求**复现 27B 的绝对分数。规模差 4 个数量级，任何「Engram 一定更好」的断言都是不诚实的。

---

## 9.2 快速跑通

```bash
# 全部实验（会训练多个小模型，CPU 几分钟）
make experiments

# 只跑端到端测试
TINYENGRAM_IMPL=solutions pytest tests/test_l9_reproduction.py -q -m slow
```

产物在 `.runs/`：

- `allocation.md`
- `train_tiny.md`
- `analysis.md`
- `offload.md`
- `scaling.md`

---

## 9.3 推荐实验矩阵

| 实验 | 变量 | 固定 | 指标 |
|---|---|---|---|
| E1 | `rho` 扫描 | 总参数、FLOPs | val loss（U 形） |
| E2 | slots 扫描 | backbone、tokens | val loss（log 线性） |
| E3 | Engram on/off | 参数量尽量匹配 | train/heldout loss、acc |
| E4 | 插入层 {1,2,4,6} | 记忆预算 | val loss |
| E5 | 去 gating / 去 conv / 去压缩 | 记忆预算 | val loss |
| E6 | 推理关掉 Engram | 训练好的模型 | 各任务保留率 |
| E7 | offload 缓存容量 | 表大小、带宽 | hit rate、stall |

每个配置至少 3 个 seed。报告 `mean ± std`，不要只报最好的一次。

---

## 9.4 合成任务的正确解读

`EntityCorpus` 的设计让 3-gram 能覆盖完整实体名，所以：

- Engram **应该**在实体属性预测上更快收敛；
- baseline 也能学会，只是需要更多层/更多步；
- 小模型（2 层）可能两者差距很小，甚至 baseline 略好（随机性）。

如果 Engram 没有更好，先检查：

1. Engram 层是否真的插入（`model.engram_modules()` 非空）；
2. gate 是否全 0（记忆没被使用）；
3. embedding 学习率是否太低；
4. 训练步数是否足够（至少几百步）；
5. 合成语料的模板是否让 baseline 太容易。

不要为了「让 Engram 赢」而调 baseline 的超参——那是作弊。

---

## 9.5 扩展到真实数据（可选）

你有一张 16GB 的 4070 Ti Super。可行配置：

```python
TinyLMConfig(
    vocab_size=32000,       # GPT-2 tokenizer
    hidden_size=512,
    n_layers=8,
    n_heads=8,
    max_seq=512,
    hc_mult=2,
    engram_layers=(1, 4),
)
EngramConfig(
    vocab_size=32000,
    hidden_size=512,
    hc_mult=2,
    max_ngram_size=3,
    n_head_per_ngram=8,
    n_embed_per_ngram=320,   # head_dim=40, dmem=640
    slots_per_head=500_000,  # 约 0.64B 记忆参数
    layer_ids=(1, 4),
)
```

显存估算：

```
embedding: 2 layers × 500k × 16 heads × 40 dims × 2 B ≈ 1.28 GB
optimizer states (Adam): ×3 ≈ 3.84 GB
backbone + activations: 几 GB
=> 16GB 卡可以放下；如果 OOM，先减 slots 或 batch。
```

用 TinyStories / Wikipedia 子集，训练 1–2 小时，观察 val loss。重点不是刷分，而是验证：

- 表能否放进显存；
- hash 是否成为 CPU 瓶颈（用 `torch.profiler`）；
- gate 分布是否合理；
- 生成是否出现重复/崩坏。

---

## 9.6 报告模板

```markdown
# Engram 小规模复现报告

## 设置
- 数据：EntityCorpus(n_entities=..., templates=...)
- 模型：layers=..., hidden=..., heads=..., hc_mult=...
- Engram：layers=..., slots/head=..., dmem=..., N=...
- 训练：steps=..., batch=..., lr=..., warmup=...
- 硬件：RTX 4070 Ti Super 16GB / CPU

## 结果
| 配置 | params | train loss | heldout loss | token acc | gate mean | gate std |
|---|---|---|---|---|---|---|

## 机制
- LogitLens KL 曲线（图/表）
- CKA 软对齐 a_j
- 门控 top-10 token

## 结论与局限
- 复现了什么机制
- 哪些数字和论文方向一致/不一致
- 规模差异带来的限制
```

---

## 9.7 常见失败模式

| 现象 | 可能原因 |
|---|---|
| loss 不降 | lr 太小 / warmup 太长 / 数据太简单 |
| loss 变 NaN | embedding lr 太大 / 没 grad clip / gate 饱和 |
| gate 全 0.5 | `W_K` 没训练 / query 没归一化 |
| gate 全 1 | 初始化过大 / 没有 signed sqrt |
| 生成重复 | 训练不足 / 采样温度太低 |
| 显存 OOM | slots 太大 / batch 太大 / 没开 bf16 |
| 速度极慢 | hash 在 CPU 上逐 token 循环 / 没用 gather |

---

## 9.8 思考题（Capstone 级别）

难度梯度：**[理解]** 能用自己的话解释；**[推导]** 需要动笔算；**[设计]** 需要改代码/做实验；**[开放]** 没有标准答案。

1. **[设计]** 设计一个「记忆 vs 计算」的合成任务，使得最优 rho 可以被测量，而不仅是拟合。
2. **[设计]** 把 Engram 和 MoE 放在同一个 backbone 里，做严格的 iso-param/iso-FLOPs 对比。
3. **[设计]** 实现一个真正的 CPU offload：`table` 放 pinned memory，独立线程预取，CUDA stream 重叠。测量端到端吞吐。
4. **[设计]** 用 Triton 写一个 fused gate+value+conv kernel，和 `torch.compile` 对比。
5. **[开放]** 研究 tokenizer 压缩对中文/代码的影响：构造一个多语言词表，测量压缩率和 n-gram 碰撞率。

---

## 9.9 结课标准

- `python scripts/grade.py --solution` 全绿（91 个测试）；
- 自己的 `src/` 实现也全绿；
- 能解释每个设计选择的「为什么」，而不只是「是什么」；
- 能诚实报告负面结果。

---

上一关：[Lab 8：机制分析](lab8-analysis.md)  
返回：[README](../README.md)
