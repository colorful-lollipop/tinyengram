# Lab 2：检索与内存布局 —— 一次 gather 拿到 16 段记忆

> 对应代码：`src/tinyengram/embedding.py`  
> 对应测试：`tests/test_l2_embedding.py`  
> 论文：§2.2 Eq. (2)

> **前置**：L1 ｜ **预计**：1 h ｜ **验收**：`pytest tests/test_l2_*.py`  
> **必做**：`MultiHeadEmbedding` / `EngramMemory`，通过 7 个测试。  
> **进阶**：实现分片 gather，和整表 gather 对比；测不同 `d_head` 的带宽。  
> **挑战**：实现稀疏 Adam；把表放 CPU 做真实 offload。

---

## 2.1 逻辑布局 vs 物理布局

逻辑上，Engram 有 `(N-1) × K = 2 × 8 = 16` 张独立的表：

```
E_{2,1}, E_{2,2}, ..., E_{2,8}, E_{3,1}, ..., E_{3,8}
```

每张表 `M_{n,k}` 行，每行 `d_head = d_mem / (N-1) / K` 维。论文 27B 配置：

```
d_mem = 1280, K = 8, N = 3  =>  d_head = 80
```

如果老老实实建 16 个 `nn.Embedding`，每个 token 要发 16 次 gather，再 concat。小 kernel 的 launch overhead 会吞掉收益。

物理上只建**一张扁平表**：

```
[ E_{2,1} | E_{2,2} | ... | E_{3,8} ]
```

总行数 `S = sum M_{n,k}`。每个 head 的起始偏移：

```
offset[h] = sum_{j<h} M_j
```

检索时：

```python
flat_id = z[h] + offset[h]
row = table[flat_id]
```

一次 `index_select` 完成 16 个 head 的 gather。

---

## 2.2 为什么偏移量要注册成 buffer

```python
self.register_buffer("offsets", torch.tensor(offsets, dtype=torch.long))
```

- 跟着 `model.to(device)` 自动搬到 GPU；
- 会进 `state_dict`，checkpoint 可复现；
- 不参与梯度。

不要用 Python list 在 forward 里现算，也不要每次 `.to(device)`。

---

## 2.3 形状约定

```python
MultiHeadEmbedding.forward(ids[..., H]) -> [..., H, d_head]
EngramMemory.retrieve(ids[B, T])       -> [B, T, d_mem]
```

`flatten(start_dim=-2)` 把 `[B,T,H,d_head]` 变成 `[B,T,H*d_head]`，顺序必须和哈希输出的 head 顺序一致。

这一步错了，后面的 gate 仍然能跑（形状对），但每个 head 的语义会错位——这是最隐蔽的 bug 之一。测试 `test_memory_retrieval_matches_manual_pipeline` 用「手动 hash → table → flatten」对照，能抓住它。

---

## 2.4 参数量与带宽

27B 配置：

```
每个 head/order 的 slot 数   2,262,400
每层总行数（2 orders × 8 heads）        36.2M
每行维度（d_head）            80
每层参数量                    2.90B
两层                          5.79B   （论文 5.7B）
```

40B 配置：

```
7,239,680 × 16 × 80 × 2 = 18.53B
```

推理时每个 token 只读 `16 × 80 = 1280` 个数：

```
bf16: 16 heads × 80 dims × 2 B = 2560 B = 2.5 KiB
fp32: 5 KiB
```

`d_mem = 1280` 是**16 个 head 拼起来的总宽度**，不是每个 head 的宽度。如果误以为每个 head 都是 1280 维，就会把流量算成 `16 × 1280 × 2 = 40 KiB`，正好放大 16 倍。读论文的 "Engram Dim 1280" 时要结合 Table 5 的 slot 总数反推：`2,262,400 × 1280 × 2 layers ≈ 5.79B`，只有总宽度 1280 才成立。

关键结论不变：**传输量 ∝ 激活 slot 数，与表大小无关**。这是 Lab 7 的核心。

---

## 2.5 梯度是稀疏的

`nn.Embedding` 的反向只给被选中的行写梯度。一个 batch 里没被任何 token 命中的行，梯度为 0（或 `None`，取决于实现）。

这对优化器有直接影响：

- 如果用带 weight decay 的 AdamW，未命中的行也会被 decay；
- 长尾 slot 可能几百万 token 才被命中一次，decay 会把它们慢慢推向 0；
- 所以论文对 embedding 用 **Adam + lr×5 + weight_decay=0**（Table 5）。

测试 `test_only_touched_rows_receive_gradient` 检查只有命中的行有梯度。

---

## 2.6 初始化

`nn.Embedding` 默认 `N(0,1)`，对 1280 维来说方差过大。本项目在 `TinyLM._init_weights` 里统一用 `std=0.02`。

初始化尺度会影响 gate：

- `e_t` 太大 → `W_K e_t` 太大 → gate 饱和；
- `e_t` 太小 → gate 接近 0.5，记忆没有区分度。

论文没有单独强调这一点，但 27B 训练里 embedding 的初始化/学习率是调过的（lr×5 说明它需要比 backbone 更大的更新）。

---

## 2.7 碰撞的后果

假设两个不同的 3-gram 落到同一个 slot：

- 它们的记忆向量完全相同；
- gate 仍然可以根据 hidden state 决定「这次要不要用」；
- 但 `W_V e` 无法区分两者，模型只能学到两者的「平均语义」。

多 head 降低的是「16 个 head 全部碰撞」的概率。若单 head 碰撞率为 `p`，全部碰撞约为 `p^16`（独立性近似）。这就是为什么 8 个 head 就够用。

---

## 2.8 自测

```bash
pytest tests/test_l2_embedding.py -x -q
```

覆盖：

- offsets 把每个 head 路由到自己的行；
- 任意前导维度；
- head 数不匹配报错；
- 只有被选中的行有梯度；
- `retrieve` 与手动 pipeline 完全一致；
- 元数据（slot 数、head 数）自洽；
- `hash_ids` 无梯度且确定性。

---

## 2.9 思考题

难度梯度：**[理解]** 能用自己的话解释；**[推导]** 需要动笔算；**[设计]** 需要改代码/做实验；**[开放]** 没有标准答案。

1. **[理解]** 如果 `d_head` 不是 8 的倍数，GPU gather 会不会变慢？为什么？
2. **[推导]** 把 16 张表拼成一张，对优化器状态（Adam m/v）的显存有什么影响？
3. **[设计]** 训练时如何实现「只更新本 batch 命中的行」的稀疏 Adam？和 dense Adam 相比省多少？
4. **[设计]** 如果 table 放在 CPU、hidden 在 GPU，`index_select` 的瓶颈在哪？
5. **[理解]** 为什么论文的 `d_mem=1280` 要拆成 16×80，而不是 1×1280 的大表？

---

## 2.10 提示

- offsets 用 `cumsum` 构造：`[0, M0, M0+M1, ...]`。
- `flatten(start_dim=-2)` 等价于把最后两维合并，不要用 `view` 破坏连续性。
- `nn.Embedding` 的输入必须是 long；哈希输出已经是 long。
- 如果要检查「没有跨 head 泄漏」，就手动比较 `out[..., h, :]` 和 `weight[ids[..., h] + offset[h]]`。

---

上一关：[Lab 1：多头部哈希](lab1-hashing.md)  
下一关：[Lab 3：上下文门控](lab3-gating.md)
