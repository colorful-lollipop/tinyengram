# Lab 5：Transformer 与训练 —— 把 Engram 装进一个能跑的 LM

> 对应代码：`src/tinyengram/transformer.py`、`src/tinyengram/train.py`、`src/tinyengram/data.py`  
> 对应测试：`tests/test_l5_transformer.py`  
> 论文：§2.4、§4.1、Table 5

> **前置**：L4 ｜ **预计**：3 h ｜ **验收**：`pytest tests/test_l5_*.py`  
> **必做**：RoPE / attention / Block / Trainer，通过 12 个测试（含 chunked KV cache）。  
> **进阶**：扫 Engram 插入层；放大 hidden/layers；换真实 tokenizer。  
> **挑战**：在 TinyStories 上做 iso-param baseline；复现有效深度趋势。

---

## 5.1 为什么自己写一个 TinyLM

论文的 backbone 是 30 层、hidden 2560、MLA + DeepSeekMoE + mHC。直接复现需要数千 GPU 小时。本项目的目标是**机制可观察**，所以：

- 2 层、hidden 48/32、2 个 head；
- 标准因果 attention + SwiGLU；
- 多分支 residual（`hc_mult=2/4`）；
- Engram 插入指定层；
- 合成语料，CPU 几分钟跑完。

我们**不复现** MLA / MoE / mHC 的数学——那些在 CS336 和姐妹项目 `mHc/` 里。

---

## 5.2 结构

```
input_ids [B,T]
  -> token embedding [B,T,D]
  -> expand to M branches [B,T,M,D]
  -> for each block:
       if layer has Engram: h += Engram(h, ids)
       h += Attention(RMSNorm(h))
       h += FFN(RMSNorm(h))
  -> final RMSNorm
  -> mean over branches
  -> LM head
```

每个分支独立做 RMSNorm/attention/FFN，但 attention/FFN 的**权重共享**。这模拟 mHC 的多流结构，又不需要实现完整的流混合矩阵。

---

## 5.3 RoPE 与多分支 reshape 的坑

RoPE 对每对维度做旋转：

```
y0 = x0 cos - x1 sin
y1 = x0 sin + x1 cos
```

频率：

```
theta_i = 10000^(-2i/d),  i = 0..d/2-1
```

关键性质：`<RoPE(q, p), RoPE(k, q)>` 只依赖相对位置 `p - q`。

实现里最容易错的是 **多分支 reshape**：

```python
# 错误！
xb = x.reshape(b * m, t, d)          # [B,T,M,D] 按内存顺序切，分支和时间混在一起
...
y.reshape(b, t, m, d)                # 同样错误

# 正确
xb = x.permute(0, 2, 1, 3).reshape(b * m, t, d)
...
y = self.out(y).reshape(b, m, t, d).permute(0, 2, 1, 3)
```

`[B,T,M,D]` 的连续内存顺序是 `B -> T -> M -> D`。直接 `reshape(B*M, T, D)` 会把 `(t, m)` 压扁成错误的 batch 维。这个 bug 不会报形状错误，只会让不同时间步/分支的数据混在一起，loss 看起来「能降」，但模型学的是错的。

测试 `test_lm_is_causal` 和 `test_kv_cache_matches_full_forward` 能间接抓住它。

---

## 5.4 KV cache 与 Engram 的上下文窗口

自回归解码时，attention 只算新 token 的 Q/K/V，缓存历史 K/V：

```
step 0: prefill [t0..tn] -> cache
step 1: q(tn+1) x cache -> logits
...
```

但 Engram 有一个额外要求：**它需要前 `N-1` 个 token 才能算 n-gram**。

如果只把新 token 传给 Engram，hash 会在左侧补 pad：

```
单 token "Great" -> 3-gram (pad, pad, Great)  ❌
完整上下文 "Alexander the Great" -> (Alexander, the, Great)  ✅
```

因此 `TinyLM.forward` 有 `context_ids` 参数：

```python
model(next_id, caches=caches, position_offset=pos, context_ids=out)
```

`generate` 会自动维护 `out` 并传入。**手动用 cache 时必须自己传**，否则 logits 会和 full forward 不一致（测试里专门注释了这一点）。

这是「架构影响推理接口」的一个好例子：Engram 不是纯 token-wise 的模块，它依赖局部历史。

---

## 5.5 训练配方

论文 Table 5 的关键项：

| 项 | 值 |
|---|---|
| backbone optimizer | Muon |
| embedding optimizer | Adam |
| base LR | 4e-4 |
| Engram LR multiplier | ×5 |
| Engram weight decay | 0.0 |
| LR scheduler | step decay |
| warmup | 有（Bi et al., 2024） |
| grad clip | 有 |

本项目用 AdamW 统一 backbone，embedding 单独一组：

```python
groups = [
  {"params": engram_embedding, "lr": 5e-3, "weight_decay": 0.0},
  {"params": decay_params,     "lr": 1e-3, "weight_decay": 0.1},
  {"params": norm_bias_params, "lr": 1e-3, "weight_decay": 0.0},
]
```

为什么 embedding 要 ×5 且不 decay？

- 每个 slot 的更新次数极少（长尾），需要更大步长；
- weight decay 会把没被命中的稀有 slot 推向 0，破坏记忆；
- 5× 是论文调出来的，不是理论值。

`make_param_groups` 用参数名匹配 `memory.table.embedding.weight`。测试检查：

- embedding 组 lr = 5×；
- weight_decay = 0；
- 所有参数恰好出现一次。

---

## 5.6 合成语料：让 Engram 有用武之地

`EntityCorpus` 生成：

```
<bos> name0_0 name0_1 name0_2 -> attr0 <eos>
<bos> who is name0_0 name0_1 name0_2 ? the answer is attr0 <eos>
<bos> the story of name0_0 name0_1 name0_2 is about attr0 . <eos>
<bos> attr0 is known as name0_0 name0_1 name0_2 <eos>
```

- 实体名 3 个 token，属性 1 个 token；
- 4 种模板，防止模型靠位置捷径；
- `n_train_entities < n_entities` 时留出 held-out 实体，测泛化。

3-gram 恰好覆盖完整实体名，所以 Engram 可以在最后一个 name token 上检索到 `attr`。baseline 必须用 attention + FFN 重建这个映射。

**注意**：这是机制 demo，不是 27B 结论。小模型可能 baseline 也很好；实验脚本会如实报告。

---

## 5.7 训练循环

```python
for step in range(steps):
    ids, targets = corpus.sample(batch_size)
    loss = trainer.step(ids, targets)   # backward + clip + step + scheduler
```

- `targets` 是右移一位的 ids，padding 位置为 `-100`（cross_entropy 忽略）；
- warmup 20 步，cosine 到 10%；
- grad clip 1.0。

`Trainer.evaluate` 返回 loss 和 token accuracy（忽略 padding）。

---

## 5.8 自测

```bash
pytest tests/test_l5_transformer.py -x -q
```

覆盖：

- 前向形状/loss；
- 因果性；
- KV cache 与 full forward 一致（含 context_ids）；
- cached generate == naive generate；
- 权重共享；
- Engram 插入层正确；
- RoPE 相对位置性质；
- 参数组；
- embedding 梯度；
- Trainer 更新参数；
- evaluate 指标。

---

## 5.9 思考题

难度梯度：**[理解]** 能用自己的话解释；**[推导]** 需要动笔算；**[设计]** 需要改代码/做实验；**[开放]** 没有标准答案。

1. **[理解]** 为什么多分支 reshape 必须用 `permute` 而不是 `reshape`？画一下内存布局。
2. **[设计]** KV cache 下，Engram 的 `context_ids` 需要保存多长？能否只保存 `N-1` 个 token？
3. **[设计]** 如果把 Engram 放在 layer 0（embedding 之后、attention 之前），gate 的 query 是什么？会有什么问题？
4. **[推导]** 为什么 embedding 用 Adam 而 backbone 用 Muon？两者的更新几何有什么不同？
5. **[开放]** 合成语料上，baseline 和 Engram 的 loss 曲线可能很接近。如何设计一个更能体现差异的任务？

---

## 5.10 提示

- `F.scaled_dot_product_attention(..., is_causal=True)` 只在没有 cache 且 `T>1` 时使用；有 cache 时所有 key 都是过去，不需要 mask。
- `position_offset` 是**新 token 的绝对位置**，等于 cache 长度。
- SwiGLU 的 hidden 宽度乘 `2/3`，保持和 4× MLP 的 FLOPs 可比。
- 训练时 `model.train()`，评估时 `model.eval()`；本项目 dropout=0，但习惯要对。

---

上一关：[Lab 4：ShortConv 与融合](lab4-fusion.md)  
下一关：[Lab 6：稀疏分配与缩放律](lab6-allocation.md)
