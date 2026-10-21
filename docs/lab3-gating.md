# Lab 3：上下文门控 —— 让静态记忆学会「闭嘴」

> 对应代码：`src/tinyengram/gating.py`  
> 对应测试：`tests/test_l3_gating.py`  
> 论文：§2.3 Eq. (3)–(4)

> **前置**：L2 ｜ **预计**：1.5 h ｜ **验收**：`pytest tests/test_l3_*.py`  
> **必做**：`rms_norm` / `signed_sqrt` / `context_gate`，通过 8 个测试。  
> **进阶**：把标量 gate 改成向量 gate；做去 gating 的消融。  
> **挑战**：推导 sigmoid 饱和区的梯度界；设计 gate 校准指标。

---

## 3.1 问题：检索到的记忆不一定是真的

`"the Great"` 的 3-gram 记忆，在下面两句话里含义完全不同：

```
Alexander the Great conquered ...
The Great Depression began ...
```

静态 embedding 只知道「the Great 后面通常接什么」，不知道当前语境。直接把它加进 residual stream 会引入噪声。论文用一个**标量门**解决：

```
k_t = W_K e_t
v_t = W_V e_t
alpha_t = sigmoid( <RMSNorm(h_t), RMSNorm(k_t)> / sqrt(d) )
u_t = alpha_t * v_t
```

- Query 来自当前 hidden state（已经过前面 attention，聚合了全局上下文）；
- Key/Value 来自检索到的记忆；
- `alpha` 是 0 到 1 的标量，对所有 1280 维共享。

这和 attention 的形式很像，但**没有 softmax、没有多 token 竞争**：每个位置独立决定「这份记忆可不可信」。

---

## 3.2 RMSNorm 的作用

RMSNorm 不减去均值，只除以均方根：

```
RMSNorm(x) = x / sqrt(mean(x^2) + eps)
```

为什么不用 LayerNorm？

- 记忆向量和 hidden state 的均值没有可比性，减均值反而破坏方向信息；
- RMSNorm 更便宜（一次 reduce + 一次乘）；
- 现代 LLM（LLaMA、DeepSeek）都用 RMSNorm。

归一化后，每个向量的 L2 范数约为 `sqrt(d)`：

```
||RMSNorm(x)||_2 ≈ sqrt(d)
```

两个这样的向量同向时点积最大为 `d`。论文除以 `sqrt(d)`，所以 logit 的范围是：

```
[-sqrt(d), +sqrt(d)]
```

而不是 `[-1, 1]`。对 `d = 2560`，范围约 `[-50, 50]`，直接 sigmoid 会严重饱和。

这就是 **signed sqrt** 存在的原因。

---

## 3.3 为什么要有 signed sqrt

直接 `sigmoid(logit)` 在 `|logit| > 10` 时梯度接近 0：

```
sigmoid'(x) = sigmoid(x)(1 - sigmoid(x)) <= 0.25
```

当 `x = 50` 时，梯度约 `1.9e-22`，门控学不动。

`signed_sqrt`：

```
f(x) = sign(x) * sqrt(|x|)
```

把 50 压成 7.07，把 0.01 压成 0.1。性质：

- 保号：负 logit 仍然抑制；
- 压缩大值：缓解饱和；
- 放大近零值：让「模棱两可」的样本有更大梯度；
- 在 0 点导数无穷，所以实现里 `clamp_min(eps)` 兜底。

论文正文没有展开这个细节，但官方 demo 明确写了：

```python
gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
gate = gate.sigmoid()
```

测试 `test_signed_sqrt_preserves_sign_and_magnitude` 和 `test_gate_is_monotone_in_similarity` 保护这些性质。

---

## 3.4 门控的三个区域

固定 `q`，改变 `k` 与 `q` 的余弦相似度 `c`：

| `c` | logit（未 sqrt） | gate | 含义 |
|---|---|---|---|
| -1 | -sqrt(d) | ≈0 | 记忆与上下文矛盾，完全抑制 |
| 0 | 0 | 0.5 | 不确定，折中 |
| +1 | +sqrt(d) | ≈1 | 记忆与上下文一致，全量注入 |

初始化时 `W_K` 很小，`k ≈ 0`，gate ≈ 0.5。训练后门控会在命名实体、固定搭配上接近 1，在功能词上接近 0（论文 Figure 7）。

---

## 3.5 多分支：每个 residual stream 一个门

mHC backbone 有 `M=4` 条并行残差流。Engram 的处理：

- **共享**：embedding 表、`W_V`；
- **独立**：每个分支一个 `W_K^(m)`、一个 gate `alpha^(m)`。

```
alpha_t^(m) = sigmoid( <RMSNorm(h_t^(m)), RMSNorm(W_K^(m) e_t)> / sqrt(d) )
u_t^(m) = alpha_t^(m) * (W_V e_t)
```

这样每个分支可以独立决定「我要不要这份记忆」。论文 §6.2 的 ablation 显示，去掉 branch-specific fusion 是最大的性能回退之一。

工程上，`W_V` 和所有 `W_K^(m)` 可以拼成一个大的 FP8 GEMM：

```
[W_V; W_K^(1); ...; W_K^(M)] @ e_t
```

一次矩阵乘出 `(M+1) × d` 个输出。这是论文提到的 "fuse into a single dense FP8 matrix multiplication"。

---

## 3.6 数值稳定性

- `rms_norm` 的 `eps` 放在 sqrt 里面：`rsqrt(mean(x^2) + eps)`，避免除零。
- `signed_sqrt` 对 0 的导数是无穷，必须 `clamp_min`。
- gate 用 `sigmoid` 而不是 hard threshold，保证可微。
- 如果训练中出现 gate 全 0 或全 1，检查：
  1. `d_mem` 初始化是否过大；
  2. 学习率是否过高；
  3. 是否忘了对 query/key 做 RMSNorm。

---

## 3.7 自测

```bash
pytest tests/test_l3_gating.py -x -q
```

覆盖：

- RMSNorm 单位 RMS；
- 权重缩放；
- signed sqrt 保号、幅值；
- 零 logit → gate=0.5；
- 单调性；
- 缩放不变性（RMSNorm 的核心性质）；
- 梯度能回到 query 和 key；
- 形状校验。

---

## 3.8 思考题

难度梯度：**[理解]** 能用自己的话解释；**[推导]** 需要动笔算；**[设计]** 需要改代码/做实验；**[开放]** 没有标准答案。

1. **[设计]** 如果把 gate 改成向量（每个维度一个 gate），参数量增加多少？会更好吗？
2. **[理解]** 为什么用 `h_t` 做 query，而不是用 `e_t` 自己做 self-gate？
3. **[推导]** `signed_sqrt` 和 `tanh` 都能压缩 logit，两者有什么本质区别？
4. **[理解]** 多分支共享 `W_V` 是否限制了表达？为什么论文仍然这么做？
5. **[开放]** 训练后 gate 的分布应该是什么样？如果所有 token 的 gate 都≈0.5，说明什么？

---

## 3.9 提示

- `rms_norm(x) = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)`。
- `signed_sqrt(x) = torch.sign(x) * x.abs().clamp_min(eps).sqrt()`。
- gate 输出形状 `[..., 1]`，方便广播到 `[..., D]`。
- 测试缩放不变性时，`q*7` 和 `k*0.2` 经过 RMSNorm 后应该几乎相同（浮点误差内）。

---

上一关：[Lab 2：检索与内存布局](lab2-retrieval.md)  
下一关：[Lab 4：ShortConv 与融合](lab4-fusion.md)
