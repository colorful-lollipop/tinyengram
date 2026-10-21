# Lab 4：ShortConv 与融合 —— 把记忆写回残差流

> 对应代码：`src/tinyengram/conv.py`、`src/tinyengram/engram.py`  
> 对应测试：`tests/test_l4_engram.py`  
> 论文：§2.3 Eq. (5)、§2.4

> **前置**：L3 ｜ **预计**：2 h ｜ **验收**：`pytest tests/test_l4_*.py`  
> **必做**：`ShortConv` / `Engram.forward`，通过 10 个测试。  
> **进阶**：扫 dilation/kernel；对比零初始化与随机初始化。  
> **挑战**：写 fused gate+value+conv Triton kernel；和 mHC 的流混合结合。

---

## 4.1 为什么需要卷积

门控后的记忆 `u_t` 只包含「当前位置的 2/3-gram」。但语言模式常常跨更长范围：

```
"as a matter of fact"     (4-gram)
"the city of New York"    (5-gram)
```

一个 kernel size 4、dilation 3 的深度卷积，让位置 `t` 看到 `t, t-3, t-6, t-9`，感受野达到 10 个 token，参数量只有 `4 × d`（每个通道一个 4-tap filter）。

```
Y = SiLU( Conv1D( RMSNorm(V~) ) ) + V~
```

- `Conv1D` 是 **depthwise**：每个通道独立卷积；
- `RMSNorm` 在卷积前稳定尺度；
- `SiLU` 提供非线性；
- **残差** `+ V~` 保证不破坏已经门控好的记忆。

---

## 4.2 因果性

卷积核在左侧补 `(k-1) × dilation` 个零，然后取前 `T` 个输出：

```
padding = (kernel_size - 1) * dilation
y = conv(x, padding=padding)[..., :T]
```

输出 `t` 的窗口是：

```
t, t - dilation, t - 2*dilation, ..., t - (k-1)*dilation
```

全部 `<= t`。不会看到未来。

**不要用 `F.pad(..., mode='circular')` 或 `torch.roll`**：它们会把序列末尾卷到开头，破坏因果性。测试 `test_short_conv_is_causal` 会改动 `t >= 7` 的输入，检查 `y[:, :7]` 不变。

---

## 4.3 零初始化 = 恒等映射

论文 Table 5 有一行：

```
Engram Conv Zero Init    True
```

把卷积权重初始化为 0：

```
Y = SiLU(0) + V~ = 0 + V~ = V~
```

于是训练第 0 步，Engram 就是一个**纯门控查表**，不会给 backbone 注入随机噪声。这对深层网络很关键：如果 conv 随机初始化，27B 模型在第一步就会收到方差不匹配的信号，可能直接训练不稳定。

测试 `test_short_conv_zero_init_is_identity` 与 `test_engram_init_is_pure_gated_lookup` 保护这个性质。

注意：`bias=False`。如果有 bias，零初始化权重仍然会输出 bias，就不是恒等了。

---

## 4.4 Engram 的完整前向

```python
e = retrieve(ids)                    # [B,T,d_mem]
v = W_V(e)                           # [B,T,D]，所有分支共享
for m in range(M):
    k_m = W_K[m](e)
    alpha_m = sigmoid(signed_sqrt(<RMSNorm(h_m), RMSNorm(k_m)> / sqrt(D)))
    u_m = alpha_m * v
u = stack(u_m, dim=2)                # [B,T,M,D]
y = u + ShortConv(u)
return y
```

调用方：

```python
h = h + engram(h, ids)
```

即 Engram 的输出是**增量**，不是替换。论文 Figure 1 的 `H^(l) <- H^(l) + Y`。

---

## 4.5 放在哪一层？

论文 §6.2 的 layer sweep（固定 1.6B 预算，单模块）：

| 插入层 | val loss |
|---|---|
| 1 | 1.778 |
| **2** | **1.770** |
| 3 | 1.780 |
| 6 | 1.783 |
| 12 | 1.808（≈ baseline） |

结论：

- **太早**（layer 1）：hidden state 还没经过 attention，gate 缺少全局上下文，门控不准；
- **太晚**（layer > 6）：backbone 已经花深度重建了局部模式，Engram 来得太迟；
- **layer 2** 是单模块最优：一轮 attention 足够提供上下文，又足够早。

但论文最终用**两个模块**（layer 2 和 15），每个分配一半容量（减小 `d_mem`）。这比单个 1.6B 模块更好（1.768 vs 1.770），因为同时获得「早期卸载」和「晚期精细门控」。

系统上也更好：两个模块让预取和计算重叠更充分（Lab 7）。

---

## 4.6 多分支集成

mHC backbone 把 residual stream 扩成 `M=4` 条。Engram 的适配：

- 一份 embedding 表（所有分支共享）；
- 一份 `W_V`（所有分支共享）；
- `M` 份 `W_K`（每分支独立 gate）；
- 一个 depthwise conv，`groups=M`（每分支独立卷积）。

为什么共享 `W_V`？因为记忆的「值」是客观的，分歧只在「要不要用」。共享让 `W_V` 和所有 `W_K` 可以拼成一个 FP8 GEMM：

```
[W_V; W_K^1; ...; W_K^M] @ e_t
```

一次矩阵乘，GPU 利用率最高。测试 `test_value_is_shared_but_keys_are_per_branch` 检查参数量关系：`key_params == M * value_params`。

论文 ablation（§6.2）显示，去掉 multi-branch fusion 是最大的回退之一——所以这不是工程细节，而是模型效果的一部分。

---

## 4.7 形状检查清单

| 张量 | 形状 |
|---|---|
| `input_ids` | `[B, T]` |
| `hidden_states` | `[B, T, M, D]` |
| `e_t` | `[B, T, d_mem]` |
| `value` | `[B, T, D]` |
| `gates` | `[B, T, M]` |
| `u` | `[B, T, M, D]` |
| `short_conv(u)` | `[B, T, M, D]` |
| 输出 | `[B, T, M, D]` |

`d_mem = (N-1) * n_embed_per_ngram`。27B: `2 × 640 = 1280`。

---

## 4.8 自测

```bash
pytest tests/test_l4_engram.py -x -q
```

覆盖：

- 卷积因果性；
- 零初始化恒等；
- dilation 感受野（脉冲响应）；
- Engram 输出形状与 gate 范围；
- 初始化时等于纯门控查表；
- 分支 gate 不同；
- token 因果性；
- 参数共享；
- tokenizer 压缩集成；
- 形状校验。

---

## 4.9 思考题

难度梯度：**[理解]** 能用自己的话解释；**[推导]** 需要动笔算；**[设计]** 需要改代码/做实验；**[开放]** 没有标准答案。

1. **[理解]** 如果把 dilation 从 3 改成 1，感受野变成多少？对 3-gram 记忆有什么影响？
2. **[推导]** 为什么 conv 前要 RMSNorm，而不是 conv 后？
3. **[推导]** 零初始化 conv 是否会让训练早期 conv 永远学不到东西？梯度会怎样？
4. **[设计]** 两个 Engram 模块（layer 2/15）如何分配 5.7B 参数？每个模块的 `d_mem` 是多少？
5. **[理解]** 如果 backbone 是单流（M=1），Engram 还需要 `M` 份 `W_K` 吗？

---

## 4.10 提示

- `nn.Conv1d(..., groups=channels)` 是 depthwise；`groups=1` 是普通卷积。
- 卷积输入是 `[B, C, T]`，记得 transpose 再 transpose 回来。
- `nn.init.zeros_(self.conv.weight)` 放在 `__init__` 最后。
- 测试脉冲响应时，把 `activation=False`，否则 SiLU 会改变零附近的值。

---

上一关：[Lab 3：上下文门控](lab3-gating.md)  
下一关：[Lab 5：Transformer 与训练](lab5-transformer.md)
