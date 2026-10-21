# Lab 1：多头部哈希 —— 把 V^N 压进常数张表

> 对应代码：`src/tinyengram/hashing.py`  
> 对应测试：`tests/test_l1_hashing.py`  
> 论文：§2.2 Eq. (1)–(2)、§2.5

> **前置**：L0 ｜ **预计**：1.5 h ｜ **验收**：`pytest tests/test_l1_*.py`  
> **必做**：素数表、奇数乘数、乘法-XOR，通过 12 个测试。  
> **进阶**：扫 head 数与 4-gram，测碰撞率和验证 loss。  
> **挑战**：证明 int64 边界；用生日悖论模拟碰撞；改成 uint64 实现并对比。

---

## 1.1 为什么必须哈希

3-gram 的可能数量是 `|V'|^3`。以压缩后 98k 词表计算：

```
98,000^3 ≈ 9.4e14
```

即使每个 slot 只存 80 维 bf16，也需要 `1.5e17` 字节。直接参数化是不可能的。

经典做法是哈希：

```
z_{t,n,k} = phi_{n,k}(g_{t,n}) in [0, M_{n,k})
e_{t,n,k} = E_{n,k}[z_{t,n,k}]
```

把任意 n-gram 映射到固定大小的表。代价是**碰撞**：不同的 n-gram 可能落到同一行，共享记忆。

---

## 1.2 为什么用 K 个 head

如果只用一个哈希函数、一张表，碰撞会直接导致语义混淆。Engram 对每个 n-gram order 用 `K = 8` 个独立哈希头：

```
e_t = concat_{n=2..N} concat_{k=1..K} e_{t,n,k}
```

- 每个 head 有独立的表 `E_{n,k}` 和独立的模数 `M_{n,k}`；
- 一次碰撞只污染一个 head 的 80 维，其他 15 个 head 仍然提供信息；
- 下游的 `W_V` 可以学习「哪个 head 更可信」。

这类似 Bloom filter / count-min sketch 的直觉：多个独立哈希降低最坏情况的影响，而不是消除碰撞。

---

## 1.3 乘法-XOR 哈希

论文说 `phi` 是 "lightweight multiplicative-XOR hash"。官方 demo 的实现是：

```
mix = x_t * m_0
for j in 1..n-1:
    mix = mix XOR (x_{t-j} * m_j)
z = mix mod M
```

其中 `m_j` 是**随机奇数**，每个 layer 独立。

### 为什么是 XOR 而不是加法？

- XOR 没有进位，结果不会因为「低位相加溢出」产生系统性偏差；
- 对固定 `m`，`x -> x*m` 在模 2^64 下是双射（`m` 为奇数），低比特也能充分混合；
- 实现只需一次乘法和一次异或，GPU 上极便宜。

### 为什么乘数必须是奇数？

如果 `m` 是偶数，`x*m` 的最低位恒为 0，哈希的低位会丢失信息。强制 `m = 2r + 1`。

### 为什么每个 layer 用不同乘数？

同一对 token 在 layer 2 和 layer 15 如果碰撞模式完全相同，两个模块会犯同样的错误。不同乘数让碰撞事件近似独立。

---

## 1.4 int64 溢出边界（重要）

PyTorch 的 int64 乘法溢出会 wrap（实际是 C++ 有符号溢出，工程上按补码回绕）。如果乘数太大，`x*m` 可能变成负数，后面的 `% M` 也会得到负索引。

官方做法：

```python
M_max = INT64_MAX // vocab_size
half  = max(1, M_max // 2)
r     = randint(0, half)
m     = 2*r + 1
```

于是

```
x * m <= (V-1) * (INT64_MAX / V) < 2^63
```

每个乘积都非负且小于 `2^63`。XOR 不会产生进位，所以 n 个乘积 XOR 后仍 `< 2^63`。`mod` 结果自然落在 `[0, M)`。

本项目把这条性质写成了测试：`test_multipliers_are_odd_and_bounded` 和 `test_large_vocab_multipliers_do_not_overflow`。

> 注意：严格来说 C++ 有符号溢出是 UB，但 PyTorch/CUDA 的实际行为是补码回绕。教学实现依赖这个事实，生产代码可以用 `uint64` 或显式掩码。

---

## 1.5 素数表大小

每个 head 的表大小 `M_{n,k}` 取「不小于请求值的下一个素数」，并且**全局互不相同**：

```python
seen = set()
for layer in layers:
    for order in orders:
        for head in range(K):
            M = next_prime_after(requested, seen)
            seen.add(M)
```

两个原因：

1. **素数模数**避免 `M` 是 2 的幂时只取低比特、丢掉高位信息；
2. **互不相同**让不同 head 的碰撞模式尽量独立。如果两个 head 用同一个 `M` 且乘数相关，它们可能同时碰撞。

论文 Table 5 的 `Engram Vocab Size = 2,262,400` 是**每个 head、每个 n-gram order** 的 slot 数（官方 demo 的 `engram_vocab_size` 也是这个口径）。因此每层总行数是：

```
2 orders × 8 heads × 2,262,400 = 36.2M rows
```

乘以 `d_head = 80`、两层：

```
36.2M × 80 × 2 = 5.79B 参数
```

`2,262,400` 接近 `1.09 × 129,280`（DeepSeek-V3 词表），说明每个 head 的表容量约为词表大小。

---

## 1.6 后缀 n-gram 的向量化

对 batch `[B, T]`，构造 `N` 个移位视图：

```
shift_0[t] = x[t]
shift_1[t] = x[t-1]   （左端补 pad）
...
```

然后对 order `n`：

```
mix = shift_0 * m_0
for j in 1..n-1:
    mix ^= shift_j * m_j
```

最后对每个 head 取模并 stack，得到 `[B, T, (N-1)*K]`。

实现要点：

- 用 `F.pad(x, (k, 0), value=pad_id)[:, :T]`，不要用 roll（roll 会把末尾卷到开头，破坏因果性）；
- 输出 dtype 必须是 int64；
- 最后一维的顺序必须固定：先 order 2 的所有 head，再 order 3 的所有 head。这个顺序要和 `MultiHeadEmbedding` 的 offsets 完全一致。

---

## 1.7 确定性 = 可预取

哈希只依赖 `input_ids`，不依赖 hidden state，也不依赖任何可学习参数。因此：

- 给定 token 序列，可以在 forward 之前算出所有 slot id；
- 训练时可以用它做 All-to-All 路由；
- 推理时可以在前一层计算时预取下一层的行（Lab 7）。

这是 Engram 相对 MoE 的核心系统优势。测试 `test_hash_is_deterministic` 和 `test_hash_is_strictly_causal` 保护这两个性质。

---

## 1.8 自测

```bash
pytest tests/test_l1_hashing.py -x -q
```

覆盖：

- 素数判定与全局唯一；
- 表大小 >= 请求值且为素数；
- 乘数奇数、上界正确；
- 同 seed 可复现、不同 layer 不同；
- 输出形状/范围；
- 未来 token 不影响过去；
- 首个位置使用 pad；
- 越界输入报错；
- 大词表不溢出。

---

## 1.9 思考题

难度梯度：**[理解]** 能用自己的话解释；**[推导]** 需要动笔算；**[设计]** 需要改代码/做实验；**[开放]** 没有标准答案。

1. **[推导]** 两个 head 的碰撞概率独立吗？如果乘数相同、模数不同，相关性有多大？
2. **[推导]** 给定 `M = 2,262,400`、每天激活 1e9 个 3-gram，期望有多少个不同 n-gram 落到同一个 slot？用生日悖论估算。
3. **[理解]** 如果把 `mod` 换成 `& (M-1)`（M 为 2 的幂），会发生什么？为什么论文坚持素数？
4. **[理解]** 为什么 Engram 不对 1-gram 建表？1-gram 记忆和词嵌入有什么区别？
5. **[开放]** 能否用两个 32-bit 哈希拼成一个 64-bit 索引来扩大表？代价是什么？

---

## 1.10 提示

- `torch.remainder(x, m)` 对正数等价于 `%`，且返回非负。
- 判断素数只需试除到 `sqrt(n)`；表大小在 1e5 量级，毫秒级完成。
- 生成随机数用 `torch.Generator().manual_seed(seed)`，不要用全局 `torch.manual_seed`，否则会污染模型初始化。
- 测试里手动复算哈希时，注意**当前 token 乘 `m[0]`，前一个 token 乘 `m[1]`**，不要写反。

---

上一关：[Lab 0：Tokenizer 压缩](lab0-tokenizer.md)  
下一关：[Lab 2：检索与内存布局](lab2-retrieval.md)
