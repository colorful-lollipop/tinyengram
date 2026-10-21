# Lab 7：推理系统与 offload —— 把 100B 参数放进主机内存

> 对应代码：`src/tinyengram/systems.py`、`src/tinyengram/inference.py`、`perf/`  
> 对应测试：`tests/test_l7_systems.py`、`tests/test_l7_perf.py`  
> 论文：§2.5 Figure 2、§6.4 Table 4

> **前置**：L5 ｜ **预计**：2.5 h ｜ **验收**：`pytest tests/test_l7_*.py`  
> **必做**：`systems.py` / `inference.py`，通过 11 个测试（GPU test 可选）。  
> **进阶**：实现 pinned-memory 真 offload；比较 LRU / Zipf 缓存策略。  
> **挑战**：做 All-to-All 通信重叠；调优 Triton kernel 的 block/grid。

---

## 7.1 确定性寻址是系统优势的根源

MoE 的路由依赖 **hidden state**：

```
router(h_t) -> top-k experts
```

h_t 要等前面的层算完才知道，所以专家加载无法提前。

Engram 的索引只依赖 **token id**：

```
hash(x_{t-2}, x_{t-1}, x_t) -> slot ids
```

token 序列在 forward 之前就已知。因此：

```
时间轴 ──────────────────────────────────────────>
layer 1 compute | layer 2 compute | layer 3 ...
     └─ prefetch layer 2 rows ─┘
```

预取可以和前一层计算完全重叠。这就是论文说的 "deterministic addressing enables runtime prefetching"。

---

## 7.2 预取的时间账

每层需要读取的字节：

```
bytes = unique_slots × head_dim × bytes_per_element
```

27B 配置、bf16：

```
16 heads × 80 dims × 2 B = 2.5 KiB / token
```

一个 batch 512 token，若每个 token 的 slot 都不同：

```
512 × 2.5 KiB = 1.25 MiB
```

PCIe 4.0 x16 有效带宽约 25–50 GB/s，1.25 MiB 的传输时间：

```
1.25e6 / 40e9 ≈ 31 μs
```

而一层 Transformer 的计算通常是毫秒级。**计算窗口远大于传输时间**，所以 stall 接近 0。

论文 Table 4 的实测：100B 表全放 host DRAM，吞吐只下降 **2.8%**（8B backbone）。

关键洞察：**传输量 ∝ 激活 slot 数，不 ∝ 表大小**。100B 表和 1B 表，每 token 都只读 16 行。

---

## 7.3 Zipf 与多级缓存

自然语言的 n-gram 频率服从 Zipf 分布：

```
P(rank r) ∝ 1 / r^s,   s ≈ 1
```

少量高频 n-gram 占了绝大多数访问。因此可以做多级缓存：

```
HBM (GPU)      <- 最热门的几万行
Host DRAM      <- 大部分表
NVMe SSD       <- 长尾
```

论文的多级缓存是「could maximize」级别的设计；Table 4 的实验故意**不做缓存**（全部走 PCIe），得到的是保守下界。

`LRUCache` / `simulate_cache` 让你量化：

- 容量 0：全 miss；
- 容量 = unique：全 hit；
- 容量介于中间：hit rate 由 Zipf 和访问顺序决定。

---

## 7.4 训练时的 All-to-All

训练时表可能装不下单卡。论文用标准模型并行：把表按行切到多卡。

前向：

1. 每张卡算出本 batch 需要的 slot id；
2. All-to-All 把请求发给持有该行的卡；
3. 目标卡 gather 后发回；
4. 拼回 `[B,T,d_mem]`。

反向：

1. 梯度按 owner 分组；
2. All-to-All dispatch 回各卡；
3. 各卡只更新自己的行。

`shard_gather` 模拟前向路由：

```python
owner = bucketize(ids, boundaries)
local = ids - offsets[owner]
out = stack([shards[o][l] for o, l in ...])
```

测试检查：分片 gather 的结果和「整张表直接 gather」完全一致。

真实实现还要考虑：

- 负载均衡（热门行所在卡会成为瓶颈）；
- 通信与计算重叠；
- 梯度累积的确定性。

---

## 7.5 offload 模拟器的模型

`simulate_offload` 的假设：

```
for layer in layers:
    unique = unique(plan[layer])
    hits = cache 命中的行
    misses = unique - hits
    transfer_ms = misses * bytes_per_slot / bandwidth
    stall = max(0, transfer_ms - compute_window_ms)
```

参数：

- `bytes_per_slot`：一行 head 的字节数（bf16 80 维 = 160 B）；
- `compute_window_ms`：前一层计算能提供的遮盖时间；
- `bandwidth_gbps`：PCIe/NVLink 带宽（decimal Gbit/s）；
- `cache_capacity`：HBM 能放多少行（best-case 全热）。

输出：

- `stalls_ms`：暴露的延迟；
- `transfer_ms`：总传输时间；
- `bytes`：总流量；
- `hit_rate`：缓存命中率。

测试检查：

- 窗口足够大 → stall=0；
- 窗口=0 → stall=transfer；
- 重复 slot 不重复计费；
- 缓存降低流量；
- 传输时间公式（1 GB @ 64 Gbps = 125 ms）。

---

## 7.6 真实 benchmark：Triton 融合 gather

`perf/kernels.py` 实现一个融合 kernel：

```
one program per token:
  for h in 0..H-1:
      row = table[hash[n,h] + offset[h]]
      out[n, h*D:(h+1)*D] = row
```

对比 eager：

```
eager:        table[hashes + offsets].flatten(-2)
```

在 RTX 4070 Ti Super 上的结果（`python perf/bench.py`）：

| 实现 | 延迟 | 带宽 |
|---|---|---|
| eager | 0.180 ms | 233 GB/s |
| torch.compile | 0.155 ms | 271 GB/s |
| Triton fused | 0.174 ms | 241 GB/s |

**融合只快 3%**。这是一个重要的负面结果：检索是**随机访问、延迟受限**的，不是纯带宽受限；eager 的 `index_select` 已经高度优化，融合省下的 launch 开销抵不过随机访存。

真正的大头在模块级：`torch.compile` 把 gate/norm/conv 融合后，forward 从 1.87 ms 降到 0.55 ms（**3.4×**）。所以性能优化的正确顺序是：

1. 先 profile，找到热点；
2. 模块级融合（compile / 手写 kernel）；
3. 只有当 gather 成为瓶颈时，才去写 gather kernel。

---

## 7.7 KV cache 与 Engram 的局部历史

`generate` 用 KV cache 避免重算前缀。但 Engram 需要前 `N-1` 个 token：

```python
logits, _, caches = model(
    next_id, caches=caches, position_offset=pos, context_ids=out
)
```

`out` 是完整序列。也可以只保留最后 `N-1` 个 token（省内存），但 `position_offset` 仍然要传绝对位置。

测试 `test_cached_generation_matches_naive_greedy` 检查两种解码结果逐 token 相同。

---

## 7.8 自测

```bash
pytest tests/test_l7_systems.py -x -q
pytest tests/test_l7_perf.py -x -q   # 需要 CUDA + Triton
python perf/bench.py --breakdown
python experiments/exp4_offload.py
```

覆盖：

- Zipf 归一化/单调；
- LRU 淘汰；
- 缓存命中率上下界；
- 预取计划 == 哈希；
- 流量与 unique slot 数成正比；
- 大窗口隐藏 stall；
- 缓存降低流量；
- 分片 gather 一致；
- 传输时间公式；
- 参数校验。

---

## 7.9 思考题

难度梯度：**[理解]** 能用自己的话解释；**[推导]** 需要动笔算；**[设计]** 需要改代码/做实验；**[开放]** 没有标准答案。

1. **[理解]** 如果 batch=1、序列长度=1，预取还有意义吗？计算窗口是多少？
2. **[推导]** PCIe 5.0 比 4.0 快一倍，offload 的瓶颈会转移吗？
3. **[设计]** 为什么论文把 Engram 放在 layer 2/15，而不是 layer 0？从预取窗口和建模两个角度回答。
4. **[推导]** 训练时 All-to-All 的通信量如何估算？和 MoE 的 All-to-All 比谁大？
5. **[开放]** 如果热门 slot 集中在少数几张卡，如何做负载均衡？重新哈希可行吗？

---

## 7.10 提示

- `simulate_offload` 的 `plan` 是 `dict[layer_id, LongTensor]`，值可以是任意形状，内部会 `unique`。
- `estimate_transfer_ms` 用**十进制 Gbit/s**，1 byte = 8 bit。
- `shard_gather` 的 shard 是连续切分，不是取模；这样可以用 `bucketize`。
- Triton kernel 只在 CUDA 上跑；CPU 环境测试会自动 skip。

---

上一关：[Lab 6：稀疏分配与缩放律](lab6-allocation.md)  
下一关：[Lab 8：机制分析](lab8-analysis.md)
