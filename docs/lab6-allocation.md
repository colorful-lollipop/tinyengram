# Lab 6：稀疏分配与缩放律 —— 给「记忆 vs 计算」算一笔账

> 对应代码：`src/tinyengram/allocation.py`  
> 对应测试：`tests/test_l6_allocation.py`  
> 论文：§3 Figure 3、Table 5

> **前置**：L2、L5 ｜ **预计**：1.5 h ｜ **验收**：`pytest tests/test_l6_*.py`  
> **必做**：`allocate` / 拟合函数，通过 12 个测试。  
> **进阶**：用小规模真实 runs 拟合 U 形；改变 sparsity 看最优点。  
> **挑战**：加入推理延迟约束重做分配；分析 40B under-training。

---

## 6.1 三个参数口径

论文用三个量把「容量」和「计算」分开：

| 符号 | 含义 | 决定什么 |
|---|---|---|
| `P_tot` | 总可训练参数（不含词嵌入/LM head） | 显存、模型容量 |
| `P_act` | 每 token 激活的参数 | FLOPs |
| `P_sparse` | `P_tot - P_act` | 「免费」容量 |

对 MoE：`P_act` 由 top-k 专家决定，未选中的专家进入 `P_sparse`。  
对 Engram：每个 token 只取固定数量的 slot，增加表大小只增加 `P_tot`，不增加 `P_act`。

定义**分配比例**：

```
rho = P_moe / P_sparse
P_engram = (1 - rho) * P_sparse
```

- `rho = 1`：纯 MoE；
- `rho = 0`：纯记忆（没有条件计算）；
- 论文最优：`rho ≈ 0.75 - 0.80`。

---

## 6.2 为什么是 U 形

两个极端都差：

**rho → 1（纯 MoE）**：模型没有专门的静态记忆，必须用深度和计算去「重建」n-gram。早期层本可以做推理，却在做查表。

**rho → 0（纯记忆）**：模型失去条件计算能力。记忆能告诉你「Paris 是法国首都」，但不能做多步推理、代码执行、数学运算。

中间存在一个最优点：把**静态、局部、模式化**的知识交给记忆，把**动态、组合、上下文相关**的计算留给 MoE。

论文在 2e20 和 6e20 FLOPs 两个预算下都观察到 U 形，最优 `rho` 稳定在 75%–80%，说明这个比例对规模不敏感（在固定 sparsity 下）。

---

## 6.3 参数换算

给定 `P_tot`、`P_act`、`rho`：

```python
sparse = P_tot - P_act
moe = rho * sparse
engram = (1 - rho) * sparse
n_experts = moe / params_per_expert
slots_per_head = engram / (d_mem * n_layers)
```

其中 `d_mem` 已经是所有 head 拼起来的总宽度（27B: 1280）。因为

```
params = slots_per_head * d_mem * layers
```

（`slots_per_head × 16 heads × 80 dims = slots_per_head × 1280`）。

27B 的账：

```
P_tot = 26.7B, P_act = 3.8B  =>  sparse = 22.9B
rho = 0.743                  =>  engram = 5.89B, moe = 17.01B
engram slots/head = 5.89e9 / (1280 * 2) = 2.30M
```

论文 Table 5 写 2,262,400，差 1.7%，来自四舍五入和 `P_tot` 的精确值。

---

## 6.4 无限记忆 regime

固定 backbone（`P_tot ≈ 3B`, `P_act = 568M`），只增加 Engram slots：

```
2.58e5 -> 1.0e7 slots
```

验证 loss 对 `log10(slots)` 近似线性：

```
L(s) = a + b * log10(s),   b < 0
```

论文的解读：

- 记忆是一个**可预测的 scaling knob**：加 slot 不需要加 FLOPs；
- OverEncoding（直接平均 n-gram embedding）也有收益，但同样预算下 Engram 的斜率更陡；
- 到 1e7 还没有饱和迹象。

本项目 `power_law_fit` / `power_law_predict` 实现这条拟合；`experiments/exp5_scaling.py` 在玩具任务上扫 slots。

---

## 6.5 复算 Table 5

```bash
python scripts/paper_math.py
```

输出（参考实现）：

```
Engram 参数量              5.79B   (论文: 5.7B)
Engram-40B 记忆参数量      18.53B  (论文: 18.5B)
每个 head/order 的 slot 数    2,262,400
总 head row 数/层        36.2M
routed experts                 55   (论文: 55)
训练 FLOPs (6PD)         5.97e21   (论文: ~6e21)
```

注意 `2,262,400` 是**每个 head** 的 slot 数。16 个 head、每个 80 维：

```
2,262,400 × 16 × 80 × 2 layers = 5.79B
```

如果你把 `2,262,400` 当成「所有 head 的总 slot 数」，算出来的参数量会少 16 倍。这是读 Table 5 最容易踩的坑。

---

## 6.6 拟合细节

### 二次拟合

```python
L(rho) = a rho^2 + b rho + c
rho* = -b / (2a)
```

用 `torch.linalg.lstsq` 解。测试用精确抛物线检查系数恢复。

### 幂律拟合

```python
x = log10(slots)
L = intercept + slope * x
```

最小二乘闭式解：

```
slope = sum((x - x̄)(y - ȳ)) / sum((x - x̄)^2)
```

测试检查斜率恢复和外推。

---

## 6.7 这个模型的局限

1. **解析 U 形不是实验结果**。它只是把论文的定性形状参数化，用来练习拟合和预算换算。
2. **iso-FLOPs 的假设**：`6 * P_act * tokens` 忽略了 attention 的 `T^2` 项、通信、重计算。
3. **rho 的最优值依赖 sparsity**。论文固定 `P_tot/P_act ≈ 10`；如果 sparsity 变成 100，最优 rho 可能移动。
4. **记忆和计算的边界不是固定的**。4-gram、更长实体、代码模板，都会改变「什么算静态」。

---

## 6.8 自测

```bash
pytest tests/test_l6_allocation.py -x -q
```

覆盖：

- 27B/40B 参数量复算；
- 预算守恒（moe + engram = sparse）；
- rho 极端值；
- 单调性（rho↑ ⇒ experts↑, slots↓）；
- U 形最小值；
- 不对称项不移动最小值；
- 二次拟合系数/argmin；
- 幂律拟合/外推；
- iso-FLOPs；
- 非法输入。

---

## 6.9 思考题

难度梯度：**[理解]** 能用自己的话解释；**[推导]** 需要动笔算；**[设计]** 需要改代码/做实验；**[开放]** 没有标准答案。

1. **[推导]** 如果 `P_tot/P_act = 100`，最优 rho 会变大还是变小？为什么？
2. **[设计]** 把 Engram 的 `d_mem` 减半、slots 翻倍，参数量不变。哪种更好？设计一个实验。
3. **[开放]** 论文说 40B 模型「还没训练充分」。如何从 loss 曲线判断是 under-training 还是容量浪费？
4. **[设计]** 如果推理时只能把 10% 的表放进 HBM，rho 的最优值会变吗？（提示：系统约束会反过来影响建模选择。）
5. **[推导]** 用 `u_shaped_loss` 拟合论文 Figure 3 的两个预算，需要哪些额外参数？

---

## 6.10 提示

- `allocate` 里所有量都是整数；先算 `slots_per_head`，再乘 `n_heads_total` 得 `slots_total`。
- U 形函数在一维上的最小值就是顶点；如果加不对称项，确保该项在顶点右侧为 0。
- 拟合前把 rho 限制在 `[0,1]`，否则顶点可能跑到区间外。
- `power_law_fit` 的输入 slots 必须都 > 0。

---

上一关：[Lab 5：Transformer 与训练](lab5-transformer.md)  
下一关：[Lab 7：推理系统与 offload](lab7-systems.md)
