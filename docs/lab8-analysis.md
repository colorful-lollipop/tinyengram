# Lab 8：机制分析 —— 记忆到底让模型「深」了多少

> 对应代码：`src/tinyengram/analysis.py`  
> 对应测试：`tests/test_l8_analysis.py`  
> 论文：§6.1 Figure 4、§6.2 Figure 5、§6.5 Figure 7

> **前置**：L5 ｜ **预计**：1.5 h ｜ **验收**：`pytest tests/test_l8_*.py`  
> **必做**：`linear_cka` / `logit_lens_kl` / `soft_alignment`，通过 9 个测试。  
> **进阶**：训练 baseline/Engram 对比；做 gating / conv / 压缩消融。  
> **挑战**：实现 unbiased CKA；用因果中介或敏感性实验验证知识存储位置。

---

## 8.1 核心假设

论文的机制假设是：

> Transformer 模拟知识检索需要消耗深度。Engram 用一次 O(1) 查表替代了这段计算，因此**等效于增加了模型有效深度**。

三个可检验的推论：

1. 中间层的预测分布更早接近最终分布（LogitLens KL 更低）；
2. Engram 的浅层表示与 baseline 的深层表示更相似（CKA 上移）；
3. 门控在静态模式上激活（可视化）。

---

## 8.2 LogitLens

对每一层 hidden state `h^(l)`，用最终的 norm + LM head 投影：

```
logits^(l) = LMHead(RMSNorm(h^(l)))
```

计算与最终层分布的 KL：

```
KL(P_final || P_l) = sum_v P_final(v) log(P_final(v) / P_l(v))
```

- `l = L` 时 KL=0；
- KL 下降越快，说明表示越早「可预测」；
- Engram 模型的曲线在**早期层**明显更低（论文 Figure 4a）。

实现注意：

- 用 `log_softmax` 而不是 `softmax`，数值更稳；
- 对 batch/seq 取平均；
- 最后一层 KL 应该≈0（测试检查）。

```python
target = log_softmax(final_logits)
kl = (target.exp() * (target - log_softmax(layer_logits))).sum(-1).mean()
```

---

## 8.3 CKA

Centered Kernel Alignment 衡量两组表示的结构相似度，对宽度/旋转不敏感。

给定 `X [N, D1]`、`Y [N, D2]`，先中心化：

```
Xc = X - mean(X)
Yc = Y - mean(Y)
```

线性核 CKA（feature-space 形式）：

```
CKA = ||Yc^T Xc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)
```

性质：

- 完全相同：1；
- 各向同性缩放：1；
- 正交变换：1；
- 独立随机：接近 0；
- 对样本数敏感：N 太小时估计噪声大。

论文用 unbiased HSIC estimator（Davari et al., 2023），本项目用更简单的 feature-space 形式，足以展示趋势。测试覆盖 identity/scaling/rotation/independent。

### 软对齐

论文 Eq. (9)：对每个 Engram 层 `j`，找最相似的 top-k 个 MoE 层，取加权平均：

```
a_j = sum_{i in I_j} S_ij * i / sum_{i in I_j} S_ij
I_j = argtopk_i S_ij
```

如果 `a_j > j`，说明 Engram 的浅层在做 baseline 深层的事。论文报告 Engram layer 5 ≈ MoE layer 12。

`soft_alignment` 的输入是 `[L_moe, L_engram]`，输出 `[L_engram]`。测试用 identity 和 shift 矩阵验证。

---

## 8.4 门控可视化

论文 Figure 7 把 `alpha_t` 画成热力图，发现：

- "Alexander the Great"：在 "Great" 上强烈激活；
- "the Milky Way"：在 "Way" 上激活；
- "Princess of Wales"：在 "Wales" 上激活；
- 中文 "四大发明"、"张仲景" 同样。

因为 Engram 用后缀 3-gram，`t` 上的高激活意味着「以 t 结尾的短语被识别为静态模式」。

`gate_stats` 给出：

- mean / std；
- `frac_above_0.5` / `frac_below_0.1`；
- 二元熵。

训练良好的门控应该是**双峰**的：大部分 token 接近 0，少数实体/搭配接近 1。如果全是 0.5，说明 gate 没学到东西。

---

## 8.5 消融实验

论文 §6.2 的结论（固定 1.6B 预算）：

| 变体 | 效果 |
|---|---|
| 去掉 multi-branch fusion | 最大回退之一 |
| 去掉 context-aware gating | 大回退 |
| 去掉 tokenizer compression | 大回退 |
| 去掉 short conv | 轻微回退 |
| 加 4-gram | 轻微变差（容量稀释） |

复现建议：

1. 用 `EngramConfig` 的开关（`hc_mult=1`、`use_gating=False` 等）做对照；
2. 固定总参数量，不要固定 slot 数；
3. 每个变体跑 3 个 seed，报告均值±方差；
4. 小规模只能复现**方向**，不要期待和 27B 一样的幅度。

本项目没有为所有消融提供开关——这是留给你的 Capstone 扩展。

---

## 8.6 敏感性分析

论文 §6.3 在推理时**完全关掉 Engram 输出**，看保留多少性能：

| 任务类型 | 保留比例 |
|---|---|
| 事实知识（TriviaQA 等） | 29–44% |
| 阅读理解（C3 等） | 81–93% |

解释：

- 事实知识主要存在 Engram 里，关掉就崩；
- 阅读理解依赖上下文 attention，Engram 只是辅助。

这是一个很强的因果证据：Engram 不是「锦上添花」，而是事实知识的主要载体。

实现上，可以在 forward 时把 gate 置 0 或直接跳过 `h + engram(...)`。注意这是**训练-推理不一致**的 ablation，会引入噪声。

---

## 8.7 自测

```bash
pytest tests/test_l8_analysis.py -x -q
```

覆盖：

- CKA identity/scaling/rotation/independent；
- 输入校验；
- 软对齐 identity/shift；
- LogitLens 最后一层 KL=0；
- 二元熵最大值；
- gate 统计范围。

---

## 8.8 思考题

难度梯度：**[理解]** 能用自己的话解释；**[推导]** 需要动笔算；**[设计]** 需要改代码/做实验；**[开放]** 没有标准答案。

1. **[推导]** CKA 对样本数 N 的依赖：N=16 和 N=1024 的估计差多少？如何做 unbiased 估计？
2. **[理解]** 如果 Engram 层和 MoE 层的宽度不同，CKA 还能算吗？为什么？
3. **[开放]** LogitLens 的 KL 降低，是否一定意味着「更好」？有没有可能是过度自信？
4. **[设计]** 门控双峰分布如何量化？设计一个「选择性分数」。
5. **[开放]** 关掉 Engram 后阅读理解保留 80%+，能否说明 Engram 对长上下文没用？结合 RULER 结果讨论。

---

## 8.9 提示

- CKA 的矩阵可能是奇异的；用 Frobenius 范数而不是行列式。
- `topk` 返回的 indices 形状是 `[k, cols]`，`gather` 时 dim=0。
- 软对齐的权重和可能为 0；`clamp_min(1e-12)`。
- LogitLens 的 hidden 要是**每层 block 输出**，不是 attention 内部。

---

上一关：[Lab 7：推理系统与 offload](lab7-systems.md)  
下一关：[Lab 9：端到端复现](lab9-capstone.md)
