# FAQ

## 环境与安装

### 需要什么依赖？

核心只需 Python 3.10+ 和 PyTorch 2.2+：

```bash
pip install torch pytest
# 或
uv sync --extra dev
```

可选：

- `triton`：L7 的 GPU kernel；
- `transformers` + `tokenizers`：用真实 tokenizer 做实验；
- `matplotlib`：画图。

### 没有 GPU 能学吗？

能。L0–L6、L8 全部 CPU 可跑；L7 的 Triton 测试会自动 skip，模拟器照常工作；L9 的训练在 CPU 上几分钟。

有 GPU 的话，L9 可以换成真实数据（见 lab9）。

### 为什么 `import tinyengram` 失败？

项目没有强制 `pip install -e .`。测试通过 `tests/conftest.py` 把 `src/` 或 `solutions/` 加进 `sys.path`：

```bash
pytest                      # src（你的实现）
TINYENGRAM_IMPL=solutions pytest   # 参考实现
```

跑脚本时：

```bash
PYTHONPATH=src python scripts/paper_math.py
# 或直接
TINYENGRAM_IMPL=solutions python experiments/exp1_allocation.py
```

---

## 测试与评分

### 一开始全红是正常的吗？

是。`src/` 里所有核心函数都是 `raise NotImplementedError`。按关卡顺序做，前面的关卡没做完，后面的关卡会因为依赖而失败。

### `pytest` 和 `scripts/grade.py` 有什么区别？

- `pytest` 给完整报错和 traceback；
- `grade.py` 按关卡汇总，显示通过/失败/跳过，适合看进度。

```bash
python scripts/grade.py
python scripts/grade.py --stage 3
python scripts/grade.py --solution
```

### 测试太严格了吗？

测试检查的是**数学性质**，不是逐行对照：

- 哈希必须因果、无溢出；
- gate 必须单调、缩放不变；
- 卷积必须因果、零初始化恒等；
- KV cache 必须和 full forward 一致。

如果你实现的方式和参考不同但满足性质，测试应该通过。如果测试本身有 bug，欢迎提 issue。

### 为什么我的实现和参考实现数值不完全一样？

浮点顺序不同（例如 `sum` 的归约顺序）会带来 1e-6 级差异。测试用 `atol=1e-5` 左右。如果差很多，通常是逻辑错误。

---

## 概念

### Engram 和 MoE 的区别？

| | MoE | Engram |
|---|---|---|
| 稀疏维度 | 条件**计算** | 条件**记忆** |
| 路由依据 | hidden state（动态） | token id（静态） |
| 每 token 计算 | top-k 专家 | 固定 16 次查表 |
| 可预取 | 否 | 是 |
| 存什么 | 变换/算法 | 静态模式/事实 |

两者互补，不是替代。

### Engram 和 RAG 的区别？

- RAG 检索**文档**，是外部、非参数化的，通常需要检索器；
- Engram 检索**参数化的 n-gram 记忆**，在模型内部，端到端训练；
- Engram 的 key 是局部 token 序列，不能做语义检索；
- RAG 可以更新知识库，Engram 需要重新训练/编辑表。

### 为什么用 n-gram 而不是整句话哈希？

- n-gram 局部、组合、泛化更好；
- 整句哈希在训练集外几乎必然 miss；
- 2/3-gram 覆盖实体、搭配、公式模板，性价比最高。

### 为什么不用 attention 做记忆？

Attention 是 `O(T^2)` 的动态计算；Engram 是 `O(1)` 的静态查表。对于「the Great 后面接什么」这种局部模式，查表更便宜。

### Engram 会取代 FFN 吗？

不会。论文的 ablation 显示，去掉条件计算（rho→0）性能大幅下降。记忆负责「知道」，FFN/MoE 负责「推理」。

---

## 实现细节

### `d_mem` 和 `head_dim` 到底怎么算？

```
head_dim = n_embed_per_ngram / n_head_per_ngram
d_mem    = (max_ngram_size - 1) * n_embed_per_ngram
```

27B：`n_embed_per_ngram=640`, `n_head=8` → `head_dim=80`, `d_mem=1280`。

`Engram Vocab Size = 2,262,400` 是**每个 head、每个 order** 的 slot 数，不是所有 head 的总数。

### 为什么 hash 要用 int64？

token id × 乘数 可能接近 2^63。int32 会溢出。项目里所有 hash id 都是 `torch.long`。

### 为什么 gate 要用 signed sqrt？

把大 logit 压小，避免 sigmoid 饱和；保号，抑制仍然是抑制。详见 lab3。

### 为什么 conv 要零初始化？

训练第 0 步 `Y = V~`，Engram 是纯查表，不注入噪声。详见 lab4。

### KV cache 下为什么 logits 对不上？

Engram 需要前 `N-1` 个 token。只传新 token 会让 hash 用 pad 补齐。传 `context_ids=out`（完整序列或最后 N-1 个 token）。详见 lab5。

### 为什么 `slots_per_head` 要取素数？

避免 2 的幂模数只取低位；全局不同的素数让各 head 碰撞模式独立。详见 lab1。

---

## 性能

### 训练很慢怎么办？

1. 用 GPU；
2. 减小 `slots_per_head`（测试用 64 就够）；
3. 减小 batch/seq；
4. 确认 hash 没有在 Python 循环里逐 token 做（本项目是向量化的）；
5. `torch.compile(model)`。

### 显存不够怎么办？

- 用 bf16/fp16；
- 减小表；
- 把表 offload 到 CPU（L7 的主题）；
- 用 gradient checkpointing（本项目没实现，作为练习）。

### Triton kernel 为什么没有大幅加速？

因为 gather 是随机访问、延迟受限，不是纯带宽受限。模块级的 `torch.compile` 收益更大（4×）。这个负面结果本身就是 L7 的教学内容。

---

## 论文与延伸

### 论文在哪里？

- 官方仓库：<https://github.com/deepseek-ai/Engram>
- 项目内精读：[`papers/Engram-conditional-memory/`](../papers/Engram-conditional-memory/)

### 还应该读什么？

- CS336: Language Modeling from Scratch（本项目风格的来源）；
- mHC 项目（多分支 backbone 的数学）；
- OverEncoding、SCONE、BLT：n-gram/byte-level 记忆的相关工作；
- DeepSeekMoE：条件计算的对照。

### 这个项目能发论文吗？

它是教学项目，不是研究贡献。但 Capstone 的扩展（真实数据、iso-param 对比、kernel 优化）可以做课程项目或 workshop 复现。

---

## 贡献

见 [`CONTRIBUTING.md`](../CONTRIBUTING.md)。新关卡必须同时提供：

1. 讲义；
2. 骨架（TODO）；
3. 测试；
4. 参考实现；
5. `TINYENGRAM_IMPL=solutions pytest` 全绿。
