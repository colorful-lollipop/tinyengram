# TinyEngram

**从 n-gram 查表出发，一步步实现 DeepSeek Engram 的条件记忆、训练、推理与系统优化。**

[English](README.en.md) | 简体中文

[![CI](https://github.com/colorful-lollipop/tinyengram/actions/workflows/ci.yml/badge.svg)](https://github.com/colorful-lollipop/tinyengram/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-91%20core-brightgreen.svg)](#测试与评分)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

这是一个 **测试驱动（TDD）** 的渐进式课程项目，风格参考 Stanford CS336 与同仓库的姐妹项目 **TinyMHC**：
每一关给你可运行的骨架、一套会失败的测试和一份讲义；你补全核心函数，测试变绿，再进入下一关。

> 目标读者：会写 `nn.Module`、知道 `loss.backward()`、能看懂张量广播，但没有系统做过
> 「读论文 → 复现模块 → 验证性质 → 剖析系统」这条链路的人。

> **项目性质**：独立教学项目，不是 DeepSeek-AI 官方实现，也不代表论文作者观点。
> 论文与代码的版权归原作者；本项目只做教学复现、注释与评论。

---

## 30 秒看懂 Engram

Transformer 没有原生的「知识查找」原语。要识别 "Alexander the Great"，它得用好几层
attention/FFN 把三个 token 组合起来——相当于**用计算模拟查表**。

Engram 把这件事件变成真正的查表：

```text
token ids ──► 压缩词表 ──► 3-gram 哈希 ──► 16 张素数表 ──► 一次 gather
                                                              │
hidden h_t ─────────────► 上下文门控 α_t ──► u_t = α_t · W_V e_t
                                                              │
                                       Y = u_t + ShortConv(u_t)
                                       H ← H + Y
```

- **静态**：key 只依赖 token 序列，不依赖 hidden state；
- **O(1)**：每个 token 固定读 16 行，不随表大小增长；
- **可预取**：索引在 forward 之前就能算出来，能藏住 PCIe 延迟；
- **可扩展**：27B 模型里 5.7B 参数是记忆，激活参数不变。

论文在严格 iso-param / iso-FLOPs 下，让 27B Engram 全面超过同规模 MoE：

| 任务 | MoE-27B | Engram-27B | Δ |
|---|---|---|---|
| MMLU | 57.4 | 60.4 | +3.0 |
| CMMLU | 57.9 | 61.9 | +4.0 |
| BBH | 50.9 | 55.9 | +5.0 |
| ARC-Challenge | 70.1 | 73.8 | +3.7 |
| HumanEval | 37.8 | 40.8 | +3.0 |
| MATH | 28.3 | 30.7 | +2.4 |

更反直觉的是：收益最大的不是知识题，而是**推理、代码、数学**。机制分析的解释是——
Engram 把早期层的静态重建卸载掉，等效于给模型增加了深度。

---

## 学习路线

| 关卡 | 主题 | 你要实现什么 | 验收 | 难度 |
|---|---|---|---|---|
| **L0** | [Tokenizer 压缩](docs/lab0-tokenizer.md) | 规范化、满射投影、23% 词表压缩 | `test_l0_tokenizer.py` | ⭐ |
| **L1** | [多头部哈希](docs/lab1-hashing.md) | 素数表、奇数乘数、乘法-XOR、int64 边界 | `test_l1_hashing.py` | ⭐⭐ |
| **L2** | [检索与内存布局](docs/lab2-retrieval.md) | 扁平表 + offsets、一次 gather 取 16 个 head | `test_l2_embedding.py` | ⭐ |
| **L3** | [上下文门控](docs/lab3-gating.md) | RMSNorm、余弦 logit、signed sqrt、分支门控 | `test_l3_gating.py` | ⭐⭐ |
| **L4** | [ShortConv 与融合](docs/lab4-fusion.md) | 因果 depthwise conv、零初始化恒等、残差 | `test_l4_engram.py` | ⭐⭐⭐ |
| **L5** | [Transformer 与训练](docs/lab5-transformer.md) | 多分支 TinyLM、RoPE、KV cache、参数组 | `test_l5_transformer.py` | ⭐⭐⭐ |
| **L6** | [稀疏分配与缩放律](docs/lab6-allocation.md) | iso-param/iso-FLOPs、U 形、幂律拟合 | `test_l6_allocation.py` | ⭐⭐ |
| **L7** | [推理系统与 offload](docs/lab7-systems.md) | 确定性预取、Zipf 缓存、分片、Triton | `test_l7_systems.py` | ⭐⭐⭐⭐ |
| **L8** | [机制分析](docs/lab8-analysis.md) | LogitLens、CKA、软对齐、门控可视化 | `test_l8_analysis.py` | ⭐⭐⭐ |
| **L9** | [端到端复现](docs/lab9-capstone.md) | 训练 baseline vs Engram、完整分析链路 | `test_l9_reproduction.py` | ⭐⭐⭐⭐ |

讲义在 [`docs/`](docs/README.md)：每关一篇，包含公式、接口约定、常见坑、思考题。
论文精读与关键段落摘录在 [`papers/`](papers/)。

---

## 安装

**环境要求**：Python 3.10+、PyTorch 2.2+。CPU 可完成全部核心关卡；GPU 只用于 L7 的 Triton 基准。

```bash
git clone https://github.com/colorful-lollipop/tinyengram.git
cd tinyengram

# 可编辑安装（推荐）
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 或使用 uv
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

如果只想先跑起来、不想安装包，也可以直接用 `PYTHONPATH`：

```bash
PYTHONPATH=src pytest -q -m "not slow and not gpu"
```

## 快速开始

```bash
# 看看当前进度（一开始 L0-L9 全红，这是正常的）
python scripts/grade.py

# 只跑一关，边写边测
pytest tests/test_l0_tokenizer.py -x -q

# 读讲义
less docs/lab0-tokenizer.md
```

跑通参考实现（用来确认测试本身没问题）：

```bash
TINYENGRAM_IMPL=solutions pytest -q -m "not slow"
python scripts/grade.py --solution
python scripts/check_docs.py
```

---

## 推荐工作流

1. **先读讲义**：`docs/lab1-hashing.md`，搞清后缀 n-gram 和溢出边界，不要直接看代码。
2. **补全 TODO**：只改 `src/tinyengram/` 下标了 `TODO(Lx)` 的地方。
3. **跑单关测试**：`pytest tests/test_l1_hashing.py -x -q`，失败信息里写了期望的数学性质。
4. **卡住看提示**：讲义末尾有 hints；再不行看 `solutions/`（建议先痛苦 20 分钟）。
5. **跑实验**：`TINYENGRAM_IMPL=solutions python experiments/exp1_allocation.py`，把数字和论文对上。
6. **进系统关**：L7 不再加新数学，而是回答「为什么快、快在哪、什么时候融合反而更慢」。

---

## 项目结构

```text
Engram/
├── src/tinyengram/            # ← 你实现的地方（骨架 + TODO）
│   ├── tokenizer.py           # L0 规范化 / 压缩投影
│   ├── hashing.py             # L1 素数表 / 乘法-XOR
│   ├── embedding.py           # L2 扁平表 / offsets / gather
│   ├── gating.py              # L3 RMSNorm / signed sqrt / gate
│   ├── conv.py                # L4 因果 depthwise conv
│   ├── engram.py              # L4 检索 + 门控 + 融合
│   ├── transformer.py         # L5 TinyLM / RoPE / KV cache
│   ├── train.py               # L5 参数组 / Trainer
│   ├── allocation.py          # L6 稀疏分配 / 缩放律
│   ├── systems.py             # L7 预取 / 缓存 / 分片 / offload
│   ├── inference.py           # L7 KV cache 生成
│   └── analysis.py            # L8 LogitLens / CKA / gate stats
├── solutions/tinyengram/      # 参考实现（同接口，可整体替换）
├── tests/                     # 每关一个文件，性质驱动
├── docs/                      # 讲义（lab0 ~ lab9 + FAQ）
├── papers/                    # 论文元数据、精读笔记、关键段落
├── perf/                      # Triton gather + benchmark
├── experiments/               # 五个教学实验
├── scripts/grade.py           # 分阶段评分
├── scripts/paper_math.py      # 复算论文数字
├── scripts/check_docs.py      # 讲义一致性检查
├── Makefile
└── pyproject.toml
```

---

## 测试与评分

测试不是「对答案」，而是**直接检查数学性质**：

- 哈希必须因果、确定性、无 int64 溢出；
- 每个 head 的表大小是素数且全局唯一；
- `CompressedTokenizer` 的投影是满射，负 padding 不被映射；
- gate 对相似度单调、对缩放不变、梯度能回到 query/key；
- 卷积因果、零初始化恒等、dilation 感受野正确；
- KV cache 与 full forward 逐位一致（包括 Engram 的 `context_ids`）；
- 预算守恒：`moe + engram = sparse`；
- 分片 gather 与整表 gather 完全一致；
- CKA 对 identity/缩放/旋转不变；
- offload 流量只随 unique slot 数增长。

```bash
python scripts/grade.py
# L0   Tokenizer 压缩        PASS     6 / 0 / 0
# L1   多头部哈希            PASS    12 / 0 / 0
# ...
# 合计: 91 passed, 0 failed, 0 skipped  (100%)
```

`--solution` 用参考实现评分；`--all` 包含 L9 的慢测试；`--stage 3` 只跑一关。

---

## 你会看到的数字

**论文复算**（`python scripts/paper_math.py`）

| 量 | 本项目 | 论文 |
|---|---|---|
| Engram-27B 记忆参数 | 5.79B | 5.7B |
| Engram-40B 记忆参数 | 18.53B | 18.5B |
| routed experts | 55 | 55 |
| 训练 FLOPs（6PD） | 5.97e21 | ~6e21 |
| 每 token 读取 | 2.5 KiB | O(1)，与表大小无关 |

**性能**（`python perf/bench.py --breakdown`，RTX 4070 Ti Super）

| 实现 | 延迟 | 相对 |
|---|---|---|
| Engram module eager | 1.87 ms | 1.0× |
| **Engram module + torch.compile** | **0.55 ms** | **3.4×** |
| gather eager | 0.180 ms | 1.0× |
| gather + Triton 融合 | 0.174 ms | 1.03× |

两个结论：

1. 模块级融合（norm/gate/conv）收益最大；
2. gather 融合只快 3%——随机访存是延迟受限的，`index_select` 已经很好。**负面结果也是结果。**

**小规模训练**（`experiments/exp2_train_tiny.py`）

合成任务：多 token 实体名 → 固定属性 token。脚本如实报告 baseline 与 Engram 的
train/heldout loss；规模太小，不保证 Engram 一定赢。这一关训练的是**实验诚实性**。

---

## 参考实现

`solutions/tinyengram/` 是一份完整、可运行的参考实现，接口与 `src/` 完全一致：

- 验证测试本身正确（CI 会跑 `TINYENGRAM_IMPL=solutions`）；
- 你卡住时对照思路；
- 做实验时快速得到 baseline。

切换方式：

```bash
TINYENGRAM_IMPL=solutions pytest
# 或直接把 solutions 加到 PYTHONPATH
```

不建议直接 `cp solutions/tinyengram/*.py src/tinyengram/`——会失去做题的意义。

---

## 论文与延伸阅读

- **Engram**：Cheng et al., *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*, 2026.
  - 官方代码：<https://github.com/deepseek-ai/Engram>
  - 项目内精读：[`papers/Engram-conditional-memory/notes.md`](papers/Engram-conditional-memory/notes.md)
  - 关键段落摘录 + 中文翻译：[`papers/Engram-conditional-memory/README.md`](papers/Engram-conditional-memory/README.md)
- **MoE**：Shazeer et al., *Outrageously Large Neural Networks* (2017)；DeepSeekMoE (2024)
- **n-gram / hash embedding**：Bojanowski et al., *Enriching Word Vectors with Subword Information* (2017)；Tito Svenstrup et al., *Hash Embeddings* (2017)
- **相关记忆工作**：OverEncoding、SCONE、Byte Latent Transformer、PEER/PKM、RETRO
- **系统**：Stanford CS336 Assignment 2（GPU/分布式/通信重叠）、nano-vLLM

---

## 与 CS336 / TinyMHC 的关系

| 项目 | 主题 | 互补点 |
|---|---|---|
| [Stanford CS336](https://stanford-cs336.github.io/) | 从零训练语言模型 | tokenizer、Transformer、GPU、scaling、数据、对齐 |
| **TinyMHC**（姐妹项目） | 多分支残差与流形约束 | Engram 的 backbone 用 mHC |
| **TinyEngram** | 条件记忆 | 稀疏容量、查表、offload、机制分析 |

三者的接口风格一致：骨架 + 性质测试 + 参考实现 + 讲义 + 评分脚本。

---

## 贡献

欢迎补新关卡、修 bug、改讲义、加真实数据实验。适合上手的方向：

- 4-gram 容量与哈希碰撞实验；
- 真实 tokenizer（GPT-2 / DeepSeek）压缩率对比；
- pinned-memory CPU offload 与 CUDA stream 重叠；
- fused gate + value + conv 的 Triton kernel；
- Engram + MoE 的 iso-param/iso-FLOPs 小规模对比。

请先读 [`CONTRIBUTING.md`](CONTRIBUTING.md)。新关卡必须同时提供讲义、骨架、测试和参考实现，
并且满足：

```bash
TINYENGRAM_IMPL=solutions pytest -q
python scripts/check_docs.py
```

参与本项目即表示同意遵守 [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)。
安全漏洞请按 [`SECURITY.md`](SECURITY.md) 私下报告，不要开公开 issue。

## 引用

如果本课程对你的学习、教学或研究有帮助，可以引用：

```bibtex
@misc{tinyengram2026,
  title        = {TinyEngram: A Test-Driven Course on Conditional Memory},
  author       = {TinyEngram contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/colorful-lollipop/tinyengram}},
  note         = {Educational implementation of DeepSeek Engram}
}
```

Engram 原论文的引用格式见
[`papers/Engram-conditional-memory/README.md`](papers/Engram-conditional-memory/README.md)。
仓库根目录也提供了 [`CITATION.cff`](CITATION.cff)，GitHub 会自动显示 “Cite this repository”。

## 致谢

- DeepSeek-AI 与北京大学的研究者公开了 Engram 论文、官方 demo 与设计细节；
- Stanford CS336 提供了「from scratch」课程的组织方式；
- TinyMHC 提供了测试驱动 + 讲义 + 参考实现的课程结构；
- PyTorch、Triton、nano-vLLM 等开源基础设施。

本项目是独立教学项目，与上述机构没有隶属关系。

## License

代码以 [MIT License](LICENSE) 发布。`papers/` 中的论文摘录、图表与引用版权归原作者，
仅用于教学、评论与研究，详见 [`papers/README.md`](papers/README.md)。
