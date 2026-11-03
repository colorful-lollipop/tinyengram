# 教学实验

所有脚本默认使用 `solutions/` 的参考实现（可用 `TINYENGRAM_IMPL=src` 切到自己的实现）。
产物写到 `../.runs/`，不纳入版本管理。

| 脚本 | 对应论文 | 内容 | 参考耗时（CPU） |
|---|---|---|---|
| `exp1_allocation.py` | Fig. 3 左 | 扫 `rho`，二次拟合最优分配点 | < 1 s |
| `exp2_train_tiny.py` | Table 1 | 合成语料上 baseline vs Engram | ~1 min |
| `exp3_analysis.py` | Fig. 4/7 | LogitLens、CKA、软对齐、门控 top token | ~1 min |
| `exp4_offload.py` | Table 4 | 100B 表、PCIe 预取、Zipf 缓存 | ~5 s |
| `exp5_scaling.py` | Fig. 3 右 | slots sweep + 幂律拟合 | ~1 min |
| `exp6_speculative.py` | 独立扩展实验 | TinyLM 草拟/验证/双 KV 回滚/接受长度与时延 | ~1 min |

投机解码的接口、状态边界、复现命令与实测见
[最小贪心投机解码实验](../docs/experiment-speculative.md)。它单独运行，不包含在
`make experiments` 中，不增加课程评分关卡。

```bash
TINYENGRAM_IMPL=solutions python experiments/exp2_train_tiny.py
# 或一次跑完
make experiments
```

## 解读原则

- 小模型只复现**机制方向**，不复现 27B 的幅度；
- 报告均值±方差，不要挑 seed；
- 如果 Engram 没赢，先检查 gate 是否学会、embedding 是否有梯度，再怀疑实现；
- 负面结果（例如 Triton gather 只快 3%）同样是课程内容。
