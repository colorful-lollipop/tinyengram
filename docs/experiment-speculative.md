# 最小贪心投机解码实验

本实验用两个 TinyLM 验证 **多 token 草拟 → 并行验证 → 拒绝回滚**。
不加载 Kimi K3、不依赖 DSpark，也不声称验证了 Engram 草稿架构的语言建模收益。
这是独立实验，不增加评分关卡，不补写 `src/` 现有 TODO。

## 运行

在仓库根目录，Python 3.10+、PyTorch 2.2+：

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[dev]'

OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TINYENGRAM_IMPL=solutions \
  python -m pytest -q tests/test_speculative.py

# 两个独立训练的模型：target 2 层/64 维，draft 1 层/32 维
TINYENGRAM_IMPL=solutions python experiments/exp6_speculative.py \
  --pair trained --repeats 5 --warmup 2 --output .runs/speculative-trained.json

# 相同目标模型 + Engram 草稿；Engram 只使用单点卷积
TINYENGRAM_IMPL=solutions python experiments/exp6_speculative.py \
  --pair trained --draft-engram --repeats 5 --warmup 2 \
  --output .runs/speculative-trained-engram.json

# 独立随机模型负对照；同权重双模型最大接受率正确性对照
TINYENGRAM_IMPL=solutions python experiments/exp6_speculative.py \
  --pair random --repeats 5 --warmup 2 --output .runs/speculative-random.json
TINYENGRAM_IMPL=solutions python experiments/exp6_speculative.py \
  --pair identical --output .runs/speculative-identical.json
```

默认固定 seed=42、CPU 单线程、贪心、batch=1，输出 32 token；prompt 长度
4/32/96，每种长度 3 个 prompt，草拟长度 K=1/2/4/8。可用 `--device cuda`，
计时会在 CUDA 前后同步；本次只有 CPU 实测。随机状态固定且启用确定性算子；
不同 PyTorch/硬件版本不保证相同浮点结果，时延也有主机噪声。

训练语料是 1..31 循环的合成序列，目标和草稿分别从头训练 120 步，每步 16×32 token。
两者用相同训练数据流；评估 prompt 也来自这个简单分布，没有独立语义泛化测试。
这能检查草稿对齐后解码机制是否成立，**不能测试长距离记忆或语言能力**。
`--pair identical` 复制目标权重，模型大小也相同，只是接受率上界控制。

JSON 保存模型配置、参数量、训练首末 loss、seed、环境、源码 SHA256、每个 prompt
的实际 token/每轮接受记录、调用次数，以及每次时延样本。基线和投机执行顺序交替，
每个 prompt 均检查普通 KV 解码、naive 全量重算、投机解码的 token 完全相等。
不要并发运行基准与训练/测试。已记录的实测见 [CPU 实验记录](experiment-speculative-results.md)。

## 现有接口与状态不变量

算法在 [experiments/speculative.py](../experiments/speculative.py)，入口为
`greedy_speculative_generate(target, draft, ids, max_new_tokens, draft_length=4)`。
返回 `SpeculativeResult`，包括输出、最终两份 KV Cache、每轮统计和调用次数。
`on_commit(ids, target_cache, draft_cache, stats)` 可只读检查每轮状态，基准不调用它。

对已经提交的序列 C，令长度为 L。每轮开始和结束时保持：

```text
target_cache = target 对 C 计算出的每层 K/V，时间维长度 L
draft_cache  = draft  对 C 计算出的每层 K/V，时间维长度 L
context_ids  = C（只含已提交 token）
position_offset = 当前 cache 的时间维长度
```

两个 cache 的张量值不必相同；它们来自各自不同的模型。
TinyLM 的 KV 是每层 `(K,V)` 列表，形状 `[B*hc_mult, heads, time, head_dim]`。
回滚沿 `time`（`-2`）截断，且不原地修改 cache 视图。

1. 预填充 C，保存目标/草稿最后位置的 next-token logits。
2. 草稿逐 token 产生 d1..dK，同时增长临时 `proposal_context=C+d1..dK`。
3. 目标模型把 **整个 `[d1..dK]` 一次传入 forward**，带原始 C 的 cache、
   `position_offset=L`、完整临时 context。TinyLM 现有 chunk causal mask 防止未来泄漏。
4. 第一个草稿由预填充 logits 验证；目标 chunk 第 j 个输出验证 d(j+1)，
   最后位置输出还提供全接受时的 bonus token。不能直接拿同位置 logits 验证 d(j)。
5. 只接受连续匹配的最长前缀 r。在第一个拒绝处，取目标 argmax 纠正；
   全接受时取目标 bonus。其后的目标预测条件包含错误草稿，必须丢弃。
6. 两份 cache 都裁到 `L+r`，context 重建为 `C+accepted`，然后把纠正/bonus
   token 分别送入两模型一次，恢复完整提交状态。**修正后的 context 里没有任何被拒绝 ID**。

EOS 只在已接受草稿或目标纠正/bonus 被提交时生效；错误草稿中的 EOS 不停止生成。
prompt 本身带 EOS 不停止生成，与单请求 `generate` 约定一致。达到输出预算时不加 bonus；
最终短块也计入统计。空 prompt、batch>1、越界 token、不同词表大小、跨设备、
超出任意模型 max_seq 都会报错。词表语义必须由调用者保证相同，大小相同不足以证明语义相同。

## Engram 的状态支持

当前实现同时缓存注意力 KV 和 ShortConv 的门控激活历史，支持非零的历史卷积 tap。
`context_ids` 只负责 n-gram token 历史，不能替代激活状态。默认 `--conv-kernel 1`
便于对照初版；加 `--conv-kernel 4 --draft-engram` 即可运行时间卷积版本。

每层保存最近 H=(kernel_size-1)×dilation 个 gated 激活。目标一次验证 chunk 的中间
回滚使用“旧 H 个＋本轮新激活”，草稿逐 token 的回滚使用每一步的小状态快照。
缓存是请求自己的对象，仍可用 `cache[i]` 读取 `(K,V)`，但不能转换为普通 list。
原理、内存边界和失败提示见 [解码状态图解](decode-state.md)。

初版只支持单点或零历史 tap 的限制已解除；[初版实测](experiment-speculative-results.md)
仍保留当时的设置和源码指纹，不应把它当成新版的时延或时间卷积结论。

## 需要历史的 recall 任务

`--task recall` 使用无需外部数据的单绑定拷贝任务：

```text
prompt:   [key, v1,v2,v3,v4, noise..., QUERY, key]
response: [v1,v2,v3,v4, EOS]
```

payload、key、noise 都随机生成。两个 prompt 可以最近后缀相同，但早期 payload 不同，
因此不能仅靠最近一个 token 答对。这仍是合成拷贝任务，不是语言能力或多实体检索基准。
训练只对四个值及 EOS 计算 loss；prompt 长度在 7..32 间变化。
评估使用独立随机流。长度 96 超出训练长度，报告时应单独标明位置/距离外推。

```bash
TINYENGRAM_IMPL=solutions python experiments/exp6_speculative.py \
  --task recall --pair trained --steps 400 --prompts 8 \
  --prompt-lengths 8 32 96 --draft-lengths 1 2 4 \
  --repeats 3 --warmup 1 --output .runs/recall-plain.json

TINYENGRAM_IMPL=solutions python experiments/exp6_speculative.py \
  --task recall --pair trained --steps 400 --prompts 8 \
  --draft-engram --conv-kernel 4 --prompt-lengths 8 32 96 \
  --draft-lengths 1 2 4 --repeats 3 --warmup 1 \
  --output .runs/recall-temporal-engram.json
```

recall 默认只输出 5 token，cycle 默认输出 32；也可以显式指定 `--new-tokens`。
JSON 额外保存预期回复及目标/草稿**独立解码**的回复 token 准确率（包括 EOS，
若只输出不足 5 token 则只统计已生成部分）。投机输出与目标一致，
不等于目标把任务答对：两者要分别报告。小模型在距离外推时失败，也是有效结果。
当前没有蒸馏，也没有对齐两草稿的参数量，不把该对照解释为架构的最终优劣。

## 指标和代价

| 指标 | 定义 |
|---|---|
| `mean_accepted_draft_tokens` | 已提交草稿 token 总数 / 轮数，范围 0..K |
| `mean_emitted_tokens` | 新输出 token 总数 / 轮数，通常 r+1，EOS/预算处可能更少 |
| `acceptance_rate` | 已提交草稿 / 全部草拟 token；包含拒绝后被浪费的后缀 |
| `rejection_rounds` | 遇到第一个拒绝的轮数；若更早遇到已接受 EOS，则不记为拒绝 |
| `discarded_draft_tokens` | 草拟数减去实际提交草稿数，也包括 EOS 后的无用草稿 |
| `speedup` | 普通 KV 解码中位耗时 / 投机解码中位耗时；小于 1 表示更慢 |

接受率分母不是“实际走到该位置的条件试验次数”，不要把它当成每位置条件接受概率。
这里的本轮输出数也不是“每次目标调用的输出数”：本实现还有一次纠正/bonus flush。

时延包括每次请求的双模型预填充、草拟、验证、回滚、纠正/bonus、Python 控制和统计，
不包括模型构建、训练、naive 检查和审计 hook。模型权重已驻留内存；没有磁盘加载或网络。
普通基线是已有 `generate` 的 KV 解码，**不是较慢的 naive 解码**。

这版每轮额外 flush 一次纠正/bonus，方便完整状态审计；因此没有实现生产解码器常见的
pending-token 优化。大致代价（忽略 EOS/尾块）为：

```text
T_round ≈ K * T_draft(1) + T_target(K)
          + T_target(1) + T_draft(1) + T_bookkeeping
outputs_per_round = r + 1
```

生产实现可以把未处理的最后 token 合并进下一次验证，此时必须重写缓存不变量。
本实验不使用理论接受率换算成实测加速。

## 测试检查什么

[test_speculative.py](../tests/test_speculative.py) 包含真实独立 TinyLM 与同权重控制，
草拟长度 1/2/4/8、prompt 长度 1/7/37，普通/Engram、多分支模型；并覆盖零输出预算、
最后短块、EOS、非法输入和非零时间卷积状态。位置可控的 TinyLM 测试强制首位、中途、
末位拒绝和全接受，但仍实际计算注意力与 Engram 的 K/V。

审计 hook 在**每轮**将目标和草稿分别对已提交序列完整重算，并逐层检查实际 K/V
数值与长度。这比只检查输出更强：即使测试 logits 可控，也能发现错误草稿污染缓存。
真实模型输出同时对照 `generate` 和 `naive_generate`；hook 还检查验证确实是多 token
一次 forward。浮点运算块形状变化可能改变近乎平票时的 argmax，因此这里的相等是已测
模型/环境下的结果，不是所有精度与硬件上的逐 bit 保证。

## 无损随机采样还缺什么

当前 API **只实现确定性贪心**。要保持目标采样分布，需要按
[Leviathan et al., §2.2–2.3 / Algorithm 1](https://proceedings.mlr.press/v202/leviathan23a.html)
实现以下逻辑，而不能仅比较两个模型随机抽到的 token：

```text
q_i = 草稿在已接受前缀下实际使用的归一化分布
p_i = 目标在同一前缀下的归一化分布
x_i ~ q_i
accept x_i with probability min(1, p_i(x_i) / q_i(x_i))

first rejection at i:
    replacement ~ normalize(max(p_i - q_i, 0))
    discard the remainder; crop both caches and context; commit replacement
all accepted:
    bonus ~ p_(K+1)
```

要保存每个草拟位置的完整 q（纠正分布需要所有词表项），并取得相应 p；temperature、
top-k/top-p 等处理必须计入**实际抽样分布**并正确归一化。还需要数值稳定的概率比、
零概率/退化残差处理、可复现且独立的草拟/接受/纠正 RNG，以及 EOS/预算规则。
若 q(x)=0，该 x 不应由 q 抽出；p=q 时拒绝事件在精确数学下不会发生，不能强行对零残差归一化。
将来应先对小离散分布解析验证接受+纠正质量守恒，再做频率检验及缓存回滚测试。
保持**分布**不等于“固定同一 seed 就逐 token 等于普通采样”，两算法消耗 RNG 的方式不同。

## 已验证、假设、尚未验证

| 层级 | 本次能说什么 |
|---|---|
| 已验证 | TinyLM 上的 chunk 验证、双 KV/context 回滚、贪心输出对齐及本机 CPU 耗时 |
| 可研究假设 | Engram 若能以足够低的成本提高目标分布拟合度，可能改善投机性价比 |
| 尚未验证 | DSpark 特征输入、Kimi K3 权重/激活提取、平均接受 4–5 token、长上下文接受率、Ascend/GPU 加速 |

cycle 的 4..96 长度扫描不需要远距离历史。recall 增加了对历史 payload 的依赖，
但仍只是小模型合成拷贝实验。要讨论 K3 或 DSpark，需要再区分草稿能力、位置外推、
上下文截断与状态错误，不能将这里的接受长度直接迁移到真实大模型。

新版的全部配置结果及局部查表对照见 [优化后实测](decode-state-results.md)。
首次运行与 Windows 命令见 [新手入口](start-here.md)。
