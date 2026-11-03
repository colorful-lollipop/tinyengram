# 第一次运行 TinyEngram

先得到一个可解释的成功结果，再开始补代码。全程可以用 CPU，不需要下载模型权重，
不需要懂 CUDA，也不需要先读完论文。

## 第一步：在仓库根目录安装

建议 Python 3.10–3.12；本项目只需要 PyTorch，其他依赖随 `dev` 安装。
先用 `git clone https://github.com/colorful-lollipop/tinyengram.git` 下载，再 `cd tinyengram`。

Linux / macOS：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch
python -m pip install -e '.[dev]'
python scripts/hello_engram.py
```

Windows PowerShell（直接调用虚拟环境解释器，无需修改脚本执行策略）：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe scripts/hello_engram.py
```

后面的 `python` 在 Windows 上可替换为 `.\.venv\Scripts\python.exe`。
只需要 CPU 的 Linux / Windows 也可用
`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`。

## 第二步：看懂这个成功结果

演示使用 `solutions/` 完整参考实现，运行两个很小、未训练的模型。你会看到类似：

```text
草拟 4，接受 1，输出 2；缓存长度 5，KV/卷积历史审计 PASS
...
PASS：普通解码 = 完整重算 = 投机解码
```

接受数可能随 PyTorch 版本或浮点结果变化；最后的正确性检查应该通过。

| 输出 | 如何理解 |
|---|---|
| 草拟 4 | 小模型先猜四个 token |
| 接受 1 | 目标模型只认可连续前缀中的第一个猜测 |
| 输出 2 | 提交一个草稿，再提交目标的纠正 token |
| KV/卷积历史审计 PASS | 拒绝后保存的状态与从头计算已提交序列一致 |
| 三种解码相等 | 加速算法没有改变本例的目标贪心输出 |

这里的 token ID 是整数，不是自然语言；随机模型不用于衡量质量或性能。
卷积权重刻意非零，避免“历史状态丢了，但零初始化让错误暂时看不出来”。

## 第三步：选择你的路径

| 目标 | 做什么 | 验收标准 |
|---|---|---|
| 先理解机制 | 读 [解码状态图解](decode-state.md)，改演示的 `draft_length` | 能解释接受数与输出数为何不同 |
| 从零做题 | `python scripts/grade.py --stage 0`，读 [L0](lab0-tokenizer.md)，改 `src/` | 本关测试从红变绿 |
| 验证安装 | `python scripts/grade.py --solution --stage 0` | 参考实现通过，而不是要求骨架通过 |
| 开始实验 | 按 [实验说明](experiment-speculative.md) 运行小规模基准 | 输出一致，并保存接受数、时延和设置 |

**做题前先确认这两个目录：**

- `src/tinyengram/`：给你补的骨架，`NotImplementedError` 是预期的。
- `solutions/tinyengram/`：完整参考实现；演示和实验默认用它，不会替你填写骨架。

评分脚本遇到未通过的关卡会返回退出码 1；这表示练习未完成，不一定是安装错误。
`--stage 9` 会运行该关慢测试；`--all` 才会在全关评分中包含它。

直接用 pytest 验证参考实现，Linux/macOS：

```bash
TINYENGRAM_IMPL=solutions python -m pytest -q tests/test_decode_state.py
```

PowerShell：

```powershell
$env:TINYENGRAM_IMPL = 'solutions'
.\.venv\Scripts\python.exe -m pytest -q tests/test_decode_state.py
Remove-Item Env:TINYENGRAM_IMPL
```

## 第四步：做四个小练习

这些是扩展练习，不另加课程评分关卡。每次只改变一件事。

1. **先手算**：草稿为 `[A,B,C,D]`，目标在这些前缀下的预测为 `[A,B,X,Y]`。
   应接受几个？应提交什么？为什么不能直接提交 Y？
2. **再看状态**：在演示中打印 `cache[i][0].shape` 和 `cache.conv_states[i].shape`，
   找出 KV 的时间维与卷积历史的时间维。
3. **比较成本**：运行 `python experiments/exp7_local_lookup.py`，解释为什么新输入
   只有一个 token 时，不需要把整个历史的 embedding 再取一次。
4. **设计反例**：读 [recall 任务](experiment-speculative.md#需要历史的-recall-任务)，
   找两个最近 token 一样、但正确答案不同的 prompt。只看最近上下文的草稿能都答对吗？

第一题提示：接受两个；提交 `[A,B,X]`。Y 是在带错误 C 的条件下得到的预测，必须丢弃。

## 遇到报错时

| 现象 | 先检查什么 |
|---|---|
| `No module named torch/pytest` | 安装与运行是否用了同一个 Python？试 `python -m pip --version` |
| `No module named tinyengram` | 是否在仓库根目录？是否执行过 `python -m pip install -e '.[dev]'`？ |
| `TODO(Lx)` / `NotImplementedError` | 若在做题，这是预期；若想先演示，用 `hello_engram.py` 或 `--solution` |
| `position_offset must equal cache length` | 裁剪缓存后，位置应取裁剪后的长度，不能继续沿用旧长度 |
| `context_ids` 不足 | 传入完整提交序列，或“本轮输入 + 前 N−1 个 token” |
| `plain list loses Engram conv history` | 保留模型返回的整个缓存对象，不要写 `list(cache)` |
| Triton/CUDA 测试被 skip | CPU 学习不需要这些依赖，可先完成其余测试 |

然后回到 [讲义总览](README.md)，按你的目标选择 L0–L9。
