# Lab 0：Tokenizer 压缩 —— 让「同一个词」真的共享一个 id

> 对应代码：`src/tinyengram/tokenizer.py`  
> 对应测试：`tests/test_l0_tokenizer.py`  
> 论文：§2.2 "Tokenizer Compression"、Appendix C

> **前置**：无（只要 PyTorch 基础） ｜ **预计**：1 h ｜ **验收**：`pytest tests/test_l0_*.py`  
> **必做**：实现 `normalize_token_text` / `CompressedTokenizer`，通过 6 个测试。  
> **进阶**：用真实 GPT-2 tokenizer 测压缩率；比较 NFC 与 NFKC 的差异。  
> **挑战**：构造多语言 + 代码词表，量化压缩率、碰撞率与下游 loss 的关系。

---

## 0.1 为什么先讲 tokenizer

Engram 的键是 **token id 序列**，不是文本。

```
"Apple"  -> [3121]
" apple" -> [742]
"APPLE"  -> [91823]
```

在 BPE 词表里这三个 token 的 id 完全不同。如果直接拿 id 做 3-gram 哈希，模型会学到三套互不相干的记忆槽位，而它们在语义上是同一个东西。论文的解决办法非常朴素：

> 先学一个**满射**（surjective）映射 `P: V -> V'`，把文本规范化后相同的 token 折叠成一个规范 id，再在 `V'` 上做 n-gram。

对 128k 词表，这一步把有效词表缩小 **23.43%**（Appendix C）。这不是小优化：n-gram 的组合空间随词表大小呈指数增长，缩小 23% 相当于把 3-gram 空间缩小到 `0.766^3 ≈ 45%`。

---

## 0.2 先把 n-gram 写清楚

给定压缩后的 token 序列 `x'_1 ... x'_T`，位置 `t` 的 **后缀 n-gram** 是

```
g_{t,n} = (x'_{t-n+1}, ..., x'_{t-1}, x'_t),   2 <= n <= N
```

注意是**后缀**：位置 `t` 只能看见自己和左边。这一点在实现里对应「左填充 + 取前 T 个位置」，也是 Lab 1 因果性测试的核心。

Engram 默认 `N = 3`，即同时使用 2-gram 和 3-gram。为什么不是更大？

- 2-gram 捕捉固定搭配（"the Great"）；
- 3-gram 捕捉完整实体（"Alexander the Great"）；
- 4-gram 在固定 1.6B 预算下反而略差（论文 §6.2）：容量被更稀疏的模式稀释了。

论文没有排除「更大记忆预算下 4-gram 会更好」，这是留给读者的开放问题。

---

## 0.3 规范化流水线

`normalize_token_text` 的顺序是有讲究的：

1. **NFKC**：全角转半角、连字展开、兼容字符归一。`Ａpple -> Apple`。
2. **NFD + 去掉 combining marks**：`café -> cafe`。先 NFD 再删组合符，比手写映射表可靠。
3. **lowercase**：`APPLE -> apple`。
4. **空白折叠**：`"hello\t\tworld" -> "hello world"`。
5. **strip**：去掉首尾空白。但**纯空白 token 要保留成 `" "`**。

第 5 条容易写错。BPE 词表里有大量 `"\n"`、`"  "`、`"\t"`、`" \n"`，它们都是高频 token。如果 `strip()` 后统一变成空串，你就会把「换行」和「空格」混为一谈；如果保留原样，又会分裂成几十个槽位。正确做法是：**纯空白折叠成一个规范空格 token**。

```python
was_space = text.strip() == ""
...
if was_space:
    return " "
```

---

## 0.4 构建投影表

`CompressedTokenizer._build_lookup_table` 做一次 O(V) 的预处理：

```python
for tid in range(vocab_size):
    text = decode([tid])
    key = normalize(text)          # 或 byte-level fallback
    if key not in key2new:
        key2new[key] = len(new_tokens)
        new_tokens.append(key)
    old2new[tid] = key2new[key]
```

性质：

- **满射**：每个新 id 至少被一个旧 id 命中（测试会检查 `set(lookup) == range(num_new)`）。
- **非单射**：多个旧 id 可以映射到同一个新 id，这正是压缩的来源。
- 复杂度 O(V)，内存 O(V)，只在初始化时做一次。
- 结果是一张 `LongTensor[V]`，推理时一次 index 就完成压缩。

### byte-level BPE 的坑

GPT-2 / DeepSeek 的 tokenizer 是 byte-level BPE。某些 token 单独 `decode` 会得到替换字符 `�`。这时不能规范化文本，而应回退到 `convert_ids_to_tokens(tid)`：

```python
if "\ufffd" in text:
    key = str(tokenizer.convert_ids_to_tokens(tid))
```

否则所有不完整字节 token 会被折叠成同一个 key，模型会丢失字节级信息。

### 特殊 token 的坑

`<pad>`、`<bos>`、`<eos>` 解码后可能都是空串或相同字符串。如果把它们折叠成一个 id，padding 和句子边界就混了。本项目的处理是给空串一个唯一 key：

```python
key = f"\x00special:{tid}"
```

这不是论文的要求，而是教学实现的防御性选择；真实系统通常显式配置 special token 不参与压缩。

---

## 0.5 接口约定

```python
normalize_token_text(text: str) -> str
CompressedTokenizer(tokenizer)
    .lookup_table : LongTensor[V]
    .num_new_token: int
    .compress(ids) -> LongTensor  # 保持形状，负数 padding 原样保留
    .compression_ratio -> float   # 1 - |V'|/|V|
    .top_merges(k) -> [(new_id, canonical, group_size)]
```

`compress` 必须：

- 保持输入形状（支持 `[B, T]`、任意维度）；
- 对 `-1` 这类负数 padding **不做映射**（否则 `-1` 会索引到词表最后一行）；
- 对越界 id 报错，而不是静默 wrap。

---

## 0.6 自测

```bash
pytest tests/test_l0_tokenizer.py -x -q
```

测试检查：

- NFKC / 大小写 / 去重音 / 空白；
- 纯空白 token 折叠为 `" "`；
- 投影满射，压缩率符合构造；
- 等价 token 共享 id；
- 形状与负 padding 保持；
- `top_merges` 排序正确。

---

## 0.7 思考题

难度梯度：**[理解]** 能用自己的话解释；**[推导]** 需要动笔算；**[设计]** 需要改代码/做实验；**[开放]** 没有标准答案。

1. **[理解]** 为什么用 **NFKC** 而不是 NFC？举一个两者结果不同的例子。
2. **[理解]** 压缩率越高越好吗？如果两个语义不同的词被错误合并，会发生什么？
3. **[设计]** 对中文 tokenizer，去重音和 lowercase 还有意义吗？应该增加哪些规范化步骤？
4. **[推导]** 假设词表从 128k 压到 98k，3-gram 的**可能组合数**缩小了多少倍？这和哈希表容量选择有什么关系？
5. **[开放]** 如果推理时 tokenizer 版本升级，`lookup_table` 不一致会怎样？工程上如何做版本校验？

---

## 0.8 提示

- `unicodedata.normalize("NFKC", s)` 和 `"NFD"` 是标准库，不需要额外依赖。
- 组合字符判断：`unicodedata.combining(ch) != 0`。
- 折叠空白最简单：`" ".join(s.split())`，但它会把纯空白变成空串，记得单独处理。
- 不要在 `compress` 里用 `lookup[ids]` 直接索引负 padding；先 mask。

---

下一关：[Lab 1：多头部哈希](lab1-hashing.md)
