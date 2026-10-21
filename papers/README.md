# 论文精读

本目录保存与 Engram 直接相关的论文元数据、关键段落摘录和中文精读笔记。
论文版权归原作者所有，这里只做教学引用与评论。

| 目录 | 论文 | 主题 |
|---|---|---|
| [`Engram-conditional-memory/`](Engram-conditional-memory/) | *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models* (2026) | Engram 模块、稀疏分配、系统效率 |

## 版权与引用

- 本目录**不重新分发论文 PDF**，只保存链接、元数据、短摘录和中文评论；
- 论文、图表、摘录的版权归原作者（DeepSeek-AI / Peking University）；
- 摘录用于教学、评论与研究，引用请以论文原文为准；
- 如果你认为某段内容不适合在此使用，请开 issue，我们会尽快删除。

## 如何使用

1. 先读 [`notes.md`](Engram-conditional-memory/notes.md) 的「一页纸总结」；
2. 对照 `docs/lab*.md` 看每个部件的实现；
3. 读 [`README.md`](Engram-conditional-memory/README.md) 的原文摘录，检查自己是否理解术语；
4. 用 `scripts/paper_math.py` 复算关键数字。

## 引用

```bibtex
@article{cheng2026engram,
  title   = {Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models},
  author  = {Cheng, Xin and Zeng, Wangding and Dai, Damai and Chen, Qinyu and Wang, Bingxuan and
             Xie, Zhenda and Huang, Kezhao and Yu, Xingkai and Hao, Zhewen and Li, Yukun and
             Zhang, Han and Zhang, Huishuai and Zhao, Dongyan and Liang, Wenfeng},
  year    = {2026},
  note    = {DeepSeek-AI and Peking University},
  url     = {https://github.com/deepseek-ai/Engram}
}
```
