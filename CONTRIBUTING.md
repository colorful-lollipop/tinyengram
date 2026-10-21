# Contributing to TinyEngram

感谢你愿意改进这个项目。它是教学项目，最重要的不是代码量，而是**每一关都能被独立理解、独立验证**。

## 新关卡的最低标准

一个关卡必须同时包含：

1. **讲义** `docs/labN-*.md`
   - 问题动机（为什么需要这个部件）
   - 数学定义 / 公式
   - 接口约定与形状
   - 常见错误
   - 3–5 道思考题
2. **骨架** `src/tinyengram/*.py`
   - 保留类型标注和 docstring
   - 核心逻辑用 `raise NotImplementedError("TODO(LN): ...")`
3. **测试** `tests/test_lN_*.py`
   - 检查性质，不是逐行对照
   - 失败信息要能指导学习者
   - 至少一个「反向测试」（非法输入报错）
4. **参考实现** `solutions/tinyengram/*.py`
   - 与骨架接口完全一致
5. **评分** `scripts/grade.py` 的 `STAGES` 表
6. **全绿**：

```bash
make check          # ruff + 文档检查 + 参考实现全量测试
python scripts/grade.py --solution
```

## 代码风格

- Python 3.10+，`from __future__ import annotations`
- 行宽 100
- 公开 API 放进 `__all__`
- 形状写在 docstring 里
- 不引入重量级依赖；`transformers`/`triton`/`matplotlib` 必须是 optional

## 测试风格

好测试：

```python
def test_hash_is_strictly_causal():
    ...
    assert torch.equal(h1[:, :3], h2[:, :3])
```

坏测试：

```python
assert module.forward(x).shape == (2, 8, 64)  # 只检查形状
```

尽量测试：

- 数学不变量（归一化、守恒、单调、因果）；
- 与慢速参考实现的一致性；
- 边界条件（空输入、越界、负 padding）。

## 文档风格

- 中文为主，术语保留英文；
- 公式用 fenced code block，避免渲染依赖；
- 不吹「SOTA」；负面结果照实写；
- 引用论文时给出章节号。

## 提交

```bash
git checkout -b lab10-your-topic
make check
git commit -m "lab10: add ..."
```

PR 描述里请写：

- 这关解决什么问题；
- 测试覆盖了哪些性质；
- 你实际跑出的数字（如果有）。

提交前请确认没有把以下内容带进仓库：

- `.runs/`、checkpoint、数据集、日志；
- 个人路径、API key、token；
- 未标注来源的论文图片或大段原文。

## 报告问题

- Bug / 新功能：使用 `.github/ISSUE_TEMPLATE/` 中的模板；
- 使用问题：开 GitHub Discussion；
- 安全漏洞：按 [`SECURITY.md`](SECURITY.md) 私下报告，不要开公开 issue。

## 文档与翻译

英文 README 是入口，完整讲义目前以中文为主。欢迎：

- 把单篇 lab 翻成英文（保持公式、代码、测试引用一致）；
- 修正术语不一致（先看 [`docs/glossary.md`](docs/glossary.md)）；
- 补图表、实验复现记录、常见失败模式。

## 行为准则

参与本项目即表示同意遵守 [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)。

- 对初学者友好；不嘲讽「这都不会」；
- 不为了刷分而调 baseline；
- 不把未经验证的结论写成事实。
