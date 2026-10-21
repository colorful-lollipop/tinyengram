#!/usr/bin/env python3
"""检查讲义的一致性：链接、关卡卡片、术语、测试引用。

    python scripts/check_docs.py

这是一个「文档测试」：不依赖 PyTorch，也不训练模型。CI 和本地提交前都应该跑。
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

REQUIRED = [
    "README.md",
    "README.en.md",
    "CONTRIBUTING.md",
    "CODE_OF_CONDUCT.md",
    "SECURITY.md",
    "CHANGELOG.md",
    "CITATION.cff",
    "LICENSE",
    ".editorconfig",
    ".github/PULL_REQUEST_TEMPLATE.md",
    ".github/ISSUE_TEMPLATE/bug_report.yml",
    ".github/ISSUE_TEMPLATE/feature_request.yml",
    ".github/ISSUE_TEMPLATE/config.yml",
    "docs/README.md",
    "docs/faq.md",
    "docs/glossary.md",
    "papers/README.md",
    "papers/Engram-conditional-memory/README.md",
    "papers/Engram-conditional-memory/notes.md",
    "experiments/README.md",
    "scripts/grade.py",
    "scripts/paper_math.py",
    "perf/bench.py",
] + [f"docs/lab{i}-{name}.md" for i, name in enumerate(
    [
        "tokenizer",
        "hashing",
        "retrieval",
        "gating",
        "fusion",
        "transformer",
        "allocation",
        "systems",
        "analysis",
        "capstone",
    ]
)]

LABS = sorted(DOCS.glob("lab[0-9]-*.md"))
TAGS = ("理解", "推导", "设计", "开放")


def strip_code(text: str) -> str:
    """Remove fenced blocks and inline code before parsing links."""
    text = re.sub(r"```.*?```", "", text, flags=re.S)
    text = re.sub(r"`[^`\n]*`", "", text)
    return text


def check_links(path: Path) -> list[str]:
    errors: list[str] = []
    text = strip_code(path.read_text())
    for target in re.findall(r"\[[^\]]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)", text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        clean = target.split("#", 1)[0]
        if not clean:
            continue
        resolved = (path.parent / clean).resolve()
        try:
            resolved.relative_to(ROOT.resolve())
        except ValueError:
            errors.append(f"{path.relative_to(ROOT)}: 链接逃出项目目录: {target}")
            continue
        if not resolved.exists():
            errors.append(f"{path.relative_to(ROOT)}: 链接不存在: {target}")
    return errors


def check_lab(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text()
    stem = path.stem
    num = int(re.search(r"lab(\d)", stem).group(1))  # type: ignore[union-attr]
    rel = path.relative_to(ROOT)

    if not text.startswith(f"# Lab {num}"):
        errors.append(f"{rel}: 标题应以 `# Lab {num}` 开头")

    for field in ("前置", "必做", "进阶", "挑战"):
        if f"**{field}**" not in text:
            errors.append(f"{rel}: 缺少本关卡片字段 `**{field}**`")

    if num < 9:
        for section in ("自测", "思考题", "提示"):
            if not re.search(rf"^## \d+\.\d+ {section}", text, re.M):
                errors.append(f"{rel}: 缺少章节 `{section}`")
    else:
        if "思考题" not in text or "结课标准" not in text:
            errors.append(f"{rel}: Capstone 需要「思考题」和「结课标准」")

    if num > 0 and "上一关" not in text:
        errors.append(f"{rel}: 缺少「上一关」链接")
    if num < 9 and "下一关" not in text:
        errors.append(f"{rel}: 缺少「下一关」链接")
    if num == 9 and "返回" not in text:
        errors.append(f"{rel}: 缺少「返回 README」链接")

    # 思考题的难度梯度
    for tag in TAGS:
        if f"**[{tag}]**" not in text and not (num == 9 and tag in ("理解", "推导")):
            errors.append(f"{rel}: 思考题缺少 `[{tag}]` 难度标记")
    if "难度梯度" not in text:
        errors.append(f"{rel}: 缺少「难度梯度」说明")

    # 引用的测试文件必须存在
    for ref in set(re.findall(r"tests/test_l\d+_[a-z0-9_]+\.py", text)):
        if not (ROOT / ref).exists():
            errors.append(f"{rel}: 引用了不存在的测试 `{ref}`")

    return errors


def check_terminology() -> list[str]:
    errors: list[str] = []
    # 论文 Table 5 的口径必须统一为「每个 head/order」
    for path in [DOCS / "lab6-allocation.md", DOCS / "faq.md"]:
        text = path.read_text()
        if "每个 head 的 slot 数" in text:
            errors.append(f"{path.relative_to(ROOT)}: 口径应为「每个 head/order 的 slot 数」")
    # 旧的、错误的 40 KiB/token 结论不能出现在结论性段落
    lab2 = (DOCS / "lab2-retrieval.md").read_text()
    if "40 KiB/token" in lab2:
        errors.append("docs/lab2-retrieval.md: 仍在使用错误的 40 KiB/token 口径")
    return errors


def check_cross_references() -> list[str]:
    errors: list[str] = []
    docs_readme = (DOCS / "README.md").read_text()
    for name in ("glossary.md", "faq.md"):
        if name not in docs_readme:
            errors.append(f"docs/README.md: 缺少对 `{name}` 的链接")

    root_readme = (ROOT / "README.md").read_text()
    for i in range(10):
        if f"docs/lab{i}-" not in root_readme:
            errors.append(f"README.md: 学习路线缺少 L{i} 的讲义链接")
    for link in ("README.en.md", "CODE_OF_CONDUCT.md", "SECURITY.md", "CITATION.cff"):
        if link not in root_readme:
            errors.append(f"README.md: 缺少对 `{link}` 的链接")
    if "README.md" not in (ROOT / "README.en.md").read_text():
        errors.append("README.en.md: 缺少返回中文 README 的链接")
    return errors


def main() -> int:
    errors: list[str] = []

    for rel in REQUIRED:
        if not (ROOT / rel).exists():
            errors.append(f"缺少必需文件: {rel}")

    if len(LABS) != 10:
        errors.append(f"应有 10 篇 lab 讲义，实际 {len(LABS)} 篇")

    for path in LABS:
        errors.extend(check_lab(path))

    for path in sorted(list(ROOT.glob("*.md")) + list(DOCS.glob("*.md"))
                       + list((ROOT / "papers").rglob("*.md"))
                       + list((ROOT / "experiments").glob("*.md"))):
        errors.extend(check_links(path))

    errors.extend(check_terminology())
    errors.extend(check_cross_references())

    if errors:
        print(f"文档检查失败：{len(errors)} 个问题\n")
        for err in errors:
            print(" -", err)
        return 1

    print("文档检查通过")
    print(f"  - {len(LABS)} 篇 lab 讲义，均含卡片 / 梯度 / 自测 / 思考题 / 提示")
    print("  - 所有相对链接均存在")
    print("  - 术语口径与测试引用一致")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
