# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned

- English translations of individual labs.
- Real-tokenizer compression study (GPT-2 / DeepSeek).
- Pinned-memory CPU offload with CUDA stream overlap.
- Fused gate + value + conv Triton kernel.

## [0.1.0] - 2026-10-22

### Added

- Ten test-driven labs (L0–L9) covering tokenizer compression, multi-head n-gram hashing,
  retrieval and memory layout, context gating, causal short convolution, TinyLM training,
  sparsity allocation, systems/offload, mechanistic analysis, and end-to-end reproduction.
- Parallel learner skeleton (`src/tinyengram/`) and reference implementation
  (`solutions/tinyengram/`).
- 91 core property-based tests plus one optional GPU/Triton test.
- Chinese lecture notes with prerequisites, derivations, common pitfalls, graded exercises, and
  hints.
- Paper notes with key excerpts, translations, critical reading, and reproduction checklist.
- Five teaching experiments: allocation sweep, tiny training comparison, mechanistic analysis,
  host offload simulation, and memory-scaling sweep.
- `scripts/grade.py` staged grader, `scripts/paper_math.py` paper arithmetic, and
  `scripts/check_docs.py` documentation consistency checker.
- Optional Triton fused gather kernel and benchmark.
- CI, contribution guide, code of conduct, security policy, citation metadata, issue/PR
  templates, and English README.

[Unreleased]: https://github.com/colorful-lollipop/tinyengram/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/colorful-lollipop/tinyengram/releases/tag/v0.1.0
