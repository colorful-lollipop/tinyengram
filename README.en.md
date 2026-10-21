# TinyEngram

**Implement DeepSeek Engram conditional memory from scratch — retrieval, gating, training, inference, and systems optimization — one test-driven lab at a time.**

[简体中文](README.md) | English

[![CI](https://github.com/colorful-lollipop/tinyengram/actions/workflows/ci.yml/badge.svg)](https://github.com/colorful-lollipop/tinyengram/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.2%2B-ee4c2c.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-91%20core-brightgreen.svg)](#testing)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Status**: an independent educational project, not an official DeepSeek-AI implementation.
> The paper, official demo, and figures remain copyright of their authors.

---

## Why Engram?

Transformers have no native knowledge-lookup primitive. Recognizing “Alexander the Great”
requires several layers of attention/FFN to compose the entity — effectively **simulating a
lookup with computation**.

Engram adds **conditional memory** as a second sparsity axis, complementary to MoE:

```text
token ids ──► compressed vocab ──► 3-gram hashes ──► 16 prime tables ──► one gather
                                                                             │
hidden h_t ─────────────► context gate α_t ──► u_t = α_t · W_V e_t
                                                                             │
                                      Y = u_t + ShortConv(u_t)
                                      H ← H + Y
```

- **Static**: keys depend only on token ids, not on hidden states.
- **O(1)**: one token reads a fixed number of rows, independent of table size.
- **Prefetchable**: indices are known before the forward pass, hiding PCIe latency.
- **Scalable**: in the 27B model, 5.7B parameters are memory while activated parameters stay fixed.

In the paper, a 27B Engram model beats an iso-parameter, iso-FLOPs MoE baseline on knowledge,
reasoning, code, and math benchmarks.

---

## Course map

Ten labs, each with a runnable skeleton, property-based tests, and a written lecture (in Chinese):

| Lab | Topic | What you implement | Tests |
|---|---|---|---|
| **L0** | Tokenizer compression | Unicode normalization + surjective projection | `test_l0_tokenizer.py` |
| **L1** | Multi-head hashing | prime tables, odd multipliers, multiplicative-XOR, int64 bounds | `test_l1_hashing.py` |
| **L2** | Retrieval & layout | flat table + offsets, one gather for 16 heads | `test_l2_embedding.py` |
| **L3** | Context gating | RMSNorm, cosine logit, signed sqrt, per-branch gates | `test_l3_gating.py` |
| **L4** | ShortConv & fusion | causal depthwise conv, zero-init identity, residual | `test_l4_engram.py` |
| **L5** | Transformer & training | multi-branch TinyLM, RoPE, KV cache, parameter groups | `test_l5_transformer.py` |
| **L6** | Allocation & scaling | iso-param/iso-FLOPs, U-shaped allocation, power law | `test_l6_allocation.py` |
| **L7** | Systems & offload | deterministic prefetch, Zipf cache, sharding, Triton | `test_l7_systems.py` |
| **L8** | Mechanistic analysis | LogitLens, CKA, soft alignment, gate stats | `test_l8_analysis.py` |
| **L9** | End-to-end reproduction | train baseline vs Engram, full analysis pipeline | `test_l9_reproduction.py` |

Each lab is graded in three levels:

1. **Understand (required)** — explain the math, pass the tests.
2. **Transfer (advanced)** — change one design knob, predict, run a controlled experiment.
3. **Research (challenge)** — pose a falsifiable question, design an experiment, report honestly.

See [`docs/README.md`](docs/README.md) for prerequisites, time budget, and the dependency graph.
Terminology: [`docs/glossary.md`](docs/glossary.md). FAQ: [`docs/faq.md`](docs/faq.md).

---

## Install

Requirements: Python 3.10+, PyTorch 2.2+.

```bash
git clone https://github.com/colorful-lollipop/tinyengram.git
cd tinyengram

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Or with `uv`:

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
```

CPU is enough for L0–L6 and L8. A GPU is only needed for the optional Triton benchmark in L7.

---

## Quick start

```bash
python scripts/grade.py              # progress by lab
pytest tests/test_l0_tokenizer.py -x -q
less docs/lab0-tokenizer.md
```

Run the reference implementation (also used by CI):

```bash
TINYENGRAM_IMPL=solutions pytest -q
python scripts/grade.py --solution --all
python scripts/check_docs.py
```

Run the experiments:

```bash
make experiments
# outputs are written to .runs/
```

---

## Verification

On the reference implementation, from a clean clone:

```text
TINYENGRAM_IMPL=solutions pytest -q     -> 92 passed
python scripts/grade.py --solution --all -> 91 / 91 (100%)
python scripts/check_docs.py             -> docs consistency ok
```

The learner skeleton (`src/tinyengram/`) intentionally fails until the corresponding lab is
implemented. `solutions/tinyengram/` is a complete parallel implementation with the same API.

Paper arithmetic (`python scripts/paper_math.py`):

| Quantity | This project | Paper |
|---|---|---|
| Engram-27B memory parameters | 5.79B | 5.7B |
| Engram-40B memory parameters | 18.53B | 18.5B |
| Routed experts | 55 | 55 |
| Training FLOPs (6PD) | 5.97e21 | ~6e21 |
| Read per token | 2.5 KiB | O(1), independent of table size |

Performance on an RTX 4070 Ti Super (`python perf/bench.py --breakdown`):

| Implementation | Latency | Relative |
|---|---|---|
| Engram module, eager | 1.87 ms | 1.0× |
| Engram module, `torch.compile` | 0.55 ms | 3.4× |
| Gather, eager | 0.180 ms | 1.0× |
| Gather, Triton fused | 0.174 ms | 1.03× |

The Triton gather only wins by ~3%: retrieval is latency-bound on random accesses, and
`index_select` is already well optimized. This negative result is part of the systems lecture.

---

## Repository layout

```text
src/tinyengram/          # learner skeleton with TODO(Lx) markers
solutions/tinyengram/    # complete reference implementation
tests/                   # property-based tests, one file per lab
docs/                    # Chinese lecture notes (lab0–lab9, glossary, FAQ)
papers/                  # paper notes, key excerpts, critical reading
experiments/             # allocation, training, analysis, offload, scaling
perf/                    # Triton kernel + benchmarks
scripts/                 # grade.py, paper_math.py, check_docs.py
```

## Paper & further reading

- **Engram**: Cheng et al., *Conditional Memory via Scalable Lookup: A New Axis of Sparsity
  for Large Language Models*, 2026. Official code: <https://github.com/deepseek-ai/Engram>
- Project notes: [`papers/Engram-conditional-memory/notes.md`](papers/Engram-conditional-memory/notes.md)
- Excerpts with translation: [`papers/Engram-conditional-memory/README.md`](papers/Engram-conditional-memory/README.md)
- Related: MoE, hash embeddings, OverEncoding, SCONE, Byte Latent Transformer, PEER/PKM, RETRO.

---

## Contributing

Contributions are welcome: new labs, bug fixes, lecture improvements, real-data experiments.
Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) and [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md).
Security issues: follow [`SECURITY.md`](SECURITY.md) and report privately.

## Citation

```bibtex
@misc{tinyengram2026,
  title        = {TinyEngram: A Test-Driven Course on Conditional Memory},
  author       = {TinyEngram contributors},
  year         = {2026},
  howpublished = {\url{https://github.com/colorful-lollipop/tinyengram}},
  note         = {Educational implementation of DeepSeek Engram}
}
```

See also [`CITATION.cff`](CITATION.cff).

## License

Code is released under the [MIT License](LICENSE). Excerpts and figures from the Engram paper
remain copyright of their authors and are included for teaching, commentary, and research only;
see [`papers/README.md`](papers/README.md).

---

**Note**: the full lecture notes are currently in Chinese. English translations of individual
labs are welcome — please open an issue or a PR.
