PY ?= python3

.PHONY: help install-dev lint test test-all solution solution-all grade bench bench-offload \
        experiments paper-math docs-check check clean

help:
	@echo "make install-dev    安装可编辑开发环境"
	@echo "make lint           ruff 检查"
	@echo "make test           只跑快速测试（L0-L9，跳过 slow/gpu）"
	@echo "make test-all       跑全部测试（含端到端复现）"
	@echo "make solution       用参考实现跑快速测试"
	@echo "make solution-all   用参考实现跑全部测试"
	@echo "make grade          分阶段显示学习进度"
	@echo "make bench          GPU kernel 基准（需要 CUDA + Triton）"
	@echo "make bench-offload  CPU/GPU offload 与缓存基准"
	@echo "make experiments    跑全部教学实验"
	@echo "make paper-math     复算论文 Table 1/5 的参数与分配比例"
	@echo "make docs-check     检查讲义链接、关卡卡片和术语一致性"
	@echo "make check          lint + docs + 参考实现全量测试"
	@echo "PY=/path/to/python make test   指定解释器"

install-dev:
	$(PY) -m pip install -e ".[dev]"

lint:
	ruff check solutions tests experiments perf scripts

test:
	$(PY) -m pytest -q -m "not slow and not gpu"

test-all:
	$(PY) -m pytest -q -m ""

solution:
	TINYENGRAM_IMPL=solutions $(PY) -m pytest -q -m "not slow and not gpu"

solution-all:
	TINYENGRAM_IMPL=solutions $(PY) -m pytest -q -m ""

grade:
	$(PY) scripts/grade.py

bench:
	$(PY) perf/bench.py --breakdown

bench-offload:
	$(PY) experiments/exp4_offload.py

experiments:
	$(PY) experiments/exp1_allocation.py
	$(PY) experiments/exp2_train_tiny.py
	$(PY) experiments/exp3_analysis.py
	$(PY) experiments/exp4_offload.py
	$(PY) experiments/exp5_scaling.py

paper-math:
	$(PY) scripts/paper_math.py

docs-check:
	$(PY) scripts/check_docs.py

check: lint docs-check solution-all

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov *.egg-info build dist .runs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
