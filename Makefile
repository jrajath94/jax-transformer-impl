.PHONY: install test bench run lint clean

# ── Installation ─────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

install-cpu:
	pip install -e ".[dev]"
	pip install "jax[cpu]"

install-gpu:
	pip install -e ".[dev]"
	pip install "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# ── Tests ─────────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --tb=short --cov=src --cov-report=term-missing

test-fast:
	pytest tests/ -v --tb=short -x

# ── Benchmarks ────────────────────────────────────────────────────────────────
bench:
	python benchmarks/bench_gqa.py

# ── Examples ──────────────────────────────────────────────────────────────────
run:
	python examples/quickstart.py

# ── CLI ───────────────────────────────────────────────────────────────────────
benchmark:
	python -m jax_transformer.cli benchmark --num-heads 8 --num-kv-heads 2

profile:
	python -m jax_transformer.cli profile --model-dim 512 --seq-len 256

# ── Linting ───────────────────────────────────────────────────────────────────
lint:
	ruff check src/ tests/
	mypy src/ --ignore-missing-imports

lint-fix:
	ruff check --fix src/ tests/

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ .coverage htmlcov/
