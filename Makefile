.PHONY: sync lint format type-check test clean

sync:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

type-check:
	uv run mypy src

test:
	uv run pytest

clean:
	rm -rf .venv __pycache__ .mypy_cache .ruff_cache .pytest_cache output/analysis
