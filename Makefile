.DEFAULT_GOAL := help

.PHONY: help sync lint format fmt typecheck test check all data

help:
	@echo "Available targets:"
	@echo "  sync       - Install dependencies (including dev) with uv"
	@echo "  lint       - Run Ruff lint checks"
	@echo "  format/fmt - Run Ruff formatter over src and tests"
	@echo "  typecheck  - Run mypy type checks"
	@echo "  test       - Run pytest test suite"
	@echo "  check      - Run lint, typecheck, and test"
	@echo "  all        - Run sync, format, and full check pipeline"
	@echo "  data       - Run preprocessing to build numpy datasets"

sync:
	uv sync --dev

lint:
	uv run ruff check src tests

format fmt:
	uv run ruff format src tests

typecheck:
	uv run mypy src tests

test:
	uv run pytest

check: lint typecheck test

all: sync format check

data:
	uv run python -m musicagent.scripts.preprocess


