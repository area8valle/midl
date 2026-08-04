PYTHON ?= python
PKG := src/midl

.DEFAULT_GOAL := help

.PHONY: help install dev lint type docker clean

help:
	@echo "targets:"
	@echo "  install  install the package"
	@echo "  dev      install with development extras"
	@echo "  lint     run ruff, black --check, isort --check"
	@echo "  type     run mypy --strict on the package"
	@echo "  docker   build the container image"
	@echo "  clean    remove caches and run artifacts"

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m black --check src tests
	$(PYTHON) -m isort --check-only src tests

type:
	$(PYTHON) -m mypy $(PKG)

docker:
	docker build -t midl .

clean:
	rm -rf .mypy_cache .ruff_cache runs
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
