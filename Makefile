UV_PYTHON := $(shell command -v uv >/dev/null 2>&1 && uv python find --no-project --system '>=3.10' 2>/dev/null)
VENV_PYTHON := $(wildcard .venv/bin/python)
PYTHON ?= $(if $(UV_PYTHON),$(UV_PYTHON),$(if $(VENV_PYTHON),$(VENV_PYTHON),python3))
SOURCE_PATH := $(CURDIR)/src
export LANG := en_US.UTF-8
export LC_ALL := en_US.UTF-8
export PYTHONUTF8 := 1

.PHONY: help test test-fast test-integration compile lint verify release verify-artifacts

help:
	@echo "make verify            Validate the source delivery without pytest"
	@echo "make test              Run the unittest suite with the project Python"
	@echo "make test-fast         Run contract and hot-path regression tests"
	@echo "make test-integration  Run native plugin and provider integration tests"
	@echo "make compile           Compile-check all Python sources"
	@echo "make lint              Run the static-analysis gate (ruff; see ruff.toml)"
	@echo "make release           Build and verify wheel, sdist, and delivery archive"
	@echo "make verify-artifacts  Re-verify existing files under dist/"

test:
	@PYTHONPATH="$(SOURCE_PATH)$${PYTHONPATH:+:$${PYTHONPATH}}" $(PYTHON) -m unittest discover -v

test-fast:
	@PYTHONPATH="$(SOURCE_PATH)$${PYTHONPATH:+:$${PYTHONPATH}}" $(PYTHON) -m unittest \
		tests.test_api_contracts tests.test_projection_contract tests.test_core \
		tests.test_dsh_tool_contracts tests.test_dsh_reconciliation tests.test_runtime_gc

test-integration:
	@node --test integrations/dsh_ecology_plugin/test/*.test.mjs \
		integrations/dsh_ecology_plugin/test/proxy_security.mjs

compile:
	@PYTHONPATH="$(SOURCE_PATH)$${PYTHONPATH:+:$${PYTHONPATH}}" $(PYTHON) -m compileall -q src scripts tests

# Ruff is a development tool, not a runtime dependency: the project declares no
# dependencies at all and must keep installing that way. `uvx` fetches a pinned
# ruff into its own cache without touching this environment; `ruff` on PATH is
# used when it is already there. Neither is required to build or run anything.
RUFF ?= $(shell command -v ruff >/dev/null 2>&1 && echo ruff || (command -v uvx >/dev/null 2>&1 && echo "uvx ruff@0.16.7"))

lint:
	@if [ -z "$(RUFF)" ]; then \
		echo "ruff not found: install it, or install uv so 'uvx ruff' can run"; exit 1; \
	fi
	@$(RUFF) check src scripts tests

verify:
	@PYTHON="$(PYTHON)" ./scripts/verify_delivery.sh --source-only

release:
	@PYTHON="$(PYTHON)" ./scripts/build_delivery.sh

verify-artifacts:
	@PYTHON="$(PYTHON)" ./scripts/verify_delivery.sh --artifacts-only
