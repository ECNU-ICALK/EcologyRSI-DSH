UV_PYTHON := $(shell command -v uv >/dev/null 2>&1 && uv python find --no-project --system '>=3.10' 2>/dev/null)
VENV_PYTHON := $(wildcard .venv/bin/python)
PYTHON ?= $(if $(UV_PYTHON),$(UV_PYTHON),$(if $(VENV_PYTHON),$(VENV_PYTHON),python3))
SOURCE_PATH := $(CURDIR)/src

.PHONY: help test verify release verify-artifacts

help:
	@echo "make verify            Validate the source delivery without pytest"
	@echo "make test              Run the unittest suite with the project Python"
	@echo "make release           Build and verify wheel, sdist, and delivery archive"
	@echo "make verify-artifacts  Re-verify existing files under dist/"

test:
	@PYTHONPATH="$(SOURCE_PATH)$${PYTHONPATH:+:$${PYTHONPATH}}" $(PYTHON) -m unittest discover -v

verify:
	@PYTHON="$(PYTHON)" ./scripts/verify_delivery.sh --source-only

release:
	@PYTHON="$(PYTHON)" ./scripts/build_delivery.sh

verify-artifacts:
	@PYTHON="$(PYTHON)" ./scripts/verify_delivery.sh --artifacts
