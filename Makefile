PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
export PYTHONPATH := $(CURDIR)/src$(if $(PYTHONPATH),:$(PYTHONPATH))
# Loopback calls (tests start local HTTP servers) must bypass any system proxy.
export no_proxy := 127.0.0.1,localhost
export NO_PROXY := 127.0.0.1,localhost

.PHONY: help test lint serve

help:
	@echo "make test     run the unittest suite (tests needing AGC data skip when it is absent)"
	@echo "make lint     run ruff"
	@echo "make serve    start the JSON API alone on http://127.0.0.1:8850/ (the UI is the DSH plugin: ecogym dsh install)"

test:
	$(PYTHON) -m unittest discover -s tests -t .

lint:
	uvx ruff@0.16.7 check src tests

serve:
	$(PYTHON) -m ecogym serve
