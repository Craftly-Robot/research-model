.PHONY: install test lint format check clean gateway cli

install:
	python3 -m venv .venv && .venv/bin/pip install -r requirements-local-dev.txt && .venv/bin/pip install -e . --no-deps && .venv/bin/pip install ruff

test:
	.venv/bin/python -m unittest

lint:
	.venv/bin/ruff check src/craftly --fix

format:
	.venv/bin/ruff format src/craftly
	.venv/bin/ruff check src/craftly --fix

check:
	.venv/bin/ruff check src/craftly
	.venv/bin/ruff format --check src/craftly

compile:
	.venv/bin/python -m compileall -q src/craftly tests deploy/gpu

clean:
	rm -rf .venv __pycache__ **/__pycache__ .pytest_cache

gateway:
	.venv/bin/python -m uvicorn src.craftly.gateway.api:app --host 127.0.0.1 --port 8090 --reload

cli:
	.venv/bin/python -m src.craftly.cli --prompt "$(PROMPT)" --workspace . --agent-backend-mode mock --no-verifier --no-security
