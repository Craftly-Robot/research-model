.PHONY: install test lint compile clean gateway cli

install:
	python3 -m venv .venv && .venv/bin/pip install -r requirements-local-dev.txt && .venv/bin/pip install -e . --no-deps

test:
	.venv/bin/python -m unittest

lint:
	.venv/bin/python -m compileall -q src/craftly tests deploy/gpu

compile:
	.venv/bin/python -m compileall -q src/craftly tests deploy/gpu

clean:
	rm -rf .venv __pycache__ **/__pycache__ .pytest_cache

gateway:
	.venv/bin/python -m uvicorn src.craftly.gateway.api:app --host 127.0.0.1 --port 8090 --reload

cli:
	.venv/bin/python -m src.craftly.cli --prompt "$(PROMPT)" --workspace . --agent-backend-mode mock --no-verifier --no-security
