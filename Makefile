.PHONY: install test lint run nodes route

install:
	python -m pip install -e '.[dev]'

test:
	pytest

lint:
	ruff check src tests

run:
	uvicorn orchestro_mesh.gateway:app --host 127.0.0.1 --port 8765

nodes:
	orchestro-mesh nodes --config mesh.yaml

route:
	orchestro-mesh route "write a private inference mesh smoke test" --config mesh.yaml --task coding
