.PHONY: setup demo serve test

setup:
	uv sync --extra dev --extra pose --extra web

demo:
	uv run skate-metrics demo --output out/demo

serve:
	uv run skate-metrics serve --host 127.0.0.1 --port 8000

test:
	uv run pytest
