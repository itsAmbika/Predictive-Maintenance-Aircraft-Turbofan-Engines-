# Every target is a one-line command you can also run by hand -- `make` is a
# convenience, not a dependency. On Windows without make, copy the command out.

.PHONY: help setup data features train evaluate extras pipeline register serve ui \
        test test-fast test-slow lint format docker-build docker-up mlflow-ui clean

CONF ?=            # extra config overrides, e.g. `make train CONF="subset=FD003"`

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-14s %s\n", $$1, $$2}'

setup:  ## Create .venv and install every pinned dependency
	uv sync

data:  ## Stage 1: raw txt -> interim parquet with the RUL target
	uv run python -m src.pipeline.prepare_data $(CONF)

features:  ## Stage 2: interim -> processed features + fitted transforms
	uv run python -m src.pipeline.build_features $(CONF)

train:  ## Stage 3: train all candidates, log to MLflow, persist the best
	uv run python -m src.pipeline.train $(CONF)

evaluate:  ## Stage 4: score the official test set + apply the quality gate
	uv run python -m src.pipeline.evaluate $(CONF)

extras:  ## Stage 5: failure-probability classifiers + RUL interval models
	uv run python -m src.pipeline.serving_extras $(CONF)

pipeline:  ## Run the whole DAG end to end
	uv run python -m src.pipeline.run_all $(CONF)

register:  ## Promote the trained model to the MLflow Model Registry
	uv run python -m src.pipeline.register mlflow.register=true $(CONF)

serve:  ## Run the API (serves the React build if frontend-react/dist exists)
	uv run uvicorn api.main:app --reload --port 8000

ui:  ## Run the React dev server against a local API
	cd frontend-react && npm run dev -- --port 5173

test:  ## Full test suite
	uv run pytest

test-fast:  ## Only tests that need no trained artifacts (what CI runs on every push)
	uv run pytest -m "not slow"

test-slow:  ## Only the artifact-dependent tests
	uv run pytest -m slow

lint:  ## Static checks
	uv run ruff check src api tests
	uv run ruff format --check src api tests

format:  ## Apply formatting + import sorting
	uv run ruff format src api tests
	uv run ruff check --fix src api tests

docker-build:  ## Build the single API + frontend image
	docker build -t rul-api:local .

docker-up:  ## API + MLflow tracking server
	docker compose up --build

mlflow-ui:  ## Browse local runs (file store) at http://localhost:5000
	# uvx, not uv run: the project pins mlflow-skinny (client only, no server UI).
	MLFLOW_ALLOW_FILE_STORE=true uvx --from mlflow mlflow ui --backend-store-uri file:./mlruns --port 5000

clean:  ## Remove derived data and caches (models/ and artifacts/ are kept)
	rm -rf data/interim data/processed .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
