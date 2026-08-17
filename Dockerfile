# syntax=docker/dockerfile:1
#
# One image that serves the API *and* the production React build, so a single
# container is the whole application:
#
#   docker build -t rul-api .
#   docker run --rm -p 8000:8000 rul-api
#
# Model artifacts are baked in from models/ and artifacts/ (they're committed, so
# a fresh clone can serve without re-running the pipeline). Mount them instead
# (`-v $PWD/models:/app/models`) to swap models without rebuilding.

# --- stage 1: build the React frontend -------------------------------------
FROM node:22-alpine AS frontend
WORKDIR /build
COPY frontend-react/package.json frontend-react/package-lock.json ./
RUN npm ci
COPY frontend-react/ ./
RUN npm run build


# --- stage 2: python runtime ------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS runtime

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so code edits don't invalidate the (slow) install layer.
# --no-dev drops mlflow/pytest/jupyter; torch is excluded because only the LSTM/GRU
# training code imports it -- the served model is XGBoost, and torch is ~200MB.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-package torch

COPY src/ ./src/
COPY api/ ./api/
COPY conf/ ./conf/
COPY models/ ./models/
COPY artifacts/ ./artifacts/
COPY reports/ ./reports/
COPY frontend/ ./frontend/
COPY --from=frontend /build/dist ./frontend-react/dist

# Run as a non-root user.
RUN useradd --create-home --uid 10001 app && chown -R app:app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
b=json.load(urllib.request.urlopen('http://localhost:8000/api/health')); \
sys.exit(0 if b['model_loaded'] else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
