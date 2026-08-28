# syntax=docker/dockerfile:1
#
# One image that serves the API *and* the production React build, so a single
# container is the whole application. Runs unchanged on Hugging Face Spaces
# (PORT=7860), Cloud Run / Render (they inject $PORT), or locally:
#
#   docker build -t rul-api .
#   docker run --rm -e PORT=8000 -p 8000:8000 rul-api
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

# Hugging Face Spaces expects uid 1000 with a writable HOME; matching that here
# keeps the same image portable across HF, Cloud Run, and local runs.
RUN useradd --create-home --uid 1000 user

# MPLCONFIGDIR / NUMBA_CACHE_DIR: shap pulls in matplotlib and numba wants a
# cache dir -- both fail noisily when HOME isn't writable, which is the classic
# Spaces container error. PORT defaults to 7860, which is what Spaces expects.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HOME=/home/user \
    MPLCONFIGDIR=/tmp/matplotlib \
    NUMBA_CACHE_DIR=/tmp/numba \
    XDG_CACHE_HOME=/tmp/.cache \
    PORT=7860

WORKDIR /app

# Dependencies first, so code edits don't invalidate the (slow) install layer.
# --no-dev drops mlflow/pytest/jupyter; torch is excluded because only the LSTM/GRU
# training code imports it -- the served model is XGBoost, and torch is ~200MB.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-package torch

COPY src/ ./src/
COPY api/ ./api/
COPY conf/ ./conf/
COPY models/ ./models/
COPY artifacts/ ./artifacts/
COPY reports/ ./reports/
COPY frontend/ ./frontend/
COPY --from=frontend /build/dist ./frontend-react/dist
# One real C-MAPSS file, served at /api/sample, so a visitor can try the demo
# without hunting down the NASA dataset first.
COPY data/raw/test_FD001.txt ./data/raw/test_FD001.txt

RUN chown -R user:user /app
USER user

EXPOSE 7860
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860}"]
