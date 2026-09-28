# InsightFlow AI - one image for both the API and the Streamlit UI (see docker-compose.yml).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so code changes don't invalidate the dependency layer.
COPY pyproject.toml README.md ./
COPY app ./app
RUN pip install -e .

COPY frontend ./frontend
COPY knowledge ./knowledge
COPY evaluations/datasets ./evaluations/datasets
COPY evaluations/baseline.json ./evaluations/baseline.json
COPY data/examples ./data/examples

# Run as an unprivileged user; data/ is the only writable location.
RUN useradd --create-home --uid 1000 insightflow \
    && mkdir -p data/raw data/processed data/traces data/sandbox \
    && chown -R insightflow:insightflow /app/data
USER insightflow

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
