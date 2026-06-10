# ─── Build stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

RUN pip install uv

COPY pyproject.toml .
COPY src/ src/

# CPU-only torch for smaller image
RUN uv pip install --system --no-cache \
    torch==2.3.0+cpu \
    --extra-index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --system --no-cache \
    torch-geometric \
    --extra-index-url https://data.pyg.org/whl/torch-2.3.0+cpu.html

RUN uv pip install --system --no-cache -e "." --no-deps

# ─── Runtime stage ────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN groupadd -r c2gnn && useradd -r -g c2gnn c2gnn

COPY --from=builder /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app/src ./src

RUN mkdir -p data/processed models/artifacts reports && \
    chown -R c2gnn:c2gnn /app

USER c2gnn

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "c2gnn.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
