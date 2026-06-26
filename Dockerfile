# Stage 1: Build requirements
FROM python:3.12-slim as builder

WORKDIR /app
RUN pip install poetry
RUN poetry self add poetry-plugin-export
COPY pyproject.toml poetry.lock* ./
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes

# Stage 2: Final image
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONPATH=/app

# Устанавливаем постоянные зависимости системы времени выполнения (этот слой отлично кэшируется)
RUN apt-get update && apt-get install -y --no-install-recommends \
    postgresql-client \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/requirements.txt .

# Устанавливаем сборочные зависимости, ставим pip-пакеты и очищаем их в одном слое (инвалидируется при изменении requirements.txt)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && pip install --no-cache-dir -r requirements.txt pandas \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["uvicorn", "polyflip.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
