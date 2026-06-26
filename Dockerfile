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

COPY --from=builder /app/requirements.txt .

# Устанавливаем зависимости системы, ставим pip-пакеты и очищаем кэш/билдеры в одном слое
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    postgresql-client \
    curl \
    && pip install --no-cache-dir -r requirements.txt pandas \
    && apt-get purge -y --auto-remove build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY . .

CMD ["uvicorn", "polyflip.api.main:app", "--host", "0.0.0.0", "--port", "8001"]
