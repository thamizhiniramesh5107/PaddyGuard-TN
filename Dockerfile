FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    OPENCV_FOR_THREADS_NUM=1 \
    TORCH_NUM_THREADS=1 \
    MODEL_PATH=/tmp/paddyguard.pt \
    DATABASE_PATH=/tmp/paddyguard.sqlite3 \
    PORT=8080

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY class_map.json official_advisories.json paddyguard_advisory_data_starter.csv ./
COPY artifacts/metrics.json ./artifacts/metrics.json

EXPOSE 8080
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1"]
