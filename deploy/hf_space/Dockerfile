# Hugging Face Spaces Docker SDK image.
# Spaces expects the app to listen on the port declared as `app_port` in
# README.md's YAML metadata (7860 here) and serves that publicly.
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY models ./models
COPY static ./static

ENV MLFLOW_TRACKING_URI=file:///app/mlruns_unused \
    PYTHONUNBUFFERED=1

# Spaces containers run as a non-root user with a writable /app
RUN useradd -m -u 1000 spaceuser && chown -R spaceuser /app
USER spaceuser

EXPOSE 7860

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
