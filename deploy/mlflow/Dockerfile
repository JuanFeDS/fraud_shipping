# Servidor de MLflow para Cloud Run: misma versión que el cliente local para evitar incompatibilidades de API
FROM python:3.12-slim

RUN pip install --no-cache-dir mlflow==3.16.1 psycopg2-binary==2.9.13 google-cloud-storage

WORKDIR /app
COPY iniciar.py .

ENV MLFLOW_DISABLE_AGENT_HINT=1
CMD ["python", "iniciar.py"]
