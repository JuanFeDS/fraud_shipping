# Servidor de MLflow para Cloud Run: misma versión que el cliente local para evitar incompatibilidades de API
FROM python:3.12-slim

# El extra auth trae Flask-WTF, que la autenticación básica necesita para la protección CSRF
RUN pip install --no-cache-dir "mlflow[auth]==3.16.1" psycopg2-binary==2.9.13 google-cloud-storage

WORKDIR /app
COPY entrypoint.py .

ENV MLFLOW_DISABLE_AGENT_HINT=1
CMD ["python", "entrypoint.py"]
