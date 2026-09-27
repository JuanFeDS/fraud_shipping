# API de scoring para Cloud Run: solo dependencias de producción y el modelo descargado del registry dentro de la imagen
FROM python:3.12-slim

# LightGBM necesita la librería de OpenMP, que la imagen slim no trae
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY deploy/api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY models/fraud_pipeline.joblib models/fraud_pipeline.json models/

ENV PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/fraud_pipeline.joblib \
    PORT=8080
CMD ["sh", "-c", "uvicorn fraude_shipping.production.api:app --host 0.0.0.0 --port ${PORT}"]
