"""API de scoring online: recibe una transacción y devuelve la probabilidad de fraude y la decisión."""

import json
import os
from contextlib import asynccontextmanager
from datetime import timedelta, timezone
from pathlib import Path
from secrets import compare_digest
from typing import Literal

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import AwareDatetime, BaseModel, Field

from fraude_shipping.production.pipeline import MODEL_PATH, FraudPipeline

API_KEY_HEADER = APIKeyHeader(name='X-API-Key', auto_error=False)
# Zona horaria que se supone para la fecha del dataset (Brasil y Argentina, UTC−3 en 2020): `hora` se aprendió así
DATASET_TIMEZONE = timezone(timedelta(hours=-3))


class Transaction(BaseModel):
    """Variables de una transacción tal como llegan al momento de decidir; las opcionales admiten nulo."""

    a: int
    b: float | None = None
    c: float | None = None
    d: float | None = None
    e: float
    f: float | None = None
    g: str | None = Field(default=None, description='País')
    h: int
    j: str
    k: float
    l: float | None = None
    m: float | None = None
    n: int = Field(ge=0, le=1)
    o: Literal['Y', 'N'] | None = None
    p: Literal['Y', 'N']
    fecha: AwareDatetime = Field(description='Fecha y hora con zona horaria, por ejemplo 2020-04-15T02:30:00-03:00')
    monto: float = Field(gt=0)
    score: int = Field(ge=0, le=100)


class Prediction(BaseModel):
    """Resultado del scoring de una transacción."""

    probabilidad_fraude: float
    decision: Literal['aprobar', 'rechazar']
    umbral: float


@asynccontextmanager
async def lifespan(application):
    """Carga el pipeline una sola vez al iniciar la API, junto con la versión registrada si viene de MLflow."""
    # Falla cerrada: si el despliegue pierde la API key, la API no arranca en lugar de quedar abierta
    if not os.environ.get('FRAUDE_API_KEY'):
        raise RuntimeError('Falta la variable de entorno FRAUDE_API_KEY')
    path = Path(os.environ.get('MODEL_PATH', MODEL_PATH))
    application.state.pipeline = FraudPipeline.load(path)
    metadata_path = path.with_suffix('.json')
    # Un pipeline entrenado localmente, sin pasar por el registry, no tiene metadata ni versión
    metadata = json.loads(metadata_path.read_text(encoding='utf-8')) if metadata_path.exists() else {}
    application.state.model_version = metadata.get('version')
    yield


app = FastAPI(title='Prevención de fraude', lifespan=lifespan)


def verify_api_key(api_key: str | None = Security(API_KEY_HEADER)):
    """Exige la API key de FRAUDE_API_KEY; si la variable falta, rechaza todo en lugar de quedar abierta."""
    expected = os.environ.get('FRAUDE_API_KEY')
    # compare_digest evita que el tiempo de respuesta revele cuántos caracteres de la key coinciden
    if not (expected and api_key and compare_digest(api_key.encode(), expected.encode())):
        raise HTTPException(status_code=401, detail='API key inválida o ausente')


@app.get('/salud')
def health(request: Request):
    """Confirma que la API está arriba y con qué modelo: versión del registry (si se conoce) y umbral."""
    return {
        'estado': 'ok',
        'version_modelo': getattr(request.app.state, 'model_version', None),
        'umbral': request.app.state.pipeline.threshold,
    }


@app.post('/predecir', response_model=Prediction, dependencies=[Depends(verify_api_key)])
def predict(transaction: Transaction, request: Request):
    """Probabilidad de fraude y decisión para una transacción."""
    pipeline = request.app.state.pipeline
    row = transaction.model_dump()
    # La misma transacción da la misma hora del día sin importar la zona horaria en la que la envíe el cliente
    row['fecha'] = transaction.fecha.astimezone(DATASET_TIMEZONE).replace(tzinfo=None)
    result = pipeline.predict(pd.DataFrame([row])).iloc[0]
    return Prediction(
        probabilidad_fraude=result['probabilidad_fraude'], decision=result['decision'], umbral=pipeline.threshold
    )
