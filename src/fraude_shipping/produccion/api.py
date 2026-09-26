"""API de scoring online: recibe una transacción y devuelve la probabilidad de fraude y la decisión."""

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, Field

from fraude_shipping.produccion.pipeline import RUTA_MODELO, PipelineFraude


class Transaccion(BaseModel):
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
    fecha: datetime
    monto: float = Field(gt=0)
    score: int = Field(ge=0, le=100)


class Prediccion(BaseModel):
    """Resultado del scoring de una transacción."""

    probabilidad_fraude: float
    decision: Literal['aprobar', 'rechazar']
    umbral: float


@asynccontextmanager
async def ciclo_de_vida(aplicacion):
    """Carga el pipeline una sola vez al iniciar la API, junto con la versión registrada si viene de MLflow."""
    ruta = Path(os.environ.get('RUTA_MODELO', RUTA_MODELO))
    aplicacion.state.pipeline = PipelineFraude.cargar(ruta)
    ruta_metadata = ruta.with_suffix('.json')
    # Un pipeline entrenado localmente, sin pasar por el registry, no tiene metadata ni versión
    metadata = json.loads(ruta_metadata.read_text(encoding='utf-8')) if ruta_metadata.exists() else {}
    aplicacion.state.version_modelo = metadata.get('version')
    yield


app = FastAPI(title='Prevención de fraude', lifespan=ciclo_de_vida)


@app.get('/salud')
def salud(request: Request):
    """Confirma que la API está arriba y con qué modelo: versión del registry (si se conoce) y umbral."""
    return {
        'estado': 'ok',
        'version_modelo': getattr(request.app.state, 'version_modelo', None),
        'umbral': request.app.state.pipeline.umbral,
    }


@app.post('/predecir', response_model=Prediccion)
def predecir(transaccion: Transaccion, request: Request):
    """Probabilidad de fraude y decisión para una transacción."""
    pipeline = request.app.state.pipeline
    resultado = pipeline.predecir(pd.DataFrame([transaccion.model_dump()])).iloc[0]
    return Prediccion(
        probabilidad_fraude=resultado['probabilidad_fraude'], decision=resultado['decision'], umbral=pipeline.umbral
    )
