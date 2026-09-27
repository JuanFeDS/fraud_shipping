"""Consistencia entre scoring batch, fila a fila y API sobre una muestra del dataset real."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraude_shipping.features import cargar_datos
from fraude_shipping.produccion.api import app
from fraude_shipping.produccion.pipeline import PARAMETROS_LIGHTGBM, PipelineFraude

RUTA_DATOS = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'dataset.csv'
TAMANO_TRAIN = 20_000
TAMANO_PRUEBA = 200


@pytest.fixture(scope='module', name='datos_reales')
def fixture_datos_reales():
    """Muestra del dataset real: train y transacciones no vistas, con los nulos y categorías reales."""
    datos = cargar_datos(RUTA_DATOS).sample(TAMANO_TRAIN + TAMANO_PRUEBA, random_state=0)
    return datos.iloc[:TAMANO_TRAIN], datos.iloc[TAMANO_TRAIN:].drop(columns='fraude')


@pytest.fixture(scope='module', name='pipeline_real')
def fixture_pipeline_real(datos_reales):
    """Pipeline con los hiperparámetros de producción y menos árboles, para que el test corra en segundos."""
    train, _ = datos_reales
    return PipelineFraude(parametros={**PARAMETROS_LIGHTGBM, 'n_estimators': 50}).ajustar(train)


def test_fila_a_fila_igual_que_batch(pipeline_real, datos_reales):
    """Predecir de a una transacción da lo mismo que predecir el lote completo."""
    _, prueba = datos_reales
    batch = pipeline_real.predecir_probabilidad(prueba)
    fila_a_fila = [pipeline_real.predecir_probabilidad(prueba.iloc[[posicion]])[0] for posicion in range(len(prueba))]
    np.testing.assert_allclose(fila_a_fila, batch)


def test_api_igual_que_batch(pipeline_real, datos_reales):
    """La API devuelve la misma probabilidad y decisión que el scoring batch."""
    _, prueba = datos_reales
    app.state.pipeline = pipeline_real
    cliente = TestClient(app)
    muestra = prueba.head(20)
    esperado = pipeline_real.predecir(muestra)
    for (_, fila), (_, prediccion) in zip(muestra.iterrows(), esperado.iterrows()):
        respuesta = cliente.post('/predecir', json=json.loads(fila.to_json(date_format='iso')))
        assert respuesta.status_code == 200
        assert respuesta.json()['probabilidad_fraude'] == pytest.approx(prediccion['probabilidad_fraude'])
        assert respuesta.json()['decision'] == prediccion['decision']
