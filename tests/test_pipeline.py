"""Consistencia del pipeline productivo entre scoring batch, fila a fila y API."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraude_shipping.features import cargar_datos
from fraude_shipping.produccion.api import app
from fraude_shipping.produccion.pipeline import PARAMETROS_LIGHTGBM, PipelineFraude

RUTA_DATOS = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'dataset.csv'
TAMANO_TRAIN = 20_000
TAMANO_PRUEBA = 200


@pytest.fixture(scope='module', name='datos_prueba')
def fixture_datos_prueba():
    """Transacciones que el pipeline no vio al entrenar, con nulos en varias columnas."""
    datos = cargar_datos(RUTA_DATOS).sample(TAMANO_TRAIN + TAMANO_PRUEBA, random_state=0)
    return datos.iloc[:TAMANO_TRAIN], datos.iloc[TAMANO_TRAIN:].drop(columns='fraude')


@pytest.fixture(scope='module', name='pipeline')
def fixture_pipeline(datos_prueba):
    """Pipeline chico (menos árboles y menos datos) para que los tests corran en segundos."""
    train, _ = datos_prueba
    return PipelineFraude(parametros={**PARAMETROS_LIGHTGBM, 'n_estimators': 50}).ajustar(train)


@pytest.fixture(scope='module', name='cliente')
def fixture_cliente(pipeline):
    """Cliente de la API con el pipeline de prueba inyectado en lugar del artefacto en disco."""
    app.state.pipeline = pipeline
    return TestClient(app)


def _a_json(fila):
    """Transacción como la enviaría un cliente de la API: nulos como None y fecha en ISO 8601."""
    return json.loads(fila.to_json(date_format='iso'))


def test_fila_a_fila_igual_que_batch(pipeline, datos_prueba):
    """Predecir de a una transacción da lo mismo que predecir el lote completo."""
    _, prueba = datos_prueba
    batch = pipeline.predecir_probabilidad(prueba)
    fila_a_fila = [pipeline.predecir_probabilidad(prueba.iloc[[posicion]])[0] for posicion in range(len(prueba))]
    np.testing.assert_allclose(fila_a_fila, batch)


def test_api_igual_que_batch(pipeline, cliente, datos_prueba):
    """La API devuelve la misma probabilidad y decisión que el scoring batch."""
    _, prueba = datos_prueba
    muestra = prueba.head(20)
    esperado = pipeline.predecir(muestra)
    for (_, fila), (_, prediccion) in zip(muestra.iterrows(), esperado.iterrows()):
        respuesta = cliente.post('/predecir', json=_a_json(fila))
        assert respuesta.status_code == 200
        assert respuesta.json()['probabilidad_fraude'] == pytest.approx(prediccion['probabilidad_fraude'])
        assert respuesta.json()['decision'] == prediccion['decision']


def test_categorias_no_vistas(pipeline, datos_prueba):
    """Una categoría de j y un país que no estaban en train usan los valores por defecto."""
    _, prueba = datos_prueba
    nueva = prueba.head(1).assign(j='cat_nueva', g='ZZ')
    features = pipeline.transformar(nueva).iloc[0]
    assert features['j_tasa_fraude'] == pytest.approx(pipeline.tasa_fraude_global)
    assert features['j_frecuencia'] == 0
    assert features['g_agrupado'] == 'Otros'
    assert 0 <= pipeline.predecir_probabilidad(nueva)[0] <= 1


def test_api_rechaza_input_invalido(cliente, datos_prueba):
    """Un score fuera de rango se rechaza en la validación del input, antes de llegar al modelo."""
    _, prueba = datos_prueba
    transaccion = {**_a_json(prueba.iloc[0]), 'score': 150}
    assert cliente.post('/predecir', json=transaccion).status_code == 422
