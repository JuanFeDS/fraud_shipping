"""API de scoring: endpoints, validación del input y carga del artefacto al iniciar."""

import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from fraude_shipping.production.api import app

API_KEY = 'clave-correcta'


@pytest.fixture(name='client')
def fixture_client(pipeline, monkeypatch):
    """Cliente de la API con el pipeline sintético inyectado y la API key configurada, que envía en cada request."""
    monkeypatch.setenv('FRAUDE_API_KEY', API_KEY)
    app.state.pipeline = pipeline
    app.state.model_version = None
    return TestClient(app, headers={'X-API-Key': API_KEY})


@pytest.fixture(name='transaction')
def fixture_transaction(new_data):
    """Una transacción válida como la enviaría un cliente: nulos como null y fecha en ISO 8601 con zona horaria."""
    transaction = json.loads(new_data.iloc[0].to_json(date_format='iso'))
    transaction['fecha'] += '-03:00'
    return transaction


def test_health(client, pipeline):
    """El endpoint de salud confirma el modelo cargado y su umbral."""
    response = client.get('/salud')
    assert response.status_code == 200
    assert response.json() == {'estado': 'ok', 'version_modelo': None, 'umbral': pipeline.threshold}


def test_predict_matches_pipeline(client, pipeline, new_data, transaction):
    """La respuesta coincide con la predicción del pipeline para la misma transacción."""
    expected = pipeline.predict(new_data.head(1)).iloc[0]
    response = client.post('/predecir', json=transaction)
    assert response.status_code == 200
    assert response.json()['probabilidad_fraude'] == pytest.approx(expected['probabilidad_fraude'])
    assert response.json()['decision'] == expected['decision']
    assert response.json()['umbral'] == pipeline.threshold


def test_predict_accepts_missing_optional_fields(client, transaction):
    """Las variables que admiten nulo pueden no enviarse."""
    required = {field: value for field, value in transaction.items() if field not in ('b', 'c', 'd', 'f', 'g', 'l', 'm', 'o')}
    assert client.post('/predecir', json=required).status_code == 200


@pytest.mark.parametrize('change', [
    {'score': 150},
    {'score': -1},
    {'n': 2},
    {'o': 'X'},
    {'p': None},
    {'monto': 0},
    {'fecha': 'no es una fecha'},
])
def test_invalid_input_is_rejected(client, transaction, change):
    """Un valor fuera de dominio se rechaza con 422 antes de llegar al modelo."""
    assert client.post('/predecir', json={**transaction, **change}).status_code == 422


def test_fecha_requires_timezone(client, transaction):
    """Una fecha sin zona horaria es ambigua entre países: se rechaza antes de calcular la hora del día."""
    naive = {**transaction, 'fecha': transaction['fecha'].removesuffix('-03:00')}
    assert client.post('/predecir', json=naive).status_code == 422


def test_same_instant_in_any_timezone_gives_same_prediction(client, transaction):
    """La fecha se convierte a la zona del dataset: el mismo instante en UTC da la misma probabilidad."""
    in_utc = pd.Timestamp(transaction['fecha']).tz_convert('UTC').isoformat()
    local = client.post('/predecir', json=transaction).json()
    utc = client.post('/predecir', json={**transaction, 'fecha': in_utc}).json()
    assert utc == local


@pytest.mark.parametrize('field', ['j', 'score', 'monto', 'fecha'])
def test_missing_required_field_is_rejected(client, transaction, field):
    """Sin una variable obligatoria no se puede decidir."""
    incomplete = {key: value for key, value in transaction.items() if key != field}
    assert client.post('/predecir', json=incomplete).status_code == 422


@pytest.mark.parametrize(('metadata', 'expected_version'), [(None, None), ({'version': '3'}, '3')])
def test_startup_loads_configured_artifact(tmp_path, monkeypatch, pipeline, metadata, expected_version):
    """La API lee el pipeline de MODEL_PATH y, si hay metadata del registry al lado, informa su versión."""
    path = tmp_path / 'pipeline.joblib'
    pipeline.save(path)
    if metadata is not None:
        path.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
    monkeypatch.setenv('MODEL_PATH', str(path))
    monkeypatch.setenv('FRAUDE_API_KEY', API_KEY)
    app.state.pipeline = None
    with TestClient(app) as client:
        health = client.get('/salud').json()
    assert health['umbral'] == pipeline.threshold
    assert health['version_modelo'] == expected_version


@pytest.mark.parametrize(('headers', 'expected_status'), [
    ({}, 401),
    ({'X-API-Key': 'otra-clave'}, 401),
    ({'X-API-Key': API_KEY}, 200),
])
@pytest.mark.usefixtures('client')
def test_api_key_required(transaction, headers, expected_status):
    """/predecir solo responde a quien envía la key de FRAUDE_API_KEY en el header X-API-Key."""
    assert TestClient(app).post('/predecir', json=transaction, headers=headers).status_code == expected_status


def test_predict_rejected_if_api_key_is_missing(client, transaction, monkeypatch):
    """Si FRAUDE_API_KEY desaparece con la API arriba, /predecir rechaza todo en lugar de quedar abierto."""
    monkeypatch.delenv('FRAUDE_API_KEY')
    assert client.post('/predecir', json=transaction).status_code == 401


def test_startup_fails_without_api_key(monkeypatch):
    """Sin FRAUDE_API_KEY la API no arranca."""
    monkeypatch.delenv('FRAUDE_API_KEY', raising=False)
    with pytest.raises(RuntimeError, match='FRAUDE_API_KEY'), TestClient(app):
        pass


@pytest.mark.usefixtures('client')
def test_health_and_docs_do_not_require_api_key():
    """/salud y /docs quedan abiertos para poder verificar el servicio y probarlo desde el navegador."""
    anonymous = TestClient(app)
    assert anonymous.get('/salud').status_code == 200
    assert anonymous.get('/docs').status_code == 200
    assert 'X-API-Key' in anonymous.get('/openapi.json').text
