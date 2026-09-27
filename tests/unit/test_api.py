"""API de scoring: endpoints, validación del input y carga del artefacto al iniciar."""

import json

import pytest
from fastapi.testclient import TestClient

from fraude_shipping.produccion.api import app


@pytest.fixture(name='cliente')
def fixture_cliente(pipeline, monkeypatch):
    """Cliente de la API con el pipeline sintético inyectado y sin API key, salvo que el test la configure."""
    monkeypatch.delenv('FRAUDE_API_KEY', raising=False)
    app.state.pipeline = pipeline
    app.state.version_modelo = None
    return TestClient(app)


@pytest.fixture(name='transaccion')
def fixture_transaccion(datos_nuevos):
    """Una transacción válida como la enviaría un cliente: nulos como null y fecha en ISO 8601."""
    return json.loads(datos_nuevos.iloc[0].to_json(date_format='iso'))


def test_salud(cliente, pipeline):
    """El endpoint de salud confirma el modelo cargado y su umbral."""
    respuesta = cliente.get('/salud')
    assert respuesta.status_code == 200
    assert respuesta.json() == {'estado': 'ok', 'version_modelo': None, 'umbral': pipeline.umbral}


def test_predecir_devuelve_lo_mismo_que_el_pipeline(cliente, pipeline, datos_nuevos, transaccion):
    """La respuesta coincide con la predicción del pipeline para la misma transacción."""
    esperado = pipeline.predecir(datos_nuevos.head(1)).iloc[0]
    respuesta = cliente.post('/predecir', json=transaccion)
    assert respuesta.status_code == 200
    assert respuesta.json()['probabilidad_fraude'] == pytest.approx(esperado['probabilidad_fraude'])
    assert respuesta.json()['decision'] == esperado['decision']
    assert respuesta.json()['umbral'] == pipeline.umbral


def test_predecir_acepta_campos_opcionales_ausentes(cliente, transaccion):
    """Las variables que admiten nulo pueden no enviarse."""
    obligatoria = {campo: valor for campo, valor in transaccion.items() if campo not in ('b', 'c', 'd', 'f', 'g', 'l', 'm', 'o')}
    assert cliente.post('/predecir', json=obligatoria).status_code == 200


@pytest.mark.parametrize('cambio', [
    {'score': 150},
    {'score': -1},
    {'n': 2},
    {'o': 'X'},
    {'p': None},
    {'monto': 0},
    {'fecha': 'no es una fecha'},
])
def test_input_invalido_se_rechaza(cliente, transaccion, cambio):
    """Un valor fuera de dominio se rechaza con 422 antes de llegar al modelo."""
    assert cliente.post('/predecir', json={**transaccion, **cambio}).status_code == 422


@pytest.mark.parametrize('campo', ['j', 'score', 'monto', 'fecha'])
def test_campo_obligatorio_faltante_se_rechaza(cliente, transaccion, campo):
    """Sin una variable obligatoria no se puede decidir."""
    incompleta = {clave: valor for clave, valor in transaccion.items() if clave != campo}
    assert cliente.post('/predecir', json=incompleta).status_code == 422


@pytest.mark.parametrize(('metadata', 'version_esperada'), [(None, None), ({'version': '3'}, '3')])
def test_al_iniciar_carga_el_artefacto_indicado(tmp_path, monkeypatch, pipeline, metadata, version_esperada):
    """La API lee el pipeline de RUTA_MODELO y, si hay metadata del registry al lado, informa su versión."""
    ruta = tmp_path / 'pipeline.joblib'
    pipeline.guardar(ruta)
    if metadata is not None:
        ruta.with_suffix('.json').write_text(json.dumps(metadata), encoding='utf-8')
    monkeypatch.setenv('RUTA_MODELO', str(ruta))
    app.state.pipeline = None
    with TestClient(app) as cliente:
        salud = cliente.get('/salud').json()
    assert salud['umbral'] == pipeline.umbral
    assert salud['version_modelo'] == version_esperada


@pytest.mark.parametrize(('encabezados', 'codigo_esperado'), [
    ({}, 401),
    ({'X-API-Key': 'otra-clave'}, 401),
    ({'X-API-Key': 'clave-correcta'}, 200),
])
def test_api_key_requerida_si_esta_configurada(cliente, transaccion, monkeypatch, encabezados, codigo_esperado):
    """Con FRAUDE_API_KEY definida, /predecir solo responde a quien envía esa key en el header X-API-Key."""
    monkeypatch.setenv('FRAUDE_API_KEY', 'clave-correcta')
    assert cliente.post('/predecir', json=transaccion, headers=encabezados).status_code == codigo_esperado


def test_salud_y_documentacion_no_requieren_api_key(cliente, monkeypatch):
    """/salud y /docs quedan abiertos para poder verificar el servicio y probarlo desde el navegador."""
    monkeypatch.setenv('FRAUDE_API_KEY', 'clave-correcta')
    assert cliente.get('/salud').status_code == 200
    assert cliente.get('/docs').status_code == 200
    assert 'X-API-Key' in cliente.get('/openapi.json').text
