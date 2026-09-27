"""Integración con MLflow: experimento, datasets y model registry, siempre sobre una base temporal."""

import json

import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient

from fraude_shipping.produccion.pipeline import PipelineFraude
from fraude_shipping.registro import (
    ALIAS_PRODUCCION,
    NOMBRE_MODELO_REGISTRADO,
    NOMBRE_RUN_VALIDACION,
    buscar_ultima_validacion,
    configurar_mlflow,
    descargar_pipeline,
    formatear_numero,
    registrar_dataset,
    registrar_pipeline,
)

URI_PRODUCCION = f'models:/{NOMBRE_MODELO_REGISTRADO}@{ALIAS_PRODUCCION}'


def test_configurar_mlflow_crea_y_reutiliza_el_experimento(mlflow_temporal):
    """La primera vez crea la base y el experimento; la segunda reutiliza el existente."""
    configurar_mlflow('prueba')
    experimento = mlflow.get_experiment_by_name('prueba')
    configurar_mlflow('prueba')
    assert (mlflow_temporal / 'mlflow.db').exists()
    assert mlflow.get_experiment_by_name('prueba').experiment_id == experimento.experiment_id


def test_configurar_mlflow_usa_el_servidor_de_la_variable_de_entorno(mlflow_temporal, monkeypatch):
    """Con MLFLOW_TRACKING_URI definida se usa ese servidor y el experimento no fija una carpeta local de artefactos."""
    uri_servidor = f'sqlite:///{(mlflow_temporal / "servidor.db").as_posix()}'
    monkeypatch.setenv('MLFLOW_TRACKING_URI', uri_servidor)
    monkeypatch.chdir(mlflow_temporal)
    configurar_mlflow('prueba')
    assert mlflow.get_tracking_uri() == uri_servidor
    assert not (mlflow_temporal / 'mlflow.db').exists()
    experimento = mlflow.get_experiment_by_name('prueba')
    # Sin ubicación explícita, el servidor usa su raíz de artefactos más el id del experimento
    assert experimento.artifact_location.endswith(f'/{experimento.experiment_id}')


@pytest.mark.usefixtures('mlflow_temporal')
def test_registrar_dataset_con_fuente_y_etiqueta(datos):
    """El run queda asociado al dataset, con su fuente, su contexto y la columna objetivo."""
    configurar_mlflow('prueba')
    with mlflow.start_run() as run:
        registrar_dataset(datos, 'dataset', 'training', fuente='data/raw/dataset.csv')
    entrada = mlflow.get_run(run.info.run_id).inputs.dataset_inputs[0]
    assert entrada.dataset.name == 'dataset'
    assert 'data/raw/dataset.csv' in entrada.dataset.source
    assert entrada.tags[0].value == 'training'


@pytest.mark.usefixtures('mlflow_temporal')
def test_registrar_pipeline_versiona_y_mueve_el_alias(tmp_path, pipeline, datos_train, datos_nuevos):
    """Cada registro crea una versión nueva, el alias apunta a la última y el modelo predice igual que el pipeline."""
    ruta = tmp_path / 'pipeline.joblib'
    pipeline.guardar(ruta)
    configurar_mlflow('prueba')
    with mlflow.start_run():
        primera = registrar_pipeline(ruta, datos_train)
    with mlflow.start_run():
        segunda = registrar_pipeline(ruta, datos_train)

    assert (int(primera), int(segunda)) == (1, 2)
    version_champion = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO_REGISTRADO, ALIAS_PRODUCCION)
    assert int(version_champion.version) == int(segunda)

    prediccion = mlflow.pyfunc.load_model(URI_PRODUCCION).predict(datos_nuevos)
    esperado = pipeline.predecir(datos_nuevos)
    np.testing.assert_allclose(prediccion['probabilidad_fraude'], esperado['probabilidad_fraude'])
    assert (prediccion['decision'] == esperado['decision']).all()


@pytest.mark.usefixtures('mlflow_temporal')
def test_modelo_registrado_acepta_nulos_en_columnas_opcionales(tmp_path, pipeline, datos_train, datos_nuevos):
    """La firma se infiere con todo train, así que las columnas que admiten nulos no se exigen al servir."""
    ruta = tmp_path / 'pipeline.joblib'
    pipeline.guardar(ruta)
    configurar_mlflow('prueba')
    with mlflow.start_run():
        registrar_pipeline(ruta, datos_train)
    con_nulos = datos_nuevos.head(3).copy()
    con_nulos.loc[:, ['b', 'c']] = np.nan
    con_nulos.loc[:, ['g', 'o']] = None
    prediccion = mlflow.pyfunc.load_model(URI_PRODUCCION).predict(con_nulos)
    assert prediccion['probabilidad_fraude'].between(0, 1).all()


@pytest.mark.usefixtures('mlflow_temporal')
def test_metodo_predict_proba_devuelve_solo_la_probabilidad(tmp_path, pipeline, datos_train, datos_nuevos):
    """Con params={'metodo': 'predict_proba'} sale solo la probabilidad; un método desconocido es un error."""
    ruta = tmp_path / 'pipeline.joblib'
    pipeline.guardar(ruta)
    configurar_mlflow('prueba')
    with mlflow.start_run():
        registrar_pipeline(ruta, datos_train)
    modelo = mlflow.pyfunc.load_model(URI_PRODUCCION)

    probabilidad = modelo.predict(datos_nuevos, params={'metodo': 'predict_proba'})
    assert list(probabilidad.columns) == ['probabilidad_fraude']
    np.testing.assert_allclose(probabilidad['probabilidad_fraude'], pipeline.predecir_probabilidad(datos_nuevos))

    with pytest.raises(Exception, match='metodo debe ser uno de'):
        modelo.predict(datos_nuevos, params={'metodo': 'decision_function'})


@pytest.mark.usefixtures('mlflow_temporal')
def test_descargar_pipeline_por_alias_o_numero(tmp_path, pipeline, datos_train, datos_nuevos):
    """Descarga la versión pedida con su metadata, y el pipeline descargado predice igual que el original."""
    ruta = tmp_path / 'pipeline.joblib'
    pipeline.guardar(ruta)
    configurar_mlflow('prueba')
    with mlflow.start_run():
        registrar_pipeline(ruta, datos_train)
    with mlflow.start_run():
        registrar_pipeline(ruta, datos_train)

    destino = tmp_path / 'descargado' / 'pipeline_fraude.joblib'
    metadata = descargar_pipeline(destino)
    assert metadata['version'] == '2'
    assert metadata['alias'] == [ALIAS_PRODUCCION]
    assert json.loads(destino.with_suffix('.json').read_text(encoding='utf-8')) == metadata
    np.testing.assert_array_equal(
        PipelineFraude.cargar(destino).predecir_probabilidad(datos_nuevos), pipeline.predecir_probabilidad(datos_nuevos)
    )
    assert descargar_pipeline(destino, '1')['version'] == '1'


@pytest.mark.usefixtures('mlflow_temporal')
def test_version_registrada_queda_documentada(tmp_path, pipeline, datos_train):
    """Sin validación, la versión describe el entrenamiento; con validación, suma sus métricas y el enlace al run."""
    ruta = tmp_path / 'pipeline.joblib'
    pipeline.guardar(ruta)
    configurar_mlflow('prueba')
    assert buscar_ultima_validacion() is None

    with mlflow.start_run():
        sin_validacion = registrar_pipeline(ruta, datos_train)
    with mlflow.start_run(run_name=NOMBRE_RUN_VALIDACION):
        mlflow.log_metrics({
            'ganancia_pct_maxima_media': 78.88, 'ganancia_pct_maxima_desvio': 1.5, 'auc_roc_media': 0.89, 'auc_pr_media': 0.474,
        })
    validacion = buscar_ultima_validacion()
    with mlflow.start_run():
        con_validacion = registrar_pipeline(ruta, datos_train, validacion)

    cliente = MlflowClient()
    assert 'LightGBM' in cliente.get_registered_model(NOMBRE_MODELO_REGISTRADO).description
    primera = cliente.get_model_version(NOMBRE_MODELO_REGISTRADO, sin_validacion)
    assert f'{formatear_numero(len(datos_train))} transacciones' in primera.description
    assert primera.tags['umbral'] == str(pipeline.umbral)
    assert 'run_validacion' not in primera.tags
    segunda = cliente.get_model_version(NOMBRE_MODELO_REGISTRADO, con_validacion)
    assert '78,9%' in segunda.description
    assert segunda.tags['run_validacion'] == validacion.info.run_id
    assert segunda.tags['ganancia_pct_maxima'] == '78.88'


@pytest.mark.parametrize(('valor', 'decimales', 'esperado'), [
    (150000, 0, '150.000'), (78.88, 1, '78,9'), (0.15, 2, '0,15'), (1234567.891, 2, '1.234.567,89'), (0.890, 3, '0,890'),
])
def test_formatear_numero_en_espanol(valor, decimales, esperado):
    """Punto de miles y coma decimal, como se escriben los números en español."""
    assert formatear_numero(valor, decimales) == esperado
