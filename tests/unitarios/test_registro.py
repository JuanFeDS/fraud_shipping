"""Integración con MLflow: experimento, datasets y model registry, siempre sobre una base temporal."""

import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient

from fraude_shipping.registro import (
    ALIAS_PRODUCCION,
    NOMBRE_MODELO_REGISTRADO,
    configurar_mlflow,
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
