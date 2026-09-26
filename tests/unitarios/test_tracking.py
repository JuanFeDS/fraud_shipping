"""Registro de experimentos en MLflow, siempre sobre una base temporal."""

import mlflow
import pytest

from fraude_shipping.experimentacion import tracking
from fraude_shipping.features import construir_features
from fraude_shipping.registro import configurar_mlflow

FEATURES = ['a', 'j', 'monto', 'score', 'perfil_onp']
PARAMETROS = {'n_estimators': 10, 'num_leaves': 4, 'min_child_samples': 5}


@pytest.mark.usefixtures('mlflow_temporal')
def test_ejecutar_experimento_registra_el_run(datos):
    """El run queda con etiquetas, parámetros, métricas, dataset y la curva de ganancia."""
    configurar_mlflow('prueba')
    resultado = tracking.ejecutar_experimento(
        construir_features(datos), 'run_prueba', FEATURES, 'lightgbm', PARAMETROS,
        columnas_tasa=['j'], etiquetas={'etapa': 'test'}, descripcion='Hipótesis y resultado del experimento',
    )
    run = mlflow.search_runs(filter_string="attributes.run_name = 'run_prueba'").iloc[0]
    assert run['tags.etapa'] == 'test'
    assert run['tags.mlflow.note.content'] == 'Hipótesis y resultado del experimento'
    assert run['params.numero_features'] == str(len(FEATURES))
    assert run['metrics.umbral'] == pytest.approx(resultado.umbral)
    artefactos = [artefacto.path for artefacto in mlflow.MlflowClient().list_artifacts(run['run_id'])]
    assert {'features.json', 'curva_ganancia.html'} <= set(artefactos)
    # j es feature y además se le calcula la tasa: con la columna repetida, registrar el dataset fallaría
    dataset = mlflow.get_run(run['run_id']).inputs.dataset_inputs[0].dataset
    assert dataset.name == 'features'
