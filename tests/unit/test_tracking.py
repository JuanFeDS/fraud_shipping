"""Registro de experimentos en MLflow, siempre sobre una base temporal."""

import mlflow
import pytest

from fraude_shipping.experimentation import tracking
from fraude_shipping.experimentation.candidate_features import build_features
from fraude_shipping.registry import setup_mlflow

FEATURES = ['a', 'j', 'monto', 'score', 'perfil_onp']
PARAMS = {'n_estimators': 10, 'num_leaves': 4, 'min_child_samples': 5}


@pytest.mark.usefixtures('temp_mlflow')
def test_run_experiment_logs_the_run(data):
    """El run queda con etiquetas, parámetros, métricas, dataset y la curva de ganancia."""
    setup_mlflow('prueba')
    result = tracking.run_experiment(
        build_features(data), 'run_prueba', FEATURES, 'lightgbm', PARAMS,
        rate_columns=['j'], tags={'etapa': 'test'}, description='Hipótesis y resultado del experimento',
    )
    run = mlflow.search_runs(filter_string="attributes.run_name = 'run_prueba'").iloc[0]
    assert run['tags.etapa'] == 'test'
    assert run['tags.mlflow.note.content'] == 'Hipótesis y resultado del experimento'
    assert run['params.numero_features'] == str(len(FEATURES))
    assert run['metrics.umbral'] == pytest.approx(result.threshold)
    artifacts = [artifact.path for artifact in mlflow.MlflowClient().list_artifacts(run['run_id'])]
    assert {'features.json', 'curva_ganancia.html'} <= set(artifacts)
    # j es feature y además se le calcula la tasa: con la columna repetida, registrar el dataset fallaría
    dataset = mlflow.get_run(run['run_id']).inputs.dataset_inputs[0].dataset
    assert dataset.name == 'features'
