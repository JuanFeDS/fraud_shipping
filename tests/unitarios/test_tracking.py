"""Registro de experimentos en MLflow, siempre sobre una base temporal."""

import mlflow
import pytest

from fraude_shipping.experimentacion import tracking
from fraude_shipping.features import construir_features

FEATURES = ['a', 'monto', 'score', 'perfil_onp']
PARAMETROS = {'n_estimators': 10, 'num_leaves': 4, 'min_child_samples': 5}


@pytest.fixture(name='raiz_temporal')
def fixture_raiz_temporal(tmp_path, monkeypatch):
    """Redirige la base y los artefactos de MLflow a una carpeta temporal para no tocar el mlflow.db real."""
    monkeypatch.setattr(tracking, 'RAIZ_PROYECTO', tmp_path)
    return tmp_path


def test_configurar_mlflow_crea_y_reutiliza_el_experimento(raiz_temporal):
    """La primera vez crea la base y el experimento; la segunda reutiliza el existente."""
    tracking.configurar_mlflow('prueba')
    experimento = mlflow.get_experiment_by_name('prueba')
    tracking.configurar_mlflow('prueba')
    assert (raiz_temporal / 'mlflow.db').exists()
    assert mlflow.get_experiment_by_name('prueba').experiment_id == experimento.experiment_id


@pytest.mark.usefixtures('raiz_temporal')
def test_ejecutar_experimento_registra_el_run(datos):
    """El run queda con etiquetas, parámetros, métricas y la curva de ganancia."""
    tracking.configurar_mlflow('prueba')
    resultado = tracking.ejecutar_experimento(
        construir_features(datos), 'run_prueba', FEATURES, 'lightgbm', PARAMETROS, etiquetas={'etapa': 'test'}
    )
    run = mlflow.search_runs(filter_string="attributes.run_name = 'run_prueba'").iloc[0]
    assert run['tags.etapa'] == 'test'
    assert run['params.numero_features'] == str(len(FEATURES))
    assert run['metrics.umbral'] == pytest.approx(resultado.umbral)
    artefactos = [artefacto.path for artefacto in mlflow.MlflowClient().list_artifacts(run['run_id'])]
    assert {'features.json', 'curva_ganancia.html'} <= set(artefactos)
