"""Flujo completo por línea de comandos: entrenar, predecir en batch y validar, sobre archivos temporales."""

import runpy
import sys
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from fraude_shipping.registro import ALIAS_PRODUCCION, NOMBRE_MODELO_REGISTRADO

CARPETA_SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'


def _ejecutar_script(monkeypatch, nombre, *argumentos):
    """Corre un script como si se llamara desde la terminal con los argumentos indicados."""
    monkeypatch.setattr(sys, 'argv', [nombre, *map(str, argumentos)])
    runpy.run_path(str(CARPETA_SCRIPTS / nombre), run_name='__main__')


@pytest.fixture(name='rutas')
def fixture_rutas(tmp_path, datos):
    """CSV etiquetado, CSV de transacciones nuevas y rutas de salida dentro de una carpeta temporal."""
    datos.to_csv(tmp_path / 'etiquetados.csv', index=False)
    datos.drop(columns='fraude').to_csv(tmp_path / 'nuevas.csv', index=False)
    return {
        'etiquetados': tmp_path / 'etiquetados.csv',
        'nuevas': tmp_path / 'nuevas.csv',
        'modelo': tmp_path / 'models' / 'pipeline.joblib',
        'predicciones': tmp_path / 'salida' / 'predicciones.csv',
    }


def test_entrenar_y_predecir(monkeypatch, capsys, rutas):
    """entrenar.py guarda el artefacto y predecir.py agrega probabilidad y decisión a cada transacción."""
    _ejecutar_script(monkeypatch, 'entrenar.py', '--datos', rutas['etiquetados'], '--modelo', rutas['modelo'])
    assert rutas['modelo'].exists()

    _ejecutar_script(
        monkeypatch, 'predecir.py',
        '--entrada', rutas['nuevas'], '--salida', rutas['predicciones'], '--modelo', rutas['modelo'],
    )
    predicciones = pd.read_csv(rutas['predicciones'])
    assert len(predicciones) == len(pd.read_csv(rutas['nuevas']))
    assert predicciones['probabilidad_fraude'].between(0, 1).all()
    assert set(predicciones['decision']) <= {'aprobar', 'rechazar'}
    assert 'transacciones evaluadas' in capsys.readouterr().out


def test_validar_pipeline(monkeypatch, capsys, rutas):
    """validar_pipeline.py imprime las métricas de cada fold y el resumen."""
    _ejecutar_script(monkeypatch, 'validar_pipeline.py', '--datos', rutas['etiquetados'])
    salida = capsys.readouterr().out
    assert 'Fold 5/5 listo' in salida
    assert 'ganancia_pct_maxima_media' in salida


@pytest.mark.usefixtures('mlflow_temporal')
def test_entrenar_y_validar_registran_en_mlflow(monkeypatch, capsys, rutas):
    """Con --mlflow, validar registra su run y entrenar publica el pipeline en el registry con el alias de producción."""
    _ejecutar_script(monkeypatch, 'validar_pipeline.py', '--datos', rutas['etiquetados'], '--mlflow')
    _ejecutar_script(
        monkeypatch, 'entrenar.py', '--datos', rutas['etiquetados'], '--modelo', rutas['modelo'], '--mlflow',
    )
    assert 'con alias "champion"' in capsys.readouterr().out

    validacion = mlflow.search_runs(filter_string="attributes.run_name = 'validacion_pipeline'").iloc[0]
    assert 'metrics.ganancia_pct_maxima_media' in validacion
    artefactos = [artefacto.path for artefacto in MlflowClient().list_artifacts(validacion['run_id'])]
    assert 'metricas_por_fold.json' in artefactos

    version = MlflowClient().get_model_version_by_alias(NOMBRE_MODELO_REGISTRADO, ALIAS_PRODUCCION)
    entrenamiento = mlflow.get_run(version.run_id)
    assert entrenamiento.info.run_name == 'entrenamiento_pipeline'
    assert entrenamiento.inputs.dataset_inputs[0].dataset.name == 'dataset'
