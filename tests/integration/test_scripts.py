"""Flujo completo por línea de comandos: entrenar, predecir en batch y validar, sobre archivos temporales."""

import runpy
import sys
from pathlib import Path

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from fraude_shipping.registry import PRODUCTION_ALIAS, REGISTERED_MODEL_NAME

SCRIPTS_DIR = Path(__file__).resolve().parents[2] / 'scripts'


def _run_script(monkeypatch, name, *args):
    """Corre un script como si se llamara desde la terminal con los argumentos indicados."""
    monkeypatch.setattr(sys, 'argv', [name, *map(str, args)])
    runpy.run_path(str(SCRIPTS_DIR / name), run_name='__main__')


@pytest.fixture(name='paths')
def fixture_paths(tmp_path, data):
    """CSV etiquetado, CSV de transacciones nuevas y rutas de salida dentro de una carpeta temporal."""
    data.to_csv(tmp_path / 'labeled.csv', index=False)
    data.drop(columns='fraude').to_csv(tmp_path / 'new.csv', index=False)
    return {
        'labeled': tmp_path / 'labeled.csv',
        'new': tmp_path / 'new.csv',
        'model': tmp_path / 'models' / 'pipeline.joblib',
        'predictions': tmp_path / 'output' / 'predictions.csv',
    }


def test_train_and_predict(monkeypatch, capsys, paths):
    """train.py guarda el artefacto y predict.py agrega probabilidad y decisión a cada transacción."""
    _run_script(monkeypatch, 'train.py', '--data', paths['labeled'], '--model', paths['model'])
    assert paths['model'].exists()

    _run_script(
        monkeypatch, 'predict.py',
        '--input', paths['new'], '--output', paths['predictions'], '--model', paths['model'],
    )
    predictions = pd.read_csv(paths['predictions'])
    assert len(predictions) == len(pd.read_csv(paths['new']))
    assert predictions['probabilidad_fraude'].between(0, 1).all()
    assert set(predictions['decision']) <= {'aprobar', 'rechazar'}
    assert 'transacciones evaluadas' in capsys.readouterr().out


def test_validate_pipeline(monkeypatch, capsys, paths):
    """validate_pipeline.py imprime las métricas de cada fold y el resumen."""
    _run_script(monkeypatch, 'validate_pipeline.py', '--data', paths['labeled'])
    output = capsys.readouterr().out
    assert 'Fold 5/5 listo' in output
    assert 'ganancia_pct_maxima_media' in output


@pytest.mark.usefixtures('temp_mlflow')
def test_train_and_validate_log_to_mlflow(monkeypatch, capsys, paths):
    """Con --mlflow, validate registra su run y train publica el pipeline en el registry con el alias de producción."""
    _run_script(monkeypatch, 'validate_pipeline.py', '--data', paths['labeled'], '--mlflow')
    _run_script(
        monkeypatch, 'train.py', '--data', paths['labeled'], '--model', paths['model'], '--mlflow',
    )
    assert 'con alias "champion"' in capsys.readouterr().out

    validation = mlflow.search_runs(filter_string="attributes.run_name = 'validacion_pipeline'").iloc[0]
    assert 'metrics.ganancia_pct_maxima_media' in validation
    artifacts = [artifact.path for artifact in MlflowClient().list_artifacts(validation['run_id'])]
    assert 'metricas_por_fold.json' in artifacts

    assert validation['tags.decision'] == 'elegido'
    assert 'Validación del pipeline productivo' in validation['tags.mlflow.note.content']

    version = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
    assert version.tags['run_validacion'] == validation['run_id']
    training = mlflow.get_run(version.run_id)
    assert training.data.tags['decision'] == 'produccion'
    assert training.info.run_name == 'entrenamiento_pipeline'
    assert training.inputs.dataset_inputs[0].dataset.name == 'dataset'

    downloaded = paths['model'].parent / 'downloaded.joblib'
    _run_script(monkeypatch, 'download_model.py', '--model', downloaded)
    assert downloaded.exists()
    assert downloaded.with_suffix('.json').exists()
    assert 'v1' in capsys.readouterr().out


@pytest.mark.usefixtures('temp_mlflow')
def test_train_without_validation_is_not_promoted(monkeypatch, capsys, paths):
    """Sin una validación del mismo modelo y datos, train.py registra la versión pero no la pasa a producción."""
    _run_script(monkeypatch, 'train.py', '--data', paths['labeled'], '--model', paths['model'], '--mlflow')
    assert 'sin alias: no supera el gate de promoción' in capsys.readouterr().out
    assert not MlflowClient().get_registered_model(REGISTERED_MODEL_NAME).aliases
