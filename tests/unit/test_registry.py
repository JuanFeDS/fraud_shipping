"""Integración con MLflow: experimento, datasets y model registry, siempre sobre una base temporal."""

import json

import mlflow
import numpy as np
import pytest
from mlflow import MlflowClient

from fraude_shipping.production.pipeline import FraudPipeline
from fraude_shipping.registry import (
    PRODUCTION_ALIAS,
    REGISTERED_MODEL_NAME,
    VALIDATION_RUN_NAME,
    download_pipeline,
    find_validation,
    format_number,
    log_dataset,
    register_pipeline,
    setup_mlflow,
    should_promote,
)

PRODUCTION_URI = f'models:/{REGISTERED_MODEL_NAME}@{PRODUCTION_ALIAS}'
PARAMS = {'n_estimators': 50, 'umbral': 0.2}


def _log_validation(data, params, profit=78.88):
    """Run de validación como el de validate_pipeline.py: parámetros, dataset y métricas resumidas."""
    with mlflow.start_run(run_name=VALIDATION_RUN_NAME) as run:
        mlflow.log_params(params)
        log_dataset(data, 'dataset', 'training')
        mlflow.log_metrics({
            'ganancia_pct_maxima_media': profit, 'ganancia_pct_maxima_desvio': 1.3, 'auc_roc_media': 0.889,
            'auc_pr_media': 0.474,
        })
    return run.info.run_id


def test_setup_mlflow_creates_and_reuses_experiment(temp_mlflow):
    """La primera vez crea la base y el experimento; la segunda reutiliza el existente."""
    setup_mlflow('prueba')
    experiment = mlflow.get_experiment_by_name('prueba')
    setup_mlflow('prueba')
    assert (temp_mlflow / 'mlflow.db').exists()
    assert mlflow.get_experiment_by_name('prueba').experiment_id == experiment.experiment_id


def test_setup_mlflow_uses_server_from_environment(temp_mlflow, monkeypatch):
    """Con MLFLOW_TRACKING_URI definida se usa ese servidor y el experimento no fija una carpeta local de artefactos."""
    server_uri = f'sqlite:///{(temp_mlflow / "server.db").as_posix()}'
    monkeypatch.setenv('MLFLOW_TRACKING_URI', server_uri)
    monkeypatch.chdir(temp_mlflow)
    setup_mlflow('prueba')
    assert mlflow.get_tracking_uri() == server_uri
    assert not (temp_mlflow / 'mlflow.db').exists()
    experiment = mlflow.get_experiment_by_name('prueba')
    # Sin ubicación explícita, el servidor usa su raíz de artefactos más el id del experimento
    assert experiment.artifact_location.endswith(f'/{experiment.experiment_id}')


@pytest.mark.usefixtures('temp_mlflow')
def test_log_dataset_with_source_and_target(data):
    """El run queda asociado al dataset, con su fuente, su contexto y la columna objetivo."""
    setup_mlflow('prueba')
    with mlflow.start_run() as run:
        log_dataset(data, 'dataset', 'training', source='data/raw/dataset.csv')
    dataset_input = mlflow.get_run(run.info.run_id).inputs.dataset_inputs[0]
    assert dataset_input.dataset.name == 'dataset'
    assert 'data/raw/dataset.csv' in dataset_input.dataset.source
    assert dataset_input.tags[0].value == 'training'


@pytest.mark.usefixtures('temp_mlflow')
def test_register_pipeline_versions_and_moves_alias(tmp_path, pipeline, train_data, new_data):
    """Cada registro crea una versión nueva, el alias apunta a la última y el modelo predice igual que el pipeline."""
    path = tmp_path / 'pipeline.joblib'
    pipeline.save(path)
    setup_mlflow('prueba')
    with mlflow.start_run():
        first = register_pipeline(path, train_data)
    with mlflow.start_run():
        second = register_pipeline(path, train_data)

    assert (int(first), int(second)) == (1, 2)
    champion_version = MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS)
    assert int(champion_version.version) == int(second)

    prediction = mlflow.pyfunc.load_model(PRODUCTION_URI).predict(new_data)
    expected = pipeline.predict(new_data)
    np.testing.assert_allclose(prediction['probabilidad_fraude'], expected['probabilidad_fraude'])
    assert (prediction['decision'] == expected['decision']).all()


@pytest.mark.usefixtures('temp_mlflow')
def test_registered_model_accepts_nulls_in_optional_columns(tmp_path, pipeline, train_data, new_data):
    """La firma se infiere con todo train, así que las columnas que admiten nulos no se exigen al servir."""
    path = tmp_path / 'pipeline.joblib'
    pipeline.save(path)
    setup_mlflow('prueba')
    with mlflow.start_run():
        register_pipeline(path, train_data)
    with_nulls = new_data.head(3).copy()
    with_nulls.loc[:, ['b', 'c']] = np.nan
    with_nulls.loc[:, ['g', 'o']] = None
    prediction = mlflow.pyfunc.load_model(PRODUCTION_URI).predict(with_nulls)
    assert prediction['probabilidad_fraude'].between(0, 1).all()


@pytest.mark.usefixtures('temp_mlflow')
def test_predict_proba_method_returns_only_probability(tmp_path, pipeline, train_data, new_data):
    """Con params={'metodo': 'predict_proba'} sale solo la probabilidad; un método desconocido es un error."""
    path = tmp_path / 'pipeline.joblib'
    pipeline.save(path)
    setup_mlflow('prueba')
    with mlflow.start_run():
        register_pipeline(path, train_data)
    model = mlflow.pyfunc.load_model(PRODUCTION_URI)

    probability = model.predict(new_data, params={'metodo': 'predict_proba'})
    assert list(probability.columns) == ['probabilidad_fraude']
    np.testing.assert_allclose(probability['probabilidad_fraude'], pipeline.predict_proba(new_data))

    with pytest.raises(Exception, match='metodo debe ser uno de'):
        model.predict(new_data, params={'metodo': 'decision_function'})


@pytest.mark.usefixtures('temp_mlflow')
def test_download_pipeline_by_alias_or_number(tmp_path, pipeline, train_data, new_data):
    """Descarga la versión pedida con su metadata, y el pipeline descargado predice igual que el original."""
    path = tmp_path / 'pipeline.joblib'
    pipeline.save(path)
    setup_mlflow('prueba')
    with mlflow.start_run():
        register_pipeline(path, train_data)
    with mlflow.start_run():
        register_pipeline(path, train_data)

    destination = tmp_path / 'downloaded' / 'fraud_pipeline.joblib'
    metadata = download_pipeline(destination)
    assert metadata['version'] == '2'
    assert metadata['alias'] == [PRODUCTION_ALIAS]
    assert json.loads(destination.with_suffix('.json').read_text(encoding='utf-8')) == metadata
    np.testing.assert_array_equal(
        FraudPipeline.load(destination).predict_proba(new_data), pipeline.predict_proba(new_data)
    )
    assert download_pipeline(destination, '1')['version'] == '1'


@pytest.mark.usefixtures('temp_mlflow')
def test_registered_version_is_documented(tmp_path, pipeline, train_data):
    """Sin validación, la versión describe el entrenamiento; con validación, suma sus métricas y el enlace al run."""
    path = tmp_path / 'pipeline.joblib'
    pipeline.save(path)
    setup_mlflow('prueba')

    with mlflow.start_run():
        without_validation = register_pipeline(path, train_data)
    _log_validation(train_data, PARAMS)
    validation = find_validation(PARAMS, train_data)
    with mlflow.start_run():
        with_validation = register_pipeline(path, train_data, validation)

    client = MlflowClient()
    assert 'LightGBM' in client.get_registered_model(REGISTERED_MODEL_NAME).description
    first = client.get_model_version(REGISTERED_MODEL_NAME, without_validation)
    assert f'{format_number(len(train_data))} transacciones' in first.description
    assert first.tags['umbral'] == str(pipeline.threshold)
    assert 'run_validacion' not in first.tags
    second = client.get_model_version(REGISTERED_MODEL_NAME, with_validation)
    assert '78,9%' in second.description
    assert second.tags['run_validacion'] == validation.info.run_id
    assert second.tags['ganancia_pct_maxima'] == '78.88'


@pytest.mark.usefixtures('temp_mlflow')
def test_find_validation_requires_same_params_and_data(train_data, data):
    """Solo sirve una validación hecha con los mismos parámetros, el mismo umbral y el mismo dataset."""
    setup_mlflow('prueba')
    assert find_validation(PARAMS, train_data) is None

    matching = _log_validation(train_data, PARAMS)
    _log_validation(train_data, {**PARAMS, 'umbral': 0.15})
    other_data = _log_validation(data, PARAMS)

    assert find_validation(PARAMS, train_data).info.run_id == matching
    assert find_validation(PARAMS, data).info.run_id == other_data
    assert find_validation({**PARAMS, 'umbral': 0.3}, train_data) is None


@pytest.mark.usefixtures('temp_mlflow')
def test_promotion_gate(tmp_path, pipeline, train_data):
    """Sin validación no se promueve; sin champion, sí; con champion, solo si no empeora más allá de la tolerancia."""
    path = tmp_path / 'pipeline.joblib'
    pipeline.save(path)
    setup_mlflow('prueba')
    assert should_promote(None)[0] is False

    _log_validation(train_data, PARAMS, profit=78.88)
    first_validation = find_validation(PARAMS, train_data)
    assert should_promote(first_validation)[0] is True
    with mlflow.start_run():
        champion = register_pipeline(path, train_data, first_validation)

    _log_validation(train_data, PARAMS, profit=78.5)
    assert should_promote(find_validation(PARAMS, train_data))[0] is True
    _log_validation(train_data, PARAMS, profit=78.0)
    worse = find_validation(PARAMS, train_data)
    promote, reason = should_promote(worse)
    assert promote is False
    assert '78.88%' in reason

    with mlflow.start_run():
        register_pipeline(path, train_data, worse, promote=promote)
    assert MlflowClient().get_model_version_by_alias(REGISTERED_MODEL_NAME, PRODUCTION_ALIAS).version == champion


@pytest.mark.parametrize(('value', 'decimals', 'expected'), [
    (150000, 0, '150.000'), (78.88, 1, '78,9'), (0.15, 2, '0,15'), (1234567.891, 2, '1.234.567,89'), (0.890, 3, '0,890'),
])
def test_format_number_in_spanish(value, decimals, expected):
    """Punto de miles y coma decimal, como se escriben los números en español."""
    assert format_number(value, decimals) == expected
