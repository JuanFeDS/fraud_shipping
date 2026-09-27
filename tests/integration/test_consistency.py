"""Consistencia entre scoring batch, fila a fila y API sobre una muestra del dataset real."""

import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from fraude_shipping.features import load_data
from fraude_shipping.production.api import app
from fraude_shipping.production.pipeline import LIGHTGBM_PARAMS, FraudPipeline

DATA_PATH = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'dataset.csv'
TRAIN_SIZE = 20_000
TEST_SIZE = 200


@pytest.fixture(scope='module', name='real_data')
def fixture_real_data():
    """Muestra del dataset real: train y transacciones no vistas, con los nulos y categorías reales."""
    data = load_data(DATA_PATH).sample(TRAIN_SIZE + TEST_SIZE, random_state=0)
    return data.iloc[:TRAIN_SIZE], data.iloc[TRAIN_SIZE:].drop(columns='fraude')


@pytest.fixture(scope='module', name='real_pipeline')
def fixture_real_pipeline(real_data):
    """Pipeline con los hiperparámetros de producción y menos árboles, para que el test corra en segundos."""
    train, _ = real_data
    return FraudPipeline(params={**LIGHTGBM_PARAMS, 'n_estimators': 50}).fit(train)


def test_row_by_row_matches_batch(real_pipeline, real_data):
    """Predecir de a una transacción da lo mismo que predecir el lote completo."""
    _, test = real_data
    batch = real_pipeline.predict_proba(test)
    row_by_row = [real_pipeline.predict_proba(test.iloc[[position]])[0] for position in range(len(test))]
    np.testing.assert_allclose(row_by_row, batch)


def test_api_matches_batch(real_pipeline, real_data):
    """La API devuelve la misma probabilidad y decisión que el scoring batch."""
    _, test = real_data
    app.state.pipeline = real_pipeline
    client = TestClient(app)
    sample = test.head(20)
    expected = real_pipeline.predict(sample)
    for (_, row), (_, prediction) in zip(sample.iterrows(), expected.iterrows()):
        response = client.post('/predecir', json=json.loads(row.to_json(date_format='iso')))
        assert response.status_code == 200
        assert response.json()['probabilidad_fraude'] == pytest.approx(prediction['probabilidad_fraude'])
        assert response.json()['decision'] == prediction['decision']
