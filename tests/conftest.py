"""Fixtures compartidas: un dataset sintético chico con la misma estructura que el dataset real."""

import numpy as np
import pandas as pd
import pytest

from fraude_shipping import registry
from fraude_shipping.production.pipeline import LIGHTGBM_PARAMS, FraudPipeline

SYNTHETIC_SIZE = 600
SYNTHETIC_TRAIN_SIZE = 500
# Menos árboles y hojas más chicas: con 500 filas los parámetros de producción casi no generan cortes
FAST_PARAMS = {**LIGHTGBM_PARAMS, 'n_estimators': 20, 'num_leaves': 8, 'min_child_samples': 5}


def make_synthetic_data(size=SYNTHETIC_SIZE, seed=0):
    """Transacciones sintéticas con nulos, países frecuentes y raros, y fraude asociado a score alto."""
    generator = np.random.default_rng(seed)
    score = generator.integers(0, 101, size)
    data = pd.DataFrame({
        'a': generator.integers(1, 5, size),
        'b': generator.random(size),
        'c': generator.uniform(0, 1e5, size),
        'd': generator.integers(0, 51, size).astype(float),
        'e': generator.uniform(0, 10, size),
        'f': generator.integers(-5, 100, size).astype(float),
        'g': generator.choice(['AR', 'BR', 'UY', 'US'], size, p=[0.5, 0.3, 0.15, 0.05]),
        'h': generator.integers(0, 58, size),
        'j': generator.choice([f'cat_{number}' for number in range(15)], size),
        'k': generator.random(size),
        'l': generator.integers(0, 3000, size).astype(float),
        'm': generator.integers(0, 1000, size).astype(float),
        'n': generator.integers(0, 2, size),
        'o': generator.choice(np.array(['Y', 'N', None], dtype=object), size),
        'p': generator.choice(['Y', 'N'], size),
        'fecha': pd.Timestamp('2020-03-01') + pd.to_timedelta(generator.integers(0, 30 * 24 * 3600, size), unit='s'),
        'monto': generator.uniform(1, 500, size).round(2),
        'score': score,
        'fraude': (generator.random(size) < np.where(score >= 90, 0.4, 0.05)).astype(int),
    })
    rows_with_nulls = data.sample(frac=0.1, random_state=seed).index
    data.loc[rows_with_nulls, ['b', 'c']] = np.nan
    data.loc[rows_with_nulls[:5], 'g'] = None
    return data


@pytest.fixture(scope='session', name='data')
def fixture_data():
    """Dataset sintético completo, con etiqueta."""
    return make_synthetic_data()


@pytest.fixture(scope='session', name='train_data')
def fixture_train_data(data):
    """Parte del dataset sintético con la que se entrena."""
    return data.iloc[:SYNTHETIC_TRAIN_SIZE]


@pytest.fixture(scope='session', name='new_data')
def fixture_new_data(data):
    """Transacciones que el pipeline no vio, sin etiqueta, como llegarían a producción."""
    return data.iloc[SYNTHETIC_TRAIN_SIZE:].drop(columns='fraude')


@pytest.fixture(scope='session', name='pipeline')
def fixture_pipeline(train_data):
    """Pipeline chico entrenado con el dataset sintético."""
    return FraudPipeline(params=FAST_PARAMS).fit(train_data)


@pytest.fixture(name='temp_mlflow')
def fixture_temp_mlflow(tmp_path, monkeypatch):
    """Redirige la base, los artefactos y el registry de MLflow a una carpeta temporal para no tocar el mlflow.db real."""
    monkeypatch.setattr(registry, 'PROJECT_ROOT', tmp_path)
    # Si la terminal apunta al MLflow de producción, los tests no deben escribir ahí
    monkeypatch.delenv('MLFLOW_TRACKING_URI', raising=False)
    return tmp_path
