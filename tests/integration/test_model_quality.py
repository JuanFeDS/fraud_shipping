"""Regresión de calidad: el pipeline productivo, entrenado con el pasado, predice la última semana del dataset."""

from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from fraude_shipping.features import load_data
from fraude_shipping.production.pipeline import FraudPipeline
from fraude_shipping.profit import compute_profit

DATA_PATH = Path(__file__).resolve().parents[2] / 'data' / 'raw' / 'dataset.csv'
TEST_START = '2020-04-15'
# Hoy el modelo da 79,5% de la ganancia máxima y AUC 0,884 en la última semana (aprobar todo: 67,1%). Los pisos dejan
# ~2 puntos de margen, más que la variación natural (±1,3), para fallar solo ante una regresión real
MIN_PROFIT_PCT = 77.5
MIN_GAIN_OVER_APPROVE_ALL = 10.0
MIN_AUC = 0.87


@pytest.fixture(scope='module', name='last_week')
def fixture_last_week():
    """Etiquetas, montos y decisiones del pipeline de producción sobre la última semana, entrenado solo con el pasado."""
    data = load_data(DATA_PATH)
    past, week = data[data['fecha'] < TEST_START], data[data['fecha'] >= TEST_START]
    pipeline = FraudPipeline().fit(past)
    probability = pipeline.predict_proba(week)
    return week['fraude'], week['monto'], probability, probability < pipeline.threshold


def _profit_pct(fraud, amount, approved):
    return compute_profit(fraud, amount, approved) / compute_profit(fraud, amount, fraud == 0) * 100


def test_profit_does_not_regress(last_week):
    """La ganancia en la semana futura no cae por debajo del piso ni se acerca a la de aprobar todo."""
    fraud, amount, _, approved = last_week
    profit = _profit_pct(fraud, amount, approved)
    approve_all = _profit_pct(fraud, amount, np.ones(len(fraud), dtype=bool))
    assert profit >= MIN_PROFIT_PCT
    assert profit - approve_all >= MIN_GAIN_OVER_APPROVE_ALL


def test_ranking_does_not_regress(last_week):
    """El modelo sigue ordenando las transacciones por riesgo como hoy."""
    fraud, _, probability, _ = last_week
    assert roc_auc_score(fraud, probability) >= MIN_AUC
