"""Función de ganancia del negocio: +25% del monto por legítima aprobada y -100% por fraude aprobado."""

import numpy as np
import pytest

from fraude_shipping.profit import THRESHOLDS, compute_profit, decision_metrics, optimal_threshold, profit_curve

FRAUD = np.array([0, 0, 1, 1])
AMOUNT = np.array([100.0, 200.0, 50.0, 40.0])


def test_profit_with_hand_computed_values():
    """Aprobar una legítima de 100 y un fraude de 50 da 0,25 * 100 - 50."""
    approved = np.array([True, False, True, False])
    assert compute_profit(FRAUD, AMOUNT, approved) == pytest.approx(-25.0)


def test_rejecting_everything_gives_zero():
    """Una transacción rechazada no suma ni resta."""
    assert compute_profit(FRAUD, AMOUNT, np.zeros(4, dtype=bool)) == 0


def test_max_profit_approves_only_legitimate():
    """La ganancia máxima es el 25% de la suma de los montos legítimos."""
    assert compute_profit(FRAUD, AMOUNT, FRAUD == 0) == pytest.approx(75.0)


def test_profit_curve_has_one_value_per_threshold():
    """La curva evalúa la ganancia en cada umbral pedido."""
    probability = np.array([0.1, 0.2, 0.8, 0.9])
    curve = profit_curve(FRAUD, AMOUNT, probability, [0.15, 0.5, 1.0])
    assert list(curve['umbral']) == [0.15, 0.5, 1.0]
    assert list(curve['ganancia']) == pytest.approx([25.0, 75.0, -15.0])


def test_optimal_threshold_separates_fraud_from_legitimate():
    """Con probabilidades que separan perfecto, el umbral óptimo aprueba solo las legítimas."""
    probability = np.array([0.1, 0.1, 0.9, 0.9])
    threshold = optimal_threshold(FRAUD, AMOUNT, probability)
    assert threshold in THRESHOLDS
    assert 0.1 < threshold <= 0.9
    assert compute_profit(FRAUD, AMOUNT, probability < threshold) == pytest.approx(75.0)


def test_decision_metrics():
    """Ganancia relativa a la máxima, % de aprobadas y % de fraudes detectados."""
    approved = np.array([True, False, True, False])
    metrics = decision_metrics(FRAUD, AMOUNT, approved)
    assert metrics['ganancia_pct_maxima'] == pytest.approx(-25.0 / 75.0 * 100)
    assert metrics['aprobadas_pct'] == pytest.approx(50.0)
    assert metrics['fraudes_detectados_pct'] == pytest.approx(50.0)
