"""Función de ganancia del negocio y métricas de decisión."""

import numpy as np
import pandas as pd

PROFIT_MARGIN = 0.25
THRESHOLDS = np.linspace(0.01, 0.99, 99)


def compute_profit(fraud, amount, approved):
    """Ganancia total: +25% del monto por legítima aprobada, -100% por fraude aprobado y 0 si se rechaza."""
    fraud = np.asarray(fraud)
    amount = np.asarray(amount)
    approved = np.asarray(approved)
    legitimate_profit = PROFIT_MARGIN * amount[approved & (fraud == 0)].sum()
    fraud_loss = amount[approved & (fraud == 1)].sum()
    return legitimate_profit - fraud_loss


def profit_curve(fraud, amount, probability, thresholds=THRESHOLDS):
    """Ganancia obtenida con cada umbral de decisión (se aprueba si la probabilidad es menor al umbral)."""
    probability = np.asarray(probability)
    return pd.DataFrame({
        'umbral': thresholds,
        'ganancia': [compute_profit(fraud, amount, probability < threshold) for threshold in thresholds],
    })


def optimal_threshold(fraud, amount, probability, thresholds=THRESHOLDS):
    """Umbral que maximiza la ganancia sobre las probabilidades dadas."""
    curve = profit_curve(fraud, amount, probability, thresholds)
    return curve.loc[curve['ganancia'].idxmax(), 'umbral']


def decision_metrics(fraud, amount, approved):
    """Resultado de negocio de una decisión: ganancia relativa a la máxima posible, aprobación y detección."""
    fraud = np.asarray(fraud)
    approved = np.asarray(approved)
    max_profit = compute_profit(fraud, amount, fraud == 0)
    return {
        'ganancia_pct_maxima': compute_profit(fraud, amount, approved) / max_profit * 100,
        'aprobadas_pct': approved.mean() * 100,
        'fraudes_detectados_pct': (~approved[fraud == 1]).mean() * 100,
    }
