"""Validación cruzada de la experimentación: predicciones out-of-fold, tasas por fold y métricas."""

import numpy as np
import pandas as pd
import pytest

from fraude_shipping.experimentation.validation import add_fraud_rates, compute_fold_metrics, cross_validate
from fraude_shipping.features import apply_fraud_rate, build_features, fit_fraud_rate, make_folds
from fraude_shipping.profit import THRESHOLDS

FEATURES = ['a', 'b', 'd', 'monto', 'score', 'g_agrupado', 'perfil_onp', 'hora']
PARAMS = {'n_estimators': 20, 'num_leaves': 8, 'min_child_samples': 5}


@pytest.fixture(scope='module', name='feature_data')
def fixture_feature_data(data):
    """Dataset sintético con las features del notebook 03."""
    return build_features(data)


@pytest.fixture(scope='module', name='result')
def fixture_result(feature_data):
    """Validación cruzada de LightGBM con la tasa de j calculada dentro de cada fold."""
    return cross_validate(feature_data, FEATURES, 'lightgbm', PARAMS, rate_columns=['j'])


def test_validation_result(result, data):
    """Una probabilidad por fila, un modelo y una fila de métricas por fold y un umbral de la grilla."""
    assert result.oof_probability.shape == (len(data),)
    assert ((result.oof_probability > 0) & (result.oof_probability < 1)).all()
    assert len(result.models) == 5
    assert len(result.fold_metrics) == 5
    assert result.threshold in THRESHOLDS


def test_summary_has_mean_std_and_threshold(result):
    """El resumen trae media y desvío de cada métrica, más el umbral."""
    summary = result.summary
    for metric in result.fold_metrics.columns:
        assert summary[f'{metric}_media'] == pytest.approx(result.fold_metrics[metric].mean())
        assert f'{metric}_desvio' in summary
    assert summary['umbral'] == result.threshold


def test_given_folds_are_used(feature_data):
    """Si se pasan folds, se usan esos en lugar de los por defecto."""
    folds = make_folds(feature_data['fraude'], n_folds=3)
    result = cross_validate(feature_data, FEATURES, 'lightgbm', PARAMS, folds=folds)
    assert len(result.fold_metrics) == 3


def test_uniform_weights_do_not_change_result(result, feature_data):
    """Entrenar con todos los pesos en 1 equivale a entrenar sin pesos."""
    weights = pd.Series(np.ones(len(feature_data)), index=feature_data.index)
    weighted = cross_validate(feature_data, FEATURES, 'lightgbm', PARAMS, rate_columns=['j'], weights=weights)
    np.testing.assert_allclose(weighted.oof_probability, result.oof_probability)


def test_fold_rates_without_leakage(feature_data):
    """En validación la tasa sale de todo train; en train, out-of-fold; y los datos de entrada no se modifican."""
    train, validation = feature_data.iloc[:500], feature_data.iloc[500:]
    train_with_rate, validation_with_rate = add_fraud_rates(train, validation, ['j'])
    rate, global_rate = fit_fraud_rate(train['j'], train['fraude'])
    np.testing.assert_allclose(validation_with_rate['j_tasa_fraude'], apply_fraud_rate(validation['j'], rate, global_rate))
    assert not np.allclose(train_with_rate['j_tasa_fraude'], apply_fraud_rate(train['j'], rate, global_rate))
    assert 'j_tasa_fraude' not in train.columns


def test_compute_fold_metrics():
    """Métricas de ranking y de negocio de un fold."""
    metrics = compute_fold_metrics(np.array([0, 0, 1, 1]), np.array([10.0, 10.0, 10.0, 10.0]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert metrics['auc_roc'] == pytest.approx(1.0)
    assert metrics['auc_pr'] == pytest.approx(1.0)
    assert metrics['ganancia_pct_maxima'] == pytest.approx(100.0)
    assert metrics['fraudes_detectados_pct'] == pytest.approx(100.0)
