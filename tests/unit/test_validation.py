"""Validación cruzada de la experimentación: predicciones out-of-fold, tasas por fold y métricas."""

import numpy as np
import pandas as pd
import pytest

from fraude_shipping.experimentacion.validacion import agregar_tasas_fraude, metricas_fold, validacion_cruzada
from fraude_shipping.features import ajustar_tasa_fraude, aplicar_tasa_fraude, construir_features, crear_folds
from fraude_shipping.ganancia import UMBRALES

FEATURES = ['a', 'b', 'd', 'monto', 'score', 'g_agrupado', 'perfil_onp', 'hora']
PARAMETROS = {'n_estimators': 20, 'num_leaves': 8, 'min_child_samples': 5}


@pytest.fixture(scope='module', name='datos_features')
def fixture_datos_features(datos):
    """Dataset sintético con las features del notebook 03."""
    return construir_features(datos)


@pytest.fixture(scope='module', name='resultado')
def fixture_resultado(datos_features):
    """Validación cruzada de LightGBM con la tasa de j calculada dentro de cada fold."""
    return validacion_cruzada(datos_features, FEATURES, 'lightgbm', PARAMETROS, columnas_tasa=['j'])


def test_resultado_de_la_validacion(resultado, datos):
    """Una probabilidad por fila, un modelo y una fila de métricas por fold y un umbral de la grilla."""
    assert resultado.probabilidad_oof.shape == (len(datos),)
    assert ((resultado.probabilidad_oof > 0) & (resultado.probabilidad_oof < 1)).all()
    assert len(resultado.modelos) == 5
    assert len(resultado.metricas_por_fold) == 5
    assert resultado.umbral in UMBRALES


def test_resumen_tiene_media_desvio_y_umbral(resultado):
    """El resumen trae media y desvío de cada métrica, más el umbral."""
    resumen = resultado.resumen
    for metrica in resultado.metricas_por_fold.columns:
        assert resumen[f'{metrica}_media'] == pytest.approx(resultado.metricas_por_fold[metrica].mean())
        assert f'{metrica}_desvio' in resumen
    assert resumen['umbral'] == resultado.umbral


def test_folds_indicados(datos_features):
    """Si se pasan folds, se usan esos en lugar de los por defecto."""
    folds = crear_folds(datos_features['fraude'], numero_folds=3)
    resultado = validacion_cruzada(datos_features, FEATURES, 'lightgbm', PARAMETROS, folds=folds)
    assert len(resultado.metricas_por_fold) == 3


def test_pesos_uniformes_no_cambian_el_resultado(resultado, datos_features):
    """Entrenar con todos los pesos en 1 equivale a entrenar sin pesos."""
    pesos = pd.Series(np.ones(len(datos_features)), index=datos_features.index)
    con_pesos = validacion_cruzada(datos_features, FEATURES, 'lightgbm', PARAMETROS, columnas_tasa=['j'], pesos=pesos)
    np.testing.assert_allclose(con_pesos.probabilidad_oof, resultado.probabilidad_oof)


def test_tasas_por_fold_sin_fuga(datos_features):
    """En validación la tasa sale de todo train; en train, out-of-fold; y los datos de entrada no se modifican."""
    train, validacion = datos_features.iloc[:500], datos_features.iloc[500:]
    train_con_tasa, validacion_con_tasa = agregar_tasas_fraude(train, validacion, ['j'])
    tasa, tasa_global = ajustar_tasa_fraude(train['j'], train['fraude'])
    np.testing.assert_allclose(validacion_con_tasa['j_tasa_fraude'], aplicar_tasa_fraude(validacion['j'], tasa, tasa_global))
    assert not np.allclose(train_con_tasa['j_tasa_fraude'], aplicar_tasa_fraude(train['j'], tasa, tasa_global))
    assert 'j_tasa_fraude' not in train.columns


def test_metricas_fold():
    """Métricas de ranking y de negocio de un fold."""
    metricas = metricas_fold(np.array([0, 0, 1, 1]), np.array([10.0, 10.0, 10.0, 10.0]), np.array([0.1, 0.2, 0.8, 0.9]), 0.5)
    assert metricas['auc_roc'] == pytest.approx(1.0)
    assert metricas['auc_pr'] == pytest.approx(1.0)
    assert metricas['ganancia_pct_maxima'] == pytest.approx(100.0)
    assert metricas['fraudes_detectados_pct'] == pytest.approx(100.0)
