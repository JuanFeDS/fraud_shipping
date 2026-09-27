"""Catálogo de modelos comparados en la experimentación."""

import pytest

from fraude_shipping.experimentation.models import build_model
from fraude_shipping.features import prepare_categoricals

FEATURES = ['a', 'b', 'd', 'monto', 'score', 'g', 'o', 'p']
# Pocos árboles o iteraciones para que cada modelo entrene en segundos
FAST_PARAMS = {
    'lightgbm': {'n_estimators': 20},
    'xgboost': {'n_estimators': 20},
    'catboost': {'iterations': 20},
    'random_forest': {'n_estimators': 20},
    'regresion_logistica': {},
}


@pytest.mark.parametrize('model_name', FAST_PARAMS)
def test_each_model_trains_and_predicts_probabilities(data, model_name):
    """Todos los modelos aceptan categóricas y nulos y devuelven una probabilidad por fila."""
    prepared = prepare_categoricals(data, FEATURES)
    model = build_model(model_name, FEATURES, FAST_PARAMS[model_name])
    model.fit(prepared[FEATURES], prepared['fraude'])
    probability = model.predict_proba(prepared[FEATURES])[:, 1]
    assert probability.shape == (len(data),)
    assert ((probability >= 0) & (probability <= 1)).all()


def test_unknown_model():
    """Pedir un modelo que no está en el catálogo es un error explícito."""
    with pytest.raises(ValueError, match='Modelo no soportado'):
        build_model('svm', FEATURES)
