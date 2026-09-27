"""Catálogo de modelos comparados en la experimentación."""

import pytest

from fraude_shipping.experimentacion.modelos import crear_modelo
from fraude_shipping.features import preparar_categoricas

FEATURES = ['a', 'b', 'd', 'monto', 'score', 'g', 'o', 'p']
# Pocos árboles o iteraciones para que cada modelo entrene en segundos
PARAMETROS_RAPIDOS = {
    'lightgbm': {'n_estimators': 20},
    'xgboost': {'n_estimators': 20},
    'catboost': {'iterations': 20},
    'random_forest': {'n_estimators': 20},
    'regresion_logistica': {},
}


@pytest.mark.parametrize('nombre_modelo', PARAMETROS_RAPIDOS)
def test_cada_modelo_entrena_y_predice_probabilidades(datos, nombre_modelo):
    """Todos los modelos aceptan categóricas y nulos y devuelven una probabilidad por fila."""
    preparados = preparar_categoricas(datos, FEATURES)
    modelo = crear_modelo(nombre_modelo, FEATURES, PARAMETROS_RAPIDOS[nombre_modelo])
    modelo.fit(preparados[FEATURES], preparados['fraude'])
    probabilidad = modelo.predict_proba(preparados[FEATURES])[:, 1]
    assert probabilidad.shape == (len(datos),)
    assert ((probabilidad >= 0) & (probabilidad <= 1)).all()


def test_modelo_desconocido():
    """Pedir un modelo que no está en el catálogo es un error explícito."""
    with pytest.raises(ValueError, match='Modelo no soportado'):
        crear_modelo('svm', FEATURES)
