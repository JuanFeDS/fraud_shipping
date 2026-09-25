"""Catálogo de modelos comparables: todos reciben un DataFrame con categóricas de dtype category."""

import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, QuantileTransformer
from xgboost import XGBClassifier

from src.features import columnas_categoricas

SEMILLA = 42
MINIMO_FRECUENCIA_CATEGORIA = 100


def _separar_columnas(columnas):
    """Divide las features en categóricas y numéricas."""
    categoricas = columnas_categoricas(columnas)
    numericas = [columna for columna in columnas if columna not in categoricas]
    return categoricas, numericas


def _crear_random_forest(columnas, parametros):
    """Random Forest con las categóricas codificadas como enteros; los nulos numéricos los maneja sklearn."""
    categoricas, numericas = _separar_columnas(columnas)
    preprocesamiento = ColumnTransformer([
        ('categoricas', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categoricas),
        ('numericas', 'passthrough', numericas),
    ])
    return make_pipeline(
        preprocesamiento, RandomForestClassifier(random_state=SEMILLA, n_jobs=-1, **parametros)
    )


def _crear_regresion_logistica(columnas, parametros):
    """Regresión logística con imputación, normalización por cuantiles y one-hot de categorías frecuentes."""
    categoricas, numericas = _separar_columnas(columnas)
    preprocesamiento = ColumnTransformer([
        ('categoricas', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=MINIMO_FRECUENCIA_CATEGORIA), categoricas),
        ('numericas', make_pipeline(
            SimpleImputer(strategy='median', add_indicator=True),
            QuantileTransformer(output_distribution='normal', random_state=SEMILLA),
        ), numericas),
    ])
    return make_pipeline(preprocesamiento, LogisticRegression(max_iter=2000, **parametros))


def crear_modelo(nombre_modelo, columnas, parametros=None):
    """Instancia el modelo pedido con sus parámetros por defecto más los indicados."""
    parametros = parametros or {}
    if nombre_modelo == 'lightgbm':
        return lgb.LGBMClassifier(random_state=SEMILLA, verbose=-1, **parametros)
    if nombre_modelo == 'xgboost':
        return XGBClassifier(enable_categorical=True, tree_method='hist', random_state=SEMILLA, **parametros)
    if nombre_modelo == 'catboost':
        return CatBoostClassifier(
            cat_features=columnas_categoricas(columnas), random_seed=SEMILLA, verbose=0, allow_writing_files=False,
            **parametros,
        )
    if nombre_modelo == 'random_forest':
        return _crear_random_forest(columnas, parametros)
    if nombre_modelo == 'regresion_logistica':
        return _crear_regresion_logistica(columnas, parametros)
    raise ValueError(f'Modelo no soportado: {nombre_modelo}')
