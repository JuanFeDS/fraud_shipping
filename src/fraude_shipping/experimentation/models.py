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

from fraude_shipping.features import SEED, categorical_columns

MIN_CATEGORY_FREQUENCY = 100


def _split_columns(columns):
    """Divide las features en categóricas y numéricas."""
    categoricals = categorical_columns(columns)
    numerics = [column for column in columns if column not in categoricals]
    return categoricals, numerics


def _build_random_forest(columns, params):
    """Random Forest con las categóricas codificadas como enteros; los nulos numéricos los maneja sklearn."""
    categoricals, numerics = _split_columns(columns)
    preprocessing = ColumnTransformer([
        ('categoricas', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categoricals),
        ('numericas', 'passthrough', numerics),
    ])
    return make_pipeline(
        preprocessing, RandomForestClassifier(random_state=SEED, n_jobs=-1, **params)
    )


def _build_logistic_regression(columns, params):
    """Regresión logística con imputación, normalización por cuantiles y one-hot de categorías frecuentes."""
    categoricals, numerics = _split_columns(columns)
    preprocessing = ColumnTransformer([
        ('categoricas', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=MIN_CATEGORY_FREQUENCY), categoricals),
        ('numericas', make_pipeline(
            SimpleImputer(strategy='median', add_indicator=True),
            QuantileTransformer(output_distribution='normal', random_state=SEED),
        ), numerics),
    ])
    return make_pipeline(preprocessing, LogisticRegression(max_iter=2000, **params))


def build_model(model_name, columns, params=None):
    """Instancia el modelo pedido con sus parámetros por defecto más los indicados."""
    params = params or {}
    if model_name == 'lightgbm':
        return lgb.LGBMClassifier(random_state=SEED, verbose=-1, **params)
    if model_name == 'xgboost':
        return XGBClassifier(enable_categorical=True, tree_method='hist', random_state=SEED, **params)
    if model_name == 'catboost':
        return CatBoostClassifier(
            cat_features=categorical_columns(columns), random_seed=SEED, verbose=0, allow_writing_files=False,
            **params,
        )
    if model_name == 'random_forest':
        return _build_random_forest(columns, params)
    if model_name == 'regresion_logistica':
        return _build_logistic_regression(columns, params)
    raise ValueError(f'Modelo no soportado: {model_name}')
