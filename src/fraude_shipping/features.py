"""Piezas compartidas por producción y experimentación: datos, folds, países y tasa de fraude por categoría."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

SEED = 42
N_FOLDS = 5
ORIGINAL_CATEGORICAL_COLUMNS = ['g', 'j', 'o', 'p']
ORIGINAL_COLUMNS = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'monto', 'score',
]
MIN_COUNTRY_TRANSACTIONS = 100
SMOOTHING = 20


def load_data(path):
    """Lee el dataset crudo con la fecha como datetime."""
    return pd.read_csv(path, parse_dates=['fecha'])


def get_frequent_countries(countries):
    """Países con al menos MIN_COUNTRY_TRANSACTIONS transacciones."""
    country_counts = countries.value_counts()
    return set(country_counts[country_counts >= MIN_COUNTRY_TRANSACTIONS].index)


def group_countries(countries, frequent_countries):
    """Reemplaza por "Otros" los países que no están entre los frecuentes; el nulo se mantiene."""
    return countries.where(countries.isin(frequent_countries) | countries.isna(), 'Otros')


def categorical_columns(columns):
    """Columnas de texto dentro de una lista de features."""
    return [column for column in columns if column in [*ORIGINAL_CATEGORICAL_COLUMNS, 'g_agrupado', 'perfil_onp']]


def prepare_categoricals(data, columns):
    """Convierte las categóricas a dtype category, con el nulo como categoría propia."""
    data = data.copy()
    for column in categorical_columns(columns):
        data[column] = data[column].fillna('nulo').astype('category')
    return data


def fit_fraud_rate(categories, fraud):
    """Tasa de fraude suavizada por categoría y tasa global, para aplicar a datos nuevos."""
    global_rate = fraud.mean()
    stats = fraud.groupby(categories, observed=True).agg(['sum', 'count'])
    smoothed_rate = (stats['sum'] + SMOOTHING * global_rate) / (stats['count'] + SMOOTHING)
    return smoothed_rate, global_rate


def apply_fraud_rate(categories, smoothed_rate, global_rate):
    """Asigna a cada fila la tasa de su categoría; las categorías no vistas reciben la tasa global."""
    return categories.map(smoothed_rate).astype(float).fillna(global_rate)


def oof_fraud_rate(categories, fraud, folds):
    """Tasa de fraude por categoría donde cada fila recibe la calculada con los folds que no la contienen."""
    result = pd.Series(np.nan, index=categories.index)
    for train_indices, validation_indices in folds:
        smoothed_rate, global_rate = fit_fraud_rate(categories.iloc[train_indices], fraud.iloc[train_indices])
        result.iloc[validation_indices] = apply_fraud_rate(
            categories.iloc[validation_indices], smoothed_rate, global_rate
        ).to_numpy()
    return result


def make_folds(fraud, n_folds=N_FOLDS, seed=SEED):
    """Folds estratificados por la etiqueta; con la semilla por defecto son los mismos del baseline."""
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(len(fraud)), fraud))
