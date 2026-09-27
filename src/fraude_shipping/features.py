"""Construcción de features a partir de los hallazgos del notebook 03."""

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
RATIOS = {
    'ratio_f_l': ('f', 'l'),
    'ratio_m_l': ('m', 'l'),
    'ratio_h_l': ('h', 'l'),
    'ratio_d_m': ('d', 'm'),
}
ACTIVITY_WINDOWS = {'j_transacciones_1h': '1h', 'j_transacciones_24h': '24h'}


def load_data(path):
    """Lee el dataset crudo con la fecha como datetime."""
    return pd.read_csv(path, parse_dates=['fecha'])


def _add_recent_activity(data):
    """Transacciones previas de la misma categoría de j en cada ventana; solo mira hacia atrás."""
    ordered = data[['fecha', 'j']].sort_values('fecha')
    original_index = ordered.index
    ordered = ordered.set_index('fecha').assign(transaction=1)
    for name, window in ACTIVITY_WINDOWS.items():
        count = ordered.groupby('j')['transaction'].transform(
            lambda transactions, window=window: transactions.rolling(window).sum() - 1
        )
        data.loc[original_index, name] = count.to_numpy()


def build_features(data):
    """Agrega las features del notebook 03 que no usan la etiqueta; la tasa de fraude de j se calcula por fold."""
    data = data.copy()
    data['hora'] = data['fecha'].dt.hour
    data['dia_semana'] = data['fecha'].dt.dayofweek

    data['j_frecuencia'] = data['j'].map(data['j'].value_counts())
    data['j_monto_relativo'] = data['monto'] / data.groupby('j')['monto'].transform('median')
    _add_recent_activity(data)

    data['bc_nulo'] = data['b'].isna().astype(int)
    data['f_negativo'] = (data['f'] < 0).astype(int)
    data['d_tope'] = (data['d'] == 50).astype(int)
    data['e_cero'] = (data['e'] == 0).astype(int)
    data['monto_entero'] = (data['monto'].mul(100).round() % 100 == 0).astype(int)

    data['g_agrupado'] = group_countries(data['g'], get_frequent_countries(data['g']))

    for name, (numerator, denominator) in RATIOS.items():
        data[name] = data[numerator] / data[denominator].replace(0, np.nan)

    data['perfil_onp'] = build_onp_profile(data)
    return data


def get_frequent_countries(countries):
    """Países con al menos MIN_COUNTRY_TRANSACTIONS transacciones."""
    country_counts = countries.value_counts()
    return set(country_counts[country_counts >= MIN_COUNTRY_TRANSACTIONS].index)


def group_countries(countries, frequent_countries):
    """Reemplaza por "Otros" los países que no están entre los frecuentes; el nulo se mantiene."""
    return countries.where(countries.isin(frequent_countries) | countries.isna(), 'Otros')


def build_onp_profile(data):
    """Combinación de o, n y p, con el nulo de o como categoría propia."""
    return data['o'].fillna('nulo') + '_' + data['n'].astype(int).astype(str) + '_' + data['p']


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
