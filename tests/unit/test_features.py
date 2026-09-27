"""Construcción de features, tasas de fraude por categoría y folds."""

import numpy as np
import pandas as pd
import pytest

from fraude_shipping.features import (
    SMOOTHING,
    apply_fraud_rate,
    build_features,
    build_onp_profile,
    categorical_columns,
    fit_fraud_rate,
    get_frequent_countries,
    group_countries,
    load_data,
    make_folds,
    oof_fraud_rate,
    prepare_categoricals,
)


def test_load_data_parses_date_as_datetime(tmp_path, data):
    """La fecha del CSV se convierte a datetime para poder extraer la hora."""
    path = tmp_path / 'data.csv'
    data.to_csv(path, index=False)
    assert pd.api.types.is_datetime64_any_dtype(load_data(path)['fecha'])


def test_time_features_and_flags(data):
    """Hora, día de la semana y flags de valores especiales."""
    features = build_features(data)
    assert (features['hora'] == data['fecha'].dt.hour).all()
    assert (features['dia_semana'] == data['fecha'].dt.dayofweek).all()
    assert (features['f_negativo'] == (data['f'] < 0)).all()
    assert (features['d_tope'] == (data['d'] == 50)).all()
    assert (features['bc_nulo'] == data['b'].isna()).all()
    assert (features['j_frecuencia'] == data['j'].map(data['j'].value_counts())).all()


def test_whole_amount():
    """Un monto sin centavos se marca como entero."""
    data = pd.DataFrame({'monto': [10.0, 10.5, 7.01]})
    features = build_features(_with_minimal_columns(data))
    assert list(features['monto_entero']) == [1, 0, 0]


def test_ratio_with_zero_denominator_is_null(data):
    """Dividir por un historial en cero da nulo, no infinito."""
    data = data.copy()
    data.loc[0, 'l'] = 0
    features = build_features(data)
    assert np.isnan(features.loc[0, 'ratio_f_l'])
    assert np.isnan(features.loc[0, 'ratio_m_l'])
    assert np.isfinite(features['ratio_f_l'].dropna()).all()


def test_recent_activity_only_looks_back():
    """Cada transacción cuenta solo las previas de su categoría de j, sin importar el orden de las filas."""
    start = pd.Timestamp('2020-03-01 10:00')
    data = _with_minimal_columns(pd.DataFrame({
        'j': ['x', 'x', 'x', 'y'],
        'fecha': [start + pd.Timedelta(minutes=120), start, start + pd.Timedelta(minutes=30), start],
    }))
    features = build_features(data)
    assert list(features['j_transacciones_1h']) == [0, 0, 1, 0]
    assert list(features['j_transacciones_24h']) == [2, 0, 1, 0]


def test_frequent_and_grouped_countries():
    """Los países con menos de 100 transacciones pasan a "Otros" y el nulo se mantiene."""
    countries = pd.Series(['AR'] * 100 + ['BR'] * 99)
    assert get_frequent_countries(countries) == {'AR'}
    grouped = group_countries(pd.Series(['AR', 'BR', None]), {'AR'})
    assert grouped.iloc[0] == 'AR'
    assert grouped.iloc[1] == 'Otros'
    assert pd.isna(grouped.iloc[2])


def test_onp_profile_with_null_and_decimal_n():
    """El nulo de o es una categoría propia y n se lee como entero aunque llegue como 1.0."""
    data = pd.DataFrame({'o': [None, 'Y'], 'n': [1.0, 0], 'p': ['Y', 'N']})
    assert list(build_onp_profile(data)) == ['nulo_1_Y', 'Y_0_N']


def test_categorical_columns():
    """Solo se reconocen como categóricas las columnas de texto conocidas."""
    assert categorical_columns(['a', 'g', 'perfil_onp', 'score', 'g_agrupado']) == ['g', 'perfil_onp', 'g_agrupado']


def test_prepare_categoricals_turns_null_into_category(data):
    """Las categóricas pasan a dtype category con "nulo" y las numéricas no cambian."""
    prepared = prepare_categoricals(data, ['o', 'score'])
    assert isinstance(prepared['o'].dtype, pd.CategoricalDtype)
    assert 'nulo' in prepared['o'].cat.categories
    assert prepared['o'].notna().all()
    assert (prepared['score'] == data['score']).all()


def test_smoothed_rate_matches_formula():
    """La tasa de cada categoría se acerca a la global según su cantidad de transacciones."""
    categories = pd.Series(['x', 'x', 'y', 'y'])
    fraud = pd.Series([1, 1, 0, 0])
    rate, global_rate = fit_fraud_rate(categories, fraud)
    assert global_rate == pytest.approx(0.5)
    assert rate['x'] == pytest.approx((2 + SMOOTHING * 0.5) / (2 + SMOOTHING))
    assert rate['y'] == pytest.approx((0 + SMOOTHING * 0.5) / (2 + SMOOTHING))


def test_unseen_category_gets_global_rate():
    """Una categoría que no estaba en train toma la tasa global."""
    rate = pd.Series({'x': 0.3})
    applied = apply_fraud_rate(pd.Series(['x', 'nueva']), rate, 0.05)
    assert list(applied) == pytest.approx([0.3, 0.05])


def test_oof_rate_does_not_use_own_label(data):
    """Cambiar la etiqueta de una fila no cambia su propia tasa out-of-fold."""
    folds = make_folds(data['fraude'])
    original = oof_fraud_rate(data['j'], data['fraude'], folds)
    modified_fraud = data['fraude'].copy()
    modified_fraud.iloc[0] = 1 - modified_fraud.iloc[0]
    modified = oof_fraud_rate(data['j'], modified_fraud, folds)
    assert modified.iloc[0] == pytest.approx(original.iloc[0])
    assert original.notna().all()


def test_folds_are_stratified_and_deterministic(data):
    """Los folds cubren todas las filas una vez, mantienen la tasa de fraude y dependen solo de la semilla."""
    folds = make_folds(data['fraude'])
    validation = np.concatenate([indices for _, indices in folds])
    assert len(folds) == 5
    assert sorted(validation) == list(range(len(data)))
    frauds_per_fold = [data['fraude'].iloc[indices].sum() for _, indices in folds]
    assert max(frauds_per_fold) - min(frauds_per_fold) <= 1
    assert all(np.array_equal(first[1], second[1]) for first, second in zip(folds, make_folds(data['fraude'])))
    assert not np.array_equal(folds[0][1], make_folds(data['fraude'], seed=7)[0][1])


def _with_minimal_columns(data):
    """Completa las columnas que build_features necesita con valores neutros."""
    defaults = {
        'fecha': pd.Timestamp('2020-03-01'), 'j': 'x', 'monto': 10.0, 'b': 0.5, 'd': 1.0, 'e': 1.0, 'f': 1.0,
        'g': 'AR', 'h': 1, 'l': 1.0, 'm': 1.0, 'n': 1, 'o': 'Y', 'p': 'Y',
    }
    for column, value in defaults.items():
        if column not in data:
            data[column] = value
    return data
