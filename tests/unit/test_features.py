"""Piezas compartidas: carga de datos, países, categóricas, tasas de fraude por categoría y folds."""

import numpy as np
import pandas as pd
import pytest

from fraude_shipping.features import (
    SMOOTHING,
    apply_fraud_rate,
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


def test_frequent_and_grouped_countries():
    """Los países con menos de 100 transacciones pasan a "Otros" y el nulo se mantiene."""
    countries = pd.Series(['AR'] * 100 + ['BR'] * 99)
    assert get_frequent_countries(countries) == {'AR'}
    grouped = group_countries(pd.Series(['AR', 'BR', None]), {'AR'})
    assert grouped.iloc[0] == 'AR'
    assert grouped.iloc[1] == 'Otros'
    assert pd.isna(grouped.iloc[2])


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
