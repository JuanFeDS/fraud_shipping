"""Variables candidatas del notebook 03: tiempo, flags, ratios, actividad reciente y perfil o/n/p."""

import numpy as np
import pandas as pd

from fraude_shipping.experimentation.candidate_features import build_features, build_onp_profile


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


def test_onp_profile_with_null_and_decimal_n():
    """El nulo de o es una categoría propia y n se lee como entero aunque llegue como 1.0."""
    data = pd.DataFrame({'o': [None, 'Y'], 'n': [1.0, 0], 'p': ['Y', 'N']})
    assert list(build_onp_profile(data)) == ['nulo_1_Y', 'Y_0_N']


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
