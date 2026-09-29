"""Variables candidatas del notebook 03, incluidas las que se descartaron: solo para experimentación, no para producción."""

import numpy as np

from fraude_shipping.features import get_frequent_countries, group_countries

RATIOS = {
    'ratio_f_l': ('f', 'l'),
    'ratio_m_l': ('m', 'l'),
    'ratio_h_l': ('h', 'l'),
    'ratio_d_m': ('d', 'm'),
}
ACTIVITY_WINDOWS = {'j_transacciones_1h': '1h', 'j_transacciones_24h': '24h'}


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


def build_onp_profile(data):
    """Combinación de o, n y p, con el nulo de o como categoría propia."""
    return data['o'].fillna('nulo') + '_' + data['n'].astype(int).astype(str) + '_' + data['p']
