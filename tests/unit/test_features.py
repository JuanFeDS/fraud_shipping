"""Construcción de features, tasas de fraude por categoría y folds."""

import numpy as np
import pandas as pd
import pytest

from fraude_shipping.features import (
    SUAVIZADO,
    agrupar_paises,
    ajustar_tasa_fraude,
    aplicar_tasa_fraude,
    cargar_datos,
    columnas_categoricas,
    construir_features,
    crear_folds,
    crear_perfil_onp,
    obtener_paises_frecuentes,
    preparar_categoricas,
    tasa_fraude_oof,
)


def test_cargar_datos_lee_la_fecha_como_datetime(tmp_path, datos):
    """La fecha del CSV se convierte a datetime para poder extraer la hora."""
    ruta = tmp_path / 'datos.csv'
    datos.to_csv(ruta, index=False)
    assert pd.api.types.is_datetime64_any_dtype(cargar_datos(ruta)['fecha'])


def test_features_de_tiempo_y_flags(datos):
    """Hora, día de la semana y flags de valores especiales."""
    features = construir_features(datos)
    assert (features['hora'] == datos['fecha'].dt.hour).all()
    assert (features['dia_semana'] == datos['fecha'].dt.dayofweek).all()
    assert (features['f_negativo'] == (datos['f'] < 0)).all()
    assert (features['d_tope'] == (datos['d'] == 50)).all()
    assert (features['bc_nulo'] == datos['b'].isna()).all()
    assert (features['j_frecuencia'] == datos['j'].map(datos['j'].value_counts())).all()


def test_monto_entero():
    """Un monto sin centavos se marca como entero."""
    datos = pd.DataFrame({'monto': [10.0, 10.5, 7.01]})
    features = construir_features(_con_columnas_minimas(datos))
    assert list(features['monto_entero']) == [1, 0, 0]


def test_ratio_con_denominador_cero_es_nulo(datos):
    """Dividir por un historial en cero da nulo, no infinito."""
    datos = datos.copy()
    datos.loc[0, 'l'] = 0
    features = construir_features(datos)
    assert np.isnan(features.loc[0, 'ratio_f_l'])
    assert np.isnan(features.loc[0, 'ratio_m_l'])
    assert np.isfinite(features['ratio_f_l'].dropna()).all()


def test_actividad_reciente_solo_mira_hacia_atras():
    """Cada transacción cuenta solo las previas de su categoría de j, sin importar el orden de las filas."""
    inicio = pd.Timestamp('2020-03-01 10:00')
    datos = _con_columnas_minimas(pd.DataFrame({
        'j': ['x', 'x', 'x', 'y'],
        'fecha': [inicio + pd.Timedelta(minutes=120), inicio, inicio + pd.Timedelta(minutes=30), inicio],
    }))
    features = construir_features(datos)
    assert list(features['j_transacciones_1h']) == [0, 0, 1, 0]
    assert list(features['j_transacciones_24h']) == [2, 0, 1, 0]


def test_paises_frecuentes_y_agrupados():
    """Los países con menos de 100 transacciones pasan a "Otros" y el nulo se mantiene."""
    paises = pd.Series(['AR'] * 100 + ['BR'] * 99)
    assert obtener_paises_frecuentes(paises) == {'AR'}
    agrupados = agrupar_paises(pd.Series(['AR', 'BR', None]), {'AR'})
    assert agrupados.iloc[0] == 'AR'
    assert agrupados.iloc[1] == 'Otros'
    assert pd.isna(agrupados.iloc[2])


def test_perfil_onp_con_nulo_y_n_decimal():
    """El nulo de o es una categoría propia y n se lee como entero aunque llegue como 1.0."""
    datos = pd.DataFrame({'o': [None, 'Y'], 'n': [1.0, 0], 'p': ['Y', 'N']})
    assert list(crear_perfil_onp(datos)) == ['nulo_1_Y', 'Y_0_N']


def test_columnas_categoricas():
    """Solo se reconocen como categóricas las columnas de texto conocidas."""
    assert columnas_categoricas(['a', 'g', 'perfil_onp', 'score', 'g_agrupado']) == ['g', 'perfil_onp', 'g_agrupado']


def test_preparar_categoricas_convierte_el_nulo_en_categoria(datos):
    """Las categóricas pasan a dtype category con "nulo" y las numéricas no cambian."""
    preparados = preparar_categoricas(datos, ['o', 'score'])
    assert isinstance(preparados['o'].dtype, pd.CategoricalDtype)
    assert 'nulo' in preparados['o'].cat.categories
    assert preparados['o'].notna().all()
    assert (preparados['score'] == datos['score']).all()


def test_tasa_suavizada_coincide_con_la_formula():
    """La tasa de cada categoría se acerca a la global según su cantidad de transacciones."""
    categorias = pd.Series(['x', 'x', 'y', 'y'])
    fraude = pd.Series([1, 1, 0, 0])
    tasa, tasa_global = ajustar_tasa_fraude(categorias, fraude)
    assert tasa_global == pytest.approx(0.5)
    assert tasa['x'] == pytest.approx((2 + SUAVIZADO * 0.5) / (2 + SUAVIZADO))
    assert tasa['y'] == pytest.approx((0 + SUAVIZADO * 0.5) / (2 + SUAVIZADO))


def test_categoria_no_vista_recibe_la_tasa_global():
    """Una categoría que no estaba en train toma la tasa global."""
    tasa = pd.Series({'x': 0.3})
    aplicada = aplicar_tasa_fraude(pd.Series(['x', 'nueva']), tasa, 0.05)
    assert list(aplicada) == pytest.approx([0.3, 0.05])


def test_tasa_oof_no_usa_la_etiqueta_de_la_propia_fila(datos):
    """Cambiar la etiqueta de una fila no cambia su propia tasa out-of-fold."""
    folds = crear_folds(datos['fraude'])
    original = tasa_fraude_oof(datos['j'], datos['fraude'], folds)
    fraude_modificado = datos['fraude'].copy()
    fraude_modificado.iloc[0] = 1 - fraude_modificado.iloc[0]
    modificada = tasa_fraude_oof(datos['j'], fraude_modificado, folds)
    assert modificada.iloc[0] == pytest.approx(original.iloc[0])
    assert original.notna().all()


def test_folds_estratificados_y_deterministas(datos):
    """Los folds cubren todas las filas una vez, mantienen la tasa de fraude y dependen solo de la semilla."""
    folds = crear_folds(datos['fraude'])
    validacion = np.concatenate([indices for _, indices in folds])
    assert len(folds) == 5
    assert sorted(validacion) == list(range(len(datos)))
    fraudes_por_fold = [datos['fraude'].iloc[indices].sum() for _, indices in folds]
    assert max(fraudes_por_fold) - min(fraudes_por_fold) <= 1
    assert all(np.array_equal(a[1], b[1]) for a, b in zip(folds, crear_folds(datos['fraude'])))
    assert not np.array_equal(folds[0][1], crear_folds(datos['fraude'], semilla=7)[0][1])


def _con_columnas_minimas(datos):
    """Completa las columnas que construir_features necesita con valores neutros."""
    base = {
        'fecha': pd.Timestamp('2020-03-01'), 'j': 'x', 'monto': 10.0, 'b': 0.5, 'd': 1.0, 'e': 1.0, 'f': 1.0,
        'g': 'AR', 'h': 1, 'l': 1.0, 'm': 1.0, 'n': 1, 'o': 'Y', 'p': 'Y',
    }
    for columna, valor in base.items():
        if columna not in datos:
            datos[columna] = valor
    return datos
