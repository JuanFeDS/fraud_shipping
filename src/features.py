"""Construcción de features a partir de los hallazgos del notebook 03."""

import numpy as np
import pandas as pd

COLUMNAS_CATEGORICAS_ORIGINALES = ['g', 'j', 'o', 'p']
COLUMNAS_ORIGINALES = [
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'monto', 'score',
]
MINIMO_TRANSACCIONES_PAIS = 100
SUAVIZADO = 20
RATIOS = {
    'ratio_f_l': ('f', 'l'),
    'ratio_m_l': ('m', 'l'),
    'ratio_h_l': ('h', 'l'),
    'ratio_d_m': ('d', 'm'),
}
VENTANAS_ACTIVIDAD = {'j_transacciones_1h': '1h', 'j_transacciones_24h': '24h'}


def cargar_datos(ruta):
    """Lee el dataset crudo con la fecha como datetime."""
    return pd.read_csv(ruta, parse_dates=['fecha'])


def _agregar_actividad_reciente(datos):
    """Transacciones previas de la misma categoría de j en cada ventana; solo mira hacia atrás."""
    ordenadas = datos[['fecha', 'j']].sort_values('fecha')
    indice_original = ordenadas.index
    ordenadas = ordenadas.set_index('fecha').assign(transaccion=1)
    for nombre, ventana in VENTANAS_ACTIVIDAD.items():
        conteo = ordenadas.groupby('j')['transaccion'].transform(
            lambda transacciones, ventana=ventana: transacciones.rolling(ventana).sum() - 1
        )
        datos.loc[indice_original, nombre] = conteo.to_numpy()


def construir_features(datos):
    """Agrega las features del notebook 03 que no usan la etiqueta; la tasa de fraude de j se calcula por fold."""
    datos = datos.copy()
    datos['hora'] = datos['fecha'].dt.hour
    datos['dia_semana'] = datos['fecha'].dt.dayofweek

    datos['j_frecuencia'] = datos['j'].map(datos['j'].value_counts())
    datos['j_monto_relativo'] = datos['monto'] / datos.groupby('j')['monto'].transform('median')
    _agregar_actividad_reciente(datos)

    datos['bc_nulo'] = datos['b'].isna().astype(int)
    datos['f_negativo'] = (datos['f'] < 0).astype(int)
    datos['d_tope'] = (datos['d'] == 50).astype(int)
    datos['e_cero'] = (datos['e'] == 0).astype(int)
    datos['monto_entero'] = (datos['monto'].mul(100).round() % 100 == 0).astype(int)

    frecuencia_pais = datos['g'].value_counts()
    paises_frecuentes = frecuencia_pais[frecuencia_pais >= MINIMO_TRANSACCIONES_PAIS].index
    datos['g_agrupado'] = datos['g'].where(datos['g'].isin(paises_frecuentes) | datos['g'].isna(), 'Otros')

    for nombre, (numerador, denominador) in RATIOS.items():
        datos[nombre] = datos[numerador] / datos[denominador].replace(0, np.nan)

    datos['perfil_onp'] = datos['o'].fillna('nulo') + '_' + datos['n'].astype(str) + '_' + datos['p']
    return datos


def columnas_categoricas(columnas):
    """Columnas de texto dentro de una lista de features."""
    return [columna for columna in columnas if columna in [*COLUMNAS_CATEGORICAS_ORIGINALES, 'g_agrupado', 'perfil_onp']]


def preparar_categoricas(datos, columnas):
    """Convierte las categóricas a dtype category, con el nulo como categoría propia."""
    datos = datos.copy()
    for columna in columnas_categoricas(columnas):
        datos[columna] = datos[columna].fillna('nulo').astype('category')
    return datos


def ajustar_tasa_fraude(categorias, fraude):
    """Tasa de fraude suavizada por categoría y tasa global, para aplicar a datos nuevos."""
    tasa_global = fraude.mean()
    estadisticas = fraude.groupby(categorias, observed=True).agg(['sum', 'count'])
    tasa_suavizada = (estadisticas['sum'] + SUAVIZADO * tasa_global) / (estadisticas['count'] + SUAVIZADO)
    return tasa_suavizada, tasa_global


def aplicar_tasa_fraude(categorias, tasa_suavizada, tasa_global):
    """Asigna a cada fila la tasa de su categoría; las categorías no vistas reciben la tasa global."""
    return categorias.map(tasa_suavizada).astype(float).fillna(tasa_global)


def tasa_fraude_oof(categorias, fraude, folds):
    """Tasa de fraude por categoría donde cada fila recibe la calculada con los folds que no la contienen."""
    resultado = pd.Series(np.nan, index=categorias.index)
    for indices_train, indices_validacion in folds:
        tasa_suavizada, tasa_global = ajustar_tasa_fraude(categorias.iloc[indices_train], fraude.iloc[indices_train])
        resultado.iloc[indices_validacion] = aplicar_tasa_fraude(
            categorias.iloc[indices_validacion], tasa_suavizada, tasa_global
        ).to_numpy()
    return resultado
