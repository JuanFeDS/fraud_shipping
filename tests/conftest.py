"""Fixtures compartidas: un dataset sintético chico con la misma estructura que el dataset real."""

import numpy as np
import pandas as pd
import pytest

from fraude_shipping.produccion.pipeline import PARAMETROS_LIGHTGBM, PipelineFraude

TAMANO_SINTETICO = 600
TAMANO_TRAIN_SINTETICO = 500
# Menos árboles y hojas más chicas: con 500 filas los parámetros de producción casi no generan cortes
PARAMETROS_RAPIDOS = {**PARAMETROS_LIGHTGBM, 'n_estimators': 20, 'num_leaves': 8, 'min_child_samples': 5}


def crear_datos_sinteticos(tamano=TAMANO_SINTETICO, semilla=0):
    """Transacciones sintéticas con nulos, países frecuentes y raros, y fraude asociado a score alto."""
    generador = np.random.default_rng(semilla)
    score = generador.integers(0, 101, tamano)
    datos = pd.DataFrame({
        'a': generador.integers(1, 5, tamano),
        'b': generador.random(tamano),
        'c': generador.uniform(0, 1e5, tamano),
        'd': generador.integers(0, 51, tamano).astype(float),
        'e': generador.uniform(0, 10, tamano),
        'f': generador.integers(-5, 100, tamano).astype(float),
        'g': generador.choice(['AR', 'BR', 'UY', 'US'], tamano, p=[0.5, 0.3, 0.15, 0.05]),
        'h': generador.integers(0, 58, tamano),
        'j': generador.choice([f'cat_{numero}' for numero in range(15)], tamano),
        'k': generador.random(tamano),
        'l': generador.integers(0, 3000, tamano).astype(float),
        'm': generador.integers(0, 1000, tamano).astype(float),
        'n': generador.integers(0, 2, tamano),
        'o': generador.choice(np.array(['Y', 'N', None], dtype=object), tamano),
        'p': generador.choice(['Y', 'N'], tamano),
        'fecha': pd.Timestamp('2020-03-01') + pd.to_timedelta(generador.integers(0, 30 * 24 * 3600, tamano), unit='s'),
        'monto': generador.uniform(1, 500, tamano).round(2),
        'score': score,
        'fraude': (generador.random(tamano) < np.where(score >= 90, 0.4, 0.05)).astype(int),
    })
    filas_con_nulos = datos.sample(frac=0.1, random_state=semilla).index
    datos.loc[filas_con_nulos, ['b', 'c']] = np.nan
    datos.loc[filas_con_nulos[:5], 'g'] = None
    return datos


@pytest.fixture(scope='session', name='datos')
def fixture_datos():
    """Dataset sintético completo, con etiqueta."""
    return crear_datos_sinteticos()


@pytest.fixture(scope='session', name='datos_train')
def fixture_datos_train(datos):
    """Parte del dataset sintético con la que se entrena."""
    return datos.iloc[:TAMANO_TRAIN_SINTETICO]


@pytest.fixture(scope='session', name='datos_nuevos')
def fixture_datos_nuevos(datos):
    """Transacciones que el pipeline no vio, sin etiqueta, como llegarían a producción."""
    return datos.iloc[TAMANO_TRAIN_SINTETICO:].drop(columns='fraude')


@pytest.fixture(scope='session', name='pipeline')
def fixture_pipeline(datos_train):
    """Pipeline chico entrenado con el dataset sintético."""
    return PipelineFraude(parametros=PARAMETROS_RAPIDOS).ajustar(datos_train)
