"""Pipeline productivo: point-in-time, categorías no vistas, umbral y persistencia."""

import copy

import numpy as np
import pandas as pd
import pytest

from fraude_shipping.produccion.pipeline import FEATURES_MODELO, PARAMETROS_LIGHTGBM, UMBRAL, PipelineFraude


def test_valores_por_defecto():
    """Sin argumentos, el pipeline usa los hiperparámetros tuneados y el umbral elegido."""
    pipeline = PipelineFraude()
    assert pipeline.parametros == PARAMETROS_LIGHTGBM
    assert pipeline.umbral == UMBRAL


def test_transformar_devuelve_las_features_del_modelo(pipeline, datos_nuevos):
    """Las columnas salen en el orden con el que se entrenó el modelo."""
    assert list(pipeline.transformar(datos_nuevos).columns) == FEATURES_MODELO


def test_frecuencia_de_j_viene_de_train(pipeline, datos_train, datos_nuevos):
    """La frecuencia de j es la de train, no la del lote que se predice."""
    features = pipeline.transformar(datos_nuevos)
    esperada = datos_nuevos['j'].map(datos_train['j'].value_counts()).fillna(0)
    assert (features['j_frecuencia'] == esperada).all()


def test_prediccion_no_depende_del_resto_del_lote(pipeline, datos_nuevos):
    """Una transacción recibe la misma probabilidad sola que acompañada de otras de su misma categoría."""
    fila = datos_nuevos.iloc[[0]]
    misma_categoria = pd.concat([datos_nuevos.assign(j=fila['j'].iloc[0])] * 3)
    sola = pipeline.predecir_probabilidad(fila)[0]
    en_lote = pipeline.predecir_probabilidad(pd.concat([fila, misma_categoria]))[0]
    assert sola == pytest.approx(en_lote)


def test_fila_a_fila_igual_que_batch(pipeline, datos_nuevos):
    """Predecir de a una transacción da lo mismo que predecir el lote completo."""
    batch = pipeline.predecir_probabilidad(datos_nuevos)
    fila_a_fila = [pipeline.predecir_probabilidad(datos_nuevos.iloc[[posicion]])[0] for posicion in range(20)]
    np.testing.assert_allclose(fila_a_fila, batch[:20])


def test_categorias_no_vistas(pipeline, datos_nuevos):
    """Una j, un país y un valor de o que no estaban en train no rompen el scoring."""
    nueva = datos_nuevos.head(1).assign(j='cat_nueva', g='ZZ', o='X')
    features = pipeline.transformar(nueva).iloc[0]
    assert features['j_tasa_fraude'] == pytest.approx(pipeline.tasa_fraude_global)
    assert features['j_frecuencia'] == 0
    assert features['g_agrupado'] == 'Otros'
    assert pd.isna(features['o'])
    assert 0 <= pipeline.predecir_probabilidad(nueva)[0] <= 1


def test_nulos_numericos_en_una_sola_fila(pipeline, datos_nuevos):
    """Con una sola fila, una columna numérica nula llega como object y se convierte a float."""
    fila = datos_nuevos.head(1).astype(object)
    fila[['b', 'c', 'l']] = None
    assert pipeline.transformar(fila)['b'].dtype == float
    assert 0 <= pipeline.predecir_probabilidad(fila)[0] <= 1


def test_decision_en_el_borde_del_umbral(pipeline, datos_nuevos):
    """Se rechaza cuando la probabilidad alcanza el umbral y se aprueba si queda por debajo."""
    fila = datos_nuevos.head(1)
    probabilidad = pipeline.predecir_probabilidad(fila)[0]
    pipeline_umbral = copy.copy(pipeline)
    pipeline_umbral.umbral = probabilidad
    assert pipeline_umbral.predecir(fila)['decision'].iloc[0] == 'rechazar'
    pipeline_umbral.umbral = probabilidad + 1e-9
    assert pipeline_umbral.predecir(fila)['decision'].iloc[0] == 'aprobar'


def test_predecir_conserva_el_indice(pipeline, datos_nuevos):
    """El resultado se puede unir a las transacciones originales por índice."""
    resultado = pipeline.predecir(datos_nuevos)
    assert resultado.index.equals(datos_nuevos.index)
    assert set(resultado['decision']) <= {'aprobar', 'rechazar'}


def test_guardar_y_cargar_da_las_mismas_predicciones(tmp_path, pipeline, datos_nuevos):
    """El artefacto guardado reproduce exactamente las predicciones, creando la carpeta si no existe."""
    ruta = tmp_path / 'modelos' / 'pipeline.joblib'
    pipeline.guardar(ruta)
    cargado = PipelineFraude.cargar(ruta)
    np.testing.assert_array_equal(cargado.predecir_probabilidad(datos_nuevos), pipeline.predecir_probabilidad(datos_nuevos))
    assert cargado.umbral == pipeline.umbral
