"""Pipeline productivo: point-in-time, categorías no vistas, umbral y persistencia."""

import copy

import numpy as np
import pandas as pd
import pytest

from fraude_shipping.production.pipeline import LIGHTGBM_PARAMS, MODEL_FEATURES, THRESHOLD, FraudPipeline


def test_defaults():
    """Sin argumentos, el pipeline usa los hiperparámetros tuneados y el umbral elegido."""
    pipeline = FraudPipeline()
    assert pipeline.params == LIGHTGBM_PARAMS
    assert pipeline.threshold == THRESHOLD


def test_transform_returns_model_features(pipeline, new_data):
    """Las columnas salen en el orden con el que se entrenó el modelo."""
    assert list(pipeline.transform(new_data).columns) == MODEL_FEATURES


def test_j_frequency_comes_from_train(pipeline, train_data, new_data):
    """La frecuencia de j es la de train, no la del lote que se predice."""
    features = pipeline.transform(new_data)
    expected = new_data['j'].map(train_data['j'].value_counts()).fillna(0)
    assert (features['j_frecuencia'] == expected).all()


def test_prediction_does_not_depend_on_rest_of_batch(pipeline, new_data):
    """Una transacción recibe la misma probabilidad sola que acompañada de otras de su misma categoría."""
    row = new_data.iloc[[0]]
    same_category = pd.concat([new_data.assign(j=row['j'].iloc[0])] * 3)
    alone = pipeline.predict_proba(row)[0]
    in_batch = pipeline.predict_proba(pd.concat([row, same_category]))[0]
    assert alone == pytest.approx(in_batch)


def test_row_by_row_matches_batch(pipeline, new_data):
    """Predecir de a una transacción da lo mismo que predecir el lote completo."""
    batch = pipeline.predict_proba(new_data)
    row_by_row = [pipeline.predict_proba(new_data.iloc[[position]])[0] for position in range(20)]
    np.testing.assert_allclose(row_by_row, batch[:20])


def test_unseen_categories(pipeline, new_data):
    """Una j, un país y un valor de o que no estaban en train no rompen el scoring."""
    new_row = new_data.head(1).assign(j='cat_nueva', g='ZZ', o='X')
    features = pipeline.transform(new_row).iloc[0]
    assert features['j_tasa_fraude'] == pytest.approx(pipeline.global_fraud_rate)
    assert features['j_frecuencia'] == 0
    assert features['g_agrupado'] == 'Otros'
    assert pd.isna(features['o'])
    assert 0 <= pipeline.predict_proba(new_row)[0] <= 1


def test_numeric_nulls_in_single_row(pipeline, new_data):
    """Con una sola fila, una columna numérica nula llega como object y se convierte a float."""
    row = new_data.head(1).astype(object)
    row[['b', 'c', 'l']] = None
    assert pipeline.transform(row)['b'].dtype == float
    assert 0 <= pipeline.predict_proba(row)[0] <= 1


def test_decision_at_threshold_boundary(pipeline, new_data):
    """Se rechaza cuando la probabilidad alcanza el umbral y se aprueba si queda por debajo."""
    row = new_data.head(1)
    probability = pipeline.predict_proba(row)[0]
    boundary_pipeline = copy.copy(pipeline)
    boundary_pipeline.threshold = probability
    assert boundary_pipeline.predict(row)['decision'].iloc[0] == 'rechazar'
    boundary_pipeline.threshold = probability + 1e-9
    assert boundary_pipeline.predict(row)['decision'].iloc[0] == 'aprobar'


def test_predict_keeps_index(pipeline, new_data):
    """El resultado se puede unir a las transacciones originales por índice."""
    result = pipeline.predict(new_data)
    assert result.index.equals(new_data.index)
    assert set(result['decision']) <= {'aprobar', 'rechazar'}


def test_save_and_load_give_same_predictions(tmp_path, pipeline, new_data):
    """El artefacto guardado reproduce exactamente las predicciones, creando la carpeta si no existe."""
    path = tmp_path / 'models' / 'pipeline.joblib'
    pipeline.save(path)
    loaded = FraudPipeline.load(path)
    np.testing.assert_array_equal(loaded.predict_proba(new_data), pipeline.predict_proba(new_data))
    assert loaded.threshold == pipeline.threshold
