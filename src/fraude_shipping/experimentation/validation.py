"""Validación cruzada estratificada con predicciones out-of-fold y métricas de negocio."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraude_shipping.experimentation.models import build_model
from fraude_shipping.features import (
    apply_fraud_rate, fit_fraud_rate, make_folds, oof_fraud_rate, prepare_categoricals,
)
from fraude_shipping.profit import decision_metrics, optimal_threshold

@dataclass
class ValidationResult:
    """Predicciones out-of-fold, umbral elegido, métricas por fold y modelos entrenados."""

    oof_probability: np.ndarray
    threshold: float
    fold_metrics: pd.DataFrame
    models: list

    @property
    def summary(self):
        """Media y desvío de cada métrica entre folds, más el umbral."""
        metrics = {}
        for column in self.fold_metrics.columns:
            metrics[f'{column}_media'] = float(self.fold_metrics[column].mean())
            metrics[f'{column}_desvio'] = float(self.fold_metrics[column].std())
        metrics['umbral'] = float(self.threshold)
        return metrics


def add_fraud_rates(train, validation, rate_columns):
    """Tasa de fraude por categoría: out-of-fold interno en train y ajustada con todo train en validación."""
    train = train.copy()
    validation = validation.copy()
    inner_folds = make_folds(train['fraude'])
    for column in rate_columns:
        name = f'{column}_tasa_fraude'
        train[name] = oof_fraud_rate(train[column], train['fraude'], inner_folds).to_numpy()
        smoothed_rate, global_rate = fit_fraud_rate(train[column], train['fraude'])
        validation[name] = apply_fraud_rate(validation[column], smoothed_rate, global_rate).to_numpy()
    return train, validation


def compute_fold_metrics(fraud, amount, probability, threshold):
    """Métricas de ranking y de negocio de un fold."""
    return {
        'auc_roc': roc_auc_score(fraud, probability),
        'auc_pr': average_precision_score(fraud, probability),
        **decision_metrics(fraud, amount, probability < threshold),
    }


def cross_validate(data, features, model_name, params=None, rate_columns=(), folds=None, weights=None):
    """Entrena el modelo en cada fold (con pesos por fila si se indican), junta las predicciones out-of-fold y elige el umbral que maximiza la ganancia."""
    model_columns = [*features, *(f'{column}_tasa_fraude' for column in rate_columns)]
    data = prepare_categoricals(data, [*features, *rate_columns])
    folds = folds or make_folds(data['fraude'])

    oof_probability = np.zeros(len(data))
    models = []
    for train_indices, validation_indices in folds:
        train, validation = add_fraud_rates(data.iloc[train_indices], data.iloc[validation_indices], rate_columns)
        model = build_model(model_name, model_columns, params)
        fit_kwargs = {} if weights is None else {'sample_weight': weights.iloc[train_indices].to_numpy()}
        model.fit(train[model_columns], train['fraude'], **fit_kwargs)
        oof_probability[validation_indices] = model.predict_proba(validation[model_columns])[:, 1]
        models.append(model)

    threshold = optimal_threshold(data['fraude'], data['monto'], oof_probability)
    fold_metrics = pd.DataFrame([
        compute_fold_metrics(
            data['fraude'].iloc[validation_indices].to_numpy(),
            data['monto'].iloc[validation_indices].to_numpy(),
            oof_probability[validation_indices],
            threshold,
        )
        for _, validation_indices in folds
    ])
    return ValidationResult(oof_probability, threshold, fold_metrics, models)
