"""Valida el pipeline productivo con CV: en cada fold se ajusta con train y predice validación como datos nuevos."""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from fraude_shipping.experimentation.validation import ValidationResult, compute_fold_metrics
from fraude_shipping.features import load_data, make_folds
from fraude_shipping.production.pipeline import LIGHTGBM_PARAMS, THRESHOLD, FraudPipeline
from fraude_shipping.profit import optimal_threshold
from fraude_shipping.registry import (
    DESCRIPTION_TAG, VALIDATION_RUN_NAME, format_number, log_dataset, setup_mlflow,
)

DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'dataset.csv'
# Los mismos folds nuevos con los que se validó el tuning en el notebook 04
VALIDATION_SEED = 7
EXPERIMENT_NAME = 'fraude_shipping'


def validate(data, folds):
    """Predicciones out-of-fold del pipeline, umbral óptimo y métricas por fold con el umbral de producción."""
    oof_probability = np.zeros(len(data))
    pipelines = []
    for number, (train_indices, validation_indices) in enumerate(folds, start=1):
        pipeline = FraudPipeline().fit(data.iloc[train_indices])
        oof_probability[validation_indices] = pipeline.predict_proba(data.iloc[validation_indices])
        pipelines.append(pipeline)
        print(f'Fold {number}/{len(folds)} listo')

    fold_metrics = pd.DataFrame([
        compute_fold_metrics(
            data['fraude'].iloc[validation_indices].to_numpy(),
            data['monto'].iloc[validation_indices].to_numpy(),
            oof_probability[validation_indices],
            THRESHOLD,
        )
        for _, validation_indices in folds
    ])
    threshold = optimal_threshold(data['fraude'], data['monto'], oof_probability)
    return ValidationResult(oof_probability, threshold, fold_metrics, pipelines)


def log_validation(data, data_path, result):
    """Registra la validación como run de MLflow: parámetros, dataset, métricas resumidas y por fold."""
    setup_mlflow(EXPERIMENT_NAME)
    summary = result.summary
    description = (
        f'Validación del pipeline productivo completo con 5 folds nuevos (semilla {VALIDATION_SEED}): en cada fold '
        'las tablas de j y de países se ajustan solo con train, como ocurriría en producción. '
        f"Ganancia {format_number(summary['ganancia_pct_maxima_media'], 1)}% "
        f"± {format_number(summary['ganancia_pct_maxima_desvio'], 1)} de la máxima, "
        f"AUC-ROC {format_number(summary['auc_roc_media'], 3)}, umbral óptimo {format_number(summary['umbral'], 2)}."
    )
    with mlflow.start_run(run_name=VALIDATION_RUN_NAME):
        mlflow.set_tags({
            'etapa': 'validacion_pipeline', 'modelo': 'lightgbm', 'decision': 'elegido', 'conjunto_features': 'candidatas',
            'validacion': 'folds_nuevos', 'origen': 'scripts/validate_pipeline.py', DESCRIPTION_TAG: description,
        })
        mlflow.log_params({**LIGHTGBM_PARAMS, 'umbral': THRESHOLD, 'semilla_folds': VALIDATION_SEED})
        log_dataset(data, 'dataset', 'training', source=data_path)
        mlflow.log_metrics(summary)
        mlflow.log_table(result.fold_metrics.rename_axis('fold').reset_index(), 'metricas_por_fold.json')


def main():
    """Corre la validación e imprime las métricas por fold y su resumen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA_PATH, help='CSV con transacciones etiquetadas')
    parser.add_argument('--mlflow', action='store_true', help='Registra la validación como run de MLflow')
    args = parser.parse_args()

    data = load_data(args.data)
    result = validate(data, make_folds(data['fraude'], seed=VALIDATION_SEED))

    print(f'\nMétricas por fold con el umbral de producción ({THRESHOLD}):')
    print(result.fold_metrics.round(3).to_string())
    print('\nResumen:')
    print(pd.Series(result.summary).round(3).to_string())
    if args.mlflow:
        log_validation(data, args.data, result)
        print('Validación registrada en MLflow (run validacion_pipeline)')


if __name__ == '__main__':
    main()
