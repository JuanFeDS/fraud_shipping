"""Entrena el pipeline productivo con todo el dataset y lo guarda como un único artefacto."""

import argparse
from pathlib import Path

import mlflow

from fraude_shipping.features import load_data
from fraude_shipping.production.pipeline import MODEL_PATH, FraudPipeline
from fraude_shipping.registry import (
    DESCRIPTION_TAG, PRODUCTION_ALIAS, REGISTERED_MODEL_NAME, find_latest_validation, format_number, log_dataset,
    register_pipeline, setup_mlflow,
)

DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'dataset.csv'
EXPERIMENT_NAME = 'fraude_shipping'


def log_training(data, data_path, pipeline, model_path):
    """Registra el entrenamiento como run de MLflow y el pipeline como nueva versión en el model registry."""
    setup_mlflow(EXPERIMENT_NAME)
    validation = find_latest_validation()
    description = (
        f'Entrenamiento del pipeline productivo con todo el dataset ({format_number(len(data))} transacciones) y los '
        f'hiperparámetros tuneados. Se registra como nueva versión de {REGISTERED_MODEL_NAME} con alias '
        f'"{PRODUCTION_ALIAS}".'
    )
    with mlflow.start_run(run_name='entrenamiento_pipeline'):
        mlflow.set_tags({
            'etapa': 'produccion', 'modelo': 'lightgbm', 'decision': 'produccion', 'conjunto_features': 'candidatas',
            'validacion': 'sin_validacion', 'origen': 'scripts/train.py', DESCRIPTION_TAG: description,
        })
        mlflow.log_params({**pipeline.params, 'umbral': pipeline.threshold})
        log_dataset(data, 'dataset', 'training', source=data_path)
        return register_pipeline(model_path, data, validation)


def main():
    """Lee los datos etiquetados, ajusta el pipeline y lo guarda."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DATA_PATH, help='CSV con transacciones etiquetadas')
    parser.add_argument('--model', type=Path, default=MODEL_PATH, help='Ruta donde se guarda el pipeline')
    parser.add_argument('--mlflow', action='store_true', help='Registra el pipeline en el model registry de MLflow')
    args = parser.parse_args()

    data = load_data(args.data)
    pipeline = FraudPipeline().fit(data)
    pipeline.save(args.model)
    print(f'Pipeline entrenado con {len(data):,} transacciones y guardado en {args.model}')
    if args.mlflow:
        registered_version = log_training(data, args.data, pipeline, args.model)
        print(f'Registrado como {REGISTERED_MODEL_NAME} v{registered_version} con alias "{PRODUCTION_ALIAS}"')


if __name__ == '__main__':
    main()
