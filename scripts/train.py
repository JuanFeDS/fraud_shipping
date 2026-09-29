"""Entrena el pipeline productivo con todo el dataset y lo guarda como un único artefacto."""

import argparse
from pathlib import Path

import mlflow

from fraude_shipping.features import load_data
from fraude_shipping.production.pipeline import MODEL_PATH, FraudPipeline
from fraude_shipping.registry import (
    DESCRIPTION_TAG, PRODUCTION_ALIAS, REGISTERED_MODEL_NAME, find_validation, format_number, log_dataset,
    register_pipeline, setup_mlflow, should_promote,
)

DATA_PATH = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'dataset.csv'
EXPERIMENT_NAME = 'fraude_shipping'


def log_training(data, data_path, pipeline, model_path):
    """Registra el entrenamiento y el pipeline como nueva versión; pasa a producción solo si supera el gate."""
    setup_mlflow(EXPERIMENT_NAME)
    params = {**pipeline.params, 'umbral': pipeline.threshold}
    validation = find_validation(params, data)
    promote, reason = should_promote(validation)
    description = (
        f'Entrenamiento del pipeline productivo con todo el dataset ({format_number(len(data))} transacciones) y los '
        f'hiperparámetros tuneados. Se registra como nueva versión de {REGISTERED_MODEL_NAME}; recibe el alias '
        f'"{PRODUCTION_ALIAS}" solo si supera el gate de promoción ({reason}).'
    )
    with mlflow.start_run(run_name='entrenamiento_pipeline'):
        mlflow.set_tags({
            'etapa': 'produccion', 'modelo': 'lightgbm', 'decision': 'produccion' if promote else 'no_promovido',
            'conjunto_features': 'candidatas_sin_perfil_onp',
            'validacion': validation.info.run_id if validation else 'sin_validacion',
            'origen': 'scripts/train.py', DESCRIPTION_TAG: description,
        })
        mlflow.log_params(params)
        log_dataset(data, 'dataset', 'training', source=data_path)
        return register_pipeline(model_path, data, validation, promote), promote, reason


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
        registered_version, promote, reason = log_training(data, args.data, pipeline, args.model)
        status = f'con alias "{PRODUCTION_ALIAS}"' if promote else 'sin alias: no supera el gate de promoción'
        print(f'Registrado como {REGISTERED_MODEL_NAME} v{registered_version} {status} ({reason})')


if __name__ == '__main__':
    main()
