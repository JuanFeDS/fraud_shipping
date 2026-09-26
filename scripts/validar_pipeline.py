"""Valida el pipeline productivo con CV: en cada fold se ajusta con train y predice validación como datos nuevos."""

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

from fraude_shipping.experimentacion.validacion import ResultadoValidacion, metricas_fold
from fraude_shipping.features import cargar_datos, crear_folds
from fraude_shipping.ganancia import umbral_optimo
from fraude_shipping.produccion.pipeline import PARAMETROS_LIGHTGBM, UMBRAL, PipelineFraude
from fraude_shipping.registro import (
    ETIQUETA_DESCRIPCION, NOMBRE_RUN_VALIDACION, configurar_mlflow, formatear_numero, registrar_dataset,
)

RUTA_DATOS = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'dataset.csv'
# Los mismos folds nuevos con los que se validó el tuning en el notebook 04
SEMILLA_VALIDACION = 7
NOMBRE_EXPERIMENTO = 'fraude_shipping'


def validar(datos, folds):
    """Predicciones out-of-fold del pipeline, umbral óptimo y métricas por fold con el umbral de producción."""
    probabilidad_oof = np.zeros(len(datos))
    pipelines = []
    for numero, (indices_train, indices_validacion) in enumerate(folds, start=1):
        pipeline = PipelineFraude().ajustar(datos.iloc[indices_train])
        probabilidad_oof[indices_validacion] = pipeline.predecir_probabilidad(datos.iloc[indices_validacion])
        pipelines.append(pipeline)
        print(f'Fold {numero}/{len(folds)} listo')

    metricas_por_fold = pd.DataFrame([
        metricas_fold(
            datos['fraude'].iloc[indices_validacion].to_numpy(),
            datos['monto'].iloc[indices_validacion].to_numpy(),
            probabilidad_oof[indices_validacion],
            UMBRAL,
        )
        for _, indices_validacion in folds
    ])
    umbral = umbral_optimo(datos['fraude'], datos['monto'], probabilidad_oof)
    return ResultadoValidacion(probabilidad_oof, umbral, metricas_por_fold, pipelines)


def registrar_validacion(datos, ruta_datos, resultado):
    """Registra la validación como run de MLflow: parámetros, dataset, métricas resumidas y por fold."""
    configurar_mlflow(NOMBRE_EXPERIMENTO)
    resumen = resultado.resumen
    descripcion = (
        f'Validación del pipeline productivo completo con 5 folds nuevos (semilla {SEMILLA_VALIDACION}): en cada fold '
        'las tablas de j y de países se ajustan solo con train, como ocurriría en producción. '
        f"Ganancia {formatear_numero(resumen['ganancia_pct_maxima_media'], 1)}% "
        f"± {formatear_numero(resumen['ganancia_pct_maxima_desvio'], 1)} de la máxima, "
        f"AUC-ROC {formatear_numero(resumen['auc_roc_media'], 3)}, umbral óptimo {formatear_numero(resumen['umbral'], 2)}."
    )
    with mlflow.start_run(run_name=NOMBRE_RUN_VALIDACION):
        mlflow.set_tags({
            'etapa': 'validacion_pipeline', 'modelo': 'lightgbm', 'decision': 'elegido', 'conjunto_features': 'candidatas',
            'validacion': 'folds_nuevos', 'origen': 'scripts/validar_pipeline.py', ETIQUETA_DESCRIPCION: descripcion,
        })
        mlflow.log_params({**PARAMETROS_LIGHTGBM, 'umbral': UMBRAL, 'semilla_folds': SEMILLA_VALIDACION})
        registrar_dataset(datos, 'dataset', 'training', fuente=ruta_datos)
        mlflow.log_metrics(resumen)
        mlflow.log_table(resultado.metricas_por_fold.rename_axis('fold').reset_index(), 'metricas_por_fold.json')


def main():
    """Corre la validación e imprime las métricas por fold y su resumen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datos', type=Path, default=RUTA_DATOS, help='CSV con transacciones etiquetadas')
    parser.add_argument('--mlflow', action='store_true', help='Registra la validación como run de MLflow')
    argumentos = parser.parse_args()

    datos = cargar_datos(argumentos.datos)
    resultado = validar(datos, crear_folds(datos['fraude'], semilla=SEMILLA_VALIDACION))

    print(f'\nMétricas por fold con el umbral de producción ({UMBRAL}):')
    print(resultado.metricas_por_fold.round(3).to_string())
    print('\nResumen:')
    print(pd.Series(resultado.resumen).round(3).to_string())
    if argumentos.mlflow:
        registrar_validacion(datos, argumentos.datos, resultado)
        print('Validación registrada en MLflow (run validacion_pipeline)')


if __name__ == '__main__':
    main()
