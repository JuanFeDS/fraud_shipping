"""Valida el pipeline productivo con CV: en cada fold se ajusta con train y predice validación como datos nuevos."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fraude_shipping.experimentacion.validacion import ResultadoValidacion, metricas_fold
from fraude_shipping.features import cargar_datos, crear_folds
from fraude_shipping.ganancia import umbral_optimo
from fraude_shipping.produccion.pipeline import UMBRAL, PipelineFraude

RUTA_DATOS = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'dataset.csv'
# Los mismos folds nuevos con los que se validó el tuning en el notebook 04
SEMILLA_VALIDACION = 7


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


def main():
    """Corre la validación e imprime las métricas por fold y su resumen."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datos', type=Path, default=RUTA_DATOS, help='CSV con transacciones etiquetadas')
    argumentos = parser.parse_args()

    datos = cargar_datos(argumentos.datos)
    resultado = validar(datos, crear_folds(datos['fraude'], semilla=SEMILLA_VALIDACION))

    print(f'\nMétricas por fold con el umbral de producción ({UMBRAL}):')
    print(resultado.metricas_por_fold.round(3).to_string())
    print('\nResumen:')
    print(pd.Series(resultado.resumen).round(3).to_string())


if __name__ == '__main__':
    main()
