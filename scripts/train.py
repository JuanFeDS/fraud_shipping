"""Entrena el pipeline productivo con todo el dataset y lo guarda como un único artefacto."""

import argparse
from pathlib import Path

import mlflow

from fraude_shipping.features import cargar_datos
from fraude_shipping.produccion.pipeline import RUTA_MODELO, PipelineFraude
from fraude_shipping.registro import (
    ALIAS_PRODUCCION, ETIQUETA_DESCRIPCION, NOMBRE_MODELO_REGISTRADO, buscar_ultima_validacion, configurar_mlflow,
    formatear_numero, registrar_dataset, registrar_pipeline,
)

RUTA_DATOS = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'dataset.csv'
NOMBRE_EXPERIMENTO = 'fraude_shipping'


def registrar_entrenamiento(datos, ruta_datos, pipeline, ruta_modelo):
    """Registra el entrenamiento como run de MLflow y el pipeline como nueva versión en el model registry."""
    configurar_mlflow(NOMBRE_EXPERIMENTO)
    validacion = buscar_ultima_validacion()
    descripcion = (
        f'Entrenamiento del pipeline productivo con todo el dataset ({formatear_numero(len(datos))} transacciones) y los '
        f'hiperparámetros tuneados. Se registra como nueva versión de {NOMBRE_MODELO_REGISTRADO} con alias '
        f'"{ALIAS_PRODUCCION}".'
    )
    with mlflow.start_run(run_name='entrenamiento_pipeline'):
        mlflow.set_tags({
            'etapa': 'produccion', 'modelo': 'lightgbm', 'decision': 'produccion', 'conjunto_features': 'candidatas',
            'validacion': 'sin_validacion', 'origen': 'scripts/entrenar.py', ETIQUETA_DESCRIPCION: descripcion,
        })
        mlflow.log_params({**pipeline.parametros, 'umbral': pipeline.umbral})
        registrar_dataset(datos, 'dataset', 'training', fuente=ruta_datos)
        return registrar_pipeline(ruta_modelo, datos, validacion)


def main():
    """Lee los datos etiquetados, ajusta el pipeline y lo guarda."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datos', type=Path, default=RUTA_DATOS, help='CSV con transacciones etiquetadas')
    parser.add_argument('--modelo', type=Path, default=RUTA_MODELO, help='Ruta donde se guarda el pipeline')
    parser.add_argument('--mlflow', action='store_true', help='Registra el pipeline en el model registry de MLflow')
    argumentos = parser.parse_args()

    datos = cargar_datos(argumentos.datos)
    pipeline = PipelineFraude().ajustar(datos)
    pipeline.guardar(argumentos.modelo)
    print(f'Pipeline entrenado con {len(datos):,} transacciones y guardado en {argumentos.modelo}')
    if argumentos.mlflow:
        version_registrada = registrar_entrenamiento(datos, argumentos.datos, pipeline, argumentos.modelo)
        print(f'Registrado como {NOMBRE_MODELO_REGISTRADO} v{version_registrada} con alias "{ALIAS_PRODUCCION}"')


if __name__ == '__main__':
    main()
