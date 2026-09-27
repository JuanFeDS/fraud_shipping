"""Scoring batch: agrega la probabilidad de fraude y la decisión a cada transacción de un CSV."""

import argparse
from pathlib import Path

from fraude_shipping.features import cargar_datos
from fraude_shipping.produccion.pipeline import RUTA_MODELO, PipelineFraude


def main():
    """Carga el pipeline, predice sobre el CSV de entrada y escribe el resultado."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entrada', type=Path, required=True, help='CSV con transacciones (misma estructura que el dataset)')
    parser.add_argument('--salida', type=Path, required=True, help='CSV de salida con probabilidad_fraude y decision')
    parser.add_argument('--modelo', type=Path, default=RUTA_MODELO, help='Pipeline guardado por entrenar.py')
    argumentos = parser.parse_args()

    transacciones = cargar_datos(argumentos.entrada)
    pipeline = PipelineFraude.cargar(argumentos.modelo)
    resultado = transacciones.join(pipeline.predecir(transacciones))

    argumentos.salida.parent.mkdir(parents=True, exist_ok=True)
    resultado.to_csv(argumentos.salida, index=False)
    rechazadas = (resultado['decision'] == 'rechazar').mean()
    print(f'{len(resultado):,} transacciones evaluadas ({rechazadas:.1%} rechazadas) -> {argumentos.salida}')


if __name__ == '__main__':
    main()
