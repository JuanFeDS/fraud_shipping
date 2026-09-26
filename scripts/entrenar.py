"""Entrena el pipeline productivo con todo el dataset y lo guarda como un único artefacto."""

import argparse
from pathlib import Path

from fraude_shipping.features import cargar_datos
from fraude_shipping.produccion.pipeline import RUTA_MODELO, PipelineFraude

RUTA_DATOS = Path(__file__).resolve().parents[1] / 'data' / 'raw' / 'dataset.csv'


def main():
    """Lee los datos etiquetados, ajusta el pipeline y lo guarda."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datos', type=Path, default=RUTA_DATOS, help='CSV con transacciones etiquetadas')
    parser.add_argument('--modelo', type=Path, default=RUTA_MODELO, help='Ruta donde se guarda el pipeline')
    argumentos = parser.parse_args()

    datos = cargar_datos(argumentos.datos)
    pipeline = PipelineFraude().ajustar(datos)
    pipeline.guardar(argumentos.modelo)
    print(f'Pipeline entrenado con {len(datos):,} transacciones y guardado en {argumentos.modelo}')


if __name__ == '__main__':
    main()
