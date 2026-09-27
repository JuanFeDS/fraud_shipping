"""Descarga el pipeline de una versión del model registry (por defecto, la de producción) para empaquetarlo en la API."""

import argparse
from pathlib import Path

from fraude_shipping.produccion.pipeline import RUTA_MODELO
from fraude_shipping.registro import ALIAS_PRODUCCION, conectar_mlflow, descargar_pipeline


def main():
    """Conecta con MLflow, descarga el pipeline pedido y muestra de qué versión viene."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default=ALIAS_PRODUCCION, help='Alias o número de versión a descargar')
    parser.add_argument('--modelo', type=Path, default=RUTA_MODELO, help='Ruta donde se guarda el pipeline')
    argumentos = parser.parse_args()

    conectar_mlflow()
    metadata = descargar_pipeline(argumentos.modelo, argumentos.version)
    print(f"{metadata['modelo']} v{metadata['version']} (alias {metadata['alias']}) descargado en {argumentos.modelo}")


if __name__ == '__main__':
    main()
