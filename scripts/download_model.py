"""Descarga el pipeline de una versión del model registry (por defecto, la de producción) para empaquetarlo en la API."""

import argparse
from pathlib import Path

from fraude_shipping.production.pipeline import MODEL_PATH
from fraude_shipping.registry import PRODUCTION_ALIAS, connect_mlflow, download_pipeline


def main():
    """Conecta con MLflow, descarga el pipeline pedido y muestra de qué versión viene."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', default=PRODUCTION_ALIAS, help='Alias o número de versión a descargar')
    parser.add_argument('--model', type=Path, default=MODEL_PATH, help='Ruta donde se guarda el pipeline')
    args = parser.parse_args()

    connect_mlflow()
    metadata = download_pipeline(args.model, args.version)
    print(f"{metadata['modelo']} v{metadata['version']} (alias {metadata['alias']}) descargado en {args.model}")


if __name__ == '__main__':
    main()
