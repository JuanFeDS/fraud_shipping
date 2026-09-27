"""Scoring batch: agrega la probabilidad de fraude y la decisión a cada transacción de un CSV."""

import argparse
from pathlib import Path

from fraude_shipping.features import load_data
from fraude_shipping.production.pipeline import MODEL_PATH, FraudPipeline


def main():
    """Carga el pipeline, predice sobre el CSV de entrada y escribe el resultado."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='CSV con transacciones (misma estructura que el dataset)')
    parser.add_argument('--output', type=Path, required=True, help='CSV de salida con probabilidad_fraude y decision')
    parser.add_argument('--model', type=Path, default=MODEL_PATH, help='Pipeline guardado por train.py')
    args = parser.parse_args()

    transactions = load_data(args.input)
    pipeline = FraudPipeline.load(args.model)
    result = transactions.join(pipeline.predict(transactions))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    rejected = (result['decision'] == 'rechazar').mean()
    print(f'{len(result):,} transacciones evaluadas ({rejected:.1%} rechazadas) -> {args.output}')


if __name__ == '__main__':
    main()
