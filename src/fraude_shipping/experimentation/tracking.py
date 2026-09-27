"""Registro de experimentos en MLflow: parámetros, métricas, dataset y curva de ganancia de cada configuración."""

import mlflow
import plotly.express as px

from fraude_shipping.experimentation.validation import cross_validate
from fraude_shipping.profit import profit_curve
from fraude_shipping.registry import DESCRIPTION_TAG, log_dataset


def _plot_profit_curve(data, result, run_name):
    """Curva de ganancia out-of-fold según el umbral, con el umbral elegido marcado."""
    curve = profit_curve(data['fraude'], data['monto'], result.oof_probability)
    fig = px.line(curve, x='umbral', y='ganancia', title=f'Curva de ganancia — {run_name}')
    fig.add_vline(x=result.threshold, line_dash='dash', annotation_text=f'umbral {result.threshold:.2f}')
    return fig


def run_experiment(
    data, run_name, features, model_name, params=None, rate_columns=(), tags=None, folds=None,
    nested=False, weights=None, description=None,
):
    """Corre la validación cruzada de una configuración y registra parámetros, métricas y artefactos en MLflow."""
    params = params or {}
    result = cross_validate(data, features, model_name, params, rate_columns, folds, weights)
    with mlflow.start_run(run_name=run_name, nested=nested):
        mlflow.set_tags({'modelo': model_name, **(tags or {})})
        if description:
            mlflow.set_tag(DESCRIPTION_TAG, description)
        mlflow.log_params({'modelo': model_name, 'numero_features': len(features), **params})
        mlflow.log_dict({'features': list(features), 'columnas_tasa': list(rate_columns)}, 'features.json')
        # dict.fromkeys evita columnas repetidas cuando j es feature y además se le calcula la tasa
        log_dataset(data[list(dict.fromkeys([*features, *rate_columns, 'fraude']))], 'features', 'training')
        mlflow.log_metrics(result.summary)
        mlflow.log_figure(_plot_profit_curve(data, result, run_name), 'curva_ganancia.html')
    return result
