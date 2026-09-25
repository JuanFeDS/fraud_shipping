"""Registro de experimentos en MLflow con backend local."""

from pathlib import Path

import mlflow
import plotly.express as px

from src.ganancia import curva_ganancia
from src.validacion import validacion_cruzada

RAIZ_PROYECTO = Path(__file__).resolve().parents[1]


def configurar_mlflow(nombre_experimento):
    """Apunta MLflow a la base SQLite del proyecto y activa el experimento, creándolo si no existe."""
    mlflow.set_tracking_uri(f'sqlite:///{(RAIZ_PROYECTO / "mlflow.db").as_posix()}')
    if mlflow.get_experiment_by_name(nombre_experimento) is None:
        mlflow.create_experiment(nombre_experimento, artifact_location=(RAIZ_PROYECTO / 'mlruns').as_uri())
    mlflow.set_experiment(nombre_experimento)


def _graficar_curva_ganancia(datos, resultado, nombre_run):
    """Curva de ganancia out-of-fold según el umbral, con el umbral elegido marcado."""
    curva = curva_ganancia(datos['fraude'], datos['monto'], resultado.probabilidad_oof)
    fig = px.line(curva, x='umbral', y='ganancia', title=f'Curva de ganancia — {nombre_run}')
    fig.add_vline(x=resultado.umbral, line_dash='dash', annotation_text=f'umbral {resultado.umbral:.2f}')
    return fig


def ejecutar_experimento(
    datos, nombre_run, features, nombre_modelo, parametros=None, columnas_tasa=(), etiquetas=None, folds=None,
    anidado=False, pesos=None,
):
    """Corre la validación cruzada de una configuración y registra parámetros, métricas y artefactos en MLflow."""
    parametros = parametros or {}
    resultado = validacion_cruzada(datos, features, nombre_modelo, parametros, columnas_tasa, folds, pesos)
    with mlflow.start_run(run_name=nombre_run, nested=anidado):
        mlflow.set_tags({'modelo': nombre_modelo, **(etiquetas or {})})
        mlflow.log_params({'modelo': nombre_modelo, 'numero_features': len(features), **parametros})
        mlflow.log_dict({'features': list(features), 'columnas_tasa': list(columnas_tasa)}, 'features.json')
        mlflow.log_metrics(resultado.resumen)
        mlflow.log_figure(_graficar_curva_ganancia(datos, resultado, nombre_run), 'curva_ganancia.html')
    return resultado
