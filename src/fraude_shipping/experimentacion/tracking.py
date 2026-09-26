"""Registro de experimentos en MLflow: parámetros, métricas, dataset y curva de ganancia de cada configuración."""

import mlflow
import plotly.express as px

from fraude_shipping.experimentacion.validacion import validacion_cruzada
from fraude_shipping.ganancia import curva_ganancia
from fraude_shipping.registro import ETIQUETA_DESCRIPCION, registrar_dataset


def _graficar_curva_ganancia(datos, resultado, nombre_run):
    """Curva de ganancia out-of-fold según el umbral, con el umbral elegido marcado."""
    curva = curva_ganancia(datos['fraude'], datos['monto'], resultado.probabilidad_oof)
    fig = px.line(curva, x='umbral', y='ganancia', title=f'Curva de ganancia — {nombre_run}')
    fig.add_vline(x=resultado.umbral, line_dash='dash', annotation_text=f'umbral {resultado.umbral:.2f}')
    return fig


def ejecutar_experimento(
    datos, nombre_run, features, nombre_modelo, parametros=None, columnas_tasa=(), etiquetas=None, folds=None,
    anidado=False, pesos=None, descripcion=None,
):
    """Corre la validación cruzada de una configuración y registra parámetros, métricas y artefactos en MLflow."""
    parametros = parametros or {}
    resultado = validacion_cruzada(datos, features, nombre_modelo, parametros, columnas_tasa, folds, pesos)
    with mlflow.start_run(run_name=nombre_run, nested=anidado):
        mlflow.set_tags({'modelo': nombre_modelo, **(etiquetas or {})})
        if descripcion:
            mlflow.set_tag(ETIQUETA_DESCRIPCION, descripcion)
        mlflow.log_params({'modelo': nombre_modelo, 'numero_features': len(features), **parametros})
        mlflow.log_dict({'features': list(features), 'columnas_tasa': list(columnas_tasa)}, 'features.json')
        # dict.fromkeys evita columnas repetidas cuando j es feature y además se le calcula la tasa
        registrar_dataset(datos[list(dict.fromkeys([*features, *columnas_tasa, 'fraude']))], 'features', 'training')
        mlflow.log_metrics(resultado.resumen)
        mlflow.log_figure(_graficar_curva_ganancia(datos, resultado, nombre_run), 'curva_ganancia.html')
    return resultado
