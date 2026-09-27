"""Validación cruzada estratificada con predicciones out-of-fold y métricas de negocio."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from fraude_shipping.experimentacion.modelos import crear_modelo
from fraude_shipping.features import (
    aplicar_tasa_fraude, ajustar_tasa_fraude, crear_folds, preparar_categoricas, tasa_fraude_oof,
)
from fraude_shipping.ganancia import metricas_decision, umbral_optimo

@dataclass
class ResultadoValidacion:
    """Predicciones out-of-fold, umbral elegido, métricas por fold y modelos entrenados."""

    probabilidad_oof: np.ndarray
    umbral: float
    metricas_por_fold: pd.DataFrame
    modelos: list

    @property
    def resumen(self):
        """Media y desvío de cada métrica entre folds, más el umbral."""
        metricas = {}
        for columna in self.metricas_por_fold.columns:
            metricas[f'{columna}_media'] = float(self.metricas_por_fold[columna].mean())
            metricas[f'{columna}_desvio'] = float(self.metricas_por_fold[columna].std())
        metricas['umbral'] = float(self.umbral)
        return metricas


def agregar_tasas_fraude(train, validacion, columnas_tasa):
    """Tasa de fraude por categoría: out-of-fold interno en train y ajustada con todo train en validación."""
    train = train.copy()
    validacion = validacion.copy()
    folds_internos = crear_folds(train['fraude'])
    for columna in columnas_tasa:
        nombre = f'{columna}_tasa_fraude'
        train[nombre] = tasa_fraude_oof(train[columna], train['fraude'], folds_internos).to_numpy()
        tasa_suavizada, tasa_global = ajustar_tasa_fraude(train[columna], train['fraude'])
        validacion[nombre] = aplicar_tasa_fraude(validacion[columna], tasa_suavizada, tasa_global).to_numpy()
    return train, validacion


def metricas_fold(fraude, monto, probabilidad, umbral):
    """Métricas de ranking y de negocio de un fold."""
    return {
        'auc_roc': roc_auc_score(fraude, probabilidad),
        'auc_pr': average_precision_score(fraude, probabilidad),
        **metricas_decision(fraude, monto, probabilidad < umbral),
    }


def validacion_cruzada(datos, features, nombre_modelo, parametros=None, columnas_tasa=(), folds=None, pesos=None):
    """Entrena el modelo en cada fold (con pesos por fila si se indican), junta las predicciones out-of-fold y elige el umbral que maximiza la ganancia."""
    columnas_modelo = [*features, *(f'{columna}_tasa_fraude' for columna in columnas_tasa)]
    datos = preparar_categoricas(datos, [*features, *columnas_tasa])
    folds = folds or crear_folds(datos['fraude'])

    probabilidad_oof = np.zeros(len(datos))
    modelos = []
    for indices_train, indices_validacion in folds:
        train, validacion = agregar_tasas_fraude(datos.iloc[indices_train], datos.iloc[indices_validacion], columnas_tasa)
        modelo = crear_modelo(nombre_modelo, columnas_modelo, parametros)
        argumentos_fit = {} if pesos is None else {'sample_weight': pesos.iloc[indices_train].to_numpy()}
        modelo.fit(train[columnas_modelo], train['fraude'], **argumentos_fit)
        probabilidad_oof[indices_validacion] = modelo.predict_proba(validacion[columnas_modelo])[:, 1]
        modelos.append(modelo)

    umbral = umbral_optimo(datos['fraude'], datos['monto'], probabilidad_oof)
    metricas_por_fold = pd.DataFrame([
        metricas_fold(
            datos['fraude'].iloc[indices_validacion].to_numpy(),
            datos['monto'].iloc[indices_validacion].to_numpy(),
            probabilidad_oof[indices_validacion],
            umbral,
        )
        for _, indices_validacion in folds
    ])
    return ResultadoValidacion(probabilidad_oof, umbral, metricas_por_fold, modelos)
