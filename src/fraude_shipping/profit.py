"""Función de ganancia del negocio y métricas de decisión."""

import numpy as np
import pandas as pd

MARGEN_GANANCIA = 0.25
UMBRALES = np.linspace(0.01, 0.99, 99)


def calcular_ganancia(fraude, monto, aprobada):
    """Ganancia total: +25% del monto por legítima aprobada, -100% por fraude aprobado y 0 si se rechaza."""
    fraude = np.asarray(fraude)
    monto = np.asarray(monto)
    aprobada = np.asarray(aprobada)
    ganancia_legitimas = MARGEN_GANANCIA * monto[aprobada & (fraude == 0)].sum()
    perdida_fraudes = monto[aprobada & (fraude == 1)].sum()
    return ganancia_legitimas - perdida_fraudes


def curva_ganancia(fraude, monto, probabilidad, umbrales=UMBRALES):
    """Ganancia obtenida con cada umbral de decisión (se aprueba si la probabilidad es menor al umbral)."""
    probabilidad = np.asarray(probabilidad)
    return pd.DataFrame({
        'umbral': umbrales,
        'ganancia': [calcular_ganancia(fraude, monto, probabilidad < umbral) for umbral in umbrales],
    })


def umbral_optimo(fraude, monto, probabilidad, umbrales=UMBRALES):
    """Umbral que maximiza la ganancia sobre las probabilidades dadas."""
    curva = curva_ganancia(fraude, monto, probabilidad, umbrales)
    return curva.loc[curva['ganancia'].idxmax(), 'umbral']


def metricas_decision(fraude, monto, aprobada):
    """Resultado de negocio de una decisión: ganancia relativa a la máxima posible, aprobación y detección."""
    fraude = np.asarray(fraude)
    aprobada = np.asarray(aprobada)
    ganancia_maxima = calcular_ganancia(fraude, monto, fraude == 0)
    return {
        'ganancia_pct_maxima': calcular_ganancia(fraude, monto, aprobada) / ganancia_maxima * 100,
        'aprobadas_pct': aprobada.mean() * 100,
        'fraudes_detectados_pct': (~aprobada[fraude == 1]).mean() * 100,
    }
